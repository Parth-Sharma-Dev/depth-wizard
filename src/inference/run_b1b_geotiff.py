from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.depthwizard_multiscale_height import (
    DepthWizardMultiScaleHeightModel,
)

MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
).view(1, 3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DepthWizard B1B tiled inference on a georeferenced RGB GeoTIFF."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--output-tif", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--limit-tiles", type=int, default=0)
    parser.add_argument("--clip-negative", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> DepthWizardMultiScaleHeightModel:
    print("\nLoading DepthWizard B1B...")

    model = DepthWizardMultiScaleHeightModel(
        freeze_backbone=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "height_head_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'height_head_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.height_head.load_state_dict(
        checkpoint["height_head_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def make_positions(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)

    return positions


def feather_window(
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    if overlap <= 0:
        return np.ones(
            (tile_size, tile_size),
            dtype=np.float32,
        )

    border = min(
        overlap,
        tile_size // 2,
    )

    one_d = np.ones(
        tile_size,
        dtype=np.float32,
    )

    ramp = np.linspace(
        0.05,
        1.0,
        border,
        dtype=np.float32,
    )

    one_d[:border] = ramp
    one_d[-border:] = ramp[::-1]

    return np.outer(
        one_d,
        one_d,
    ).astype(np.float32)


def read_tile(
    src: rasterio.io.DatasetReader,
    x: int,
    y: int,
    tile_size: int,
) -> tuple[np.ndarray, int, int]:
    actual_h = min(
        tile_size,
        src.height - y,
    )
    actual_w = min(
        tile_size,
        src.width - x,
    )

    window = Window(
        col_off=x,
        row_off=y,
        width=actual_w,
        height=actual_h,
    )

    rgb = src.read(
        [1, 2, 3],
        window=window,
        out_dtype="uint8",
    )

    rgb = np.transpose(
        rgb,
        (1, 2, 0),
    )

    if actual_h != tile_size or actual_w != tile_size:
        padded = np.zeros(
            (tile_size, tile_size, 3),
            dtype=np.uint8,
        )
        padded[:actual_h, :actual_w] = rgb
        rgb = padded

    return rgb, actual_h, actual_w


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    batch: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    tensor = torch.from_numpy(
        batch.transpose(0, 3, 1, 2).copy()
    ).float() / 255.0

    mean = IMAGE_MEAN.to(
        device=device,
        dtype=tensor.dtype,
    )
    std = IMAGE_STD.to(
        device=device,
        dtype=tensor.dtype,
    )

    tensor = tensor.to(
        device,
        non_blocking=True,
    )

    tensor = (
        tensor - mean
    ) / std

    if use_amp and device.type == "cuda":
        with autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            prediction = model(tensor)
    else:
        prediction = model(tensor)

    return prediction.float().cpu().numpy()


def write_agl_geotiff(
    output_path: Path,
    reference: rasterio.io.DatasetReader,
    agl: np.ndarray,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            agl.astype(np.float32),
            1,
        )
        dst.set_band_description(
            1,
            "DepthWizard predicted AGL",
        )


def main() -> None:
    args = parse_args()

    for path, label in [
        (args.input, "Input GeoTIFF"),
        (args.checkpoint, "Checkpoint"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} does not exist: {path}"
            )

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be > 0")

    if not 0 <= args.overlap < args.tile_size:
        raise ValueError(
            "--overlap must satisfy 0 <= overlap < tile-size"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("DepthWizard - B1B GeoTIFF Tiled Inference")
    print("=" * 80)
    print(f"Input       : {args.input}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Device      : {device}")
    print(f"Tile        : {args.tile_size}")
    print(f"Overlap     : {args.overlap}")
    print(f"Batch       : {args.batch_size}")
    print(f"AMP         : {not args.no_amp}")

    if device.type == "cuda":
        print(
            f"GPU         : {torch.cuda.get_device_name(0)}"
        )

    with rasterio.open(args.input) as src:
        if src.count < 3:
            raise ValueError(
                "Input GeoTIFF must contain at least 3 bands."
            )

        if src.crs is None:
            raise ValueError(
                "Input GeoTIFF does not have a CRS."
            )

        height = src.height
        width = src.width

        stride = args.tile_size - args.overlap

        xs = make_positions(
            width,
            args.tile_size,
            stride,
        )
        ys = make_positions(
            height,
            args.tile_size,
            stride,
        )

        coordinates = [
            (x, y)
            for y in ys
            for x in xs
        ]

        total_tiles = len(coordinates)

        if args.limit_tiles > 0:
            coordinates = coordinates[:args.limit_tiles]

        print(
            f"Image grid  : {width} x {height}"
        )
        print(
            f"Tile grid   : {len(xs)} x {len(ys)}"
        )
        print(
            f"Tiles       : {len(coordinates)} / {total_tiles}"
        )

        model = load_model(
            args.checkpoint,
            device,
        )

        prediction_sum = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        weight_sum = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        blend = feather_window(
            args.tile_size,
            args.overlap,
        )

        batch_images: list[np.ndarray] = []
        batch_meta: list[
            tuple[int, int, int, int]
        ] = []

        processed = 0
        start = time.perf_counter()

        def flush_batch() -> None:
            nonlocal processed

            if not batch_images:
                return

            batch = np.stack(
                batch_images,
                axis=0,
            )

            predictions = predict_batch(
                model,
                batch,
                device,
                use_amp=(
                    not args.no_amp
                ),
            )

            for prediction, meta in zip(
                predictions,
                batch_meta,
            ):
                x, y, actual_h, actual_w = meta

                pred = prediction[
                    :actual_h,
                    :actual_w,
                ]

                if args.clip_negative:
                    pred = np.maximum(
                        pred,
                        0.0,
                    )

                local_blend = blend[
                    :actual_h,
                    :actual_w,
                ]

                prediction_sum[
                    y:y + actual_h,
                    x:x + actual_w,
                ] += pred * local_blend

                weight_sum[
                    y:y + actual_h,
                    x:x + actual_w,
                ] += local_blend

                processed += 1

            batch_images.clear()
            batch_meta.clear()

            elapsed = time.perf_counter() - start
            rate = processed / max(elapsed, 1e-6)

            print(
                f"  {processed:4d}/{len(coordinates):4d} tiles "
                f"| {rate:.2f} tiles/s"
            )

        for x, y in coordinates:
            tile, actual_h, actual_w = read_tile(
                src,
                x,
                y,
                args.tile_size,
            )

            batch_images.append(tile)
            batch_meta.append(
                (
                    x,
                    y,
                    actual_h,
                    actual_w,
                )
            )

            if len(batch_images) >= args.batch_size:
                flush_batch()

        flush_batch()

        agl = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        valid = weight_sum > 0

        agl[valid] = (
            prediction_sum[valid]
            / weight_sum[valid]
        )

        if args.clip_negative:
            agl[valid] = np.maximum(
                agl[valid],
                0.0,
            )

        args.output_npy.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            args.output_npy,
            agl,
        )

        write_agl_geotiff(
            args.output_tif,
            src,
            agl,
        )

        valid_values = agl[
            np.isfinite(agl)
        ]

        elapsed = time.perf_counter() - start

        print("\n" + "=" * 80)
        print("B1B GeoTIFF inference complete")
        print("=" * 80)
        print(f"Tiles processed : {processed}")
        print(f"Total tiles     : {total_tiles}")
        print(f"Elapsed         : {elapsed:.1f} s")

        if valid_values.size:
            print(
                f"AGL min        : {valid_values.min():.3f} m"
            )
            print(
                f"AGL median     : {np.median(valid_values):.3f} m"
            )
            print(
                f"AGL max        : {valid_values.max():.3f} m"
            )
            print(
                f"AGL mean       : {valid_values.mean():.3f} m"
            )

        print(f"AGL NPY        : {args.output_npy}")
        print(f"AGL GeoTIFF    : {args.output_tif}")

        if args.limit_tiles > 0:
            print(
                "\nWARNING: --limit-tiles was used. "
                "Output is only partially predicted."
            )


if __name__ == "__main__":
    main()
