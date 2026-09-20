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

from src.models.depthwizard_multiscale_height import DepthWizardMultiScaleHeightModel

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Improved B1B GeoTIFF inference with Gaussian overlap blending and optional footprint masking."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-npy", type=Path, required=True)
    p.add_argument("--output-tif", type=Path, required=True)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--blend-sigma", type=float, default=0.18)
    p.add_argument("--mask-white", action="store_true")
    p.add_argument("--white-threshold", type=int, default=250)
    p.add_argument("--clip-negative", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--limit-tiles", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    print("\nLoading DepthWizard B1B...")
    model = DepthWizardMultiScaleHeightModel(freeze_backbone=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "height_head_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing height_head_state_dict. Keys: {list(ckpt)}")
    model.height_head.load_state_dict(ckpt["height_head_state_dict"], strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    if "metrics" in ckpt:
        print(f"Checkpoint metrics: {ckpt['metrics']}")
    return model


def make_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    pos = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if pos[-1] != last:
        pos.append(last)
    return pos


def gaussian_blend_window(tile_size: int, sigma_fraction: float) -> np.ndarray:
    if not 0 < sigma_fraction <= 1:
        raise ValueError("--blend-sigma must be in (0,1].")
    center = (tile_size - 1) / 2.0
    sigma = max(tile_size * sigma_fraction, 1.0)
    axis = np.arange(tile_size, dtype=np.float32)
    one_d = np.exp(-0.5 * ((axis - center) / sigma) ** 2).astype(np.float32)
    one_d /= max(float(one_d.max()), 1e-12)
    one_d = np.maximum(one_d, 1e-4)
    return np.outer(one_d, one_d).astype(np.float32)


def read_tile(src, x: int, y: int, tile_size: int):
    h = min(tile_size, src.height - y)
    w = min(tile_size, src.width - x)
    window = Window(x, y, w, h)
    rgb = src.read([1, 2, 3], window=window, out_dtype="uint8")
    rgb = np.transpose(rgb, (1, 2, 0))
    if h != tile_size or w != tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        padded[:h, :w] = rgb
        rgb = padded
    return rgb, h, w


def predict_batch(model, batch, device, use_amp):
    x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).float() / 255.0
    mean = IMAGE_MEAN.to(device=device, dtype=x.dtype)
    std = IMAGE_STD.to(device=device, dtype=x.dtype)
    x = (x.to(device, non_blocking=True) - mean) / std
    if use_amp and device.type == "cuda":
        with autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)
    else:
        y = model(x)
    return y.detach().float().cpu().numpy()


def write_agl_geotiff(path: Path, ref, agl: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref.profile.copy()
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=np.nan,
                   compress="deflate", predictor=3)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(agl.astype(np.float32), 1)
        dst.set_band_description(1, "DepthWizard B1B predicted AGL")


def main() -> None:
    a = parse_args()
    for p, label in ((a.input, "Input"), (a.checkpoint, "Checkpoint")):
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p}")
    if a.tile_size <= 0 or not (0 <= a.overlap < a.tile_size):
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if a.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if not 0 <= a.white_threshold <= 255:
        raise ValueError("white-threshold must be in [0,255]")

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("DepthWizard - Improved B1B GeoTIFF Inference")
    print("=" * 80)
    print(f"Input           : {a.input}")
    print(f"Checkpoint      : {a.checkpoint}")
    print(f"Device          : {device}")
    print(f"Tile            : {a.tile_size}")
    print(f"Overlap         : {a.overlap}")
    print(f"Blend sigma     : {a.blend_sigma}")
    print(f"Batch           : {a.batch_size}")
    print(f"White mask      : {a.mask_white}")
    print(f"White threshold : {a.white_threshold}")
    print(f"AMP             : {not a.no_amp}")
    if device.type == "cuda":
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    with rasterio.open(a.input) as src:
        if src.count < 3:
            raise ValueError("Input GeoTIFF must contain at least 3 bands.")
        if src.crs is None:
            raise ValueError("Input GeoTIFF does not have a CRS.")

        H, W = src.height, src.width
        stride = a.tile_size - a.overlap
        xs = make_positions(W, a.tile_size, stride)
        ys = make_positions(H, a.tile_size, stride)
        coords = [(x, y) for y in ys for x in xs]
        total_tiles = len(coords)
        if a.limit_tiles > 0:
            coords = coords[:a.limit_tiles]

        print(f"Image grid      : {W} x {H}")
        print(f"Tile grid       : {len(xs)} x {len(ys)}")
        print(f"Tiles           : {len(coords)} / {total_tiles}")

        footprint = np.ones((H, W), dtype=bool)
        if a.mask_white:
            rgb_full = src.read([1, 2, 3], out_dtype="uint8")
            white = np.all(rgb_full >= a.white_threshold, axis=0)
            footprint &= ~white
        print(f"Valid footprint : {footprint.mean() * 100.0:.2f}%")

        model = load_model(a.checkpoint, device)
        accum = np.zeros((H, W), dtype=np.float32)
        weights = np.zeros((H, W), dtype=np.float32)
        blend = gaussian_blend_window(a.tile_size, a.blend_sigma)

        batch_images = []
        batch_meta = []
        processed = 0
        start = time.perf_counter()

        def flush():
            nonlocal processed
            if not batch_images:
                return
            preds = predict_batch(model, np.stack(batch_images), device, not a.no_amp)
            for pred, meta in zip(preds, batch_meta):
                x, y, h, w = meta
                p = pred[:h, :w]
                if a.clip_negative:
                    p = np.maximum(p, 0.0)
                bw = blend[:h, :w].copy()
                if a.mask_white:
                    bw[~footprint[y:y+h, x:x+w]] = 0.0
                accum[y:y+h, x:x+w] += p * bw
                weights[y:y+h, x:x+w] += bw
                processed += 1
            batch_images.clear()
            batch_meta.clear()
            elapsed = time.perf_counter() - start
            print(f"  {processed:4d}/{len(coords):4d} tiles | {processed/max(elapsed,1e-6):.2f} tiles/s")

        for x, y in coords:
            tile, h, w = read_tile(src, x, y, a.tile_size)
            batch_images.append(tile)
            batch_meta.append((x, y, h, w))
            if len(batch_images) >= a.batch_size:
                flush()
        flush()

        agl = np.full((H, W), np.nan, dtype=np.float32)
        valid = weights > 0
        agl[valid] = accum[valid] / weights[valid]
        if a.mask_white:
            agl[~footprint] = np.nan
        if a.clip_negative:
            finite = np.isfinite(agl)
            agl[finite] = np.maximum(agl[finite], 0.0)

        a.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.output_npy, agl)
        write_agl_geotiff(a.output_tif, src, agl)

        vals = agl[np.isfinite(agl)]
        elapsed = time.perf_counter() - start
        print("\n" + "=" * 80)
        print("Improved B1B inference complete")
        print("=" * 80)
        print(f"Tiles processed : {processed}")
        print(f"Total tiles     : {total_tiles}")
        print(f"Elapsed         : {elapsed:.1f} s")
        if vals.size:
            print(f"AGL min        : {vals.min():.3f} m")
            print(f"AGL median     : {np.median(vals):.3f} m")
            print(f"AGL mean       : {vals.mean():.3f} m")
            print(f"AGL P95        : {np.percentile(vals,95):.3f} m")
            print(f"AGL P99        : {np.percentile(vals,99):.3f} m")
            print(f"AGL max        : {vals.max():.3f} m")
        print(f"AGL NPY        : {a.output_npy}")
        print(f"AGL GeoTIFF    : {a.output_tif}")
        if a.limit_tiles > 0:
            print("WARNING: --limit-tiles was used. Output is only partially predicted.")


if __name__ == "__main__":
    main()
