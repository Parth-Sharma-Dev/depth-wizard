from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "baseline05"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


def load_model():
    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )

    model = model.to(DEVICE)
    model.eval()

    return processor, model


def infer(image, processor, model):
    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        if DEVICE == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    depth = processed[0]["predicted_depth"]

    # Convert FP16 → FP32 before NumPy operations.
    depth = depth.float().cpu().numpy()

    depth = np.squeeze(depth)

    return depth, elapsed


def save_depth(depth, output_prefix):
    np.save(
        output_prefix.with_suffix(".npy"),
        depth,
    )

    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max > depth_min:
        normalized = (
            (depth - depth_min)
            / (depth_max - depth_min)
        )
    else:
        normalized = np.zeros_like(depth)

    depth_uint8 = (
        normalized * 255
    ).astype(np.uint8)

    Image.fromarray(depth_uint8).save(
        output_prefix.with_suffix(".png")
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DepthWizard - Baseline 0.5")
    print("=" * 70)

    print(f"Device : {DEVICE}")

    if DEVICE == "cuda":
        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {INPUT_DIR}"
        )

    print(
        f"\nFound {len(image_paths)} input images."
    )

    processor, model = load_model()

    results = []

    for index, image_path in enumerate(image_paths, start=1):

        print("\n" + "-" * 70)
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        image = Image.open(image_path).convert("RGB")

        print(f"Image size: {image.size}")

        depth, elapsed = infer(
            image,
            processor,
            model,
        )

        print(
            f"Inference time : {elapsed:.4f} s"
        )

        print(
            f"Min            : {depth.min():.4f}"
        )

        print(
            f"Max            : {depth.max():.4f}"
        )

        print(
            f"Mean           : {depth.mean():.4f}"
        )

        print(
            f"Std            : {depth.std():.4f}"
        )

        output_prefix = (
            OUTPUT_DIR /
            image_path.stem
        )

        save_depth(
            depth,
            output_prefix,
        )

        results.append({
            "image": image_path.name,
            "width": image.width,
            "height": image.height,
            "inference_seconds": elapsed,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_mean": float(depth.mean()),
            "depth_std": float(depth.std()),
        })

    # Save summary
    import csv

    csv_path = OUTPUT_DIR / "summary.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("Baseline 0.5 completed.")
    print(f"Results: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()