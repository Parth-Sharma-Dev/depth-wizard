from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# Model and output paths
MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "input" / "satellite.png"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEPTH_NPY_PATH = OUTPUT_DIR / "baseline0_depth.npy"
DEPTH_PNG_PATH = OUTPUT_DIR / "baseline0_depth.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:

    print("=" * 60)
    print("DepthWizard - Baseline 0")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_PATH}\n"
            "Place a satellite image at data/input/satellite.jpg"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load input image
    print("\nLoading image...")

    image = Image.open(INPUT_PATH).convert("RGB")

    print(f"Image size: {image.size}")

    # Load depth model
    print("\nLoading depth model...")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    # Run depth estimation
    print("\nRunning inference...")

    inputs = processor(
        images=image,
        return_tensors="pt"
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
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start


    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed[0]["predicted_depth"]

    depth = predicted_depth.float().cpu().numpy()

    depth = np.squeeze(depth)

    # Inspect depth stats
    print("\nDepth statistics:")

    print(f"Inference time: {elapsed:.4f} seconds")
    print(f"Shape : {depth.shape}")
    print(f"Min   : {depth.min():.6f}")
    print(f"Max   : {depth.max():.6f}")
    print(f"Mean  : {depth.mean():.6f}")
    print(f"Std   : {depth.std():.6f}")

    # Save raw and visual outputs
    np.save(DEPTH_NPY_PATH, depth)

    depth_min = depth.min()
    depth_max = depth.max()

    if depth_max > depth_min:

        depth_normalized = (
            (depth - depth_min)
            / (depth_max - depth_min)
        )

    else:

        depth_normalized = np.zeros_like(depth)

    depth_uint8 = (
        depth_normalized * 255
    ).astype(np.uint8)

    depth_image = Image.fromarray(depth_uint8)

    depth_image.save(DEPTH_PNG_PATH)

    print("\nOutput files:")
    print(f"Raw depth : {DEPTH_NPY_PATH}")
    print(f"Visual map: {DEPTH_PNG_PATH}")

    print("\nBaseline 0 completed successfully.")


if __name__ == "__main__":
    main()