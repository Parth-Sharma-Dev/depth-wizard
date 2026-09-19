from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = PROJECT_ROOT / "data" / "input" / "snowy_mountains.png"
DEPTH_PATH = PROJECT_ROOT / "data" / "output" / "baseline05" / "snowy_mountains.npy"


def main() -> None:

    image = Image.open(IMAGE_PATH).convert("RGB")
    depth = np.load(DEPTH_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(image)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(depth, cmap="turbo")
    axes[1].set_title("Baseline 0.5 Relative Depth")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()