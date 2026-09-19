from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "gamus_inspection"
)

RGB_PATH = ROOT / "images" / "test" / "DC_03_26_RGB.h5"
AGL_PATH = ROOT / "heights" / "test" / "DC_03_26_AGL.h5"
CLS_PATH = ROOT / "classes" / "test" / "DC_03_26_CLS.h5"


CLASS_NAMES = {
    0: "Others",
    1: "Ground",
    2: "Low vegetation",
    3: "Buildings",
    4: "Water",
    5: "Road",
    6: "Tree",
}


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["image"][:]


def main():

    rgb = load_h5(RGB_PATH)
    agl = load_h5(AGL_PATH)
    cls = load_h5(CLS_PATH)

    cls = cls.astype(np.int64)

    # -----------------------------------------------------
    # Clean AGL for visualization
    # -----------------------------------------------------

    # Negative values are displayed separately.
    agl_valid = agl.copy()
    agl_valid[agl_valid < 0] = np.nan

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 6),
    )

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    # AGL
    im = axes[1].imshow(
        agl_valid,
        cmap="turbo",
    )

    axes[1].set_title(
        "AGL (negative values hidden)"
    )

    axes[1].axis("off")

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Height",
    )

    # Classes
    class_image = axes[2].imshow(
        cls,
        cmap="tab10",
        vmin=0,
        vmax=6,
    )

    axes[2].set_title("Semantic Classes")
    axes[2].axis("off")

    # Class boundaries / high-level overlay
    axes[3].imshow(rgb)

    # Highlight building pixels
    building_mask = cls == 3

    axes[3].contour(
        building_mask,
        levels=[0.5],
        linewidths=0.8,
    )

    axes[3].set_title(
        "RGB + Building Boundaries"
    )

    axes[3].axis("off")

    plt.tight_layout()

    output_path = (
        ROOT / "gamus_sample_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"Saved visualization to:\n"
        f"{output_path}"
    )

    plt.show()

    # -----------------------------------------------------
    # Inspect high-AGL road pixels
    # -----------------------------------------------------

    road_mask = cls == 5

    suspicious_roads = (
        road_mask &
        np.isfinite(agl) &
        (agl > 10)
    )

    count = suspicious_roads.sum()

    print(
        "\nRoad pixels with AGL > 10 m:"
    )
    print(f"Count: {count}")

    if count > 0:

        values = agl[suspicious_roads]

        print(
            f"Min: {values.min():.3f}"
        )

        print(
            f"Max: {values.max():.3f}"
        )

        print(
            f"Mean: {values.mean():.3f}"
        )

        print(
            f"Median: {np.median(values):.3f}"
        )

if __name__ == "__main__":
    main()