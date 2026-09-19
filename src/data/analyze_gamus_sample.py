from pathlib import Path

import h5py
import numpy as np


ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "gamus_inspection"
)

RGB_PATH = ROOT / "images" / "test" / "DC_03_26_RGB.h5"
AGL_PATH = ROOT / "heights" / "test" / "DC_03_26_AGL.h5"
CLS_PATH = ROOT / "classes" / "test" / "DC_03_26_CLS.h5"


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["image"][:]


def print_statistics(name, array):

    array = np.asarray(array)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Shape       : {array.shape}")
    print(f"Dtype       : {array.dtype}")

    if np.issubdtype(array.dtype, np.number):

        finite_mask = np.isfinite(array)

        print(f"Finite      : {finite_mask.mean() * 100:.4f}%")
        print(f"NaN count   : {np.isnan(array).sum()}")
        print(f"Inf count   : {np.isinf(array).sum()}")

        finite_values = array[finite_mask]

        print(f"Min         : {finite_values.min()}")
        print(f"Max         : {finite_values.max()}")
        print(f"Mean        : {finite_values.mean()}")
        print(f"Std         : {finite_values.std()}")

        percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]

        print("\nPercentiles:")

        for p in percentiles:
            print(
                f"  P{p:>3}: "
                f"{np.percentile(finite_values, p):.6f}"
            )


def main():

    print("=" * 70)
    print("GAMUS Sample Numerical Analysis")
    print("=" * 70)

    rgb = load_h5(RGB_PATH)
    agl = load_h5(AGL_PATH)
    cls = load_h5(CLS_PATH)

    print_statistics("RGB", rgb)
    print_statistics("AGL", agl)
    print_statistics("CLS", cls)

    # -----------------------------------------------------
    # RGB channel information
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RGB CHANNELS")
    print("=" * 70)

    for channel, name in enumerate(["R", "G", "B"]):

        values = rgb[:, :, channel]

        print(
            f"{name}: "
            f"min={values.min()}, "
            f"max={values.max()}, "
            f"mean={values.mean():.3f}, "
            f"std={values.std():.3f}"
        )

    # -----------------------------------------------------
    # Unique class values
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    class_values, counts = np.unique(
        cls,
        return_counts=True
    )

    print(f"Number of unique values: {len(class_values)}")

    for value, count in zip(class_values, counts):

        percentage = (
            count / cls.size * 100
        )

        print(
            f"Class {value}: "
            f"{count:,} pixels "
            f"({percentage:.3f}%)"
        )

    # -----------------------------------------------------
    # Height by class
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("AGL STATISTICS BY CLASS")
    print("=" * 70)

    for value in class_values:

        mask = cls == value

        class_heights = agl[mask]

        class_heights = class_heights[
            np.isfinite(class_heights)
        ]

        if len(class_heights) == 0:
            continue

        print(
            f"Class {value}: "
            f"n={len(class_heights):,}, "
            f"min={class_heights.min():.3f}, "
            f"max={class_heights.max():.3f}, "
            f"mean={class_heights.mean():.3f}, "
            f"median={np.median(class_heights):.3f}"
        )


if __name__ == "__main__":
    main()