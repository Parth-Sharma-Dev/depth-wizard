from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
)


def fill_nodata_nearest(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr)

    if valid.all():
        return arr.copy(), valid

    # For visualization only, fill invalid pixels with the nearest valid
    # value using repeated dilation-like propagation. We keep the original
    # validity mask so invalid footprint can still be restored afterward.
    filled = arr.copy()

    if not valid.any():
        raise ValueError("DSM contains no valid pixels.")

    # Fast nearest-neighbour style fill using rasterio's resampling machinery
    # is cumbersome in-place, so use iterative neighbor propagation.
    for _ in range(32):
        missing = ~np.isfinite(filled)
        if not missing.any():
            break

        candidate = filled.copy()

        for dy, dx in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            shifted = np.full_like(filled, np.nan)

            y_src0 = max(0, -dy)
            y_src1 = filled.shape[0] - max(0, dy)
            x_src0 = max(0, -dx)
            x_src1 = filled.shape[1] - max(0, dx)

            y_dst0 = max(0, dy)
            y_dst1 = filled.shape[0] - max(0, -dy)
            x_dst0 = max(0, dx)
            x_dst1 = filled.shape[1] - max(0, -dx)

            shifted[
                y_dst0:y_dst1,
                x_dst0:x_dst1,
            ] = filled[
                y_src0:y_src1,
                x_src0:x_src1,
            ]

            use = missing & np.isfinite(shifted)
            candidate[use] = shifted[use]

        if np.array_equal(
            np.isfinite(candidate),
            np.isfinite(filled),
        ):
            break

        filled = candidate

    # Any remaining isolated invalid pixels get the global median.
    remaining = ~np.isfinite(filled)
    if remaining.any():
        filled[remaining] = np.nanmedian(filled)

    return filled, valid


def robust_clip_to_local(
    arr: np.ndarray,
    valid: np.ndarray,
    radius: int,
    max_positive_deviation: float,
    max_negative_deviation: float,
) -> np.ndarray:
    """
    Suppress isolated spikes relative to a local median.

    This is a visualization-only operation.

    A point is replaced by the local median when it is farther from its
    neighbourhood median than the configured positive/negative limits.
    """
    if radius <= 0:
        return arr.copy()

    size = 2 * radius + 1

    local_median = median_filter(
        arr,
        size=size,
        mode="nearest",
    )

    result = arr.copy()

    delta = arr - local_median

    positive_outlier = (
        valid
        & (delta > max_positive_deviation)
    )

    negative_outlier = (
        valid
        & (delta < -max_negative_deviation)
    )

    result[
        positive_outlier | negative_outlier
    ] = local_median[
        positive_outlier | negative_outlier
    ]

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a visualization-only smoothed DSM from DepthWizard's "
            "SRTM-anchored estimated DSM. The original DSM is never modified."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw DSM GeoTIFF.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Visualization-only DSM GeoTIFF.",
    )

    parser.add_argument(
        "--mesh-width",
        type=int,
        default=512,
        help="Horizontal resolution for the visualization DSM.",
    )

    parser.add_argument(
        "--median-radius",
        type=int,
        default=2,
        help="Median filter radius after downsampling.",
    )

    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.25,
        help="Gaussian smoothing sigma in visualization pixels.",
    )

    parser.add_argument(
        "--spike-radius",
        type=int,
        default=2,
        help="Local median radius used for spike suppression.",
    )

    parser.add_argument(
        "--max-positive-spike",
        type=float,
        default=12.0,
        help=(
            "Maximum allowed local positive deviation in metres "
            "before replacing a pixel by the local median."
        ),
    )

    parser.add_argument(
        "--max-negative-spike",
        type=float,
        default=8.0,
        help=(
            "Maximum allowed local negative deviation in metres "
            "before replacing a pixel by the local median."
        ),
    )

    parser.add_argument(
        "--keep-footprint-only",
        action="store_true",
        help=(
            "Restore the original invalid footprint as NoData after "
            "smoothing."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    if args.mesh_width < 64:
        raise ValueError(
            "--mesh-width is too small; use at least 64."
        )

    with rasterio.open(args.input) as src:
        raw = src.read(1).astype(np.float32)
        original_profile = src.profile.copy()
        original_crs = src.crs

        if original_crs is None:
            raise ValueError(
                "Input DSM has no CRS."
            )

        original_valid = np.isfinite(raw)

        target_width = min(
            args.mesh_width,
            src.width,
        )
        target_height = max(
            2,
            round(
                src.height
                * target_width
                / src.width
            ),
        )

        downsampled = src.read(
            1,
            out_shape=(
                target_height,
                target_width,
            ),
            resampling=Resampling.bilinear,
            masked=True,
        )

        values = downsampled.filled(np.nan).astype(
            np.float32
        )

        resized_valid = np.isfinite(values)

        filled, valid = fill_nodata_nearest(values)

        print("=" * 80)
        print("DepthWizard - Visualization DSM Smoothing")
        print("=" * 80)
        print(f"Input              : {args.input}")
        print(f"Output             : {args.output}")
        print(
            f"Output grid        : "
            f"{target_width} x {target_height}"
        )
        print(
            f"Median radius      : "
            f"{args.median_radius}"
        )
        print(
            f"Gaussian sigma     : "
            f"{args.gaussian_sigma}"
        )
        print(
            f"Spike radius       : "
            f"{args.spike_radius}"
        )
        print(
            f"Positive threshold : "
            f"{args.max_positive_spike:.2f} m"
        )
        print(
            f"Negative threshold : "
            f"{args.max_negative_spike:.2f} m"
        )

        # 1) Suppress isolated high/low spikes.
        filtered = robust_clip_to_local(
            filled,
            valid,
            radius=args.spike_radius,
            max_positive_deviation=args.max_positive_spike,
            max_negative_deviation=args.max_negative_spike,
        )

        # 2) Median smoothing removes remaining small needle-like features.
        if args.median_radius > 0:
            filtered = median_filter(
                filtered,
                size=2 * args.median_radius + 1,
                mode="nearest",
            )

        # 3) Gaussian smoothing makes canopy/terrain transitions coherent.
        if args.gaussian_sigma > 0:
            filtered = gaussian_filter(
                filtered,
                sigma=args.gaussian_sigma,
                mode="nearest",
            )

        # Use the original raster footprint projected onto the visualization
        # grid. For the current prototype this is generally adequate because
        # the DSM and RGB grids are identical.
        if args.keep_footprint_only:
            footprint = src.read_masks(1)
            footprint_ds = src.read(
                1,
                out_shape=(
                    target_height,
                    target_width,
                ),
                resampling=Resampling.nearest,
            )

            footprint_valid = (
                footprint_ds != 0
            )

            filtered[
                ~footprint_valid
            ] = np.nan

        profile = {
            "driver": "GTiff",
            "height": target_height,
            "width": target_width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform
            * src.transform.scale(
                src.width / target_width,
                src.height / target_height,
            ),
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
        }

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            args.output,
            "w",
            **profile,
        ) as dst:
            dst.write(
                filtered.astype(np.float32),
                1,
            )

            dst.set_band_description(
                1,
                "Visualization-smoothed DepthWizard DSM",
            )

            dst.update_tags(
                1,
                source_dsm=str(args.input),
                processing=(
                    "visualization_only:"
                    "local_spike_suppression+"
                    "median+gaussian"
                ),
            )

    finite = filtered[np.isfinite(filtered)]

    print("\nResult:")
    print(
        f"  min    : {finite.min():.3f} m"
    )
    print(
        f"  median : {np.median(finite):.3f} m"
    )
    print(
        f"  P95    : {np.percentile(finite, 95):.3f} m"
    )
    print(
        f"  P99    : {np.percentile(finite, 99):.3f} m"
    )
    print(
        f"  max    : {finite.max():.3f} m"
    )

    print(
        "\nIMPORTANT: "
        "This file is for visualization only. "
        "The original analytical DSM is unchanged."
    )


if __name__ == "__main__":
    main()
