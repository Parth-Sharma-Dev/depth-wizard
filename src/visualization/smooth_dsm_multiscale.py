from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, median_filter


def nearest_fill(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(arr)

    if valid.all():
        return arr.copy(), valid

    filled = arr.copy()

    if not valid.any():
        raise ValueError("DSM contains no valid pixels.")

    # Iterative 4-neighbour propagation. This is only used to avoid NaNs
    # contaminating convolution near the outside footprint.
    for _ in range(64):
        missing = ~np.isfinite(filled)
        if not missing.any():
            break

        propagated = filled.copy()

        for dy, dx in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            shifted = np.full_like(
                filled,
                np.nan,
            )

            sy0 = max(0, -dy)
            sy1 = filled.shape[0] - max(0, dy)
            sx0 = max(0, -dx)
            sx1 = filled.shape[1] - max(0, dx)

            dy0 = max(0, dy)
            dy1 = filled.shape[0] - max(0, -dy)
            dx0 = max(0, dx)
            dx1 = filled.shape[1] - max(0, -dx)

            shifted[
                dy0:dy1,
                dx0:dx1,
            ] = filled[
                sy0:sy1,
                sx0:sx1,
            ]

            use = missing & np.isfinite(shifted)
            propagated[use] = shifted[use]

        if np.array_equal(
            np.isfinite(propagated),
            np.isfinite(filled),
        ):
            break

        filled = propagated

    remaining = ~np.isfinite(filled)
    if remaining.any():
        filled[remaining] = np.nanmedian(filled)

    return filled, valid


def multiscale_visual_surface(
    raw: np.ndarray,
    base_sigma: float,
    detail_strength: float,
    median_radius: int,
) -> np.ndarray:
    """
    Build a presentation-only 2.5D surface.

    The key idea is multi-scale relief decomposition:

        coarse/base = Gaussian(raw)
        detail      = raw - coarse/base
        output      = base + alpha * detail

    Keeping alpha well below 1 suppresses narrow spikes while preserving
    broad terrain/building forms.
    """
    filled, valid = nearest_fill(raw)

    if median_radius > 0:
        filled = median_filter(
            filled,
            size=2 * median_radius + 1,
            mode="nearest",
        )

    base = gaussian_filter(
        filled,
        sigma=base_sigma,
        mode="nearest",
    )

    detail = filled - base

    output = (
        base
        + detail_strength * detail
    )

    # One small final smoothing pass makes the remaining canopy ripples
    # visually coherent without destroying large structures.
    output = gaussian_filter(
        output,
        sigma=max(
            0.6,
            base_sigma * 0.35,
        ),
        mode="nearest",
    )

    # Keep the actual source footprint.
    output[~valid] = np.nan

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a visualization-only DSM using multi-scale relief "
            "compression. The analytical DSM is never modified."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mesh-width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--base-sigma",
        type=float,
        default=4.0,
        help=(
            "Gaussian scale in visualization pixels. "
            "Larger = smoother canopy/terrain."
        ),
    )

    parser.add_argument(
        "--detail-strength",
        type=float,
        default=0.20,
        help=(
            "Fraction of high-frequency detail retained. "
            "0 = base only, 1 = raw detail."
        ),
    )

    parser.add_argument(
        "--median-radius",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keep-footprint-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    if args.base_sigma <= 0:
        raise ValueError(
            "--base-sigma must be > 0."
        )

    if not 0 <= args.detail_strength <= 1:
        raise ValueError(
            "--detail-strength must be in [0, 1]."
        )

    with rasterio.open(args.input) as src:
        raw = src.read(1).astype(
            np.float32
        )

        output_width = min(
            args.mesh_width,
            src.width,
        )

        output_height = max(
            2,
            round(
                src.height
                * output_width
                / src.width
            ),
        )

        downsampled = src.read(
            1,
            out_shape=(
                output_height,
                output_width,
            ),
            resampling=Resampling.bilinear,
            masked=True,
        )

        values = downsampled.filled(
            np.nan
        ).astype(np.float32)

        print("=" * 80)
        print(
            "DepthWizard - Multi-Scale Visualization DSM"
        )
        print("=" * 80)
        print(
            f"Input           : {args.input}"
        )
        print(
            f"Output          : {args.output}"
        )
        print(
            f"Output grid     : "
            f"{output_width} x {output_height}"
        )
        print(
            f"Base sigma      : {args.base_sigma}"
        )
        print(
            f"Detail retained : {args.detail_strength}"
        )
        print(
            f"Median radius   : {args.median_radius}"
        )

        visual = multiscale_visual_surface(
            values,
            base_sigma=args.base_sigma,
            detail_strength=args.detail_strength,
            median_radius=args.median_radius,
        )

        if args.keep_footprint_only:
            footprint = src.read(
                1,
                out_shape=(
                    output_height,
                    output_width,
                ),
                resampling=Resampling.nearest,
                masked=True,
            )

            visual[
                footprint.mask
            ] = np.nan

        transform = (
            src.transform
            * src.transform.scale(
                src.width / output_width,
                src.height / output_height,
            )
        )

        profile = {
            "driver": "GTiff",
            "height": output_height,
            "width": output_width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": transform,
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
                visual.astype(
                    np.float32
                ),
                1,
            )

            dst.set_band_description(
                1,
                "DepthWizard visualization DSM - multiscale smoothed",
            )

            dst.update_tags(
                1,
                source_dsm=str(args.input),
                processing=(
                    "visualization_only:"
                    "multiscale_base_plus_compressed_detail"
                ),
                base_sigma=str(args.base_sigma),
                detail_strength=str(args.detail_strength),
            )

    finite = visual[
        np.isfinite(visual)
    ]

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
        "\nOriginal analytical DSM was not modified."
    )


if __name__ == "__main__":
    main()
