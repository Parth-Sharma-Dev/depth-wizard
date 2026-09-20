from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a quick visual QC report for DepthWizard AGL GeoTIFF."
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--agl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.rgb) as rgb_src:
        rgb = rgb_src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb_crs = rgb_src.crs
        rgb_shape = (rgb_src.height, rgb_src.width)

    with rasterio.open(args.agl) as agl_src:
        agl = agl_src.read(1).astype(np.float32)
        agl_crs = agl_src.crs

    if agl.shape != rgb_shape:
        raise ValueError(
            f"RGB/AGL shape mismatch: {rgb_shape} vs {agl.shape}"
        )

    if rgb_crs != agl_crs:
        raise ValueError(
            f"RGB/AGL CRS mismatch: {rgb_crs} vs {agl_crs}"
        )

    # Robust display range so a few extreme pixels do not flatten the map.
    finite = agl[np.isfinite(agl)]

    if finite.size == 0:
        raise ValueError("AGL raster contains no finite values.")

    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))

    if vmax <= vmin:
        vmax = float(finite.max())

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im = axes[1].imshow(
        agl,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(
        f"Predicted AGL\n"
        f"median={np.median(finite):.2f} m | "
        f"max={finite.max():.2f} m"
    )
    axes[1].axis("off")
    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
        label="Estimated AGL (m)",
    )

    axes[2].imshow(rgb)
    axes[2].imshow(
        np.ma.masked_where(
            ~np.isfinite(agl),
            agl,
        ),
        cmap="turbo",
        alpha=0.52,
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("RGB + AGL Overlay")
    axes[2].axis("off")

    plt.suptitle(
        "DepthWizard B1B Georeferenced Inference QC",
        fontsize=16,
    )

    plt.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        args.output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 70)
    print("DepthWizard - AGL Prediction QC")
    print("=" * 70)
    print(f"RGB        : {args.rgb}")
    print(f"AGL        : {args.agl}")
    print(f"CRS        : {rgb_crs}")
    print(f"Shape      : {agl.shape}")
    print(f"Min        : {finite.min():.3f} m")
    print(f"Median     : {np.median(finite):.3f} m")
    print(f"Mean       : {finite.mean():.3f} m")
    print(f"P95        : {np.percentile(finite, 95):.3f} m")
    print(f"P99        : {np.percentile(finite, 99):.3f} m")
    print(f"Max        : {finite.max():.3f} m")
    print(f"Preview    : {args.output}")


if __name__ == "__main__":
    main()
