from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


def read_rgb(path: Path):
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        shape = (src.height, src.width)
    return rgb, profile, crs, transform, shape


def read_raster(path: Path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        return (
            data,
            src.crs,
            src.transform,
            src.width,
            src.height,
            src.nodata,
        )


def align_to_reference(
    data: np.ndarray,
    src_crs,
    src_transform,
    dst_crs,
    dst_transform,
    dst_width: int,
    dst_height: int,
    src_nodata,
) -> np.ndarray:
    out = np.full(
        (dst_height, dst_width),
        np.nan,
        dtype=np.float32,
    )

    reproject(
        source=data,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return out


def stats(name: str, arr: np.ndarray):
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        print(f"{name}: no finite values")
        return

    print(
        f"{name}: "
        f"min={finite.min():.3f} m | "
        f"median={np.median(finite):.3f} m | "
        f"mean={finite.mean():.3f} m | "
        f"P95={np.percentile(finite, 95):.3f} m | "
        f"P99={np.percentile(finite, 99):.3f} m | "
        f"max={finite.max():.3f} m"
    )


def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard DSM quality-control visualization."
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--agl", type=Path, required=True)
    parser.add_argument("--srtm", type=Path, required=True)
    parser.add_argument("--dsm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rgb, _, rgb_crs, rgb_transform, shape = read_rgb(args.rgb)

    agl, agl_crs, agl_transform, agl_w, agl_h, _ = read_raster(args.agl)
    srtm, srtm_crs, srtm_transform, srtm_w, srtm_h, srtm_nodata = read_raster(args.srtm)
    dsm, dsm_crs, dsm_transform, dsm_w, dsm_h, _ = read_raster(args.dsm)

    if (agl_h, agl_w) != shape:
        raise ValueError(
            f"AGL shape mismatch: {agl.shape} vs RGB shape {shape}"
        )

    if (dsm_h, dsm_w) != shape:
        raise ValueError(
            f"DSM shape mismatch: {dsm.shape} vs RGB shape {shape}"
        )

    # Resample SRTM onto the RGB/DSM grid for comparison.
    srtm_aligned = align_to_reference(
        srtm,
        srtm_crs,
        srtm_transform,
        rgb_crs,
        rgb_transform,
        shape[1],
        shape[0],
        srtm_nodata,
    )

    print("=" * 80)
    print("DepthWizard - DSM Quality Control")
    print("=" * 80)
    print(f"RGB CRS  : {rgb_crs}")
    print(f"AGL CRS  : {agl_crs}")
    print(f"SRTM CRS : {srtm_crs}")
    print(f"DSM CRS  : {dsm_crs}")
    print(f"RGB grid : {shape[1]} x {shape[0]}")

    stats("SRTM", srtm_aligned)
    stats("AGL", agl)
    stats("DSM", dsm)

    finite = np.concatenate(
        [
            srtm_aligned[np.isfinite(srtm_aligned)],
            dsm[np.isfinite(dsm)],
        ]
    )

    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))

    if vmax <= vmin:
        vmax = float(finite.max())

    agl_finite = agl[np.isfinite(agl)]
    agl_vmin = float(np.percentile(agl_finite, 1))
    agl_vmax = float(np.percentile(agl_finite, 99))

    if agl_vmax <= agl_vmin:
        agl_vmax = float(agl_finite.max())

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input RGB")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(
        srtm_aligned,
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title("SRTM Aligned to RGB Grid")
    axes[0, 1].axis("off")
    fig.colorbar(
        im1,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label="Elevation (m)",
    )

    im2 = axes[0, 2].imshow(
        agl,
        cmap="turbo",
        vmin=agl_vmin,
        vmax=agl_vmax,
    )
    axes[0, 2].set_title("B1B Estimated AGL")
    axes[0, 2].axis("off")
    fig.colorbar(
        im2,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
        label="AGL (m)",
    )

    im3 = axes[1, 0].imshow(
        dsm,
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title("SRTM-Anchored Estimated DSM")
    axes[1, 0].axis("off")
    fig.colorbar(
        im3,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04,
        label="DSM Elevation (m)",
    )

    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(
        np.ma.masked_where(
            ~np.isfinite(dsm),
            dsm,
        ),
        cmap="terrain",
        alpha=0.50,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title("RGB + DSM Overlay")
    axes[1, 1].axis("off")

    # Difference between DSM and aligned SRTM should approximately recover AGL.
    difference = dsm - srtm_aligned

    diff_finite = difference[np.isfinite(difference)]
    diff_vmin = float(np.percentile(diff_finite, 1))
    diff_vmax = float(np.percentile(diff_finite, 99))

    if diff_vmax <= diff_vmin:
        diff_vmax = float(diff_finite.max())

    im4 = axes[1, 2].imshow(
        difference,
        cmap="turbo",
        vmin=diff_vmin,
        vmax=diff_vmax,
    )
    axes[1, 2].set_title("DSM − Aligned SRTM")
    axes[1, 2].axis("off")
    fig.colorbar(
        im4,
        ax=axes[1, 2],
        fraction=0.046,
        pad=0.04,
        label="Height Difference (m)",
    )

    plt.suptitle(
        "DepthWizard SRTM-Anchored DSM QC",
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

    print(f"\nQC preview: {args.output}")
    print("DSM − SRTM should visually resemble the B1B AGL map.")


if __name__ == "__main__":
    main()
