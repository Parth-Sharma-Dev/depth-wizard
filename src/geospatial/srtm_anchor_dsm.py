from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def load_agl(path: Path) -> np.ndarray:
    agl = np.load(path).astype(np.float32)

    if agl.ndim != 2:
        raise ValueError(
            f"Expected AGL .npy to be 2-D, got {agl.shape}"
        )

    return agl


def resample_dem_to_rgb(
    dem_path: Path,
    rgb_src: rasterio.io.DatasetReader,
) -> np.ndarray:
    if rgb_src.crs is None:
        raise ValueError(
            "RGB GeoTIFF has no CRS; cannot align SRTM."
        )

    with rasterio.open(dem_path) as dem:
        destination = np.full(
            (rgb_src.height, rgb_src.width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=rasterio.band(dem, 1),
            destination=destination,
            src_transform=dem.transform,
            src_crs=dem.crs,
            dst_transform=rgb_src.transform,
            dst_crs=rgb_src.crs,
            src_nodata=dem.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype SRTM-anchored DSM generator.\n\n"
            "Inputs:\n"
            "  1) georeferenced RGB GeoTIFF\n"
            "  2) B1B predicted AGL .npy on the same pixel grid\n"
            "  3) local SRTM DEM GeoTIFF\n\n"
            "Output:\n"
            "  SRTM + predicted AGL DSM GeoTIFF"
        )
    )

    parser.add_argument(
        "--rgb",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--agl",
        type=Path,
        required=True,
        help="B1B predicted AGL NumPy array, HxW.",
    )
    parser.add_argument(
        "--srtm",
        type=Path,
        required=True,
        help="Local SRTM DEM GeoTIFF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale on predicted AGL.",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.0,
        help="Optional additive offset on predicted AGL.",
    )
    parser.add_argument(
        "--clip-negative-agl",
        action="store_true",
        help="Set negative predicted AGL to zero.",
    )

    args = parser.parse_args()

    for path in (args.rgb, args.agl, args.srtm):
        if not path.exists():
            raise FileNotFoundError(path)

    agl = load_agl(args.agl)

    with rasterio.open(args.rgb) as rgb:

        if rgb.crs is None:
            raise ValueError(
                "RGB GeoTIFF must contain a CRS."
            )

        if agl.shape != (rgb.height, rgb.width):
            raise ValueError(
                "AGL shape does not match RGB raster grid: "
                f"AGL={agl.shape}, "
                f"RGB={(rgb.height, rgb.width)}"
            )

        dem = resample_dem_to_rgb(
            args.srtm,
            rgb,
        )

        adjusted_agl = (
            agl * args.height_scale
            + args.height_offset
        )

        if args.clip_negative_agl:
            adjusted_agl = np.maximum(
                adjusted_agl,
                0.0,
            )

        valid = (
            np.isfinite(dem)
            & np.isfinite(adjusted_agl)
        )

        dsm = np.full_like(
            adjusted_agl,
            np.nan,
            dtype=np.float32,
        )

        dsm[valid] = (
            dem[valid]
            + adjusted_agl[valid]
        )

        profile = rgb.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress="deflate",
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            args.output,
            "w",
            **profile,
        ) as dst:
            dst.write(dsm, 1)

            dst.set_band_description(
                1,
                "SRTM-anchored estimated DSM",
            )

        valid_dsm = dsm[np.isfinite(dsm)]

    print("=" * 70)
    print("DepthWizard - Prototype Metric DSM")
    print("=" * 70)
    print(f"RGB             : {args.rgb}")
    print(f"SRTM            : {args.srtm}")
    print(f"AGL             : {args.agl}")
    print(f"Output          : {args.output}")
    print(f"CRS             : {rgb.crs}")
    print(f"Pixel size      : {rgb.res}")

    if valid_dsm.size:
        print(
            f"DSM min        : {valid_dsm.min():.3f} m"
        )
        print(
            f"DSM median     : {np.median(valid_dsm):.3f} m"
        )
        print(
            f"DSM max        : {valid_dsm.max():.3f} m"
        )

    print(
        "\nPrototype formulation:"
    )
    print(
        "  DSM(x,y) = SRTM(x,y) + predicted_AGL(x,y)"
    )
    print(
        "\nNOTE: This is a prototype metric anchoring step. "
        "It is not yet a validated high-precision absolute DSM."
    )


if __name__ == "__main__":
    main()
