from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a georeferenced RGB GeoTIFF for DepthWizard."
    )
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    path = args.image

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        print("=" * 70)
        print("DepthWizard - GeoTIFF Inspection")
        print("=" * 70)

        print(f"File        : {path}")
        print(f"Driver      : {src.driver}")
        print(f"Size        : {src.width} x {src.height}")
        print(f"Bands       : {src.count}")
        print(f"Dtypes      : {src.dtypes}")
        print(f"CRS         : {src.crs}")
        print(f"Transform   : {src.transform}")
        print(f"Resolution  : {src.res}")
        print(f"Bounds      : {src.bounds}")
        print(f"Width       : {src.width}")
        print(f"Height      : {src.height}")
        print(f"NoData      : {src.nodata}")

        if src.crs is not None:
            try:
                wgs84_bounds = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *src.bounds,
                    densify_pts=21,
                )
                print(
                    "WGS84 bounds: "
                    f"left={wgs84_bounds[0]:.8f}, "
                    f"bottom={wgs84_bounds[1]:.8f}, "
                    f"right={wgs84_bounds[2]:.8f}, "
                    f"top={wgs84_bounds[3]:.8f}"
                )
            except Exception as exc:
                print(f"WGS84 bounds: unavailable ({exc})")

        if src.count >= 3:
            rgb = src.read([1, 2, 3])
            print("\nRGB sample statistics:")
            for i, name in enumerate(("R", "G", "B")):
                band = rgb[i]
                finite = band[np.isfinite(band)]
                print(
                    f"  {name}: "
                    f"min={finite.min():.3f}, "
                    f"max={finite.max():.3f}, "
                    f"mean={finite.mean():.3f}"
                )

        if src.crs is None:
            print(
                "\nWARNING: This file has no CRS. "
                "SRTM alignment cannot be geospatially reliable."
            )

        print("\nGeoTIFF inspection complete.")


if __name__ == "__main__":
    main()
