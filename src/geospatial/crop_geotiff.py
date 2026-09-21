from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Crop a georeferenced GeoTIFF using a WGS84 "
            "longitude/latitude bounding box while preserving "
            "the raster CRS and georeferencing."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left", type=float, required=True)
    parser.add_argument("--bottom", type=float, required=True)
    parser.add_argument("--right", type=float, required=True)
    parser.add_argument("--top", type=float, required=True)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.left >= args.right:
        raise ValueError("--left must be smaller than --right.")
    if args.bottom >= args.top:
        raise ValueError("--bottom must be smaller than --top.")

    bbox = {
        "type": "Polygon",
        "coordinates": [[
            [args.left, args.bottom],
            [args.right, args.bottom],
            [args.right, args.top],
            [args.left, args.top],
            [args.left, args.bottom],
        ]],
    }

    with rasterio.open(args.input) as src:
        if src.crs is None:
            raise ValueError("Input GeoTIFF has no CRS.")

        geom = (
            bbox
            if src.crs.to_epsg() == 4326
            else transform_geom("EPSG:4326", src.crs, bbox)
        )

        cropped, transform = mask(
            src,
            [geom],
            crop=True,
            filled=True,
            nodata=src.nodata if src.nodata is not None else 0,
        )

        profile = src.profile.copy()
        profile.update(
            height=cropped.shape[1],
            width=cropped.shape[2],
            transform=transform,
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
            dst.write(cropped)
            dst.update_tags(
                crop_source=str(args.input),
                crop_bbox_wgs84=(
                    f"{args.left},{args.bottom},"
                    f"{args.right},{args.top}"
                ),
            )

        print("=" * 70)
        print("DepthWizard - GeoTIFF Crop")
        print("=" * 70)
        print(f"Input       : {args.input}")
        print(f"Output      : {args.output}")
        print(f"Input CRS   : {src.crs}")
        print(
            "WGS84 BBOX  : "
            f"{args.left}, {args.bottom}, "
            f"{args.right}, {args.top}"
        )
        print(
            f"Cropped     : "
            f"{cropped.shape[2]} x {cropped.shape[1]}"
        )
        print(f"Transform   : {transform}")
        print("\nCrop complete.")


if __name__ == "__main__":
    main()
