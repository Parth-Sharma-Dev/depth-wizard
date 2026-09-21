from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


def encode_height_rgb(
    height: np.ndarray,
    min_height: float,
    max_height: float,
) -> np.ndarray:
    span = max(max_height - min_height, 1e-9)
    normalized = np.clip(
        (height - min_height) / span,
        0.0,
        1.0,
    )

    encoded = np.round(
        normalized * (2**24 - 1)
    ).astype(np.uint32)

    r = (encoded >> 16).astype(np.uint8)
    g = (encoded >> 8).astype(np.uint8)
    b = encoded.astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def reproject_dem_to_rgb_mesh(
    dem_path: Path,
    rgb_crs,
    rgb_bounds,
    mesh_width: int,
    mesh_height: int,
) -> np.ndarray:
    with rasterio.open(dem_path) as dem_src:
        if dem_src.crs is None:
            raise ValueError("Terrain DEM has no CRS.")

        destination = np.full(
            (mesh_height, mesh_width),
            np.nan,
            dtype=np.float32,
        )

        mesh_transform = from_bounds(
            rgb_bounds.left,
            rgb_bounds.bottom,
            rgb_bounds.right,
            rgb_bounds.top,
            mesh_width,
            mesh_height,
        )

        reproject(
            source=rasterio.band(dem_src, 1),
            destination=destination,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            src_nodata=dem_src.nodata,
            dst_transform=mesh_transform,
            dst_crs=rgb_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DepthWizard RGB + terrain DEM assets for the "
            "Three.js viewer. The DEM may use a different CRS and "
            "resolution from the RGB GeoTIFF."
        )
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--dsm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-width", type=int, default=1600)
    parser.add_argument("--mesh-width", type=int, default=512)
    args = parser.parse_args()

    if not args.rgb.exists():
        raise FileNotFoundError(args.rgb)
    if not args.dsm.exists():
        raise FileNotFoundError(args.dsm)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(args.rgb) as rgb_src:
        if rgb_src.count < 3:
            raise ValueError("RGB file must contain at least 3 bands.")
        if rgb_src.crs is None:
            raise ValueError("RGB file has no CRS.")

        rgb_crs = rgb_src.crs
        rgb_bounds = rgb_src.bounds
        rgb_width = rgb_src.width
        rgb_height = rgb_src.height

        texture_width = min(
            args.texture_width,
            rgb_width,
        )
        texture_height = max(
            2,
            round(
                rgb_height
                * texture_width
                / rgb_width
            ),
        )

        rgb = rgb_src.read(
            [1, 2, 3],
            out_shape=(
                3,
                texture_height,
                texture_width,
            ),
            resampling=Resampling.bilinear,
        )

        rgb = np.transpose(
            rgb,
            (1, 2, 0),
        ).astype(np.uint8)

        Image.fromarray(
            rgb,
            mode="RGB",
        ).save(
            args.output_dir / "rgb_texture.png"
        )

    mesh_width = int(args.mesh_width)
    mesh_height = max(
        2,
        round(
            mesh_width
            * rgb_height
            / rgb_width
        ),
    )

    with rasterio.open(args.dsm) as dem_src:
        terrain_source_crs = (
            dem_src.crs.to_string()
            if dem_src.crs
            else None
        )
        terrain_source_resolution = [
            float(dem_src.res[0]),
            float(dem_src.res[1]),
        ]

    terrain = reproject_dem_to_rgb_mesh(
        args.dsm,
        rgb_crs,
        rgb_bounds,
        mesh_width,
        mesh_height,
    )

    valid_mask = np.isfinite(terrain)

    finite = terrain[valid_mask]
    if finite.size == 0:
        raise ValueError(
            "No valid DEM pixels overlap the RGB footprint."
        )

    min_height = float(finite.min())
    max_height = float(finite.max())

    terrain_for_encoding = terrain.copy()
    terrain_for_encoding[~valid_mask] = min_height

    Image.fromarray(
        encode_height_rgb(
            terrain_for_encoding,
            min_height,
            max_height,
        ),
        mode="RGB",
    ).save(
        args.output_dir / "height_map.png"
    )

    Image.fromarray(
        (valid_mask.astype(np.uint8) * 255),
        mode="L",
    ).save(
        args.output_dir / "terrain_mask.png"
    )

    metadata = {
        "source_rgb": str(args.rgb),
        "source_dem": str(args.dsm),
        "source_width": rgb_width,
        "source_height": rgb_height,
        "texture_width": texture_width,
        "texture_height": texture_height,
        "mesh_width": mesh_width,
        "mesh_height": mesh_height,
        "min_elevation_m": min_height,
        "max_elevation_m": max_height,
        "elevation_span_m": max_height - min_height,
        "valid_mesh_percent": float(
            valid_mask.mean() * 100.0
        ),
        "rgb_crs": rgb_crs.to_string(),
        "terrain_source_crs": terrain_source_crs,
        "terrain_source_resolution": terrain_source_resolution,
        "terrain_reprojected_to_rgb_crs": True,
        "terrain_resampling": "bilinear",
    }

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("DepthWizard - 3D Terrain Asset Preparation")
    print("=" * 80)
    print(f"RGB source       : {args.rgb}")
    print(f"Terrain source   : {args.dsm}")
    print(f"RGB CRS          : {rgb_crs}")
    print(f"Terrain CRS      : {terrain_source_crs}")
    print("Terrain handling : Reprojected to RGB CRS/footprint")
    print(f"Texture          : {texture_width} x {texture_height}")
    print(f"Height map       : {mesh_width} x {mesh_height}")
    print(
        f"Elevation range  : "
        f"{min_height:.3f} -> {max_height:.3f} m"
    )
    print(
        f"Valid terrain    : "
        f"{valid_mask.mean() * 100.0:.2f}%"
    )
    print("\nAssets created:")
    print("  rgb_texture.png")
    print("  height_map.png")
    print("  terrain_mask.png")
    print("  metadata.json")


if __name__ == "__main__":
    main()
