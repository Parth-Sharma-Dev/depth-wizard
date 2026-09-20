from __future__ import annotations

import argparse
import math


def srtm_tile_name(lat: float, lon: float) -> str:
    """
    Return the conventional one-degree SRTM tile name.

    The tile name identifies the southwest corner of the 1-degree cell,
    e.g. N26E075 for 26-27N, 75-76E.
    """
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)

    lat_prefix = "N" if lat_floor >= 0 else "S"
    lon_prefix = "E" if lon_floor >= 0 else "W"

    return (
        f"{lat_prefix}{abs(lat_floor):02d}"
        f"{lon_prefix}{abs(lon_floor):03d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive SRTM 1-degree tile names from WGS84 bounds."
        )
    )
    parser.add_argument("--left", type=float, required=True)
    parser.add_argument("--bottom", type=float, required=True)
    parser.add_argument("--right", type=float, required=True)
    parser.add_argument("--top", type=float, required=True)

    args = parser.parse_args()

    lat_start = math.floor(args.bottom)
    lat_end = math.ceil(args.top) - 1

    lon_start = math.floor(args.left)
    lon_end = math.ceil(args.right) - 1

    tiles = []

    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(
                srtm_tile_name(lat, lon)
            )

    print("Required SRTM tiles:")
    for tile in tiles:
        print(f"  {tile}")

    print(
        "\nUse SRTM 1 Arc-Second Global tiles where available."
    )


if __name__ == "__main__":
    main()
