from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an initial clean GAMUS scene manifest from the "
            "quality-scan CSV."
        )
    )

    parser.add_argument(
        "--quality-csv",
        type=Path,
        required=True,
        help="Path to gamus_quality_*_sample.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "gamus_quality"
            / "gamus_clean_manifest.csv"
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--max-negative-pct",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-gt50-pct",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--no-strict-flags",
        action="store_true",
        help=(
            "Allow scenes with diagnostic flags if they satisfy the "
            "numeric thresholds. Default is strict for Baseline 1A."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.quality_csv)

    required = {
        "scene",
        "split",
        "negative_pct",
        "gt50_pct",
        "flags",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Quality CSV is missing columns: {sorted(missing)}"
        )

    df = df[df["split"].astype(str) == args.split].copy()

    df["negative_pct"] = pd.to_numeric(
        df["negative_pct"],
        errors="coerce",
    )
    df["gt50_pct"] = pd.to_numeric(
        df["gt50_pct"],
        errors="coerce",
    )

    mask = (
        df["negative_pct"].notna()
        & (df["negative_pct"] <= args.max_negative_pct)
        & df["gt50_pct"].notna()
        & (df["gt50_pct"] <= args.max_gt50_pct)
    )

    strict_flags = not args.no_strict_flags

    if strict_flags:
        mask &= df["flags"].astype(str).eq("NONE")

    clean = df.loc[mask].copy()

    keep_cols = [
        "scene",
        "split",
        "negative_pct",
        "gt50_pct",
        "flags",
    ]

    clean = clean[keep_cols].sort_values("scene")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        args.output,
        index=False,
    )

    print("=" * 70)
    print("GAMUS Clean Manifest")
    print("=" * 70)
    print(f"Input scenes : {len(df)}")
    print(f"Output scenes: {len(clean)}")
    print(f"Output       : {args.output}")

    print("\nRules:")
    print(
        f"  negative_pct <= {args.max_negative_pct:.3f}%"
    )
    print(
        f"  gt50_pct     <= {args.max_gt50_pct:.3f}%"
    )
    print(
        f"  strict flags : {strict_flags}"
    )


if __name__ == "__main__":
    main()
