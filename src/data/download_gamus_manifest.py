from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "earthflow/GAMUS"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RGB + AGL + CLS HDF5 files for a GAMUS manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV produced by build_gamus_manifest.py",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "validation", "test"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "gamus_local",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
    )
    parser.add_argument(
        "--skip-classes",
        action="store_true",
        help="Download only RGB + AGL. Useful for the first height-only training run.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    if "scene" not in df.columns:
        raise ValueError("Manifest must contain a 'scene' column.")

    scenes = sorted(df["scene"].astype(str).unique())

    print("=" * 70)
    print("GAMUS Manifest Downloader")
    print("=" * 70)
    print(f"Scenes      : {len(scenes)}")
    print(f"Split       : {args.split}")
    print(f"Output root : {args.output_root}")
    print(f"Classes     : {not args.skip_classes}")

    downloaded = 0
    failed = []

    for idx, scene in enumerate(scenes, start=1):
        print(f"\n[{idx}/{len(scenes)}] {scene}")

        files = [
            f"images/{args.split}/{scene}_RGB.h5",
            f"heights/{args.split}/{scene}_AGL.h5",
        ]

        if not args.skip_classes:
            files.append(
                f"classes/{args.split}/{scene}_CLS.h5"
            )

        for repo_path in files:
            try:
                local_path = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=repo_path,
                    local_dir=args.output_root,
                )
                print(f"  OK: {repo_path}")
                print(f"      -> {local_path}")
                downloaded += 1
            except Exception as exc:
                print(f"  ERROR: {repo_path}")
                print(f"         {exc}")
                failed.append((scene, repo_path, str(exc)))

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successful files : {downloaded}")
    print(f"Failed files     : {len(failed)}")

    if failed:
        failure_path = args.output_root / "download_failures.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            failed,
            columns=["scene", "repo_path", "error"],
        ).to_csv(
            failure_path,
            index=False,
        )
        print(f"Failures saved to: {failure_path}")


if __name__ == "__main__":
    main()
