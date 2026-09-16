from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download


REPO_ID = "earthflow/GAMUS"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gamus_inspection"
    / "quality_cases"
)

CLASS_NAMES = {
    0: "Others",
    1: "Ground",
    2: "Low vegetation",
    3: "Buildings",
    4: "Water",
    5: "Road",
    6: "Tree",
}

DEFAULT_SCENES = [
    "DC_40_31",
    "DC_50_26",
    "NYC_30770",
    "NYC_27766",
    "PHL_2814",
    "PHL_1804",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect representative GAMUS height-quality cases using "
            "RGB + AGL + semantic class data."
        )
    )

    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help=(
            "Scene IDs to inspect. "
            "Default: DC_40_31 DC_50_26 NYC_30770 "
            "NYC_27766 PHL_2814 PHL_1804"
        ),
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "validation", "test"],
        help="GAMUS split. Default: train",
    )

    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face dataset repository. Default: {REPO_ID}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )

    return parser.parse_args()


def download_scene(
    repo_id: str,
    scene: str,
    split: str,
    output_dir: Path,
) -> tuple[Path, Path, Path]:

    local_cache = output_dir / "h5_cache"
    local_cache.mkdir(parents=True, exist_ok=True)

    rgb_repo = f"images/{split}/{scene}_RGB.h5"
    agl_repo = f"heights/{split}/{scene}_AGL.h5"
    cls_repo = f"classes/{split}/{scene}_CLS.h5"

    print(f"\nDownloading/locating {scene}...")

    rgb_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=rgb_repo,
            local_dir=local_cache,
        )
    )

    agl_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=agl_repo,
            local_dir=local_cache,
        )
    )

    cls_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=cls_repo,
            local_dir=local_cache,
        )
    )

    return rgb_path, agl_path, cls_path


def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as file:
        if "image" not in file:
            raise KeyError(
                f"{path} does not contain the expected 'image' dataset. "
                f"Available keys: {list(file.keys())}"
            )
        return file["image"][:]


def compute_case_stats(
    agl: np.ndarray,
    cls: np.ndarray,
) -> dict[str, object]:

    agl = np.asarray(agl, dtype=np.float32)
    cls = np.asarray(cls)

    if agl.shape != cls.shape:
        raise ValueError(
            f"AGL/CLS shape mismatch: {agl.shape} vs {cls.shape}"
        )

    finite = np.isfinite(agl)

    negative = finite & (agl < 0)
    minus5 = finite & np.isclose(agl, -5.0)
    gt10 = finite & (agl > 10)
    gt20 = finite & (agl > 20)
    gt30 = finite & (agl > 30)
    gt40 = finite & (agl > 40)
    gt50 = finite & (agl > 50)

    valid_values = agl[finite]

    stats: dict[str, object] = {
        "pixels": int(agl.size),
        "negative_pct": float(negative.mean() * 100),
        "minus5_pct": float(minus5.mean() * 100),
        "gt10_pct": float(gt10.mean() * 100),
        "gt20_pct": float(gt20.mean() * 100),
        "gt30_pct": float(gt30.mean() * 100),
        "gt40_pct": float(gt40.mean() * 100),
        "gt50_pct": float(gt50.mean() * 100),
        "min": float(valid_values.min()) if valid_values.size else np.nan,
        "max": float(valid_values.max()) if valid_values.size else np.nan,
        "p50": float(np.percentile(valid_values, 50))
        if valid_values.size
        else np.nan,
        "p95": float(np.percentile(valid_values, 95))
        if valid_values.size
        else np.nan,
        "p99": float(np.percentile(valid_values, 99))
        if valid_values.size
        else np.nan,
    }

    class_stats: dict[str, dict[str, float | int]] = {}

    for class_id, class_name in CLASS_NAMES.items():
        mask = (cls == class_id) & finite
        values = agl[mask]

        entry = {
            "pixels": int(mask.sum()),
            "pct_scene": float(mask.mean() * 100),
        }

        if values.size:
            entry.update(
                {
                    "mean_agl": float(values.mean()),
                    "median_agl": float(np.median(values)),
                    "max_agl": float(values.max()),
                    "negative_pct": float((values < 0).mean() * 100),
                    "gt10_pct": float((values > 10).mean() * 100),
                    "gt40_pct": float((values > 40).mean() * 100),
                }
            )
        else:
            entry.update(
                {
                    "mean_agl": np.nan,
                    "median_agl": np.nan,
                    "max_agl": np.nan,
                    "negative_pct": np.nan,
                    "gt10_pct": np.nan,
                    "gt40_pct": np.nan,
                }
            )

        class_stats[f"{class_id}:{class_name}"] = entry

    stats["class_stats"] = class_stats
    return stats


def save_case_visualization(
    scene: str,
    rgb: np.ndarray,
    agl: np.ndarray,
    cls: np.ndarray,
    stats: dict[str, object],
    output_dir: Path,
    show: bool,
) -> Path:

    cls = cls.astype(np.int64)

    # Separate negative values so we can visually see where they occur.
    agl_positive = agl.astype(np.float32).copy()
    agl_positive[agl_positive < 0] = np.nan

    negative_mask = np.isfinite(agl) & (agl < 0)
    gt40_mask = np.isfinite(agl) & (agl > 40)
    gt50_mask = np.isfinite(agl) & (agl > 50)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    # --------------------------------------------------
    # RGB
    # --------------------------------------------------
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{scene} — RGB")
    axes[0, 0].axis("off")

    # --------------------------------------------------
    # AGL
    # --------------------------------------------------
    im = axes[0, 1].imshow(
        agl_positive,
        cmap="turbo",
    )
    axes[0, 1].set_title(
        f"AGL\n"
        f"min={stats['min']:.2f}, "
        f"median={stats['p50']:.2f}, "
        f"max={stats['max']:.2f} m"
    )
    axes[0, 1].axis("off")

    fig.colorbar(
        im,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label="AGL",
    )

    # --------------------------------------------------
    # Semantic classes
    # --------------------------------------------------
    axes[0, 2].imshow(
        cls,
        cmap="tab10",
        vmin=0,
        vmax=6,
    )
    axes[0, 2].set_title("Semantic Classes")
    axes[0, 2].axis("off")

    # --------------------------------------------------
    # Negative-value locations
    # --------------------------------------------------
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(
        np.ma.masked_where(
            ~negative_mask,
            negative_mask,
        ),
        cmap="Reds",
        alpha=0.8,
    )
    axes[1, 0].set_title(
        f"Negative AGL\n"
        f"{stats['negative_pct']:.3f}% of pixels"
    )
    axes[1, 0].axis("off")

    # --------------------------------------------------
    # >40m locations
    # --------------------------------------------------
    axes[1, 1].imshow(rgb)
    axes[1, 1].imshow(
        np.ma.masked_where(
            ~gt40_mask,
            gt40_mask,
        ),
        cmap="autumn",
        alpha=0.8,
    )
    axes[1, 1].set_title(
        f"AGL > 40 m\n"
        f"{stats['gt40_pct']:.3f}% of pixels"
    )
    axes[1, 1].axis("off")

    # --------------------------------------------------
    # >50m locations + class contours
    # --------------------------------------------------
    axes[1, 2].imshow(rgb)

    axes[1, 2].imshow(
        np.ma.masked_where(
            ~gt50_mask,
            gt50_mask,
        ),
        cmap="Reds",
        alpha=0.85,
    )

    # Building contours
    building_mask = cls == 3
    axes[1, 2].contour(
        building_mask,
        levels=[0.5],
        linewidths=0.7,
    )

    # Tree contours
    tree_mask = cls == 6
    axes[1, 2].contour(
        tree_mask,
        levels=[0.5],
        linewidths=0.7,
    )

    axes[1, 2].set_title(
        f"AGL > 50 m + Building/Tree Boundaries\n"
        f"{stats['gt50_pct']:.3f}% of pixels"
    )
    axes[1, 2].axis("off")

    plt.suptitle(
        f"GAMUS Quality Case: {scene}",
        fontsize=16,
    )

    plt.tight_layout()

    output_path = output_dir / f"{scene}_quality_case.png"

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    if show:
        plt.show()

    plt.close(fig)

    return output_path


def write_case_report(
    scene: str,
    stats: dict[str, object],
    output_dir: Path,
) -> Path:

    report_path = output_dir / f"{scene}_quality_report.txt"

    lines = [
        f"GAMUS Quality Case Report: {scene}",
        "=" * 70,
        "",
        "Scene-level AGL statistics",
        "-" * 70,
        f"Pixels       : {stats['pixels']:,}",
        f"Min          : {stats['min']:.6f} m",
        f"Median       : {stats['p50']:.6f} m",
        f"P95          : {stats['p95']:.6f} m",
        f"P99          : {stats['p99']:.6f} m",
        f"Max          : {stats['max']:.6f} m",
        f"Negative     : {stats['negative_pct']:.6f}%",
        f"Exact -5     : {stats['minus5_pct']:.6f}%",
        f">10 m        : {stats['gt10_pct']:.6f}%",
        f">20 m        : {stats['gt20_pct']:.6f}%",
        f">30 m        : {stats['gt30_pct']:.6f}%",
        f">40 m        : {stats['gt40_pct']:.6f}%",
        f">50 m        : {stats['gt50_pct']:.6f}%",
        "",
        "Class-wise AGL statistics",
        "-" * 70,
    ]

    class_stats = stats["class_stats"]

    for key, entry in class_stats.items():
        lines.extend(
            [
                "",
                key,
                f"  pixels          : {entry['pixels']:,}",
                f"  scene %         : {entry['pct_scene']:.4f}",
                f"  mean AGL        : {entry['mean_agl']:.4f}",
                f"  median AGL      : {entry['median_agl']:.4f}",
                f"  max AGL         : {entry['max_agl']:.4f}",
                f"  negative %      : {entry['negative_pct']:.4f}",
                f"  >10 m %         : {entry['gt10_pct']:.4f}",
                f"  >40 m %         : {entry['gt40_pct']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation note",
            "-" * 70,
            "These statistics are diagnostic only.",
            "A large height is not automatically invalid.",
            "Review the RGB/AGL/CLS visualization before excluding data.",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("DepthWizard - GAMUS Representative Quality Cases")
    print("=" * 80)
    print(f"Repository : {args.repo_id}")
    print(f"Split      : {args.split}")
    print(f"Scenes     : {len(args.scenes)}")
    print(f"Output     : {output_dir}")
    print("=" * 80)

    summary_rows = []

    for index, scene in enumerate(args.scenes, start=1):

        print(
            f"\n[{index}/{len(args.scenes)}] {scene}"
        )

        try:
            rgb_path, agl_path, cls_path = download_scene(
                args.repo_id,
                scene,
                args.split,
                output_dir,
            )

            rgb = load_h5(rgb_path)
            agl = load_h5(agl_path)
            cls = load_h5(cls_path)

            if rgb.shape[:2] != agl.shape:
                raise ValueError(
                    f"RGB/AGL shape mismatch: "
                    f"{rgb.shape} vs {agl.shape}"
                )

            if cls.shape != agl.shape:
                raise ValueError(
                    f"CLS/AGL shape mismatch: "
                    f"{cls.shape} vs {agl.shape}"
                )

            stats = compute_case_stats(
                agl,
                cls,
            )

            image_path = save_case_visualization(
                scene,
                rgb,
                agl,
                cls,
                stats,
                output_dir,
                args.show,
            )

            report_path = write_case_report(
                scene,
                stats,
                output_dir,
            )

            summary_rows.append(
                {
                    "scene": scene,
                    "split": args.split,
                    "shape": "x".join(map(str, agl.shape)),
                    "min_m": stats["min"],
                    "median_m": stats["p50"],
                    "p95_m": stats["p95"],
                    "p99_m": stats["p99"],
                    "max_m": stats["max"],
                    "negative_pct": stats["negative_pct"],
                    "minus5_pct": stats["minus5_pct"],
                    "gt10_pct": stats["gt10_pct"],
                    "gt20_pct": stats["gt20_pct"],
                    "gt30_pct": stats["gt30_pct"],
                    "gt40_pct": stats["gt40_pct"],
                    "gt50_pct": stats["gt50_pct"],
                    "visualization": image_path.name,
                    "report": report_path.name,
                }
            )

            print(
                f"Range: {stats['min']:.3f} -> "
                f"{stats['max']:.3f} m"
            )
            print(
                f"Negative: {stats['negative_pct']:.3f}%"
            )
            print(
                f">40 m: {stats['gt40_pct']:.3f}%"
            )
            print(
                f">50 m: {stats['gt50_pct']:.3f}%"
            )
            print(
                f"Saved: {image_path.name}"
            )

        except Exception as exc:
            print(
                f"ERROR: {scene}: {exc}"
            )

            summary_rows.append(
                {
                    "scene": scene,
                    "split": args.split,
                    "error": str(exc),
                }
            )

    # ------------------------------------------------------
    # Write combined summary
    # ------------------------------------------------------
    summary_path = output_dir / "quality_cases_summary.csv"

    import csv

    # Keep the summary simple and tolerant of error rows.
    fieldnames = sorted(
        {
            key
            for row in summary_rows
            for key in row.keys()
        }
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("QUALITY CASE INSPECTION COMPLETE")
    print("=" * 80)
    print(f"Output directory : {output_dir}")
    print(f"Summary CSV      : {summary_path}")


if __name__ == "__main__":
    main()
