from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "earthflow/GAMUS"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "gamus_quality"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "hf_cache"

SUPPORTED_SPLITS = ("train", "val", "validation", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan GAMUS AGL height files for data-quality issues. "
            "Default mode samples scenes so the whole split is not downloaded."
        )
    )
    p.add_argument("--split", default="train", choices=SUPPORTED_SPLITS)
    p.add_argument("--mode", default="sample", choices=("sample", "full"))
    p.add_argument("--num-scenes", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-classes", action="store_true")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def discover_scene_files(repo_id: str, split: str) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefix = f"heights/{split}/"
    paths = sorted(
        f for f in files
        if f.startswith(prefix) and f.endswith("_AGL.h5")
    )
    if not paths:
        available = sorted(
            {f.split("/")[1] for f in files if f.startswith("heights/") and len(f.split("/")) >= 3}
        )
        raise RuntimeError(
            f"No AGL files found for split '{split}'. Available: {available}"
        )
    return paths


def select_scene_files(paths: list[str], mode: str, n: int, seed: int) -> list[str]:
    if mode == "full" or n >= len(paths):
        return paths
    if n <= 0:
        raise ValueError("--num-scenes must be > 0")
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n))


def download_h5(repo_id: str, repo_path: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=cache_dir,
        )
    )


def read_h5_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "image" not in f:
            raise KeyError(f"Expected dataset key 'image' in {path}; found {list(f.keys())}")
        return f["image"][:]


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: np.nan for k in (
            "finite_pct", "min", "max", "mean", "std",
            "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p999"
        )}
    return {
        "finite_pct": float(finite.size / values.size * 100),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "p999": float(np.percentile(finite, 99.9)),
    }


def analyze_agl(agl: np.ndarray) -> dict[str, object]:
    agl = np.asarray(agl, dtype=np.float32)
    finite = np.isfinite(agl)
    stats = finite_stats(agl)
    result: dict[str, object] = {
        "pixels": int(agl.size),
        "nan_count": int(np.isnan(agl).sum()),
        "inf_count": int(np.isinf(agl).sum()),
        "negative_pct": float((finite & (agl < 0)).mean() * 100),
        "minus5_pct": float((finite & np.isclose(agl, -5.0)).mean() * 100),
        "zero_pct": float((finite & np.isclose(agl, 0.0)).mean() * 100),
        "gt10_pct": float((finite & (agl > 10)).mean() * 100),
        "gt20_pct": float((finite & (agl > 20)).mean() * 100),
        "gt30_pct": float((finite & (agl > 30)).mean() * 100),
        "gt40_pct": float((finite & (agl > 40)).mean() * 100),
        "gt50_pct": float((finite & (agl > 50)).mean() * 100),
        **stats,
    }

    flags: list[str] = []
    if result["finite_pct"] < 100:
        flags.append("NONFINITE_VALUES")
    if result["negative_pct"] > 0.1:
        flags.append("NEGATIVE_AGL_PRESENT")
    if result["minus5_pct"] > 0.1:
        flags.append("MINUS5_PRESENT")

    p99 = float(result["p99"])
    mx = float(result["max"])
    if np.isfinite(p99) and np.isfinite(mx) and mx > p99:
        if (mx - p99) > max(10.0, 0.5 * max(abs(p99), 1.0)):
            flags.append("POTENTIAL_HIGH_VALUE_SPIKES")
    if float(result["gt50_pct"]) > 1.0:
        flags.append("MANY_VALUES_ABOVE_50M")

    result["flags"] = ";".join(flags) if flags else "NONE"
    return result


def class_aware_analysis(agl: np.ndarray, cls: np.ndarray) -> dict[str, object]:
    agl = np.asarray(agl, dtype=np.float32)
    cls = np.asarray(cls)
    if agl.shape != cls.shape:
        raise ValueError(f"AGL/CLS shape mismatch: {agl.shape} vs {cls.shape}")

    result: dict[str, object] = {}
    for class_id in range(7):
        mask = (cls == class_id) & np.isfinite(agl)
        count = int(mask.sum())
        result[f"class_{class_id}_pixels"] = count
        result[f"class_{class_id}_pct"] = float(count / cls.size * 100)
        if count == 0:
            for suffix in ("mean", "median", "max", "gt10_pct", "negative_pct"):
                result[f"class_{class_id}_{suffix}"] = np.nan
            continue
        values = agl[mask]
        result[f"class_{class_id}_mean"] = float(values.mean())
        result[f"class_{class_id}_median"] = float(np.median(values))
        result[f"class_{class_id}_max"] = float(values.max())
        result[f"class_{class_id}_gt10_pct"] = float((values > 10).mean() * 100)
        result[f"class_{class_id}_negative_pct"] = float((values < 0).mean() * 100)
    return result


def scene_name(repo_path: str) -> str:
    return Path(repo_path).name.replace("_AGL.h5", "")


def process_scene(repo_id: str, agl_repo_path: str, cache_dir: Path, with_classes: bool) -> dict[str, object]:
    scene = scene_name(agl_repo_path)
    split = Path(agl_repo_path).parts[1]
    agl_local = download_h5(repo_id, agl_repo_path, cache_dir)
    agl = read_h5_array(agl_local)
    if agl.ndim != 2:
        raise ValueError(f"Expected 2-D AGL, got {agl.shape}")

    row: dict[str, object] = {
        "scene": scene,
        "split": split,
        "agl_shape": "x".join(map(str, agl.shape)),
        "agl_dtype": str(agl.dtype),
    }
    row.update(analyze_agl(agl))

    if with_classes:
        cls_repo_path = f"classes/{split}/{scene}_CLS.h5"
        cls_local = download_h5(repo_id, cls_repo_path, cache_dir)
        cls = read_h5_array(cls_local)
        row["cls_shape"] = "x".join(map(str, cls.shape))
        row["cls_dtype"] = str(cls.dtype)
        row["class_alignment"] = "OK" if cls.shape == agl.shape else "MISMATCH"
        if cls.shape == agl.shape:
            row.update(class_aware_analysis(agl, cls))

    print(
        f"{scene}: range={row['min']:.3f}->{row['max']:.3f} m, "
        f"P50={row['p50']:.3f}, P95={row['p95']:.3f}, "
        f"negative={row['negative_pct']:.3f}%, >40m={row['gt40_pct']:.3f}%, "
        f"flags={row['flags']}"
    )
    return row


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "hf_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DepthWizard - GAMUS Quality Scan")
    print("=" * 80)
    print(f"Repository : {args.repo_id}")
    print(f"Split      : {args.split}")
    print(f"Mode       : {args.mode}")
    print(f"Scenes     : {args.num_scenes if args.mode == 'sample' else 'ALL'}")
    print(f"Classes    : {args.with_classes}")
    print(f"Output     : {output_dir}")
    print("=" * 80)

    agl_files = discover_scene_files(args.repo_id, args.split)
    selected = select_scene_files(agl_files, args.mode, args.num_scenes, args.seed)
    print(f"Discovered {len(agl_files)} AGL files; selected {len(selected)}.")

    if args.mode == "full":
        print(
            "WARNING: full mode downloads every AGL HDF5 in the split. "
            "Use sample mode first."
        )

    rows: list[dict[str, object]] = []
    for i, repo_path in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {repo_path}")
        try:
            rows.append(process_scene(args.repo_id, repo_path, cache_dir, args.with_classes))
        except Exception as exc:
            print(f"ERROR: {exc}")
            rows.append({
                "scene": scene_name(repo_path),
                "split": args.split,
                "error": str(exc),
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"gamus_quality_{args.split}_{args.mode}.csv"
    summary_path = output_dir / f"gamus_quality_{args.split}_{args.mode}_summary.txt"
    df.to_csv(csv_path, index=False)

    lines = [
        "DepthWizard GAMUS Quality Scan",
        "=" * 60,
        f"Repository: {args.repo_id}",
        f"Split: {args.split}",
        f"Mode: {args.mode}",
        f"Scenes selected: {len(selected)}",
        f"Rows written: {len(df)}",
        "",
    ]
    if "negative_pct" in df.columns:
        lines += [
            f"Mean scene negative AGL %: {df['negative_pct'].mean():.4f}",
            f"Max scene negative AGL %: {df['negative_pct'].max():.4f}",
            f"Max scene >40m %: {df['gt40_pct'].max():.4f}",
            f"Max scene >50m %: {df['gt50_pct'].max():.4f}",
            "",
            "Flag counts:",
        ]
        counts: dict[str, int] = {}
        for value in df.get("flags", pd.Series(dtype=str)).fillna("ERROR"):
            for flag in str(value).split(";"):
                counts[flag] = counts.get(flag, 0) + 1
        for flag, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {flag}: {count}")

    lines += [
        "",
        f"CSV: {csv_path}",
        "Flags are diagnostic indicators, not automatic exclusion rules.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("QUALITY SCAN COMPLETE")
    print("=" * 80)
    print(f"CSV     : {csv_path}")
    print(f"Summary : {summary_path}")
    print(f"Cache   : {cache_dir}")


if __name__ == "__main__":
    main()
