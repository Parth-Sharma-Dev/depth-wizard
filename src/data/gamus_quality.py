from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityPolicy:
    """
    Initial conservative policy for the first clean GAMUS baseline.

    This policy is intentionally simple. It should be treated as an
    experimental baseline, not as the final scientific filtering policy.
    """
    max_negative_pct: float = 5.0
    max_gt50_pct: float = 0.05
    use_scanner_flags: bool = True
    allowed_flags: tuple[str, ...] = ("NONE",)

    # Pixel-level rules.
    invalid_negative: bool = True
    invalid_nonfinite: bool = True

    # Suspicious pixels get a smaller loss weight rather than being deleted.
    suspicious_gt40_weight: float = 0.25


CLASS_NAMES = {
    0: "others",
    1: "ground",
    2: "low_vegetation",
    3: "buildings",
    4: "water",
    5: "road",
    6: "tree",
}


def scene_is_clean(row: dict, policy: QualityPolicy) -> bool:
    """
    Decide whether a scene is eligible for the first clean training run.

    Expected fields come from gamus_quality_scan.py:
      negative_pct, gt50_pct, flags
    """
    try:
        negative_pct = float(row.get("negative_pct", 0.0))
        gt50_pct = float(row.get("gt50_pct", 0.0))
    except (TypeError, ValueError):
        return False

    if negative_pct > policy.max_negative_pct:
        return False

    if gt50_pct > policy.max_gt50_pct:
        return False

    if policy.use_scanner_flags:
        flags = str(row.get("flags", "NONE"))
        if flags not in policy.allowed_flags:
            return False

    return True


def build_height_targets(
    height: np.ndarray,
    policy: QualityPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build:
      target: float32 AGL target
      weight: float32 pixel-wise loss weight

    Initial policy:
      - non-finite pixels -> weight 0
      - negative AGL -> weight 0
      - >40 m pixels -> weight reduced, not deleted

    The >40 m rule is intentionally a training-confidence heuristic,
    not a statement that heights >40 m are invalid.
    """
    height = np.asarray(height, dtype=np.float32)

    target = height.copy()
    weight = np.ones_like(target, dtype=np.float32)

    if policy.invalid_nonfinite:
        weight[~np.isfinite(target)] = 0.0

    if policy.invalid_negative:
        weight[target < 0.0] = 0.0

    suspicious = (
        np.isfinite(target)
        & (target > 40.0)
        & (weight > 0.0)
    )
    weight[suspicious] = policy.suspicious_gt40_weight

    return target, weight
