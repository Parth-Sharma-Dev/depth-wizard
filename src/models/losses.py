from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:

    per_pixel = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )

    weighted = per_pixel * weight

    denominator = weight.sum().clamp_min(1.0)

    return weighted.sum() / denominator
