from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gamus_dataset import GAMUSDataset
from src.data.gamus_quality import QualityPolicy
from src.models.depthwizard_multiscale_height import (
    DepthWizardMultiScaleHeightModel,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "gamus_quality"
    / "gamus_clean_manifest.csv"
)

GAMUS_ROOT = PROJECT_ROOT / "data" / "gamus_local"

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "baseline1b"
)

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
HISTORY_PATH = EXPERIMENT_DIR / "history.csv"

SEED = 42

CROP_SIZE = 512

# Batch 8 worked comfortably for Baseline 1A, so keep it for the
# first comparison. If memory becomes an issue, drop to 4.
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8

SAMPLES_PER_SCENE = 8

EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0
VAL_FRACTION = 0.20

GRAD_CLIP_NORM = 1.0

USE_AMP = True

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
).view(1, 3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
).view(1, 3, 1, 1)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def load_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    if "scene" not in df.columns:
        raise ValueError(
            "Manifest must contain a 'scene' column."
        )

    df = (
        df.drop_duplicates("scene")
        .reset_index(drop=True)
    )

    if len(df) < 2:
        raise ValueError(
            "Need at least two scenes."
        )

    rng = np.random.default_rng(SEED)

    indices = np.arange(len(df))
    rng.shuffle(indices)

    val_count = max(
        1,
        int(round(len(df) * VAL_FRACTION)),
    )

    val_indices = indices[:val_count]

    train_mask = np.ones(
        len(df),
        dtype=bool,
    )
    train_mask[val_indices] = False

    train_df = (
        df.iloc[
            np.flatnonzero(train_mask)
        ]
        .reset_index(drop=True)
    )

    val_df = (
        df.iloc[val_indices]
        .reset_index(drop=True)
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        EXPERIMENT_DIR / "train_manifest.csv",
        index=False,
    )

    val_df.to_csv(
        EXPERIMENT_DIR / "val_manifest.csv",
        index=False,
    )

    print(f"Total scenes : {len(df)}")
    print(f"Train scenes : {len(train_df)}")
    print(f"Val scenes   : {len(val_df)}")

    return train_df, val_df


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    policy = QualityPolicy()

    train_dataset = GAMUSDataset(
        manifest=train_df,
        data_root=GAMUS_ROOT,
        split="train",
        crop_size=CROP_SIZE,
        samples_per_scene=SAMPLES_PER_SCENE,
        training=True,
        augment=True,
        quality_policy=policy,
        max_cached_scenes=1,
        seed=SEED,
    )

    val_dataset = GAMUSDataset(
        manifest=val_df,
        data_root=GAMUS_ROOT,
        split="train",
        crop_size=CROP_SIZE,
        samples_per_scene=1,
        training=False,
        augment=False,
        quality_policy=policy,
        max_cached_scenes=1,
        seed=SEED,
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------


def weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:

    loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )

    weighted = loss * weight

    denominator = weight.sum().clamp_min(1.0)

    return weighted.sum() / denominator


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    total_weight = 0.0
    total_abs_error = 0.0
    total_squared_error = 0.0

    predictions = []
    targets = []

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        heights = batch["height"].to(
            device,
            non_blocking=True,
        )
        weights = batch["weight"].to(
            device,
            non_blocking=True,
        )

        mean = IMAGE_MEAN.to(
            device=device,
            dtype=images.dtype,
        )
        std = IMAGE_STD.to(
            device=device,
            dtype=images.dtype,
        )

        images = (images - mean) / std

        if USE_AMP and device.type == "cuda":
            with autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                prediction = model(images)
        else:
            prediction = model(images)

        mask = weights > 0

        pred = prediction.float()[mask]
        true = heights.float()[mask]
        weight = weights.float()[mask]

        if pred.numel() == 0:
            continue

        error = pred - true

        total_abs_error += float(
            (error.abs() * weight)
            .sum()
            .item()
        )

        total_squared_error += float(
            ((error ** 2) * weight)
            .sum()
            .item()
        )

        total_weight += float(
            weight.sum().item()
        )

        predictions.append(
            pred.detach().cpu()
        )
        targets.append(
            true.detach().cpu()
        )

    if total_weight <= 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "mean_pred": float("nan"),
            "mean_target": float("nan"),
        }

    mae = (
        total_abs_error / total_weight
    )

    rmse = float(
        np.sqrt(
            total_squared_error
            / total_weight
        )
    )

    pred_all = torch.cat(predictions)
    true_all = torch.cat(targets)

    pred_centered = (
        pred_all - pred_all.mean()
    )
    true_centered = (
        true_all - true_all.mean()
    )

    denominator = torch.sqrt(
        (pred_centered ** 2).sum()
        * (true_centered ** 2).sum()
    )

    if denominator.item() > 1e-12:
        corr = float(
            (
                (pred_centered * true_centered).sum()
                / denominator
            ).item()
        )
    else:
        corr = float("nan")

    return {
        "mae": float(mae),
        "rmse": rmse,
        "corr": corr,
        "mean_pred": float(
            pred_all.mean().item()
        ),
        "mean_target": float(
            true_all.mean().item()
        ),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
) -> float:

    model.train()

    # The Large backbone is frozen for Baseline 1B.
    model.backbone_model.eval()
    model.height_head.train()

    weighted_loss_sum = 0.0
    total_weight = 0.0

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        heights = batch["height"].to(
            device,
            non_blocking=True,
        )
        weights = batch["weight"].to(
            device,
            non_blocking=True,
        )

        mean = IMAGE_MEAN.to(
            device=device,
            dtype=images.dtype,
        )
        std = IMAGE_STD.to(
            device=device,
            dtype=images.dtype,
        )

        images = (images - mean) / std

        optimizer.zero_grad(
            set_to_none=True,
        )

        if USE_AMP and device.type == "cuda":

            with autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                prediction = model(images)

                loss_map = F.smooth_l1_loss(
                    prediction,
                    heights,
                    reduction="none",
                    beta=1.0,
                )

                weighted_loss = (
                    loss_map * weights
                )

                denominator = (
                    weights.sum()
                    .clamp_min(1.0)
                )

                loss = (
                    weighted_loss.sum()
                    / denominator
                )

            if scaler is None:
                raise RuntimeError(
                    "AMP enabled but GradScaler is None."
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.height_head.parameters(),
                GRAD_CLIP_NORM,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            prediction = model(images)

            loss_map = F.smooth_l1_loss(
                prediction,
                heights,
                reduction="none",
                beta=1.0,
            )

            denominator = (
                weights.sum()
                .clamp_min(1.0)
            )

            loss = (
                (loss_map * weights).sum()
                / denominator
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.height_head.parameters(),
                GRAD_CLIP_NORM,
            )

            optimizer.step()

        batch_weight = float(
            weights.sum().detach().item()
        )

        weighted_loss_sum += (
            float(loss.detach().item())
            * batch_weight
        )

        total_weight += batch_weight

        if batch_idx % 20 == 0:
            print(
                f"  batch {batch_idx:4d}/{len(loader):4d} "
                f"loss={loss.item():.5f}"
            )

    return (
        weighted_loss_sum
        / max(total_weight, 1.0)
    )


# ---------------------------------------------------------------------
# Compact checkpointing
# ---------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "height_head_state_dict": (
                model.height_head.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "metrics": metrics,
            "model_name": MODEL_NAME,
            "crop_size": CROP_SIZE,
            "frozen_backbone": True,
            "decoder": "multi_scale",
        },
        path,
    )


def append_history(
    row: dict[str, object],
) -> None:

    exists = HISTORY_PATH.exists()

    with HISTORY_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:

    seed_everything(SEED)

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("DepthWizard - Baseline 1B Training")
    print("=" * 80)

    print(f"Device       : {device}")

    if device.type == "cuda":

        print(
            "GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "VRAM         : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"Crop         : {CROP_SIZE}x{CROP_SIZE}"
    )
    print(
        f"Batch        : {TRAIN_BATCH_SIZE}"
    )
    print(
        f"Epochs       : {EPOCHS}"
    )
    print(
        f"LR           : {LEARNING_RATE}"
    )

    # --------------------------------------------------------------
    # Data
    # --------------------------------------------------------------

    train_df, val_df = load_manifest()

    train_dataset, val_dataset = (
        build_datasets(
            train_df,
            val_df,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    print(
        f"Train samples : {len(train_dataset)}"
    )
    print(
        f"Val samples   : {len(val_dataset)}"
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    print("\nLoading multi-scale model...")

    model = (
        DepthWizardMultiScaleHeightModel(
            freeze_backbone=True,
        )
        .to(device)
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in trainable_params
    )

    print(
        f"Total parameters     : {total_params:,}"
    )
    print(
        f"Trainable parameters : {trainable_count:,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )
    )

    scaler = (
        GradScaler("cuda")
        if USE_AMP and device.type == "cuda"
        else None
    )

    best_mae = float("inf")

    # --------------------------------------------------------------
    # Loop
    # --------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("\n" + "=" * 80)
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print("=" * 80)

        start = time.perf_counter()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
        )

        metrics = validate(
            model,
            val_loader,
            device,
        )

        scheduler.step()

        elapsed = (
            time.perf_counter()
            - start
        )

        print("\nEpoch summary")
        print(
            f"Train loss : {train_loss:.6f}"
        )
        print(
            f"Val MAE    : {metrics['mae']:.4f} m"
        )
        print(
            f"Val RMSE   : {metrics['rmse']:.4f} m"
        )
        print(
            f"Val Corr   : {metrics['corr']:.4f}"
        )
        print(
            f"Mean pred  : {metrics['mean_pred']:.4f} m"
        )
        print(
            f"Mean target: {metrics['mean_target']:.4f} m"
        )
        print(
            f"Time       : {elapsed:.1f} s"
        )
        print(
            "LR         : "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        append_history(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_corr": metrics["corr"],
                "mean_pred": metrics["mean_pred"],
                "mean_target": metrics["mean_target"],
                "epoch_seconds": elapsed,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        epoch_path = (
            CHECKPOINT_DIR
            / f"baseline1b_epoch_{epoch:02d}.pt"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            metrics,
            epoch_path,
        )

        if np.isfinite(metrics["mae"]) and (
            metrics["mae"] < best_mae
        ):
            best_mae = metrics["mae"]

            best_path = (
                CHECKPOINT_DIR
                / "baseline1b_best.pt"
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                metrics,
                best_path,
            )

            print(
                "✓ New best checkpoint saved."
            )

        if device.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated(
                    device
                )
                / 1024**3
            )

            reserved = (
                torch.cuda.memory_reserved(
                    device
                )
                / 1024**3
            )

            print(
                f"VRAM allocated: "
                f"{allocated:.2f} GB | "
                f"reserved: "
                f"{reserved:.2f} GB"
            )

    print("\n" + "=" * 80)
    print("Baseline 1B training complete.")
    print("=" * 80)
    print(
        f"Best validation MAE: {best_mae:.4f} m"
    )
    print(
        f"History             : {HISTORY_PATH}"
    )
    print(
        f"Best checkpoint     : "
        f"{CHECKPOINT_DIR / 'baseline1b_best.pt'}"
    )


if __name__ == "__main__":
    main()
