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

# ---------------------------------------------------------------------
# Make project root importable when running:
# python src/training/train_baseline1a.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gamus_dataset import GAMUSDataset
from src.data.gamus_quality import QualityPolicy
from src.models.depthwizard_height import DepthWizardHeightModel


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "gamus_quality"
    / "gamus_clean_manifest.csv"
)

GAMUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "gamus_local"
)

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "baseline1a"
)

CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

HISTORY_PATH = EXPERIMENT_DIR / "history.csv"

SEED = 42

CROP_SIZE = 512

# Start conservatively for the RTX 5060 8 GB.
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2

# Random crops generated per scene per epoch.
SAMPLES_PER_SCENE = 8

EPOCHS = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0  # Windows-safe starting point.

VAL_FRACTION = 0.20

GRAD_CLIP_NORM = 1.0

USE_AMP = True

# DINOv2 / ImageNet-style normalization used by the pretrained backbone.
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


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}\n\n"
            "Build it first with build_gamus_manifest.py."
        )

    df = pd.read_csv(MANIFEST_PATH)

    if "scene" not in df.columns:
        raise ValueError("Manifest must contain a 'scene' column.")

    df = df.drop_duplicates("scene").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "Need at least 2 scenes for a train/validation split."
        )

    # Deterministic scene-level split.
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    val_count = max(
        1,
        int(round(len(df) * VAL_FRACTION)),
    )

    val_indices = indices[:val_count]

    train_mask = np.ones(len(df), dtype=bool)
    train_mask[val_indices] = False

    train_df = df.iloc[np.flatnonzero(train_mask)].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)

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
) -> tuple[GAMUSDataset, GAMUSDataset]:

    quality_policy = QualityPolicy()

    train_dataset = GAMUSDataset(
        manifest=train_df,
        data_root=GAMUS_ROOT,
        split="train",
        crop_size=CROP_SIZE,
        samples_per_scene=SAMPLES_PER_SCENE,
        training=True,
        augment=True,
        quality_policy=quality_policy,
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
        quality_policy=quality_policy,
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

    per_pixel = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )

    weighted = per_pixel * weight

    denominator = weight.sum().clamp_min(1.0)

    return weighted.sum() / denominator


@torch.no_grad()
def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:

    mask = weight > 0

    pred = prediction[mask].float()
    true = target[mask].float()

    if pred.numel() == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "valid_pct": 0.0,
            "mean_pred": float("nan"),
            "mean_target": float("nan"),
        }

    error = pred - true

    mae = error.abs().mean()
    rmse = torch.sqrt((error ** 2).mean())

    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()

    corr_denominator = torch.sqrt(
        (pred_centered ** 2).sum()
        * (true_centered ** 2).sum()
    )

    if corr_denominator.item() > 1e-12:
        corr = (
            (pred_centered * true_centered).sum()
            / corr_denominator
        )
    else:
        corr = torch.tensor(
            float("nan"),
            device=pred.device,
        )

    valid_pct = (
        mask.float().mean() * 100.0
    )

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "corr": float(corr.item()),
        "valid_pct": float(valid_pct.item()),
        "mean_pred": float(pred.mean().item()),
        "mean_target": float(true.mean().item()),
    }


# ---------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------


def normalize_images(
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    mean = IMAGE_MEAN.to(
        device=device,
        dtype=images.dtype,
    )

    std = IMAGE_STD.to(
        device=device,
        dtype=images.dtype,
    )

    return (images - mean) / std


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    # Backbone is frozen. Keep it in eval mode so any backbone-specific
    # normalization/dropout behavior remains deterministic.
    model.backbone_model.eval()
    model.height_head.train()

    running_loss = 0.0
    total_weight = 0.0

    for batch_idx, batch in enumerate(loader, start=1):

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

        images = normalize_images(
            images,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        if USE_AMP and device.type == "cuda":
            with autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                prediction = model(images)

                loss = weighted_smooth_l1(
                    prediction,
                    heights,
                    weights,
                )

            if scaler is None:
                raise RuntimeError(
                    "AMP is enabled on CUDA but GradScaler is unavailable."
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

            loss = weighted_smooth_l1(
                prediction,
                heights,
                weights,
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

        running_loss += (
            float(loss.detach().item())
            * batch_weight
        )
        total_weight += batch_weight

        if batch_idx % 20 == 0:
            print(
                f"  batch {batch_idx:4d}/{len(loader):4d} "
                f"loss={loss.item():.5f}"
            )

    epoch_loss = (
        running_loss / max(total_weight, 1.0)
    )

    return epoch_loss, total_weight


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

    predictions_for_corr = []
    targets_for_corr = []

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

        images = normalize_images(
            images,
            device,
        )

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

        if pred.numel() == 0:
            continue

        w = weights.float()[mask]

        error = pred - true

        total_abs_error += float(
            (error.abs() * w).sum().item()
        )

        total_squared_error += float(
            ((error ** 2) * w).sum().item()
        )

        total_weight += float(
            w.sum().item()
        )

        # Correlation does not need the loss weights when reporting the
        # initial baseline. We collect valid pixels across validation crops.
        predictions_for_corr.append(pred.detach().cpu())
        targets_for_corr.append(true.detach().cpu())

    if total_weight <= 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "corr": float("nan"),
            "valid_pct": 0.0,
            "mean_pred": float("nan"),
            "mean_target": float("nan"),
        }

    mae = total_abs_error / total_weight
    rmse = float(
        np.sqrt(
            total_squared_error / total_weight
        )
    )

    pred_all = torch.cat(
        predictions_for_corr
    )
    true_all = torch.cat(
        targets_for_corr
    )

    pred_centered = pred_all - pred_all.mean()
    true_centered = true_all - true_all.mean()

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

    mean_pred = float(pred_all.mean().item())
    mean_target = float(true_all.mean().item())

    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "valid_pct": float(
            total_weight
            / max(len(loader.dataset) * CROP_SIZE * CROP_SIZE, 1)
            * 100.0
        ),
        "mean_pred": mean_pred,
        "mean_target": mean_target,
    }


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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def append_history(
    row: dict[str, object],
) -> None:

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = HISTORY_PATH.exists()

    with HISTORY_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not file_exists:
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
    print("DepthWizard - Baseline 1A Training")
    print("=" * 80)
    print(f"Device       : {device}")

    if device.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )
        print(
            f"VRAM         : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(f"Manifest     : {MANIFEST_PATH}")
    print(f"GAMUS root   : {GAMUS_ROOT}")
    print(f"Crop         : {CROP_SIZE}x{CROP_SIZE}")
    print(f"Batch        : {TRAIN_BATCH_SIZE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"LR           : {LEARNING_RATE}")
    print(f"AMP          : {USE_AMP}")

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    train_df, val_df = load_manifest()

    print("\nBuilding datasets...")

    train_dataset, val_dataset = build_datasets(
        train_df,
        val_df,
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

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    print("\nLoading model...")

    model = DepthWizardHeightModel(
        freeze_backbone=True,
    ).to(device)

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable_parameters
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters     : {total_count:,}"
    )
    print(
        f"Trainable parameters : {trainable_count:,}"
    )

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = (
        GradScaler("cuda")
        if USE_AMP and device.type == "cuda"
        else None
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 80)
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print("=" * 80)

        start = time.perf_counter()

        train_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        elapsed = (
            time.perf_counter() - start
        )

        print("\nEpoch summary")
        print(
            f"Train loss : {train_loss:.6f}"
        )
        print(
            f"Val MAE    : {val_metrics['mae']:.4f} m"
        )
        print(
            f"Val RMSE   : {val_metrics['rmse']:.4f} m"
        )
        print(
            f"Val Corr   : {val_metrics['corr']:.4f}"
        )
        print(
            f"Mean pred  : {val_metrics['mean_pred']:.4f} m"
        )
        print(
            f"Mean target: {val_metrics['mean_target']:.4f} m"
        )
        print(
            f"Time       : {elapsed:.1f} s"
        )
        print(
            f"LR         : {optimizer.param_groups[0]['lr']:.8f}"
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_corr": val_metrics["corr"],
            "mean_pred": val_metrics["mean_pred"],
            "mean_target": val_metrics["mean_target"],
            "epoch_seconds": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }

        append_history(history_row)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            path=CHECKPOINT_DIR / "baseline1a_last.pt",
        )

        if np.isfinite(val_metrics["mae"]) and (
            val_metrics["mae"] < best_mae
        ):
            best_mae = val_metrics["mae"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=CHECKPOINT_DIR / "baseline1a_best.pt",
            )

            print(
                "✓ New best checkpoint saved."
            )

        # Keep an eye on VRAM during development.
        if device.type == "cuda":
            allocated = (
                torch.cuda.memory_allocated(device)
                / 1024**3
            )
            reserved = (
                torch.cuda.memory_reserved(device)
                / 1024**3
            )

            print(
                f"VRAM allocated: {allocated:.2f} GB | "
                f"reserved: {reserved:.2f} GB"
            )

    print("\n" + "=" * 80)
    print("Baseline 1A training complete.")
    print("=" * 80)
    print(f"Best validation MAE: {best_mae:.4f} m")
    print(f"History             : {HISTORY_PATH}")
    print(
        f"Best checkpoint     : "
        f"{CHECKPOINT_DIR / 'baseline1a_best.pt'}"
    )
    print(
        f"Last checkpoint     : "
        f"{CHECKPOINT_DIR / 'baseline1a_last.pt'}"
    )


if __name__ == "__main__":
    main()
