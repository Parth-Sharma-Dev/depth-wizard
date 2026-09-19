from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from gamus_dataset import GAMUSDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "gamus_quality"
    / "gamus_clean_manifest.csv"
)

# Adjust if your local GAMUS cache lives elsewhere.
GAMUS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "gamus_local"
)


def main():
    manifest = pd.read_csv(MANIFEST)

    dataset = GAMUSDataset(
        manifest=manifest,
        data_root=GAMUS_ROOT,
        split="train",
        crop_size=512,
        samples_per_scene=2,
        training=True,
        augment=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("Dataset smoke test")
    print("==================")
    print("Image       :", batch["image"].shape)
    print("Height      :", batch["height"].shape)
    print("Valid mask  :", batch["valid_mask"].shape)
    print("Weight      :", batch["weight"].shape)
    print("Scenes      :", batch["scene"])

    print(
        "Image range:",
        float(batch["image"].min()),
        "->",
        float(batch["image"].max()),
    )

    valid = batch["valid_mask"]

    if valid.any():
        print(
            "Valid height range:",
            float(batch["height"][valid].min()),
            "->",
            float(batch["height"][valid].max()),
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
