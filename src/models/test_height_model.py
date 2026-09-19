from __future__ import annotations

import torch

from depthwizard_height import DepthWizardHeightModel


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("DepthWizard Height Model Smoke Test")
    print("=" * 70)

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = DepthWizardHeightModel(
        freeze_backbone=True,
    ).to(device)

    model.eval()

    dummy = torch.randn(
        2,
        3,
        512,
        512,
        device=device,
    )

    with torch.no_grad():

        prediction = model(
            dummy,
        )

    print(
        "Input :",
        dummy.shape,
    )

    print(
        "Output:",
        prediction.shape,
    )

    print(
        "Output min:",
        float(prediction.min()),
    )

    print(
        "Output max:",
        float(prediction.max()),
    )

    print(
        "Output mean:",
        float(prediction.mean()),
    )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()