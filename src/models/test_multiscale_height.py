import torch

from depthwizard_multiscale_height import (
    DepthWizardMultiScaleHeightModel,
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DepthWizard Multi-Scale Height Model Smoke Test")
    print("=" * 70)

    print("Device:", device)

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = DepthWizardMultiScaleHeightModel(
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
            dummy
        )

    print(
        "\nInput:",
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

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total:,}",
    )

    print(
        "Trainable parameters:",
        f"{trainable:,}",
    )

    if device == "cuda":

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            "VRAM allocated:",
            f"{allocated:.2f} GB",
        )

        print(
            "VRAM reserved:",
            f"{reserved:.2f} GB",
        )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()