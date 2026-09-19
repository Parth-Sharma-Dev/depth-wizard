from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"


class HeightHead(nn.Module):
    """
    Lightweight decoder for AGL regression.

    The backbone provides a spatial feature map.
    The decoder predicts one height value per pixel.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:

        x = self.decoder(features)

        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return x[:, 0]


class DepthWizardHeightModel(nn.Module):
    """
    Depth Anything V2 Large backbone + dedicated AGL regression head.

    For the first experiment the backbone is frozen and only the
    height regression head is trainable.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone_model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_NAME,
        )

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        # Depth Anything V2 Large's final backbone feature-map channels
        # correspond to the backbone hidden size.
        hidden_size = (
            self.backbone_model
            .config
            .backbone_config
            .hidden_size
        )

        self.height_head = HeightHead(
            in_channels=hidden_size,
        )

    def _extract_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the deepest spatial backbone feature map.

        Current Transformers returns a BackboneOutput from the
        AutoBackbone path. Its feature_maps field contains spatial
        feature maps from the configured backbone stages.
        """

        backbone_outputs = self.backbone_model.backbone(
            pixel_values=pixel_values,
        )

        if hasattr(backbone_outputs, "feature_maps"):
            feature_maps = backbone_outputs.feature_maps

            if feature_maps is None or len(feature_maps) == 0:
                raise RuntimeError(
                    "Backbone returned an empty feature_maps collection."
                )

            features = feature_maps[-1]

        elif hasattr(backbone_outputs, "last_hidden_state"):
            # Compatibility fallback for older Transformers variants.
            features = backbone_outputs.last_hidden_state

            if features.ndim == 3:
                batch, tokens, channels = features.shape

                patch_size = (
                    self.backbone_model
                    .config
                    .backbone_config
                    .patch_size
                )

                height = pixel_values.shape[-2] // patch_size
                width = pixel_values.shape[-1] // patch_size

                expected_tokens = height * width

                if tokens != expected_tokens:
                    raise RuntimeError(
                        f"Unexpected token count: {tokens}; "
                        f"expected {expected_tokens}."
                    )

                features = features.transpose(1, 2).reshape(
                    batch,
                    channels,
                    height,
                    width,
                )

        else:
            raise RuntimeError(
                "Unsupported backbone output. Expected either "
                "'feature_maps' or 'last_hidden_state'. "
                f"Got: {type(backbone_outputs).__name__}"
            )

        # Some current Transformers/DINOv2 configurations expose the
        # deepest feature map as tokens: [B, N, C]. For a 512x512 input
        # here, the observed shape is [B, 1297, 1024] = 1 CLS token +
        # 36*36 patch tokens.
        if features.ndim == 3:
            batch, tokens, channels = features.shape

            # DINOv2 uses one prefix/class token by default.
            prefix_tokens = 1
            spatial_tokens = tokens - prefix_tokens

            grid_size = int(spatial_tokens ** 0.5)

            if grid_size * grid_size != spatial_tokens:
                raise RuntimeError(
                    "Cannot infer a square spatial feature map from "
                    f"token count {tokens}. After removing {prefix_tokens} "
                    f"prefix token(s), got {spatial_tokens} tokens."
                )

            features = features[:, prefix_tokens:, :]
            features = features.transpose(1, 2).contiguous()
            features = features.reshape(
                batch,
                channels,
                grid_size,
                grid_size,
            )

        elif features.ndim != 4:
            raise RuntimeError(
                "Expected backbone features with 3 dimensions "
                "[B,N,C] or 4 dimensions [B,C,H,W], "
                f"got {tuple(features.shape)}."
            )

        return features

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        features = self._extract_features(
            pixel_values
        )

        prediction = self.height_head(
            features,
            target_size=pixel_values.shape[-2:],
        )

        return prediction
