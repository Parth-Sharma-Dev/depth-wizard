from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"


class FeatureProjection(nn.Module):
    """
    Converts a DINOv2 feature map to a common decoder width.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class FusionBlock(nn.Module):
    """
    Refines a fused multi-scale representation.
    """

    def __init__(
        self,
        channels: int = 256,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleHeightHead(nn.Module):
    """
    DPT-inspired multi-scale AGL decoder.

    Four feature maps are projected to a common width and fused from
    coarse to fine spatial scales.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        decoder_channels: int = 256,
    ):
        super().__init__()

        self.projections = nn.ModuleList([
            FeatureProjection(
                in_channels,
                decoder_channels,
            )
            for _ in range(4)
        ])

        self.fuse_24 = FusionBlock(
            decoder_channels
        )

        self.fuse_18 = FusionBlock(
            decoder_channels
        )

        self.fuse_12 = FusionBlock(
            decoder_channels
        )

        self.fuse_5 = FusionBlock(
            decoder_channels
        )

        self.output_head = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                64,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        target_size: tuple[int, int],
    ):

        if len(feature_maps) != 4:
            raise RuntimeError(
                "Expected four intermediate feature maps, "
                f"got {len(feature_maps)}."
            )

        projected = [
            projection(feature)
            for projection, feature in zip(
                self.projections,
                feature_maps,
            )
        ]

        # Expected order:
        # stage5, stage12, stage18, stage24
        f5, f12, f18, f24 = projected

        # Start from the deepest feature.
        x = self.fuse_24(f24)

        # Stage 18
        x = F.interpolate(
            x,
            size=f18.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = x + f18
        x = self.fuse_18(x)

        # Stage 12
        x = F.interpolate(
            x,
            size=f12.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = x + f12
        x = self.fuse_12(x)

        # Stage 5
        x = F.interpolate(
            x,
            size=f5.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = x + f5
        x = self.fuse_5(x)

        # Final restoration to RGB resolution.
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.output_head(x)[:, 0]


class DepthWizardMultiScaleHeightModel(nn.Module):
    """
    Depth Anything V2 Large + multi-scale AGL decoder.

    Baseline 1B initially freezes the pretrained backbone and trains only
    the new multi-scale decoder.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        decoder_channels: int = 256,
    ):
        super().__init__()

        self.backbone_model = (
            AutoModelForDepthEstimation.from_pretrained(
                MODEL_NAME,
            )
        )

        if freeze_backbone:
            for parameter in self.backbone_model.parameters():
                parameter.requires_grad = False

        backbone_config = (
            self.backbone_model
            .config
            .backbone_config
        )

        hidden_size = backbone_config.hidden_size

        self.height_head = MultiScaleHeightHead(
            in_channels=hidden_size,
            decoder_channels=decoder_channels,
        )

    @staticmethod
    def _tokens_to_map(
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert DINOv2 token features:

            [B, 1297, 1024]

        into:

            [B, 1024, 36, 36]

        for a 512x512 input.

        The first token is the CLS/prefix token.
        """

        if features.ndim == 4:
            return features

        if features.ndim != 3:
            raise RuntimeError(
                "Expected [B,N,C] or [B,C,H,W], "
                f"got {tuple(features.shape)}."
            )

        batch, tokens, channels = features.shape

        # Remove CLS token.
        spatial_tokens = tokens - 1

        grid = math.isqrt(spatial_tokens)

        if grid * grid != spatial_tokens:
            raise RuntimeError(
                f"Cannot reshape {spatial_tokens} patch tokens "
                f"into a square grid."
            )

        features = features[:, 1:, :]

        features = features.transpose(
            1,
            2,
        ).contiguous()

        features = features.reshape(
            batch,
            channels,
            grid,
            grid,
        )

        return features

    def _extract_intermediate_features(
        self,
        pixel_values: torch.Tensor,
    ) -> list[torch.Tensor]:

        outputs = self.backbone_model.backbone(
            pixel_values=pixel_values,
        )

        if not hasattr(
            outputs,
            "feature_maps",
        ):
            raise RuntimeError(
                "Current Transformers backbone does not expose "
                "'feature_maps'."
            )

        feature_maps = outputs.feature_maps

        if feature_maps is None:
            raise RuntimeError(
                "Backbone returned feature_maps=None."
            )

        if len(feature_maps) != 4:
            raise RuntimeError(
                "Expected four intermediate feature maps from "
                "Depth Anything V2 Large, "
                f"got {len(feature_maps)}."
            )

        return [
            self._tokens_to_map(feature)
            for feature in feature_maps
        ]

    def forward(
        self,
        pixel_values: torch.Tensor,
    ):

        feature_maps = (
            self._extract_intermediate_features(
                pixel_values
            )
        )

        prediction = self.height_head(
            feature_maps,
            target_size=pixel_values.shape[-2:],
        )

        return prediction