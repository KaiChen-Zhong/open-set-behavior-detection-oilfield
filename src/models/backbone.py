"""
Shared backbone networks for feature extraction.

Supports:
- CSPDarknet (from YOLOv5/v8)
- Swin Transformer (from timm)

Both backbones output multi-scale feature maps compatible with FPN/PANet necks.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Conv + BN + Activation building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1e-3)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard CSP Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBnAct(in_channels, hidden, 1)
        self.cv2 = ConvBnAct(hidden, out_channels, 3, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """
    CSP bottleneck with 2 convolutions (C2f) - used in YOLOv8.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBnAct(in_channels, 2 * hidden, 1)
        self.cv2 = ConvBnAct((2 + num_blocks) * hidden, out_channels, 1)
        self.bottlenecks = nn.ModuleList(
            [
                Bottleneck(hidden, hidden, shortcut, groups, expansion=1.0)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone as used in YOLOv8.

    Outputs multi-scale features at strides [8, 16, 32].

    Args:
        depth_multiple: Controls number of bottleneck blocks.
        width_multiple: Controls channel width multiplier.
        in_channels: Input image channels (default: 3).
    """

    # Architecture table: [from, num_repeats, module, args]
    DEPTH_SCALE = {
        "n": (0.33, 0.25),
        "s": (0.33, 0.50),
        "m": (0.67, 0.75),
        "l": (1.00, 1.00),
        "x": (1.00, 1.25),
    }

    def __init__(
        self,
        variant: str = "n",
        in_channels: int = 3,
    ):
        super().__init__()
        d, w = self.DEPTH_SCALE.get(variant, (0.33, 0.25))

        def make_c(c: int) -> int:
            return max(round(c * w), 1)

        def make_n(n: int) -> int:
            return max(round(n * d), 1)

        # Stem
        self.stem = ConvBnAct(in_channels, make_c(64), 3, 2)

        # Stage 1 -> stride 4
        self.stage1 = nn.Sequential(
            ConvBnAct(make_c(64), make_c(128), 3, 2),
            C2f(make_c(128), make_c(128), make_n(3), shortcut=True),
        )

        # Stage 2 -> stride 8
        self.stage2 = nn.Sequential(
            ConvBnAct(make_c(128), make_c(256), 3, 2),
            C2f(make_c(256), make_c(256), make_n(6), shortcut=True),
        )

        # Stage 3 -> stride 16
        self.stage3 = nn.Sequential(
            ConvBnAct(make_c(256), make_c(512), 3, 2),
            C2f(make_c(512), make_c(512), make_n(6), shortcut=True),
        )

        # Stage 4 -> stride 32
        self.stage4 = nn.Sequential(
            ConvBnAct(make_c(512), make_c(1024), 3, 2),
            C2f(make_c(1024), make_c(1024), make_n(3), shortcut=True),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            if False
            else nn.Identity(),
        )

        self.out_channels = [make_c(256), make_c(512), make_c(1024)]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns features at strides [8, 16, 32]."""
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)  # stride 8
        c4 = self.stage3(c3)  # stride 16
        c5 = self.stage4(c4)  # stride 32
        return [c3, c4, c5]


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone using timm library.

    Outputs multi-scale features at strides [8, 16, 32] (indices 1, 2, 3).

    Args:
        variant: One of "swin_tiny", "swin_small", "swin_base".
        pretrained: Load ImageNet pretrained weights.
        out_indices: Feature map indices to return.
        frozen_stages: Number of stages to freeze.
    """

    def __init__(
        self,
        variant: str = "swin_tiny",
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        frozen_stages: int = 1,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for SwinTransformerBackbone. "
                "Install with: pip install timm"
            ) from e

        self.out_indices = out_indices
        self.model = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.out_channels = self.model.feature_info.channels()
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze first num_stages stages of the backbone."""
        if num_stages <= 0:
            return
        for name, param in self.model.named_parameters():
            for stage_idx in range(num_stages):
                if f"layers.{stage_idx}" in name or name.startswith("patch_embed"):
                    param.requires_grad = False
                    break

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps at selected stages."""
        return self.model(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network neck.

    Fuses multi-scale backbone features into a unified feature pyramid.

    Args:
        in_channels: List of input channel sizes from backbone.
        out_channels: Number of output channels for all pyramid levels.
        num_outs: Number of output feature maps.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 5,
    ):
        super().__init__()
        assert len(in_channels) >= 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(ch, out_channels, 1, bias=False)
            )
            self.fpn_convs.append(
                ConvBnAct(out_channels, out_channels, 3)
            )

        # Extra layers for P6, P7 if needed
        self.extra_convs = nn.ModuleList()
        for _ in range(max(0, num_outs - len(in_channels))):
            self.extra_convs.append(
                ConvBnAct(out_channels, out_channels, 3, stride=2)
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == len(self.in_channels)

        # Lateral connections
        laterals = [l(x) for l, x in zip(self.lateral_convs, inputs)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        # Build output pyramid
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # Add extra levels
        if self.extra_convs:
            x = outs[-1]
            for extra in self.extra_convs:
                x = extra(x)
                outs.append(x)

        return outs[: self.num_outs]


def build_backbone(cfg: Dict) -> nn.Module:
    """
    Factory function to build backbone from config dict.

    Args:
        cfg: Backbone config with keys: type, variant, pretrained, etc.

    Returns:
        Backbone module.
    """
    backbone_type = cfg.get("type", "CSPDarknet")
    variant = cfg.get("variant", "n")
    pretrained = cfg.get("pretrained", True)
    out_indices = tuple(cfg.get("out_indices", [1, 2, 3]))
    frozen_stages = cfg.get("frozen_stages", 1)

    if backbone_type == "CSPDarknet":
        return CSPDarknet(variant=variant)
    elif backbone_type == "SwinTransformer":
        return SwinTransformerBackbone(
            variant=variant,
            pretrained=pretrained,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
