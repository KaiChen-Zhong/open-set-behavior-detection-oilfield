"""
Multi-Task Fusion Model integrating detection, classification, and spatial reasoning.

Architecture:
    Image -> Backbone -> FPN -> [Detection Head, Classification Head, Spatial Reasoning Head]
                                    └──────────── Attention Fusion ────────────┘
                                                     └─ Final Predictions
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPDarknet, FPN, SwinTransformerBackbone, build_backbone
from .detection_head import OpenSetDetectionHead
from .spatial_reasoning import SpatialReasoningModule


class ClassificationHead(nn.Module):
    """
    Global/RoI classification head for violation type classification.

    Args:
        in_channels: Input feature channels.
        num_classes: Number of output classes (including unknown).
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of FPN feature maps.

        Returns:
            logits: (N, num_classes) classification logits.
        """
        # Use the middle FPN level (P4) as the primary classification feature
        feat = features[len(features) // 2]
        pooled = self.pool(feat)
        return self.classifier(pooled)


class AttentionFusion(nn.Module):
    """
    Cross-task attention fusion module.

    Fuses features from multiple task heads using multi-head cross-attention
    to enable information sharing between tasks.

    Args:
        in_channels: Feature dimension from each task head.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, task_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fuse features across tasks via cross-attention.

        Args:
            task_features: List of (N, D) task feature tensors.

        Returns:
            fused_features: List of fused (N, D) tensors.
        """
        if len(task_features) <= 1:
            return task_features

        # Stack as (N, T, D) where T = number of tasks
        stacked = torch.stack(task_features, dim=1)  # (N, T, D)

        # Self-attention across task features
        attn_out, _ = self.attn(stacked, stacked, stacked)
        stacked = self.norm(stacked + attn_out)
        stacked = self.norm2(stacked + self.ff(stacked))

        # Split back into per-task features
        return [stacked[:, i, :] for i in range(stacked.shape[1])]


class MultiTaskDetector(nn.Module):
    """
    Multi-Task Detector for open-set safety violation detection.

    Performs simultaneous:
    1. Open-set object detection (detection head)
    2. Violation type classification (classification head)
    3. Spatial relationship reasoning (spatial reasoning head)

    Args:
        backbone_cfg: Backbone configuration dict.
        fpn_out_channels: FPN output channels.
        num_known_classes: Number of known violation classes.
        num_classes: Total number of classes (including unknown).
        enable_spatial_reasoning: Whether to enable spatial reasoning head.
        enable_classification: Whether to enable classification head.
    """

    def __init__(
        self,
        backbone_cfg: Optional[Dict] = None,
        fpn_out_channels: int = 256,
        num_known_classes: int = 4,
        num_classes: int = 5,
        enable_spatial_reasoning: bool = True,
        enable_classification: bool = True,
    ):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.num_classes = num_classes
        self.enable_spatial_reasoning = enable_spatial_reasoning
        self.enable_classification = enable_classification

        # Build backbone
        if backbone_cfg is None:
            backbone_cfg = {"type": "CSPDarknet", "variant": "n"}
        self.backbone = build_backbone(backbone_cfg)

        # Build FPN neck
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_out_channels,
            num_outs=4,
        )

        # Task heads
        self.detection_head = OpenSetDetectionHead(
            in_channels=fpn_out_channels,
            num_known_classes=num_known_classes,
            feat_channels=fpn_out_channels,
        )

        if enable_classification:
            self.cls_head = ClassificationHead(
                in_channels=fpn_out_channels,
                num_classes=num_classes,
            )
        else:
            self.cls_head = None

        if enable_spatial_reasoning:
            self.spatial_head = SpatialReasoningModule(
                in_channels=fpn_out_channels,
                hidden_dim=fpn_out_channels,
            )
        else:
            self.spatial_head = None

        # Cross-task attention fusion
        self.fusion = AttentionFusion(
            in_channels=fpn_out_channels,
            num_heads=8,
        )

        # Global average pool for getting task-level features
        self.gap = nn.AdaptiveAvgPool2d(1)

    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from backbone + FPN."""
        backbone_feats = self.backbone(images)
        fpn_feats = self.fpn(backbone_feats)
        return fpn_feats

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            images: (N, 3, H, W) batch of images.
            targets: Optional list of target dicts (for training).

        Returns:
            Dictionary with task-specific predictions:
                - det_cls_scores: Detection classification scores per FPN level.
                - det_bbox_preds: Bounding box predictions per FPN level.
                - det_centernesses: Centerness predictions per FPN level.
                - cls_logits: Global classification logits (if enabled).
                - spatial_results: Spatial reasoning results (if enabled).
        """
        # Extract multi-scale features
        fpn_feats = self.extract_features(images)

        # Get task-level global features for fusion
        task_feats = [self.gap(f).flatten(1) for f in fpn_feats[:3]]
        # Ensure all have the same dimension by using first 3 FPN levels
        fused = self.fusion(task_feats)

        # Task 1: Open-set detection
        det_cls_scores, det_bbox_preds, det_centernesses, det_objectness = (
            self.detection_head(fpn_feats)
        )

        output: Dict[str, Any] = {
            "det_cls_scores": det_cls_scores,
            "det_bbox_preds": det_bbox_preds,
            "det_centernesses": det_centernesses,
            "det_objectness": det_objectness,
            "fpn_features": fpn_feats,
        }

        # Task 2: Global violation classification
        if self.cls_head is not None:
            cls_logits = self.cls_head(fpn_feats)
            output["cls_logits"] = cls_logits

        # Task 3: Spatial reasoning (applied to detected objects during inference)
        # During training, spatial reasoning uses ground-truth boxes
        if self.spatial_head is not None and targets is not None:
            spatial_results_batch = []
            for i, target in enumerate(targets):
                boxes = target.get("boxes", torch.zeros(0, 4))
                if boxes.shape[0] > 0:
                    # Use pooled FPN features as object features (ROI align-like)
                    # In a full implementation, use actual ROI Align
                    obj_feats = fused[0][i : i + 1].expand(boxes.shape[0], -1)
                    result = self.spatial_head(
                        obj_feats, boxes.to(images.device)
                    )
                else:
                    result = {
                        "relation_scores": torch.zeros(0, 0, self.spatial_head.num_relation_types),
                        "distance_preds": torch.zeros(0, 0, 1),
                        "violation_pairs": torch.zeros(0, 2, dtype=torch.long),
                        "node_features": torch.zeros(0, self.spatial_head.distance_threshold.__class__),
                    }
                spatial_results_batch.append(result)
            output["spatial_results"] = spatial_results_batch

        return output

    def get_num_parameters(self) -> Dict[str, int]:
        """Return parameter counts per component."""
        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone": count_params(self.backbone),
            "fpn": count_params(self.fpn),
            "detection_head": count_params(self.detection_head),
            "classification_head": count_params(self.cls_head) if self.cls_head else 0,
            "spatial_head": count_params(self.spatial_head) if self.spatial_head else 0,
            "fusion": count_params(self.fusion),
            "total": count_params(self),
        }


def build_model(cfg: Dict) -> MultiTaskDetector:
    """
    Build MultiTaskDetector from configuration dictionary.

    Args:
        cfg: Model configuration dictionary.

    Returns:
        MultiTaskDetector instance.
    """
    backbone_cfg = cfg.get("backbone", {"type": "CSPDarknet", "variant": "n"})
    fpn_out_channels = cfg.get("fpn_out_channels", 256)
    num_known_classes = cfg.get("num_known_classes", 4)
    num_classes = cfg.get("num_classes", 5)
    enable_spatial = cfg.get("enable_spatial_reasoning", True)
    enable_cls = cfg.get("enable_classification", True)

    return MultiTaskDetector(
        backbone_cfg=backbone_cfg,
        fpn_out_channels=fpn_out_channels,
        num_known_classes=num_known_classes,
        num_classes=num_classes,
        enable_spatial_reasoning=enable_spatial,
        enable_classification=enable_cls,
    )
