"""
Open-Set Detection Head with Unknown-Aware Classification.

Implements OpenDet-style unknown-aware logit adjustment and prototype-based
open-set recognition for detecting both known and unknown safety violations.

References:
- OpenDet: Expanding Open-Set Object Detection (Han et al., 2022)
- Prototype-based Out-of-Distribution Detection (Ming et al., 2022)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    """
    Prototype memory bank for open-set detection.

    Stores per-class feature prototypes updated via exponential moving average.
    Unknown classes are detected by low similarity to any known prototype.
    """

    def __init__(
        self,
        num_known_classes: int,
        prototype_dim: int = 256,
        num_prototypes_per_class: int = 10,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.prototype_dim = prototype_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.momentum = momentum

        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(num_known_classes, num_prototypes_per_class, prototype_dim),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.prototypes)

    def compute_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between features and prototypes.

        Args:
            features: (N, D) feature vectors.

        Returns:
            similarity: (N, num_known_classes) max similarity per class.
        """
        # Normalize features and prototypes
        features = F.normalize(features, dim=-1)  # (N, D)
        protos = F.normalize(self.prototypes, dim=-1)  # (C, K, D)

        # Compute cosine similarity: (N, C, K)
        sim = torch.einsum("nd,ckd->nck", features, protos)
        # Max pooling over prototypes
        return sim.max(dim=-1).values  # (N, C)

    def update_prototypes(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Update prototypes using exponential moving average.

        Args:
            features: (N, D) feature vectors.
            labels: (N,) class labels (known classes only).
        """
        with torch.no_grad():
            for c in range(self.num_known_classes):
                mask = labels == c
                if mask.sum() == 0:
                    continue
                class_feats = F.normalize(features[mask], dim=-1)  # (M, D)
                # Update first prototype slot with running mean
                current_proto = self.prototypes[c, 0].detach()
                mean_feat = class_feats.mean(0)
                updated = self.momentum * current_proto + (1 - self.momentum) * mean_feat
                self.prototypes.data[c, 0] = updated


class UnknownAwareClassifier(nn.Module):
    """
    Unknown-aware classifier that adjusts logits for open-set detection.

    Uses the PROSER approach: models unknown as a placeholder class with
    logits derived from the margin of known class predictions.
    """

    def __init__(
        self,
        in_features: int,
        num_known_classes: int,
        unknown_weight: float = 1.0,
    ):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.unknown_weight = unknown_weight

        # Standard classifier for known classes
        self.cls_linear = nn.Linear(in_features, num_known_classes)

        # Objectness head for unknown detection
        self.obj_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (N, D) RoI features.

        Returns:
            cls_scores: (N, num_known_classes + 1) class scores including unknown.
            objectness: (N, 1) objectness score.
        """
        # Known class logits
        known_logits = self.cls_linear(features)  # (N, C)

        # Unknown score: based on maximum known class score (unknown = low max score)
        # Using negative softmax entropy as known-ness score
        known_probs = torch.softmax(known_logits, dim=-1)
        unknown_score = 1.0 - known_probs.max(dim=-1, keepdim=True).values
        unknown_logit = unknown_score * self.unknown_weight

        # Concatenate [known_logits, unknown_logit]
        cls_scores = torch.cat([known_logits, unknown_logit], dim=-1)

        # Objectness score
        objectness = torch.sigmoid(self.obj_head(features))

        return cls_scores, objectness


class OpenSetDetectionHead(nn.Module):
    """
    Open-Set Detection Head combining:
    1. Multi-scale feature aggregation
    2. Unknown-aware classification
    3. Bounding box regression
    4. Prototype-based open-set recognition

    Args:
        in_channels: Number of input feature channels (from FPN).
        num_known_classes: Number of known (closed-set) classes.
        feat_channels: Internal feature channels.
        stacked_convs: Number of stacked convolutions before prediction heads.
        use_prototypes: Whether to use prototype memory for open-set detection.
        prototype_dim: Dimension of prototype features.
        num_prototypes_per_class: Number of stored prototypes per class.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_known_classes: int = 4,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_prototypes: bool = True,
        prototype_dim: int = 256,
        num_prototypes_per_class: int = 10,
    ):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.num_classes = num_known_classes + 1  # +1 for unknown
        self.use_prototypes = use_prototypes

        # Shared feature extraction layers
        cls_convs = []
        reg_convs = []
        for i in range(stacked_convs):
            in_ch = in_channels if i == 0 else feat_channels
            cls_convs.extend([
                nn.Conv2d(in_ch, feat_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, feat_channels),
                nn.ReLU(inplace=True),
            ])
            reg_convs.extend([
                nn.Conv2d(in_ch, feat_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, feat_channels),
                nn.ReLU(inplace=True),
            ])

        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)

        # Classification head (unknown-aware)
        self.cls_head = UnknownAwareClassifier(
            in_features=feat_channels,
            num_known_classes=num_known_classes,
        )

        # Regression head (DFL-style with 4 * reg_max channels)
        self.reg_max = 16
        self.reg_head = nn.Conv2d(feat_channels, 4 * self.reg_max, 1)

        # Centerness / objectness head
        self.centerness_head = nn.Conv2d(feat_channels, 1, 1)

        # Prototype memory
        if use_prototypes:
            self.prototype_memory = PrototypeMemory(
                num_known_classes=num_known_classes,
                prototype_dim=prototype_dim,
                num_prototypes_per_class=num_prototypes_per_class,
            )
        else:
            self.prototype_memory = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize head weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias init for classification (prior probability 0.01)
        if hasattr(self.cls_head, "cls_linear"):
            nn.init.constant_(self.cls_head.cls_linear.bias, -4.0)

    def forward_single(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single feature map level."""
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)

        # For classification, flatten spatial dims and apply unknown-aware head
        N, C, H, W = cls_feat.shape
        cls_flat = cls_feat.permute(0, 2, 3, 1).reshape(-1, C)
        cls_scores, objectness_flat = self.cls_head(cls_flat)
        cls_scores = cls_scores.reshape(N, H, W, self.num_classes).permute(0, 3, 1, 2)
        objectness = objectness_flat.reshape(N, H, W, 1).permute(0, 3, 1, 2)

        # Regression
        bbox_pred = self.reg_head(reg_feat)

        # Centerness
        centerness = torch.sigmoid(self.centerness_head(reg_feat))

        return cls_scores, bbox_pred, centerness, objectness

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass over all FPN levels.

        Args:
            features: List of feature maps from FPN.

        Returns:
            Tuple of (cls_scores, bbox_preds, centernesses, objectnesses) lists.
        """
        all_cls, all_bbox, all_ctr, all_obj = [], [], [], []
        for feat in features:
            cls, bbox, ctr, obj = self.forward_single(feat)
            all_cls.append(cls)
            all_bbox.append(bbox)
            all_ctr.append(ctr)
            all_obj.append(obj)
        return all_cls, all_bbox, all_ctr, all_obj

    def get_prototype_similarity(
        self, roi_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute prototype similarity for open-set scoring.

        Args:
            roi_features: (N, D) RoI feature vectors.

        Returns:
            similarity: (N, num_known_classes) or None.
        """
        if self.prototype_memory is None:
            return None
        return self.prototype_memory.compute_similarity(roi_features)
