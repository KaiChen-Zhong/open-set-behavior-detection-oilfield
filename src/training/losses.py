"""
Custom loss functions for multi-task open-set detection.

Includes:
- FocalLoss for class-imbalanced detection
- GIoULoss for bounding box regression
- SupConLoss for prototype-based contrastive learning
- MultiTaskLoss combining all task losses
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Args:
        alpha: Weighting factor for rare class (positive samples).
        gamma: Focusing parameter (0 = CE loss, higher = more focus on hard examples).
        reduction: Loss reduction mode ("mean", "sum", "none").
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) raw logits.
            targets: (N,) class indices or (N, C) one-hot labels.

        Returns:
            Focal loss scalar.
        """
        if targets.dim() == 1:
            targets_oh = F.one_hot(targets, num_classes=inputs.shape[-1]).float()
        else:
            targets_oh = targets.float()

        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets_oh, reduction="none")
        p_t = probs * targets_oh + (1 - probs) * (1 - targets_oh)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets_oh + (1 - self.alpha) * (1 - targets_oh)
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.

    Reference: "Generalized Intersection over Union" (Rezatofighi et al., 2019)

    Args:
        reduction: Loss reduction mode.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) predicted boxes in xyxy format.
            target_boxes: (N, 4) target boxes in xyxy format.

        Returns:
            GIoU loss scalar.
        """
        giou = self._compute_giou(pred_boxes, target_boxes)
        loss = 1 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_giou(
        self, boxes1: torch.Tensor, boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute GIoU between two sets of boxes."""
        eps = 1e-7
        # Intersection
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter + eps
        iou = inter / union

        # Enclosing box
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + eps

        giou = iou - (enc_area - union) / enc_area
        return giou


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for prototype-based open-set detection.

    Reference: "Supervised Contrastive Learning" (Khosla et al., 2020)

    Args:
        temperature: Contrastive temperature parameter.
        base_temperature: Base temperature for scaling.
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (N, D) normalized feature vectors.
            labels: (N,) class labels.

        Returns:
            SupCon loss scalar.
        """
        N = features.shape[0]
        device = features.device

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (N, N)

        # Build mask: 1 where same class (excluding self)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        positive_mask = labels_eq & ~self_mask

        # Compute log-sum-exp denominator (excluding self)
        exp_sim = torch.exp(sim_matrix) * (~self_mask).float()
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)

        # Loss for each anchor: mean over positives
        num_positives = positive_mask.sum(dim=1).float()
        loss_per_anchor = -(positive_mask.float() * log_prob).sum(dim=1)
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = (loss_per_anchor[valid] / num_positives[valid]).mean()
        return loss * (self.temperature / self.base_temperature)


class SpatialRelationLoss(nn.Module):
    """
    Loss for spatial relationship reasoning head.

    Combines:
    - Relation classification loss (BCE)
    - Distance regression loss (MSE)
    """

    def __init__(
        self,
        relation_weight: float = 1.0,
        distance_weight: float = 0.5,
    ):
        super().__init__()
        self.relation_weight = relation_weight
        self.distance_weight = distance_weight

    def forward(
        self,
        relation_scores: torch.Tensor,
        distance_preds: torch.Tensor,
        relation_targets: Optional[torch.Tensor],
        distance_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            relation_scores: (N, N, R) predicted relation scores.
            distance_preds: (N, N, 1) predicted distances.
            relation_targets: (N, N, R) binary relation labels.
            distance_targets: (N, N, 1) ground truth distances.

        Returns:
            Combined spatial loss scalar.
        """
        loss = torch.tensor(0.0, device=relation_scores.device)

        if relation_targets is not None:
            rel_loss = F.binary_cross_entropy_with_logits(
                relation_scores, relation_targets.float()
            )
            loss = loss + self.relation_weight * rel_loss

        if distance_targets is not None:
            dist_loss = F.mse_loss(distance_preds, distance_targets)
            loss = loss + self.distance_weight * dist_loss

        return loss


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for the open-set safety violation detector.

    Combines:
    1. Detection loss (classification + box regression)
    2. Violation classification loss
    3. Spatial reasoning loss
    4. Prototype contrastive loss (optional)

    Args:
        num_known_classes: Number of known classes.
        cls_weight: Weight for detection classification loss.
        bbox_weight: Weight for box regression loss.
        task_cls_weight: Weight for global classification task.
        spatial_weight: Weight for spatial reasoning task.
        contrastive_weight: Weight for contrastive learning.
        focal_alpha: Focal loss alpha.
        focal_gamma: Focal loss gamma.
    """

    def __init__(
        self,
        num_known_classes: int = 4,
        cls_weight: float = 1.0,
        bbox_weight: float = 2.0,
        task_cls_weight: float = 0.5,
        spatial_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.task_cls_weight = task_cls_weight
        self.spatial_weight = spatial_weight
        self.contrastive_weight = contrastive_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.giou_loss = GIoULoss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.supcon_loss = SupConLoss()
        self.spatial_loss_fn = SpatialRelationLoss()

    def compute_detection_loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses across FPN levels.

        For simplicity, uses image-level ground truth boxes.
        A full implementation would use anchor/point assignment.
        """
        det_cls_loss = torch.tensor(0.0)
        det_bbox_loss = torch.tensor(0.0)
        num_valid_levels = 0

        for level_cls, level_bbox in zip(cls_scores, bbox_preds):
            device = level_cls.device
            # Flatten spatial dimensions: (N, C, H, W) -> (N*H*W, C)
            N, C, H, W = level_cls.shape
            cls_flat = level_cls.permute(0, 2, 3, 1).reshape(-1, C)

            # Create pseudo-labels (background = 0) for demonstration
            # A real implementation would use label assignment algorithms
            pseudo_labels = torch.zeros(N * H * W, dtype=torch.long, device=device)

            # Sample foreground from targets
            for i, target in enumerate(targets):
                boxes = target.get("boxes", torch.zeros(0, 4)).to(device)
                labels = target.get("labels", torch.zeros(0, dtype=torch.long)).to(device)
                if len(boxes) > 0:
                    # Simple: mark center of feature map as foreground
                    cx, cy = H // 2, W // 2
                    idx = i * H * W + cy * W + cx
                    if idx < N * H * W and len(labels) > 0:
                        pseudo_labels[idx] = labels[0]

            # Focal loss on all positions
            targets_oh = F.one_hot(pseudo_labels.clamp(0, C - 1), num_classes=C).float()
            level_cls_loss = self.focal_loss(cls_flat, targets_oh)
            det_cls_loss = det_cls_loss + level_cls_loss
            num_valid_levels += 1

        if num_valid_levels > 0:
            det_cls_loss = det_cls_loss / num_valid_levels

        return {
            "det_cls_loss": det_cls_loss,
            "det_bbox_loss": det_bbox_loss,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all task losses.

        Args:
            outputs: Model output dict from MultiTaskDetector.forward().
            targets: List of target dicts per image.

        Returns:
            Dictionary of individual losses and total weighted loss.
        """
        losses: Dict[str, torch.Tensor] = {}
        # Determine device from any tensor in outputs
        device = torch.device("cpu")
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                device = v.device
                break
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                if isinstance(v[0], torch.Tensor):
                    device = v[0].device
                    break
        total_loss = torch.tensor(0.0, device=device)

        # Detection loss
        if "det_cls_scores" in outputs and "det_bbox_preds" in outputs:
            det_losses = self.compute_detection_loss(
                outputs["det_cls_scores"],
                outputs["det_bbox_preds"],
                targets,
            )
            losses.update(det_losses)
            total_loss = total_loss + (
                self.cls_weight * det_losses.get("det_cls_loss", 0)
                + self.bbox_weight * det_losses.get("det_bbox_loss", 0)
            )

        # Global classification loss
        if "cls_logits" in outputs:
            # Gather image-level labels from targets
            img_labels = []
            for target in targets:
                labels = target.get("labels", torch.tensor([]))
                if len(labels) > 0:
                    img_labels.append(labels[0].long())
                else:
                    img_labels.append(torch.tensor(0, dtype=torch.long))
            img_labels = torch.stack(img_labels).to(device)

            cls_loss = self.ce_loss(outputs["cls_logits"], img_labels)
            losses["cls_loss"] = cls_loss
            total_loss = total_loss + self.task_cls_weight * cls_loss

        losses["total_loss"] = total_loss
        return losses
