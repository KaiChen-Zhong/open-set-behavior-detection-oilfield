"""
Spatial Relationship Reasoning Module for distance-based safety violation detection.

Detects violations that require understanding the spatial relationships
between objects in the scene, such as:
- Two workers standing too close to a gas cylinder
- A crane operating without proper outrigger deployment radius
- Objects in prohibited zones relative to other objects
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationEmbedding(nn.Module):
    """
    Embed pairwise geometric relationships between bounding boxes.

    Encodes relative position, size, and IoU information into a feature vector.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Input: [dx, dy, dw, dh, iou, area_ratio, cx_rel, cy_rel] = 8 dims
        self.fc = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_dim),
        )

    def forward(
        self, boxes_i: torch.Tensor, boxes_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute geometric relation features for pairs of boxes.

        Args:
            boxes_i: (N, 4) boxes in xyxy format.
            boxes_j: (N, 4) boxes in xyxy format.

        Returns:
            rel_feats: (N, out_dim) relation feature vectors.
        """
        # Centers and sizes
        wi = boxes_i[:, 2] - boxes_i[:, 0]
        hi = boxes_i[:, 3] - boxes_i[:, 1]
        cxi = boxes_i[:, 0] + wi / 2
        cyi = boxes_i[:, 1] + hi / 2

        wj = boxes_j[:, 2] - boxes_j[:, 0]
        hj = boxes_j[:, 3] - boxes_j[:, 1]
        cxj = boxes_j[:, 0] + wj / 2
        cyj = boxes_j[:, 1] + hj / 2

        eps = 1e-6
        dx = (cxj - cxi) / (wi + eps)
        dy = (cyj - cyi) / (hi + eps)
        dw = torch.log((wj + eps) / (wi + eps))
        dh = torch.log((hj + eps) / (hi + eps))

        # IoU
        inter_x1 = torch.max(boxes_i[:, 0], boxes_j[:, 0])
        inter_y1 = torch.max(boxes_i[:, 1], boxes_j[:, 1])
        inter_x2 = torch.min(boxes_i[:, 2], boxes_j[:, 2])
        inter_y2 = torch.min(boxes_i[:, 3], boxes_j[:, 3])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area_i = wi * hi
        area_j = wj * hj
        union = area_i + area_j - inter + eps
        iou = inter / union

        # Area ratio (log scale)
        area_ratio = torch.log((area_j + eps) / (area_i + eps))

        # Relative center position (normalized by image size assumption ~1.0)
        cx_rel = cxj / (cxi + eps)
        cy_rel = cyj / (cyi + eps)

        geom = torch.stack([dx, dy, dw, dh, iou, area_ratio, cx_rel, cy_rel], dim=-1)
        return self.fc(geom)


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for object relationship reasoning.
    Treats each detected object as a node; edges encode pairwise relations.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4):
        super().__init__()
        assert out_features % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.out_proj = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: (N, D) node features.
            edge_feats: (N, N, E) optional edge features (unused in base).

        Returns:
            out: (N, D) updated node features.
        """
        N, D = node_feats.shape
        Q = self.q_proj(node_feats).reshape(N, self.num_heads, self.head_dim)
        K = self.k_proj(node_feats).reshape(N, self.num_heads, self.head_dim)
        V = self.v_proj(node_feats).reshape(N, self.num_heads, self.head_dim)

        # Attention weights: (num_heads, N, N)
        scale = self.head_dim ** -0.5
        attn = torch.einsum("nhd,mhd->hnm", Q, K) * scale  # (H, N, N)
        attn = F.softmax(attn, dim=-1)

        # Aggregate values: (num_heads, N, head_dim)
        out = torch.einsum("hnm,mhd->nhd", attn, V)
        out = out.reshape(N, -1)
        out = self.out_proj(out)
        return self.norm(out + node_feats if D == out.shape[-1] else out)


class SpatialReasoningModule(nn.Module):
    """
    Spatial Relationship Reasoning Module for oil depot safety violations.

    Operates on detected objects (boxes + features) to identify spatial
    relationship violations such as unsafe proximity, blocking, and
    zone violations.

    Args:
        in_channels: Feature dimension for each detected object.
        hidden_dim: Hidden dimension for relation reasoning.
        num_relation_types: Number of spatial relation categories.
        distance_threshold: Pixel distance threshold for "too close" detection.
        num_heads: Number of attention heads in graph attention.
    """

    RELATION_TYPES = ["too_close", "blocking", "improper_zone", "height_violation"]

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        num_relation_types: int = 4,
        distance_threshold: float = 50.0,
        num_heads: int = 4,
    ):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.num_relation_types = num_relation_types

        # Object feature projection
        self.obj_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Relation embedding
        self.rel_embed = RelationEmbedding(out_dim=64)

        # Graph attention for reasoning
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, num_heads)

        # Relation classification head
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relation_types),
        )

        # Distance regression head (predict actual distance in scene units)
        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(),  # Distance is non-negative
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_distance_matrix(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise center distances between bounding boxes.

        Args:
            boxes: (N, 4) boxes in xyxy format.

        Returns:
            dist_matrix: (N, N) pairwise center distances.
        """
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        centers = torch.stack([cx, cy], dim=-1)  # (N, 2)
        diff = centers.unsqueeze(0) - centers.unsqueeze(1)  # (N, N, 2)
        return torch.norm(diff, dim=-1)  # (N, N)

    def forward(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        obj_classes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for spatial relationship reasoning.

        Args:
            obj_features: (N, in_channels) per-object feature vectors.
            boxes: (N, 4) bounding boxes in xyxy format.
            obj_classes: (N,) optional class labels for type-aware reasoning.

        Returns:
            Dictionary with:
                - relation_scores: (N, N, num_relation_types)
                - distance_preds: (N, N, 1) predicted distances
                - violation_pairs: (K, 2) pairs with spatial violations
                - node_features: (N, hidden_dim) updated node features
        """
        N = obj_features.shape[0]

        if N == 0:
            return {
                "relation_scores": torch.zeros(0, 0, self.num_relation_types),
                "distance_preds": torch.zeros(0, 0, 1),
                "violation_pairs": torch.zeros(0, 2, dtype=torch.long),
                "node_features": obj_features,
            }

        # Project object features
        node_feats = self.obj_proj(obj_features)  # (N, hidden_dim)

        # Graph attention reasoning
        node_feats = self.gat1(node_feats)
        node_feats = self.gat2(node_feats)

        # Pairwise relation prediction
        # Expand for all pairs (N, N, hidden_dim)
        feats_i = node_feats.unsqueeze(1).expand(N, N, -1)
        feats_j = node_feats.unsqueeze(0).expand(N, N, -1)

        # Geometric relation features
        boxes_i = boxes.unsqueeze(1).expand(N, N, 4).reshape(N * N, 4)
        boxes_j = boxes.unsqueeze(0).expand(N, N, 4).reshape(N * N, 4)
        rel_feats = self.rel_embed(boxes_i, boxes_j).reshape(N, N, 64)

        # Concatenate pair features
        pair_feats = torch.cat([feats_i, feats_j, rel_feats], dim=-1)  # (N, N, 2H+64)

        # Predict relations
        pair_flat = pair_feats.reshape(N * N, -1)
        relation_scores = self.relation_head(pair_flat).reshape(N, N, self.num_relation_types)
        distance_preds = self.distance_head(pair_flat).reshape(N, N, 1)

        # Find violation pairs based on predicted distances and relations
        dist_matrix = self.compute_distance_matrix(boxes)
        proximity_violations = (dist_matrix < self.distance_threshold) & (
            torch.eye(N, device=boxes.device).bool().logical_not()
        )
        violation_pairs = proximity_violations.nonzero(as_tuple=False)

        return {
            "relation_scores": relation_scores,
            "distance_preds": distance_preds,
            "violation_pairs": violation_pairs,
            "node_features": node_feats,
            "distance_matrix": dist_matrix,
        }
