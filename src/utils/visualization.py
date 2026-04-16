"""
Visualization utilities for safety violation detection results.

Provides functions for:
- Drawing detection bounding boxes on images
- Visualizing spatial relationship graphs
- Generating violation heatmaps
- Creating annotation overlays for reports
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# Color palette (BGR for OpenCV)
VIOLATION_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "no_helmet": (0, 0, 255),          # Red
    "phone_usage": (0, 140, 255),       # Orange
    "gas_cylinder_viol": (0, 215, 255), # Yellow
    "crane_outrigger": (180, 105, 255), # Pink
    "unknown_violation": (128, 0, 128), # Purple
    "unknown": (200, 200, 200),         # Gray
}

CLASS_NAMES = [
    "no_helmet",
    "phone_usage",
    "gas_cylinder_viol",
    "crane_outrigger",
    "unknown_violation",
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_unknown: bool = True,
) -> np.ndarray:
    """
    Draw detection bounding boxes on an image.

    Args:
        image: (H, W, 3) BGR or RGB numpy array.
        boxes: (N, 4) boxes in xyxy format.
        labels: (N,) class indices.
        scores: (N,) optional confidence scores.
        class_names: Optional list of class names.
        thickness: Line thickness for bounding boxes.
        font_scale: Font scale for labels.
        show_unknown: Whether to draw unknown class detections.

    Returns:
        Annotated image.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).clip(0, 255).astype(np.uint8)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        if label >= len(class_names):
            cls_name = "unknown"
        else:
            cls_name = class_names[label]

        if cls_name == "unknown_violation" and not show_unknown:
            continue

        color = VIOLATION_COLORS_BGR.get(cls_name, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        label_text = cls_name.replace("_", " ")
        if scores is not None:
            label_text += f" {scores[i]:.2f}"

        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        label_y1 = max(y1 - text_h - 4, 0)
        cv2.rectangle(
            vis, (x1, label_y1), (x1 + text_w + 2, y1), color, cv2.FILLED
        )
        cv2.putText(
            vis, label_text, (x1 + 1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def draw_spatial_relations(
    image: np.ndarray,
    boxes: np.ndarray,
    violation_pairs: np.ndarray,
    relation_types: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Draw spatial relationship edges between detected objects.

    Args:
        image: (H, W, 3) input image.
        boxes: (N, 4) bounding boxes in xyxy format.
        violation_pairs: (K, 2) pairs of object indices with violations.
        relation_types: Optional list of relation type names per pair.

    Returns:
        Annotated image with relationship edges.
    """
    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).clip(0, 255).astype(np.uint8)

    # Draw edges between violation pairs
    for k, (i, j) in enumerate(violation_pairs):
        if i >= len(boxes) or j >= len(boxes):
            continue
        # Compute centers
        cx_i = int((boxes[i][0] + boxes[i][2]) / 2)
        cy_i = int((boxes[i][1] + boxes[i][3]) / 2)
        cx_j = int((boxes[j][0] + boxes[j][2]) / 2)
        cy_j = int((boxes[j][1] + boxes[j][3]) / 2)

        # Draw dashed line (warning color)
        cv2.line(vis, (cx_i, cy_i), (cx_j, cy_j), (0, 0, 255), 2, cv2.LINE_AA)

        # Draw warning icon at midpoint
        mid_x = (cx_i + cx_j) // 2
        mid_y = (cy_i + cy_j) // 2
        cv2.circle(vis, (mid_x, mid_y), 8, (0, 0, 255), cv2.FILLED)
        cv2.putText(
            vis, "!",
            (mid_x - 4, mid_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1,
        )

        # Relation type label
        if relation_types is not None and k < len(relation_types):
            cv2.putText(
                vis, relation_types[k],
                (mid_x + 10, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (0, 0, 255), 1,
            )

    return vis


def draw_violation_heatmap(
    image: np.ndarray,
    detection_scores: np.ndarray,
    boxes: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a violation density heatmap on the image.

    Args:
        image: (H, W, 3) input image.
        detection_scores: (N,) confidence scores for each detection.
        boxes: (N, 4) bounding boxes in xyxy format.
        alpha: Heatmap opacity.

    Returns:
        Image with heatmap overlay.
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    for score, (x1, y1, x2, y2) in zip(detection_scores, boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            heatmap[y1:y2, x1:x2] += float(score)

    # Normalize and apply colormap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend with original image
    vis = image.copy()
    if vis.dtype != np.uint8:
        vis = (vis * 255).clip(0, 255).astype(np.uint8)
    blended = cv2.addWeighted(vis, 1 - alpha, heatmap_color, alpha, 0)
    return blended


class Visualizer:
    """
    High-level visualization class for the safety detection system.

    Provides convenient methods for visualizing detection results,
    creating annotated video frames, and generating report-quality images.

    Args:
        class_names: List of class name strings.
        conf_threshold: Minimum confidence to display a detection.
        show_unknown: Whether to display unknown class detections.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        show_unknown: bool = True,
    ):
        self.class_names = class_names or CLASS_NAMES
        self.conf_threshold = conf_threshold
        self.show_unknown = show_unknown

    def visualize(
        self,
        image: np.ndarray,
        prediction: Dict,
        show_heatmap: bool = False,
        show_spatial: bool = True,
    ) -> np.ndarray:
        """
        Visualize a single image prediction.

        Args:
            image: Input image (H, W, 3).
            prediction: Dict with keys: boxes, labels, scores,
                        and optionally violation_pairs.
            show_heatmap: Whether to overlay detection heatmap.
            show_spatial: Whether to draw spatial violation edges.

        Returns:
            Annotated image.
        """
        boxes = np.array(prediction.get("boxes", []), dtype=np.float32)
        labels = np.array(prediction.get("labels", []), dtype=np.int64)
        scores = np.array(prediction.get("scores", []), dtype=np.float32)

        # Apply confidence threshold
        if len(scores) > 0:
            keep = scores >= self.conf_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

        vis = image.copy()

        # Heatmap overlay
        if show_heatmap and len(boxes) > 0:
            vis = draw_violation_heatmap(vis, scores, boxes)

        # Detection boxes
        if len(boxes) > 0:
            vis = draw_detections(
                vis, boxes, labels, scores,
                class_names=self.class_names,
                show_unknown=self.show_unknown,
            )

        # Spatial relations
        if show_spatial and "violation_pairs" in prediction:
            vis = draw_spatial_relations(
                vis,
                boxes,
                np.array(prediction["violation_pairs"]),
            )

        # Add info overlay
        vis = self._add_info_overlay(vis, len(boxes), prediction)
        return vis

    def _add_info_overlay(
        self,
        image: np.ndarray,
        num_detections: int,
        prediction: Dict,
    ) -> np.ndarray:
        """Add status overlay to image."""
        vis = image.copy()
        # Status bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 35), (0, 0, 0), cv2.FILLED)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
        cv2.putText(
            vis,
            f"Detections: {num_detections}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
        return vis

    def save(self, image: np.ndarray, path: str) -> None:
        """Save annotated image to disk."""
        cv2.imwrite(path, image)

    def show(self, image: np.ndarray, window_name: str = "Detections") -> None:
        """Display annotated image."""
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
