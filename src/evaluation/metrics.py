"""
Evaluation metrics for open-set safety violation detection.

Includes standard object detection metrics (mAP, Precision, Recall, F1)
and open-set specific metrics:
- WI (Wilderness Impact): Effect of unknown objects on known-class performance
- A-OSE (Absolute Open-Set Error): Number of unknown objects classified as known

References:
- "Towards Open World Object Detection" (Joseph et al., CVPR 2021)
- PASCAL VOC AP computation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: (N, 4) array.
        boxes2: (M, 4) array.

    Returns:
        iou: (N, M) IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using the 11-point interpolation method (VOC 2010+).

    Args:
        recall: Array of recall values.
        precision: Array of precision values.

    Returns:
        ap: Average precision scalar.
    """
    # Append sentinel values
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])

    # Compute precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Compute area under P-R curve
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])
    return float(ap)


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) across all classes.

    Args:
        predictions: List of per-image dicts with keys:
            - boxes: (N, 4) predicted boxes (xyxy).
            - scores: (N,) confidence scores.
            - labels: (N,) predicted class indices.
        ground_truths: List of per-image dicts with keys:
            - boxes: (M, 4) ground truth boxes (xyxy).
            - labels: (M,) ground truth class indices.
        iou_threshold: IoU threshold for a detection to count as TP.
        num_classes: Number of classes to evaluate.

    Returns:
        Dictionary with per-class AP and overall mAP.
    """
    # Collect all predictions and GTs per class
    class_preds: Dict[int, List] = {c: [] for c in range(num_classes)}
    class_gts: Dict[int, int] = {c: 0 for c in range(num_classes)}

    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)
        pred_labels = np.array(pred.get("labels", []), dtype=np.int64)

        gt_boxes = np.array(gt.get("boxes", []), dtype=np.float32)
        gt_labels = np.array(gt.get("labels", []), dtype=np.int64)

        # Track matched GTs per class
        matched: Dict[int, np.ndarray] = {}
        for c in range(num_classes):
            gt_mask = gt_labels == c
            matched[c] = np.zeros(gt_mask.sum(), dtype=bool)
            class_gts[c] += gt_mask.sum()

        # Sort predictions by score (descending)
        if len(pred_scores) > 0:
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
            pred_labels = pred_labels[sort_idx]

        for j, (box, score, label) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            if label >= num_classes or label < 0:
                continue
            gt_mask = gt_labels == label
            gt_class_boxes = gt_boxes[gt_mask]

            is_tp = False
            if len(gt_class_boxes) > 0:
                ious = compute_iou(box[None], gt_class_boxes)[0]
                best_iou_idx = int(np.argmax(ious))
                if ious[best_iou_idx] >= iou_threshold and not matched[label][best_iou_idx]:
                    matched[label][best_iou_idx] = True
                    is_tp = True

            class_preds[label].append(
                {"img_idx": img_idx, "score": score, "tp": is_tp}
            )

    # Compute AP per class
    aps: Dict[str, float] = {}
    for c in range(num_classes):
        preds_c = sorted(class_preds[c], key=lambda x: -x["score"])
        n_gt = class_gts[c]

        if n_gt == 0:
            aps[f"AP_class_{c}"] = float("nan")
            continue

        tp_cumsum = np.cumsum([p["tp"] for p in preds_c]).astype(float)
        fp_cumsum = np.cumsum([not p["tp"] for p in preds_c]).astype(float)

        recall = tp_cumsum / (n_gt + 1e-7)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        ap = voc_ap(recall, precision)
        aps[f"AP_class_{c}"] = ap

    valid_aps = [v for v in aps.values() if not np.isnan(v)]
    aps["mAP"] = float(np.mean(valid_aps)) if valid_aps else 0.0
    return aps


def compute_precision_recall_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 at a fixed score threshold.

    Args:
        predictions: Per-image prediction dicts.
        ground_truths: Per-image ground truth dicts.
        iou_threshold: IoU threshold for TP.
        score_threshold: Score threshold for filtering predictions.
        num_classes: Number of known classes.

    Returns:
        Dict with precision, recall, f1 (macro-averaged over classes).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)
        pred_labels = np.array(pred.get("labels", []), dtype=np.int64)
        gt_boxes = np.array(gt.get("boxes", []), dtype=np.float32)
        gt_labels = np.array(gt.get("labels", []), dtype=np.int64)

        # Apply score threshold
        if len(pred_scores) > 0:
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

        matched_gt = np.zeros(len(gt_labels), dtype=bool)

        for box, label in zip(pred_boxes, pred_labels):
            if label >= num_classes:
                continue
            gt_mask = gt_labels == label
            gt_class_boxes = gt_boxes[gt_mask]
            gt_class_indices = np.where(gt_mask)[0]

            if len(gt_class_boxes) > 0:
                ious = compute_iou(box[None], gt_class_boxes)[0]
                best = int(np.argmax(ious))
                if ious[best] >= iou_threshold and not matched_gt[gt_class_indices[best]]:
                    matched_gt[gt_class_indices[best]] = True
                    total_tp += 1
                else:
                    total_fp += 1
            else:
                total_fp += 1

        total_fn += (~matched_gt).sum()

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {"precision": precision, "recall": recall, "f1": f1}


class WildernessImpact:
    """
    Wilderness Impact (WI) metric for open-set object detection.

    WI measures how much the presence of unknown objects degrades the
    precision of known-class detections.

    WI = (P_known_without_unknown - P_known_with_unknown) / P_known_without_unknown

    Reference: "Towards Open World Object Detection" (Joseph et al., 2021)
    """

    @staticmethod
    def compute(
        predictions_closed: List[Dict],
        predictions_open: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5,
        num_known_classes: int = 4,
    ) -> float:
        """
        Compute WI score.

        Args:
            predictions_closed: Predictions from closed-set model.
            predictions_open: Predictions from open-set model (includes unknowns).
            ground_truths: Ground truth annotations (known classes only).
            iou_threshold: IoU threshold for TP.
            num_known_classes: Number of known classes.

        Returns:
            WI score (0 = no impact, positive = degradation).
        """
        metrics_closed = compute_precision_recall_f1(
            predictions_closed, ground_truths,
            iou_threshold=iou_threshold, num_classes=num_known_classes,
        )
        metrics_open = compute_precision_recall_f1(
            predictions_open, ground_truths,
            iou_threshold=iou_threshold, num_classes=num_known_classes,
        )

        p_closed = metrics_closed["precision"]
        p_open = metrics_open["precision"]

        if p_closed < 1e-7:
            return 0.0
        return float((p_closed - p_open) / p_closed)


class AbsoluteOpenSetError:
    """
    Absolute Open-Set Error (A-OSE) metric.

    Counts the number of unknown class instances that are wrongly
    classified as any known class.

    A-OSE = number of unknown objects detected with a known-class label
    """

    @staticmethod
    def compute(
        predictions: List[Dict],
        ground_truths_with_unknowns: List[Dict],
        num_known_classes: int = 4,
        unknown_class_id: int = 4,
        iou_threshold: float = 0.5,
    ) -> int:
        """
        Compute A-OSE.

        Args:
            predictions: Per-image prediction dicts with predicted labels.
            ground_truths_with_unknowns: GT including unknown class annotations.
            num_known_classes: Number of known classes.
            unknown_class_id: Class ID for unknown/open-set class.
            iou_threshold: IoU threshold for a match.

        Returns:
            A-OSE count (int).
        """
        ose_count = 0

        for pred, gt in zip(predictions, ground_truths_with_unknowns):
            pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
            pred_labels = np.array(pred.get("labels", []), dtype=np.int64)
            gt_boxes = np.array(gt.get("boxes", []), dtype=np.float32)
            gt_labels = np.array(gt.get("labels", []), dtype=np.int64)

            # Filter to unknown GT boxes
            unk_mask = gt_labels == unknown_class_id
            unk_boxes = gt_boxes[unk_mask]

            if len(unk_boxes) == 0 or len(pred_boxes) == 0:
                continue

            # Count predictions on unknown GT regions classified as known
            ious = compute_iou(pred_boxes, unk_boxes)  # (N_pred, N_unk)
            for j, (label, iou_row) in enumerate(zip(pred_labels, ious)):
                if label < num_known_classes and iou_row.max() >= iou_threshold:
                    ose_count += 1

        return ose_count


def compute_open_set_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    predictions_closed: Optional[List[Dict]] = None,
    num_known_classes: int = 4,
    unknown_class_id: int = 4,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all open-set detection metrics.

    Args:
        predictions: Open-set model predictions.
        ground_truths: Ground truths (including unknown class).
        predictions_closed: Closed-set baseline predictions (for WI computation).
        num_known_classes: Number of known classes.
        unknown_class_id: Unknown class index.
        iou_threshold: IoU threshold.

    Returns:
        Dictionary with WI, A-OSE, and standard metrics.
    """
    metrics = {}

    # Standard mAP on known classes
    gt_known = [
        {
            "boxes": np.array([
                b for b, l in zip(gt.get("boxes", []), gt.get("labels", []))
                if l < num_known_classes
            ], dtype=np.float32).reshape(-1, 4),
            "labels": np.array([
                l for l in gt.get("labels", []) if l < num_known_classes
            ], dtype=np.int64),
        }
        for gt in ground_truths
    ]

    map_metrics = compute_map(
        predictions, gt_known,
        iou_threshold=iou_threshold,
        num_classes=num_known_classes,
    )
    metrics.update(map_metrics)

    # A-OSE
    metrics["A-OSE"] = AbsoluteOpenSetError.compute(
        predictions, ground_truths,
        num_known_classes=num_known_classes,
        unknown_class_id=unknown_class_id,
        iou_threshold=iou_threshold,
    )

    # WI (requires closed-set baseline predictions)
    if predictions_closed is not None:
        metrics["WI"] = WildernessImpact.compute(
            predictions_closed, predictions, gt_known,
            iou_threshold=iou_threshold,
            num_known_classes=num_known_classes,
        )

    # Precision/Recall/F1 on known classes
    prf = compute_precision_recall_f1(
        predictions, gt_known,
        iou_threshold=iou_threshold,
        num_classes=num_known_classes,
    )
    metrics.update(prf)

    return metrics
