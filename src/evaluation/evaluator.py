"""
Main evaluator for the open-set safety violation detection system.

Runs inference on a test set and computes all metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_map, compute_open_set_metrics, compute_precision_recall_f1


class Evaluator:
    """
    Evaluates a trained open-set detector on a test dataset.

    Runs inference, aggregates predictions, and computes:
    - mAP@50, mAP@75
    - Precision, Recall, F1
    - Open-set metrics: WI, A-OSE

    Args:
        model: Trained MultiTaskDetector.
        data_loader: Test DataLoader.
        num_known_classes: Number of known classes.
        unknown_class_id: Unknown class ID.
        device: Computation device.
        score_threshold: Minimum confidence for predictions.
        iou_threshold: IoU threshold for evaluation.
        baseline_model: Optional closed-set baseline for WI computation.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_known_classes: int = 4,
        unknown_class_id: int = 4,
        device: Optional[torch.device] = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        baseline_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.num_known_classes = num_known_classes
        self.unknown_class_id = unknown_class_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.baseline_model = baseline_model

        self.model.to(self.device)
        if baseline_model is not None:
            baseline_model.to(self.device)

    def _decode_predictions(
        self, outputs: Dict[str, Any], image_sizes: torch.Tensor
    ) -> List[Dict]:
        """
        Decode model outputs to per-image prediction dicts.

        For a full implementation, this would use NMS and box decoding.
        Returns simplified predictions for demonstration.

        Args:
            outputs: Model forward outputs dict.
            image_sizes: (N, 2) tensor of [H, W] per image.

        Returns:
            List of per-image prediction dicts.
        """
        N = image_sizes.shape[0]
        predictions = []

        for i in range(N):
            # Extract classification logits from the first FPN level
            if "det_cls_scores" in outputs and len(outputs["det_cls_scores"]) > 0:
                # Shape: (N, C, H, W) -> flatten to get scores
                cls_map = outputs["det_cls_scores"][0][i]  # (C, H, W)
                C, H, W = cls_map.shape

                # Max pooling to get top predictions (simplified NMS)
                scores_flat = cls_map.reshape(C, -1).permute(1, 0)  # (H*W, C)
                max_scores, pred_labels = scores_flat.sigmoid().max(dim=-1)

                # Apply score threshold
                keep = max_scores > self.score_threshold

                # Build pseudo-boxes from grid positions
                ys = torch.arange(H, device=self.device).float()
                xs = torch.arange(W, device=self.device).float()
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
                grid_flat = torch.stack([
                    grid_x.reshape(-1),
                    grid_y.reshape(-1),
                    grid_x.reshape(-1) + 1,
                    grid_y.reshape(-1) + 1,
                ], dim=-1)  # (H*W, 4)

                # Scale to image size
                h, w = image_sizes[i]
                scale = torch.tensor(
                    [w / W, h / H, w / W, h / H], device=self.device
                )
                grid_boxes = grid_flat * scale

                boxes = grid_boxes[keep].cpu().numpy()
                scores = max_scores[keep].cpu().numpy()
                labels = pred_labels[keep].cpu().numpy()
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
                labels = np.zeros(0, dtype=np.int64)

            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        return predictions

    @torch.no_grad()
    def run(self) -> Dict[str, float]:
        """
        Run evaluation on the full test set.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        all_predictions: List[Dict] = []
        all_ground_truths: List[Dict] = []
        baseline_predictions: List[Dict] = []

        if self.baseline_model is not None:
            self.baseline_model.eval()

        print("Running inference...")
        for batch_idx, batch in enumerate(self.data_loader):
            images = batch["images"].to(self.device)
            targets = batch["targets"]
            image_sizes = torch.stack([t["orig_size"] for t in targets])

            # Main model predictions
            outputs = self.model(images)
            preds = self._decode_predictions(outputs, image_sizes)
            all_predictions.extend(preds)

            # Baseline model predictions (for WI)
            if self.baseline_model is not None:
                base_outputs = self.baseline_model(images)
                base_preds = self._decode_predictions(base_outputs, image_sizes)
                baseline_predictions.extend(base_preds)

            # Ground truths
            for target in targets:
                gt = {
                    "boxes": target["boxes"].cpu().numpy(),
                    "labels": target["labels"].cpu().numpy(),
                }
                all_ground_truths.append(gt)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(self.data_loader)} batches")

        print("Computing metrics...")
        metrics = compute_open_set_metrics(
            predictions=all_predictions,
            ground_truths=all_ground_truths,
            predictions_closed=baseline_predictions if baseline_predictions else None,
            num_known_classes=self.num_known_classes,
            unknown_class_id=self.unknown_class_id,
            iou_threshold=self.iou_threshold,
        )

        # Additional mAP@75
        metrics_75 = compute_map(
            all_predictions,
            all_ground_truths,
            iou_threshold=0.75,
            num_classes=self.num_known_classes,
        )
        metrics["mAP@75"] = metrics_75["mAP"]

        self.print_results(metrics)
        return metrics

    def print_results(self, metrics: Dict[str, float]) -> None:
        """Print formatted evaluation results."""
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  mAP@50        : {metrics.get('mAP', 0):.4f}")
        print(f"  mAP@75        : {metrics.get('mAP@75', 0):.4f}")
        print(f"  Precision     : {metrics.get('precision', 0):.4f}")
        print(f"  Recall        : {metrics.get('recall', 0):.4f}")
        print(f"  F1            : {metrics.get('f1', 0):.4f}")
        print("-" * 60)
        print("  Open-Set Metrics:")
        print(f"  A-OSE         : {metrics.get('A-OSE', 'N/A')}")
        print(f"  WI            : {metrics.get('WI', 'N/A')}")
        print("-" * 60)
        print("  Per-Class AP (@50):")
        for k, v in metrics.items():
            if k.startswith("AP_class_"):
                cls_id = int(k.split("_")[-1])
                cls_names = [
                    "no_helmet", "phone_usage", "gas_cylinder_viol", "crane_outrigger"
                ]
                cls_name = cls_names[cls_id] if cls_id < len(cls_names) else f"class_{cls_id}"
                print(f"    {cls_name:<25} : {v:.4f}")
        print("=" * 60)
