"""
Basic tests for the open-set safety violation detection project.

Tests cover:
- Data pipeline (dataset, augmentation)
- Model architecture (backbone, detection head, multitask model)
- Training components (losses, optimizer, scheduler)
- Evaluation metrics (mAP, open-set metrics)
- Continual learning (replay buffer)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch


# ============================================================
# Data pipeline tests
# ============================================================

class TestDataAugmentation:
    """Tests for the augmentation pipeline."""

    def test_train_transforms_returns_tensor(self):
        from src.data.augmentation import get_train_transforms
        transforms = get_train_transforms((640, 640))
        # Create dummy image and boxes
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[10, 10, 100, 100], [200, 200, 350, 350]]
        labels = [0, 1]
        result = transforms(image=image, bboxes=boxes, class_labels=labels)
        assert "image" in result
        assert hasattr(result["image"], "shape"), "Image should be a tensor"

    def test_val_transforms_deterministic(self):
        from src.data.augmentation import get_val_transforms
        transforms = get_val_transforms((640, 640))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[10, 10, 100, 100]]
        labels = [0]
        # Run twice - val transforms should be deterministic
        r1 = transforms(image=image, bboxes=boxes, class_labels=labels)
        r2 = transforms(image=image, bboxes=boxes, class_labels=labels)
        # Boxes should be the same (deterministic transforms)
        assert r1["bboxes"] == r2["bboxes"], "Val transforms should be deterministic"

    def test_mosaic_augmentation_shape(self):
        from src.data.augmentation import MosaicAugmentation
        mosaic = MosaicAugmentation(img_size=640)
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
        all_boxes = [np.array([[10, 10, 100, 100]], dtype=np.float32) for _ in range(4)]
        all_labels = [np.array([0], dtype=np.int64) for _ in range(4)]
        mosaic_img, mosaic_boxes, mosaic_labels = mosaic(images, all_boxes, all_labels)
        assert mosaic_img.shape == (640, 640, 3)
        assert mosaic_boxes.ndim == 2 and mosaic_boxes.shape[1] == 4

    def test_violation_augmentation_train_mode(self):
        from src.data.augmentation import ViolationAugmentation
        aug = ViolationAugmentation(img_size=(640, 640), mode="train")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = aug(image=image, bboxes=[], class_labels=[])
        assert "image" in result


class TestDatasetModule:
    """Tests for dataset class and utilities."""

    def test_category_maps_consistent(self):
        from src.data.dataset import (
            CATEGORY_ID_TO_NAME,
            CATEGORY_NAME_TO_ID,
            VIOLATION_CATEGORIES,
        )
        # Every category name should be in both maps
        for name, info in VIOLATION_CATEGORIES.items():
            assert name in CATEGORY_NAME_TO_ID, f"{name} missing from NAME_TO_ID"
            assert CATEGORY_NAME_TO_ID[name] == info["id"]
            assert info["id"] in CATEGORY_ID_TO_NAME
            assert CATEGORY_ID_TO_NAME[info["id"]] == name

    def test_collate_fn_stacks_images(self):
        from src.data.dataset import collate_fn
        batch = [
            {
                "image": torch.zeros(3, 64, 64),
                "target": {
                    "boxes": torch.zeros(2, 4),
                    "labels": torch.zeros(2, dtype=torch.long),
                    "image_id": torch.tensor([1]),
                    "area": torch.zeros(2),
                    "iscrowd": torch.zeros(2, dtype=torch.uint8),
                    "orig_size": torch.tensor([480, 640]),
                },
                "img_id": 1,
            }
            for _ in range(4)
        ]
        result = collate_fn(batch)
        assert result["images"].shape == (4, 3, 64, 64)
        assert len(result["targets"]) == 4


# ============================================================
# Model architecture tests
# ============================================================

class TestCSPDarknet:
    """Tests for CSPDarknet backbone."""

    def test_output_shapes(self):
        from src.models.backbone import CSPDarknet
        model = CSPDarknet(variant="n")
        x = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            feats = model(x)
        assert len(feats) == 3, "Should output 3 feature maps"
        # Check downsampling ratios (stride 8, 16, 32)
        assert feats[0].shape[2] == 640 // 8
        assert feats[1].shape[2] == 640 // 16
        assert feats[2].shape[2] == 640 // 32

    def test_out_channels_attribute(self):
        from src.models.backbone import CSPDarknet
        model = CSPDarknet(variant="n")
        assert len(model.out_channels) == 3


class TestFPN:
    """Tests for Feature Pyramid Network."""

    def test_fpn_output_count(self):
        from src.models.backbone import FPN
        fpn = FPN(in_channels=[64, 128, 256], out_channels=256, num_outs=4)
        feats = [
            torch.randn(2, 64, 80, 80),
            torch.randn(2, 128, 40, 40),
            torch.randn(2, 256, 20, 20),
        ]
        with torch.no_grad():
            outs = fpn(feats)
        assert len(outs) == 4


class TestOpenSetDetectionHead:
    """Tests for the open-set detection head."""

    def test_forward_shapes(self):
        from src.models.detection_head import OpenSetDetectionHead
        head = OpenSetDetectionHead(
            in_channels=64,
            num_known_classes=4,
            feat_channels=64,
            stacked_convs=2,
        )
        feats = [torch.randn(2, 64, 20, 20), torch.randn(2, 64, 10, 10)]
        with torch.no_grad():
            cls_scores, bbox_preds, centernesses, objectness = head(feats)

        assert len(cls_scores) == 2
        # Check num_classes = num_known + 1 unknown
        assert cls_scores[0].shape[1] == 5  # 4 known + 1 unknown
        assert bbox_preds[0].shape[1] == 4 * head.reg_max

    def test_prototype_memory(self):
        from src.models.detection_head import PrototypeMemory
        memory = PrototypeMemory(num_known_classes=4, prototype_dim=64)
        features = torch.randn(8, 64)
        sim = memory.compute_similarity(features)
        assert sim.shape == (8, 4)


class TestSpatialReasoning:
    """Tests for the spatial reasoning module."""

    def test_forward_with_objects(self):
        from src.models.spatial_reasoning import SpatialReasoningModule
        module = SpatialReasoningModule(in_channels=64, hidden_dim=64)
        obj_feats = torch.randn(3, 64)
        boxes = torch.tensor([
            [10.0, 10.0, 100.0, 100.0],
            [50.0, 50.0, 150.0, 150.0],
            [200.0, 200.0, 300.0, 300.0],
        ])
        with torch.no_grad():
            result = module(obj_feats, boxes)
        assert "relation_scores" in result
        assert result["relation_scores"].shape == (3, 3, 4)  # (N, N, R)

    def test_forward_with_empty_objects(self):
        from src.models.spatial_reasoning import SpatialReasoningModule
        module = SpatialReasoningModule(in_channels=64, hidden_dim=64)
        obj_feats = torch.zeros(0, 64)
        boxes = torch.zeros(0, 4)
        with torch.no_grad():
            result = module(obj_feats, boxes)
        assert result["violation_pairs"].shape == (0, 2)


class TestMultiTaskDetector:
    """Tests for the full multi-task model."""

    @pytest.fixture
    def model(self):
        from src.models.multitask_model import MultiTaskDetector
        return MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
            num_known_classes=4,
            num_classes=5,
            enable_spatial_reasoning=False,  # Skip for speed
            enable_classification=True,
        )

    def test_forward_inference(self, model):
        images = torch.randn(2, 3, 320, 320)
        with torch.no_grad():
            outputs = model(images)
        assert "det_cls_scores" in outputs
        assert "det_bbox_preds" in outputs
        assert "cls_logits" in outputs

    def test_cls_logits_shape(self, model):
        images = torch.randn(2, 3, 320, 320)
        with torch.no_grad():
            outputs = model(images)
        assert outputs["cls_logits"].shape == (2, 5)

    def test_parameter_count_nonzero(self, model):
        counts = model.get_num_parameters()
        assert counts["total"] > 0
        assert counts["backbone"] > 0

    def test_build_model_factory(self):
        from src.models.multitask_model import build_model
        model = build_model({
            "backbone": {"type": "CSPDarknet", "variant": "n"},
            "num_known_classes": 4,
            "num_classes": 5,
        })
        assert model is not None


# ============================================================
# Training component tests
# ============================================================

class TestLosses:
    """Tests for loss functions."""

    def test_focal_loss_shape(self):
        from src.training.losses import FocalLoss
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_giou_loss_perfect_match(self):
        from src.training.losses import GIoULoss
        loss_fn = GIoULoss()
        boxes = torch.tensor([[10.0, 10.0, 100.0, 100.0]])
        # Perfect match should give loss close to 0
        loss = loss_fn(boxes, boxes)
        assert loss.item() < 0.01

    def test_supcon_loss_same_class(self):
        from src.training.losses import SupConLoss
        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 64)
        labels = torch.zeros(8, dtype=torch.long)  # All same class
        loss = loss_fn(features, labels)
        assert not torch.isnan(loss)

    def test_multitask_loss(self):
        from src.models.multitask_model import MultiTaskDetector
        from src.training.losses import MultiTaskLoss

        model = MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
            num_known_classes=4,
            enable_spatial_reasoning=False,
        )
        criterion = MultiTaskLoss(num_known_classes=4)

        images = torch.randn(2, 3, 320, 320)
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([0]),
            }
            for _ in range(2)
        ]

        with torch.no_grad():
            outputs = model(images, targets)
        losses = criterion(outputs, targets)

        assert "total_loss" in losses
        assert not torch.isnan(losses["total_loss"])


class TestOptimizer:
    """Tests for optimizer and scheduler builders."""

    def test_build_adamw_optimizer(self):
        from src.models.multitask_model import MultiTaskDetector
        from src.training.optimizer import build_optimizer

        model = MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
        )
        optimizer = build_optimizer(model, {"name": "AdamW", "lr": 1e-4})
        assert optimizer is not None
        assert len(optimizer.param_groups) >= 1

    def test_build_cosine_scheduler(self):
        from src.models.multitask_model import MultiTaskDetector
        from src.training.optimizer import build_optimizer, build_scheduler

        model = MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
        )
        optimizer = build_optimizer(model, {"name": "AdamW", "lr": 1e-4})
        scheduler = build_scheduler(
            optimizer,
            {"name": "CosineAnnealingLR", "T_max": 100, "warmup_epochs": 0},
        )
        assert scheduler is not None


# ============================================================
# Evaluation metric tests
# ============================================================

class TestMetrics:
    """Tests for evaluation metrics."""

    @pytest.fixture
    def sample_predictions(self):
        return [
            {
                "boxes": np.array([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=np.float32),
                "labels": np.array([0, 1], dtype=np.int64),
                "scores": np.array([0.9, 0.8], dtype=np.float32),
            }
        ]

    @pytest.fixture
    def sample_ground_truths(self):
        return [
            {
                "boxes": np.array([[12, 12, 98, 98], [210, 210, 295, 295]], dtype=np.float32),
                "labels": np.array([0, 1], dtype=np.int64),
            }
        ]

    def test_compute_iou(self):
        from src.evaluation.metrics import compute_iou
        b1 = np.array([[0, 0, 100, 100]])
        b2 = np.array([[50, 50, 150, 150]])
        iou = compute_iou(b1, b2)
        assert 0 < iou[0, 0] < 1

    def test_perfect_iou(self):
        from src.evaluation.metrics import compute_iou
        b = np.array([[10, 10, 100, 100]])
        iou = compute_iou(b, b)
        assert abs(iou[0, 0] - 1.0) < 1e-5

    def test_compute_map_returns_map(
        self, sample_predictions, sample_ground_truths
    ):
        from src.evaluation.metrics import compute_map
        results = compute_map(
            sample_predictions, sample_ground_truths,
            iou_threshold=0.5, num_classes=4,
        )
        assert "mAP" in results
        assert 0.0 <= results["mAP"] <= 1.0

    def test_precision_recall_f1(
        self, sample_predictions, sample_ground_truths
    ):
        from src.evaluation.metrics import compute_precision_recall_f1
        metrics = compute_precision_recall_f1(
            sample_predictions, sample_ground_truths,
            iou_threshold=0.5, num_classes=4,
        )
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_aose_no_unknowns(self, sample_predictions):
        from src.evaluation.metrics import AbsoluteOpenSetError
        gt_with_unknown = [
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
            }
        ]
        aose = AbsoluteOpenSetError.compute(
            sample_predictions, gt_with_unknown,
            num_known_classes=4, unknown_class_id=4,
        )
        assert aose == 0


# ============================================================
# Continual learning tests
# ============================================================

class TestReplayBuffer:
    """Tests for the replay buffer."""

    def test_add_and_sample(self):
        from src.models.continual_learning import ReplayBuffer
        buffer = ReplayBuffer(max_size=100, samples_per_class=10)
        samples = [{"image": np.zeros((3, 64, 64)), "target": {"labels": [i]}} for i in range(15)]
        buffer.add_samples(samples, class_id=0, features=None)
        assert buffer.total_size <= 10  # Capped at samples_per_class

    def test_reservoir_sampling(self):
        from src.models.continual_learning import ReplayBuffer
        buffer = ReplayBuffer(max_size=50, samples_per_class=5, selection_strategy="reservoir")
        samples = [{"image": np.zeros((3, 64, 64)), "target": {}} for _ in range(20)]
        buffer.add_samples(samples, class_id=0)
        assert buffer.total_size <= 5

    def test_sample_empty_buffer(self):
        from src.models.continual_learning import ReplayBuffer
        buffer = ReplayBuffer()
        result = buffer.sample(10)
        assert result == []

    def test_multi_class_buffer(self):
        from src.models.continual_learning import ReplayBuffer
        buffer = ReplayBuffer(max_size=100, samples_per_class=10)
        for cls_id in range(4):
            samples = [{"image": np.zeros((3, 64, 64)), "target": {}} for _ in range(10)]
            buffer.add_samples(samples, class_id=cls_id)
        assert buffer.num_classes == 4
        sampled = buffer.sample(20)
        assert len(sampled) == 20

    def test_herding_selection(self):
        from src.models.continual_learning import ReplayBuffer
        buffer = ReplayBuffer(
            max_size=100, samples_per_class=5,
            selection_strategy="herding"
        )
        samples = [{"id": i} for i in range(20)]
        features = torch.randn(20, 64)
        buffer.add_samples(samples, class_id=0, features=features)
        assert buffer.total_size <= 5

    def test_continual_learner_stats(self):
        from src.models.continual_learning import ContinualLearner, ReplayBuffer
        from src.models.multitask_model import MultiTaskDetector

        model = MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
        )
        buffer = ReplayBuffer(max_size=100, samples_per_class=10)
        learner = ContinualLearner(model=model, strategy="replay", replay_buffer=buffer)
        stats = learner.get_buffer_stats()
        assert "total_samples" in stats
        assert "strategy" in stats
        assert stats["strategy"] == "replay"


# ============================================================
# Integration test
# ============================================================

class TestEndToEndForward:
    """Integration test for the full forward pass."""

    def test_full_forward_pass(self):
        """Test that the model runs end-to-end without errors."""
        from src.models.multitask_model import MultiTaskDetector
        from src.training.losses import MultiTaskLoss

        model = MultiTaskDetector(
            backbone_cfg={"type": "CSPDarknet", "variant": "n"},
            fpn_out_channels=64,
            num_known_classes=4,
            num_classes=5,
            enable_spatial_reasoning=False,
            enable_classification=True,
        )
        criterion = MultiTaskLoss(num_known_classes=4)

        images = torch.randn(2, 3, 320, 320)
        targets = [
            {
                "boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([320, 320]),
            }
            for _ in range(2)
        ]

        # Forward pass
        outputs = model(images, targets)
        losses = criterion(outputs, targets)

        # Backward pass
        losses["total_loss"].backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "Gradients should flow through the model"
