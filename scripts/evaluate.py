#!/usr/bin/env python3
"""
Evaluation script for open-set safety violation detection.

Usage:
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/multitask/multitask_config.yaml \\
        --split test

    # With baseline comparison (for WI metric)
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best_model.pth \\
        --baseline-checkpoint checkpoints/yolov8_baseline.pth \\
        --config configs/multitask/multitask_config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data import build_dataloader, get_val_transforms
from src.evaluation import Evaluator
from src.models import build_model
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate open-set safety violation detector"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_config.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for predictions",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Optional closed-set baseline checkpoint for WI computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        config_dir = Path(config_path).parent
        base_abs = (config_dir / base_path).resolve()
        if base_abs.exists():
            with open(base_abs, "r") as f:
                base_cfg = yaml.safe_load(f)
            merged = base_cfg.copy()
            merged.update(cfg)
            cfg = merged
    return cfg


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint."""
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    model = build_model({
        "backbone": model_cfg.get("backbone", {"type": "CSPDarknet", "variant": "n"}),
        "fpn_out_channels": 256,
        "num_known_classes": data_cfg.get("num_known_classes", 4),
        "num_classes": data_cfg.get("num_classes", 5),
    })

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("evaluate")
    device = torch.device(args.device)

    data_cfg = cfg.get("data", {})

    # Get annotation file and image directory for the split
    split_ann_map = {
        "train": (data_cfg.get("train_ann", "data/annotations/train.json"),
                  data_cfg.get("train_img_dir", "data/images/train")),
        "val": (data_cfg.get("val_ann", "data/annotations/val.json"),
                data_cfg.get("val_img_dir", "data/images/val")),
        "test": (data_cfg.get("test_ann", "data/annotations/test.json"),
                 data_cfg.get("test_img_dir", "data/images/test")),
    }
    ann_file, img_dir = split_ann_map[args.split]

    if not Path(ann_file).exists():
        logger.error(f"Annotation file not found: {ann_file}")
        sys.exit(1)

    # Build data loader
    img_size = data_cfg.get("img_size", [640, 640])
    transforms = get_val_transforms(tuple(img_size))
    data_loader = build_dataloader(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=transforms,
        batch_size=args.batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=False,
        num_known_classes=data_cfg.get("num_known_classes", 4),
    )
    logger.info(f"Evaluating on {args.split} split: {len(data_loader.dataset)} samples")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, device)

    # Load baseline model (optional)
    baseline_model = None
    if args.baseline_checkpoint:
        logger.info(f"Loading baseline: {args.baseline_checkpoint}")
        baseline_model = load_model(cfg, args.baseline_checkpoint, device)

    # Run evaluation
    evaluator = Evaluator(
        model=model,
        data_loader=data_loader,
        num_known_classes=data_cfg.get("num_known_classes", 4),
        unknown_class_id=data_cfg.get("num_known_classes", 4),
        device=device,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        baseline_model=baseline_model,
    )

    metrics = evaluator.run()

    # Save results
    if args.output:
        import json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Convert non-serializable values
        serializable = {
            k: (v if not (isinstance(v, float) and v != v) else "NaN")
            for k, v in metrics.items()
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
