#!/usr/bin/env python3
"""
Baseline comparison script: compares closed-set YOLOv8 baseline vs.
the proposed open-set multi-task detector.

Usage:
    python scripts/compare_baselines.py \\
        --baseline-checkpoint checkpoints/yolov8_baseline.pth \\
        --proposed-checkpoint checkpoints/best_model.pth \\
        --config configs/multitask/multitask_config.yaml \\
        --output results/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data import build_dataloader, get_val_transforms
from src.evaluation import Evaluator, compute_open_set_metrics
from src.models import build_model
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs. proposed open-set detector"
    )
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--proposed-checkpoint", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_config.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="results/comparison.json")
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
            base_cfg.update(cfg)
            cfg = base_cfg
    return cfg


def load_model(cfg: dict, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    model = build_model({
        "backbone": model_cfg.get("backbone", {"type": "CSPDarknet", "variant": "n"}),
        "fpn_out_channels": 256,
        "num_known_classes": data_cfg.get("num_known_classes", 4),
        "num_classes": data_cfg.get("num_classes", 5),
    })
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def print_comparison(
    baseline_metrics: dict, proposed_metrics: dict
) -> None:
    """Print a comparison table."""
    print("\n" + "=" * 70)
    print("  BASELINE vs. PROPOSED MODEL COMPARISON")
    print("=" * 70)
    metrics_to_compare = [
        "mAP", "mAP@75", "precision", "recall", "f1", "WI", "A-OSE"
    ]
    print(f"  {'Metric':<25} {'Baseline':>12} {'Proposed':>12} {'Delta':>10}")
    print("-" * 70)
    for metric in metrics_to_compare:
        base_val = baseline_metrics.get(metric, float("nan"))
        prop_val = proposed_metrics.get(metric, float("nan"))
        try:
            delta = prop_val - base_val
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        except (TypeError, ValueError):
            delta_str = "N/A"
        print(
            f"  {metric:<25} {str(base_val)[:10]:>12} {str(prop_val)[:10]:>12} {delta_str:>10}"
        )
    print("=" * 70)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger("compare")
    device = torch.device(args.device)

    data_cfg = cfg.get("data", {})
    img_size = data_cfg.get("img_size", [640, 640])
    transforms = get_val_transforms(tuple(img_size))

    ann_file = data_cfg.get("test_ann", "data/annotations/test.json")
    img_dir = data_cfg.get("test_img_dir", "data/images/test")

    if not Path(ann_file).exists():
        logger.error(f"Test annotation file not found: {ann_file}")
        sys.exit(1)

    loader = build_dataloader(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=transforms,
        batch_size=args.batch_size,
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=False,
        num_known_classes=data_cfg.get("num_known_classes", 4),
    )

    logger.info("Loading models...")
    baseline = load_model(cfg, args.baseline_checkpoint, device)
    proposed = load_model(cfg, args.proposed_checkpoint, device)

    # Evaluate baseline
    logger.info("Evaluating baseline...")
    baseline_evaluator = Evaluator(
        model=baseline, data_loader=loader,
        num_known_classes=data_cfg.get("num_known_classes", 4),
        device=device,
    )
    baseline_metrics = baseline_evaluator.run()

    # Evaluate proposed (with baseline for WI)
    logger.info("Evaluating proposed model...")
    proposed_evaluator = Evaluator(
        model=proposed, data_loader=loader,
        num_known_classes=data_cfg.get("num_known_classes", 4),
        device=device,
        baseline_model=baseline,
    )
    proposed_metrics = proposed_evaluator.run()

    # Print comparison table
    print_comparison(baseline_metrics, proposed_metrics)

    # Save results
    results = {
        "baseline": {k: (v if not (isinstance(v, float) and v != v) else "NaN")
                     for k, v in baseline_metrics.items()},
        "proposed": {k: (v if not (isinstance(v, float) and v != v) else "NaN")
                     for k, v in proposed_metrics.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
