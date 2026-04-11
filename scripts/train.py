#!/usr/bin/env python3
"""
Main training script for open-set safety violation detection.

Usage:
    python scripts/train.py --config configs/multitask/multitask_config.yaml
    python scripts/train.py --config configs/detection/yolov8_baseline.yaml
    python scripts/train.py --config configs/continual/continual_config.yaml \\
        --use-wandb --device cuda

For YOLOv8 baseline (uses Ultralytics API):
    python scripts/train.py --config configs/detection/yolov8_baseline.yaml \\
        --use-ultralytics
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from src.data import build_dataloader, get_train_transforms, get_val_transforms
from src.models import MultiTaskDetector, build_model, ContinualLearner, ReplayBuffer
from src.training import Trainer, MultiTaskLoss, build_optimizer, build_scheduler
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train open-set safety violation detector"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device (cuda/cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable W&B experiment tracking",
    )
    parser.add_argument(
        "--use-ultralytics",
        action="store_true",
        help="Use Ultralytics API for YOLOv8 baseline training",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use a small subset of data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and merge YAML config with base config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load base config if specified
    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        config_dir = Path(config_path).parent
        base_abs = (config_dir / base_path).resolve()
        if base_abs.exists():
            with open(base_abs, "r") as f:
                base_cfg = yaml.safe_load(f)
            # Merge: current config overrides base
            merged = base_cfg.copy()
            merged.update(cfg)
            cfg = merged

    return cfg


def train_with_ultralytics(cfg: dict, args: argparse.Namespace) -> None:
    """Train YOLOv8 baseline using Ultralytics API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_cfg = cfg.get("model", {})
    arch = model_cfg.get("arch", "yolov8n")
    pretrained_weights = model_cfg.get("pretrained_weights", f"{arch}.pt")
    training_cfg = cfg.get("training", {})

    model = YOLO(pretrained_weights)
    model.train(
        data=cfg["data"].get("dataset_yaml", "data.yaml"),
        epochs=training_cfg.get("epochs", 100),
        batch=training_cfg.get("batch_size", 16),
        imgsz=training_cfg.get("img_size", [640, 640])[0],
        device=args.device,
        project=cfg.get("project", {}).get("name", "yolov8_baseline"),
        amp=training_cfg.get("amp", True),
        save=True,
    )


def train_custom_model(cfg: dict, args: argparse.Namespace) -> None:
    """Train the custom multi-task open-set detector."""
    logger = setup_logger("train", log_file=f"{cfg.get('training', {}).get('log_dir', 'logs/')}/train.log")
    device = torch.device(args.device)

    # Override W&B setting from args
    if args.use_wandb:
        cfg.setdefault("wandb", {})["enabled"] = True

    # Override output dir from args
    if args.output_dir:
        cfg.setdefault("training", {})["checkpoint_dir"] = args.output_dir

    # Build data loaders
    logger.info("Building datasets...")
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    img_size = data_cfg.get("img_size", [640, 640])
    max_samples = 100 if args.debug else None

    train_transforms = get_train_transforms(tuple(img_size))
    val_transforms = get_val_transforms(tuple(img_size))

    ann_file_train = data_cfg.get("train_ann", "data/annotations/train.json")
    ann_file_val = data_cfg.get("val_ann", "data/annotations/val.json")
    img_dir_train = data_cfg.get("train_img_dir", "data/images/train")
    img_dir_val = data_cfg.get("val_img_dir", "data/images/val")

    # Check data exists
    if not Path(ann_file_train).exists():
        logger.warning(
            f"Training annotation file not found: {ann_file_train}. "
            "Please prepare your dataset first. See README for instructions."
        )
        return

    train_loader = build_dataloader(
        ann_file=ann_file_train,
        img_dir=img_dir_train,
        transforms=train_transforms,
        batch_size=training_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=True,
        num_known_classes=data_cfg.get("num_known_classes", 4),
        max_samples=max_samples,
    )
    val_loader = build_dataloader(
        ann_file=ann_file_val,
        img_dir=img_dir_val,
        transforms=val_transforms,
        batch_size=training_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 4),
        shuffle=False,
        num_known_classes=data_cfg.get("num_known_classes", 4),
        max_samples=max_samples,
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Build model
    logger.info("Building model...")
    model_cfg = cfg.get("model", {})
    model = build_model({
        "backbone": model_cfg.get("backbone", {"type": "CSPDarknet", "variant": "n"}),
        "fpn_out_channels": 256,
        "num_known_classes": data_cfg.get("num_known_classes", 4),
        "num_classes": data_cfg.get("num_classes", 5),
        "enable_spatial_reasoning": True,
        "enable_classification": True,
    })

    param_counts = model.get_num_parameters()
    logger.info(f"Model parameters: {param_counts}")

    # Build criterion, optimizer, scheduler
    criterion = MultiTaskLoss(
        num_known_classes=data_cfg.get("num_known_classes", 4),
    )
    optimizer = build_optimizer(model, cfg.get("optimizer", {}))
    scheduler = build_scheduler(
        optimizer,
        cfg.get("scheduler", {"name": "CosineAnnealingLR", "T_max": training_cfg.get("epochs", 100)}),
    )

    # Build continual learner if using continual config
    continual_learner = None
    if "continual_learning" in cfg:
        cl_cfg = cfg["continual_learning"]
        buffer_cfg = cl_cfg.get("replay_buffer", {})
        replay_buffer = ReplayBuffer(
            max_size=buffer_cfg.get("max_size", 2000),
            samples_per_class=buffer_cfg.get("samples_per_class", 50),
            selection_strategy=buffer_cfg.get("selection_strategy", "herding"),
        )
        continual_learner = ContinualLearner(
            model=model,
            strategy=cl_cfg.get("strategy", "replay"),
            replay_buffer=replay_buffer,
            replay_ratio=training_cfg.get("replay_ratio", 0.3),
        )
        logger.info(f"Continual learning: strategy={cl_cfg.get('strategy', 'replay')}")

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg={**training_cfg, "wandb": cfg.get("wandb", {})},
        device=device,
        continual_learner=continual_learner,
    )

    # Resume from checkpoint if specified
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Train
    logger.info("Starting training...")
    history = trainer.fit()
    logger.info("Training complete!")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.use_ultralytics:
        train_with_ultralytics(cfg, args)
    else:
        train_custom_model(cfg, args)


if __name__ == "__main__":
    main()
