"""
Training pipeline for multi-task open-set safety violation detection.

Supports:
- Standard training with W&B experiment tracking
- Automatic Mixed Precision (AMP)
- Gradient clipping
- Model checkpointing
- Continual learning with replay buffer integration
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class Trainer:
    """
    Training coordinator for the multi-task detector.

    Handles the training loop, validation, checkpointing, and
    optional W&B experiment tracking.

    Args:
        model: MultiTaskDetector model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: MultiTaskLoss instance.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        cfg: Training configuration dict.
        device: Computation device.
        continual_learner: Optional ContinualLearner for incremental learning.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        cfg: Dict[str, Any],
        device: Optional[torch.device] = None,
        continual_learner: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.continual_learner = continual_learner

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Training settings
        self.epochs = cfg.get("epochs", 100)
        self.gradient_clip = cfg.get("gradient_clip", 10.0)
        self.amp = cfg.get("amp", True) and self.device.type == "cuda"
        self.save_every = cfg.get("save_every", 10)
        self.val_every = cfg.get("val_every", 5)
        self.checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler
        self.scaler = GradScaler() if self.amp else None

        # W&B logging
        self.use_wandb = cfg.get("wandb", {}).get("enabled", False)
        self._wandb = None
        if self.use_wandb:
            self._init_wandb(cfg)

        # Tracking
        self.best_metric = 0.0
        self.current_epoch = 0
        self.global_step = 0

    def _init_wandb(self, cfg: Dict) -> None:
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            wandb_cfg = cfg.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "oilfield-safety-detection"),
                entity=wandb_cfg.get("entity"),
                name=cfg.get("project", {}).get("name", "run"),
                config=cfg,
                tags=wandb_cfg.get("tags", []),
            )
            self._wandb = wandb
        except ImportError:
            print("wandb not installed. Skipping W&B logging.")
            self.use_wandb = False

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in batch["targets"]
            ]

            # Mix in replay samples if using continual learning
            if self.continual_learner is not None:
                replay_samples = self.continual_learner.get_replay_samples(
                    batch_size=images.shape[0]
                )
                # In a real implementation, prepend replay to batch
                # For simplicity, we skip actual image loading here

            self.optimizer.zero_grad()

            if self.amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images, targets)
                    losses = self.criterion(outputs, targets)
                    loss = losses["total_loss"]

                    # Continual learning forgetting prevention
                    if self.continual_learner is not None:
                        forgetting_loss = self.continual_learner.compute_forgetting_loss(
                            outputs, images
                        )
                        loss = loss + forgetting_loss

                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, targets)
                losses = self.criterion(outputs, targets)
                loss = losses["total_loss"]

                if self.continual_learner is not None:
                    forgetting_loss = self.continual_learner.compute_forgetting_loss(
                        outputs, images
                    )
                    loss = loss + forgetting_loss

                loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else v
                )
            num_batches += 1
            self.global_step += 1

            # Log per-step
            if self._wandb is not None and batch_idx % 10 == 0:
                self._wandb.log(
                    {f"train/{k}": v / (batch_idx + 1) for k, v in epoch_losses.items()},
                    step=self.global_step,
                )

        # Average over batches
        if num_batches > 0:
            epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in batch["targets"]
            ]

            if self.amp:
                with autocast():
                    outputs = self.model(images, targets)
                    losses = self.criterion(outputs, targets)
            else:
                outputs = self.model(images, targets)
                losses = self.criterion(outputs, targets)

            for k, v in losses.items():
                val_losses[k] = val_losses.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else v
                )
            num_batches += 1

        if num_batches > 0:
            val_losses = {k: v / num_batches for k, v in val_losses.items()}

        return val_losses

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> str:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved to {best_path}")

        return str(path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint["epoch"]

    def fit(self) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Returns:
            Training history dict with loss curves.
        """
        history: Dict[str, List[float]] = {}
        print(f"\nTraining on {self.device}")
        print(f"Epochs: {self.epochs} | AMP: {self.amp}")
        print("=" * 60)

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_losses = self.train_epoch()

            # Update scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

            # Validate
            val_losses = {}
            if (epoch + 1) % self.val_every == 0:
                val_losses = self.validate()

            # Record history
            for k, v in {**train_losses, **{f"val_{k}": v for k, v in val_losses.items()}}.items():
                history.setdefault(k, []).append(v)

            # W&B logging
            if self._wandb is not None:
                log_dict = {f"train/{k}": v for k, v in train_losses.items()}
                log_dict.update({f"val/{k}": v for k, v in val_losses.items()})
                log_dict["epoch"] = epoch
                try:
                    lrs = self.scheduler.get_last_lr()
                    log_dict["lr"] = lrs[0] if lrs else 0
                except Exception:
                    pass
                self._wandb.log(log_dict, step=self.global_step)

            # Checkpointing
            current_metric = 1.0 / (train_losses.get("total_loss", 1.0) + 1e-7)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(
                    epoch + 1,
                    {**train_losses, **val_losses},
                    is_best=is_best,
                )

            # Print progress
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:>4}/{self.epochs}] "
                f"Loss: {train_losses.get('total_loss', 0):.4f} "
                f"| Val Loss: {val_losses.get('total_loss', 0):.4f} "
                f"| Time: {elapsed:.1f}s"
            )

        if self._wandb is not None:
            self._wandb.finish()

        print("=" * 60)
        print("Training complete!")
        return history
