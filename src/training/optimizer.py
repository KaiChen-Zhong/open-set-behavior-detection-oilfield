"""
Optimizer and learning rate scheduler builders for training pipeline.
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


def build_optimizer(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Build optimizer from configuration dictionary.

    Supports different learning rates for backbone vs. head parameters
    (common practice in fine-tuning detection models).

    Args:
        model: Model to optimize.
        cfg: Optimizer config with keys: name, lr, weight_decay, etc.

    Returns:
        Configured optimizer.
    """
    name = cfg.get("name", "AdamW")
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-4)
    backbone_lr_scale = cfg.get("backbone_lr_scale", 0.1)

    # Separate backbone and non-backbone parameters
    backbone_params = []
    other_params = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in param_name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        {
            "params": backbone_params,
            "lr": lr * backbone_lr_scale,
            "weight_decay": weight_decay,
        },
    ]

    if name == "AdamW":
        betas = cfg.get("betas", [0.9, 0.999])
        return AdamW(param_groups, betas=betas)
    elif name == "Adam":
        betas = cfg.get("betas", [0.9, 0.999])
        return Adam(param_groups, betas=betas)
    elif name == "SGD":
        momentum = cfg.get("momentum", 0.937)
        return SGD(param_groups, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    num_training_steps: Optional[int] = None,
) -> Any:
    """
    Build learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer instance.
        cfg: Scheduler config with keys: name, warmup_epochs, etc.
        num_training_steps: Total training steps (for OneCycleLR).

    Returns:
        Scheduler instance or SequentialLR with warmup.
    """
    name = cfg.get("name", "CosineAnnealingLR")
    warmup_epochs = cfg.get("warmup_epochs", 5)
    warmup_lr_init = cfg.get("warmup_lr_init", 1e-6)
    T_max = cfg.get("T_max", 100)
    eta_min = cfg.get("eta_min", 1e-6)
    step_size = cfg.get("step_size", 30)
    gamma = cfg.get("gamma", 0.1)

    if name == "CosineAnnealingLR":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_epochs, eta_min=eta_min
        )
    elif name == "StepLR":
        main_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            verbose=True,
        )
    elif name == "OneCycleLR":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for OneCycleLR")
        max_lr = [g["lr"] for g in optimizer.param_groups]
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=0.1,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_lr_init / max(cfg.get("lr", 1e-4), 1e-8),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler


class WarmupCosineScheduler:
    """
    Manual warmup + cosine annealing scheduler.
    Useful when finer control over LR is needed.

    Args:
        optimizer: Optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs.
        eta_min: Minimum learning rate.
        warmup_lr_init: Starting LR for warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        eta_min: float = 1e-6,
        warmup_lr_init: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.warmup_lr_init = warmup_lr_init
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate."""
        if epoch is None:
            epoch = self.current_epoch
        self.current_epoch = epoch + 1

        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(self.warmup_epochs, 1)
            lrs = [
                self.warmup_lr_init + (base_lr - self.warmup_lr_init) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lrs = [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                for base_lr in self.base_lrs
            ]

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def get_last_lr(self) -> List[float]:
        """Return current learning rates."""
        return [g["lr"] for g in self.optimizer.param_groups]
