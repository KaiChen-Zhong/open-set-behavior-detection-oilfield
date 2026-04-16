"""
Continual / Incremental Learning Module with Replay Buffer.

Supports:
- Reservoir sampling replay buffer
- Herding-based exemplar selection (iCaRL-style)
- Elastic Weight Consolidation (EWC)
- Learning Without Forgetting (LwF)

Designed for incrementally adding new violation categories
without forgetting previously learned ones.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores exemplar samples from past tasks to prevent catastrophic forgetting.
    Supports multiple selection strategies: reservoir sampling, herding, random.

    Args:
        max_size: Maximum total buffer capacity.
        samples_per_class: Maximum exemplars per class.
        selection_strategy: How to select exemplars ("reservoir", "herding", "random").
    """

    def __init__(
        self,
        max_size: int = 2000,
        samples_per_class: int = 50,
        selection_strategy: str = "reservoir",
    ):
        self.max_size = max_size
        self.samples_per_class = samples_per_class
        self.selection_strategy = selection_strategy

        # Buffer stores per-class lists of (image, target) tuples
        self._buffer: Dict[int, List[Dict]] = defaultdict(list)
        self._total_seen: Dict[int, int] = defaultdict(int)  # For reservoir sampling

    @property
    def total_size(self) -> int:
        """Total number of samples in buffer."""
        return sum(len(v) for v in self._buffer.values())

    @property
    def num_classes(self) -> int:
        """Number of classes currently in buffer."""
        return len(self._buffer)

    def add_samples(
        self,
        samples: List[Dict],
        class_id: int,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add samples for a specific class to the buffer.

        Args:
            samples: List of sample dicts with keys: image, target.
            class_id: Class ID for these samples.
            features: Optional feature vectors for herding selection (N, D).
        """
        if self.selection_strategy == "herding" and features is not None:
            selected = self._herding_select(samples, features)
        elif self.selection_strategy == "reservoir":
            selected = self._reservoir_select(samples, class_id)
        else:
            selected = random.sample(
                samples, min(len(samples), self.samples_per_class)
            )

        # Enforce per-class limit
        self._buffer[class_id] = selected[: self.samples_per_class]

        # Enforce global buffer limit by trimming oldest classes
        while self.total_size > self.max_size and len(self._buffer) > 1:
            # Remove samples from the most populated class
            largest_class = max(
                self._buffer.keys(), key=lambda k: len(self._buffer[k])
            )
            if len(self._buffer[largest_class]) > 1:
                self._buffer[largest_class].pop()
            else:
                break

    def _herding_select(
        self, samples: List[Dict], features: torch.Tensor
    ) -> List[Dict]:
        """
        Herding-based exemplar selection (iCaRL-style).

        Selects samples whose mean feature is closest to the class mean.
        """
        n_select = min(len(samples), self.samples_per_class)
        # Normalize features
        feats = F.normalize(features.float(), dim=-1)
        class_mean = feats.mean(0)

        selected_indices = []
        selected_sum = torch.zeros_like(class_mean)

        for _ in range(n_select):
            remaining = [i for i in range(len(samples)) if i not in selected_indices]
            if not remaining:
                break
            # Find exemplar closest to class mean
            candidates = feats[remaining]
            diff = (class_mean - (selected_sum + candidates) / (len(selected_indices) + 1))
            best_local = diff.norm(dim=-1).argmin().item()
            best_global = remaining[best_local]
            selected_indices.append(best_global)
            selected_sum += feats[best_global]

        return [samples[i] for i in selected_indices]

    def _reservoir_select(self, samples: List[Dict], class_id: int) -> List[Dict]:
        """
        Reservoir sampling for uniform random exemplar selection.
        """
        buffer = list(self._buffer[class_id])  # Copy current buffer
        for sample in samples:
            self._total_seen[class_id] += 1
            if len(buffer) < self.samples_per_class:
                buffer.append(sample)
            else:
                j = random.randint(0, self._total_seen[class_id] - 1)
                if j < self.samples_per_class:
                    buffer[j] = sample
        return buffer

    def sample(
        self, num_samples: int, class_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Sample from the replay buffer.

        Args:
            num_samples: Number of samples to retrieve.
            class_ids: Optional list of class IDs to sample from.

        Returns:
            List of sampled dicts.
        """
        if class_ids is None:
            class_ids = list(self._buffer.keys())

        available = []
        for cid in class_ids:
            available.extend(self._buffer.get(cid, []))

        if not available:
            return []

        return random.sample(available, min(num_samples, len(available)))

    def get_class_samples(self, class_id: int) -> List[Dict]:
        """Get all stored samples for a specific class."""
        return list(self._buffer.get(class_id, []))

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._total_seen.clear()

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer("
            f"total={self.total_size}/{self.max_size}, "
            f"classes={list(self._buffer.keys())}, "
            f"strategy={self.selection_strategy})"
        )


class EWC(nn.Module):
    """
    Elastic Weight Consolidation (EWC) for continual learning.

    Adds a regularization term that prevents important parameters
    (as measured by the Fisher Information Matrix) from changing too much.

    Args:
        model: The neural network model.
        ewc_lambda: Regularization strength.
        online: Use online EWC (single Fisher accumulation).
        gamma: Online EWC decay factor.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 5000.0,
        online: bool = True,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Registered buffers for Fisher and optimal params
        self._means: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._task_count = 0

    def compute_fisher(
        self,
        dataloader: Any,
        device: torch.device,
        num_samples: int = 200,
    ) -> None:
        """
        Compute Fisher Information Matrix approximation using a dataset.

        Args:
            dataloader: DataLoader for computing Fisher.
            device: Computation device.
            num_samples: Max samples for Fisher estimation.
        """
        self.model.eval()
        fisher: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            images = batch["images"].to(device)
            self.model.zero_grad()

            # Forward pass and compute log-likelihood gradient
            outputs = self.model(images)
            # Use classification logits if available
            if "cls_logits" in outputs:
                log_prob = F.log_softmax(outputs["cls_logits"], dim=-1)
                loss = -log_prob.sum(dim=-1).mean()
            elif "det_cls_scores" in outputs:
                scores = outputs["det_cls_scores"][0]
                log_prob = F.log_softmax(scores.reshape(-1, scores.shape[-1]), dim=-1)
                loss = -log_prob.sum(dim=-1).mean()
            else:
                continue

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            count += images.shape[0]

        # Normalize
        for name in fisher:
            fisher[name] /= max(count, 1)

        # Online EWC: decay old Fisher and accumulate new
        if self.online and self._fisher:
            for name in fisher:
                if name in self._fisher:
                    self._fisher[name] = (
                        self.gamma * self._fisher[name] + fisher[name]
                    )
                else:
                    self._fisher[name] = fisher[name]
        else:
            self._fisher = fisher

        # Store optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._means[name] = param.data.clone()

        self._task_count += 1
        self.model.train()

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty.

        Returns:
            Scalar EWC penalty loss.
        """
        if not self._fisher:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._fisher and name in self._means:
                fisher = self._fisher[name]
                mean = self._means[name]
                loss = loss + (fisher * (param - mean).pow(2)).sum()

        return self.ewc_lambda * loss * 0.5


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss for Learning Without Forgetting (LwF).

    Preserves old task knowledge by distilling from a frozen teacher model.

    Args:
        temperature: Distillation temperature (higher = softer distributions).
        alpha: Balance between distillation and new task loss.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        new_task_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined distillation + new task loss.

        Args:
            student_logits: (N, C_old) student logits for old classes.
            teacher_logits: (N, C_old) teacher logits for old classes.
            labels: (N,) ground truth labels (for new task CE loss).
            new_task_loss: Pre-computed new task loss scalar.

        Returns:
            Combined loss scalar.
        """
        T = self.temperature
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / T, dim=-1).detach()
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        distill_loss = -(soft_targets * student_soft).sum(dim=-1).mean() * (T * T)

        if new_task_loss is not None:
            return self.alpha * distill_loss + (1 - self.alpha) * new_task_loss
        return distill_loss


class ContinualLearner:
    """
    High-level continual learning coordinator.

    Manages task-incremental learning, replay buffer updates,
    and optional EWC/LwF regularization.

    Args:
        model: The detection model.
        strategy: Learning strategy ("replay", "ewc", "lwf", "icarl").
        replay_buffer: Replay buffer instance.
        ewc_lambda: EWC regularization strength (if using EWC).
        distill_temperature: LwF distillation temperature.
        distill_alpha: LwF distillation weight.
        replay_ratio: Fraction of batch from replay buffer.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "replay",
        replay_buffer: Optional[ReplayBuffer] = None,
        ewc_lambda: float = 5000.0,
        distill_temperature: float = 4.0,
        distill_alpha: float = 0.5,
        replay_ratio: float = 0.3,
    ):
        self.model = model
        self.strategy = strategy
        self.replay_ratio = replay_ratio
        self.current_task = 0
        self.seen_classes: List[int] = []

        # Replay buffer
        if replay_buffer is None:
            self.replay_buffer = ReplayBuffer()
        else:
            self.replay_buffer = replay_buffer

        # EWC
        if strategy in ("ewc", "combined"):
            self.ewc = EWC(model, ewc_lambda=ewc_lambda)
        else:
            self.ewc = None

        # LwF: frozen teacher model
        if strategy in ("lwf", "combined"):
            self.teacher_model: Optional[nn.Module] = None
            self.distill_loss_fn = KnowledgeDistillationLoss(
                temperature=distill_temperature,
                alpha=distill_alpha,
            )
        else:
            self.teacher_model = None
            self.distill_loss_fn = None

    def start_new_task(
        self,
        new_class_ids: List[int],
        dataloader: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Prepare for a new incremental task.

        Args:
            new_class_ids: List of new class IDs being added.
            dataloader: DataLoader for Fisher computation (EWC).
            device: Computation device.
        """
        # Update EWC Fisher after previous task
        if self.ewc is not None and dataloader is not None and device is not None:
            self.ewc.compute_fisher(dataloader, device)

        # Freeze teacher model for LwF
        if self.strategy in ("lwf", "combined"):
            import copy
            self.teacher_model = copy.deepcopy(self.model)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()

        self.seen_classes.extend(new_class_ids)
        self.current_task += 1

    def get_replay_samples(self, batch_size: int) -> List[Dict]:
        """
        Get replay samples to mix into the current training batch.

        Args:
            batch_size: Main batch size.

        Returns:
            List of replay samples (up to replay_ratio * batch_size).
        """
        n_replay = int(batch_size * self.replay_ratio)
        return self.replay_buffer.sample(n_replay, class_ids=self.seen_classes)

    def compute_forgetting_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anti-forgetting regularization loss.

        Args:
            outputs: Current model outputs.
            images: Input images for teacher forward pass.

        Returns:
            Forgetting prevention loss scalar.
        """
        loss = torch.tensor(0.0, device=images.device)

        # EWC penalty
        if self.ewc is not None:
            loss = loss + self.ewc.penalty().to(images.device)

        # LwF distillation
        if self.teacher_model is not None and self.distill_loss_fn is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(images)

            if "cls_logits" in outputs and "cls_logits" in teacher_outputs:
                student_logits = outputs["cls_logits"][:, : len(self.seen_classes)]
                teacher_logits = teacher_outputs["cls_logits"][
                    :, : len(self.seen_classes)
                ]
                distill = self.distill_loss_fn(student_logits, teacher_logits)
                loss = loss + distill

        return loss

    def update_buffer(
        self,
        samples: List[Dict],
        class_id: int,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        """Update replay buffer with new exemplars."""
        self.replay_buffer.add_samples(samples, class_id, features)

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Return replay buffer statistics."""
        return {
            "total_samples": self.replay_buffer.total_size,
            "max_capacity": self.replay_buffer.max_size,
            "num_classes": self.replay_buffer.num_classes,
            "seen_classes": self.seen_classes,
            "current_task": self.current_task,
            "strategy": self.strategy,
        }
