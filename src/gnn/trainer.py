"""
Task 9 - Part C: GNN training loop with early stopping and dual loss.
"""
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import HeteroFootballGNN, goal_diff_to_class

logger = logging.getLogger(__name__)

CLASS_WEIGHT_REVISION = "train_sqrt_inverse_frequency_mean1_v1"
FIXED_LOSS_REVISION = "sqrt_invfreq_ce_fixed_half_v2"
UNCERTAINTY_LOSS_REVISION = "sqrt_invfreq_ce_homoscedastic_half_v3"


def loss_revision_for(loss_mode: str) -> str:
    """Return the immutable optimisation-objective revision."""
    if loss_mode == "fixed":
        return FIXED_LOSS_REVISION
    if loss_mode == "uncertainty":
        return UNCERTAINTY_LOSS_REVISION
    raise ValueError("loss_mode must be 'fixed' or 'uncertainty'")


def training_class_weights(goal_diffs) -> torch.Tensor:
    """Square-root inverse-frequency weights derived from training labels."""
    targets = torch.as_tensor(goal_diffs, dtype=torch.float32).reshape(-1)
    labels = goal_diff_to_class(targets)
    counts = torch.bincount(labels, minlength=3).float()
    if bool((counts <= 0).any()):
        raise ValueError(
            f"every outcome class must occur in training: {counts.tolist()}")
    weights = torch.sqrt(targets.numel() / (3.0 * counts))
    return weights / weights.mean()


class GNNTrainer:
    """
    Trainer for HeteroFootballGNN.
    Manages training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: HeteroFootballGNN,
        lr: float = 0.001,
        weight_cls: float = 0.5,
        weight_reg: float = 0.5,
        patience: int = 20,
        device: Optional[str] = None,
        loss_mode: str = "fixed",
        auxiliary_weight: float = 0.01,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.patience = patience
        self.loss_mode = loss_mode
        self.auxiliary_weight = auxiliary_weight
        if loss_mode not in {"fixed", "uncertainty"}:
            raise ValueError("loss_mode must be 'fixed' or 'uncertainty'")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.class_weights = (None if class_weights is None else
                              torch.as_tensor(class_weights,
                                              dtype=torch.float32,
                                              device=self.device))

        self.model = self.model.to(self.device)
        self.task_log_vars = nn.Parameter(torch.zeros(2, device=self.device))
        parameters = list(model.parameters())
        if loss_mode == "uncertainty":
            parameters.append(self.task_log_vars)
        self.optimizer = optim.Adam(parameters, lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.cls_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.reg_criterion = nn.MSELoss()

        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        self.best_val_loss = float("inf")
        self.best_val_task_loss = float("inf")
        self.best_val_auxiliary_loss = float("nan")
        self.best_epoch = 0
        self.best_task_log_vars = None
        self.patience_counter = 0

    def _compute_loss(
        self,
        cls_logits: torch.Tensor,
        reg_out: torch.Tensor,
        goal_diff: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute combined classification + regression loss."""
        targets = goal_diff.reshape(-1).float()
        cls_labels = goal_diff_to_class(targets)
        loss_cls = self.cls_criterion(cls_logits, cls_labels)
        loss_reg = self.reg_criterion(reg_out.reshape(-1), targets)
        if self.loss_mode == "uncertainty":
            precision = torch.exp(-self.task_log_vars)
            # The factor 1/2 makes the uncertainty objective exactly equal to
            # the primary 0.5*CE + 0.5*MSE objective when both log variances
            # are initialised at zero.  This isolates learned task weighting
            # from an otherwise confounded two-fold change in loss scale.
            loss = 0.5 * (
                precision[0] * loss_cls + self.task_log_vars[0]
                + precision[1] * loss_reg + self.task_log_vars[1])
        else:
            loss = self.weight_cls * loss_cls + self.weight_reg * loss_reg
        if hasattr(self.model, "auxiliary_loss"):
            loss = loss + self.auxiliary_weight * self.model.auxiliary_loss()
        return loss, float(loss_cls), float(loss_reg)

    def train_epoch(self, loader) -> Dict:
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            cls_logits, reg_out = self.model(batch)
            goal_diff = batch.y.to(self.device)

            loss, cls_l, reg_l = self._compute_loss(cls_logits, reg_out, goal_diff)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss)
            total_cls_loss += cls_l
            total_reg_loss += reg_l
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "cls_loss": total_cls_loss / max(n_batches, 1),
            "reg_loss": total_reg_loss / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate_epoch(self, loader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        total_task_loss = 0.0
        total_auxiliary_loss = 0.0
        all_preds_cls, all_labels_cls = [], []
        all_preds_reg, all_labels_reg = [], []
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            cls_logits, reg_out = self.model(batch)
            goal_diff = batch.y.to(self.device)

            loss, _, _ = self._compute_loss(cls_logits, reg_out, goal_diff)
            total_loss += float(loss)
            auxiliary = (self.model.auxiliary_loss()
                         if hasattr(self.model, "auxiliary_loss")
                         else torch.zeros((), device=self.device))
            auxiliary_value = float(auxiliary)
            total_auxiliary_loss += auxiliary_value
            total_task_loss += float(loss) - (
                self.auxiliary_weight * auxiliary_value)

            cls_preds = cls_logits.argmax(dim=-1).cpu().numpy()
            targets = goal_diff.reshape(-1)
            cls_labels = goal_diff_to_class(targets).cpu().numpy()
            all_preds_cls.extend(cls_preds.tolist())
            all_labels_cls.extend(cls_labels.tolist())

            all_preds_reg.extend(reg_out.reshape(-1).cpu().numpy().tolist())
            all_labels_reg.extend(targets.cpu().numpy().tolist())

            n_batches += 1

        acc = np.mean(np.array(all_preds_cls) == np.array(all_labels_cls))
        mae = np.mean(np.abs(np.array(all_preds_reg) - np.array(all_labels_reg)))

        return {
            "loss": total_loss / max(n_batches, 1),
            "task_loss": total_task_loss / max(n_batches, 1),
            "auxiliary_loss": total_auxiliary_loss / max(n_batches, 1),
            "accuracy": float(acc),
            "mae": float(mae),
        }

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 200,
        save_path: Optional[str] = None,
    ) -> Dict:
        """Full training loop with early stopping."""
        logger.info(f"Training on device: {self.device}")

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            self.scheduler.step(val_metrics["loss"])

            self.train_history.append({"epoch": epoch, **train_metrics})
            self.val_history.append({"epoch": epoch, **val_metrics})

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics['loss']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.3f} | "
                    f"val_mae={val_metrics['mae']:.3f}"
                )

            # Early stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_task_loss = val_metrics["task_loss"]
                self.best_val_auxiliary_loss = val_metrics["auxiliary_loss"]
                self.best_epoch = epoch
                if self.loss_mode == "uncertainty":
                    self.best_task_log_vars = self.task_log_vars.detach().clone()
                self.patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"  -> Saved best model (val_loss={self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best weights
        if save_path and Path(save_path).exists():
            self.model.load_state_dict(torch.load(save_path, weights_only=True))
        if self.best_task_log_vars is not None:
            with torch.no_grad():
                self.task_log_vars.copy_(self.best_task_log_vars)

        task_log_vars = (self.task_log_vars.detach().cpu().tolist()
                         if self.loss_mode == "uncertainty" else [np.nan, np.nan])
        task_precisions = ([float(np.exp(-value)) for value in task_log_vars]
                           if self.loss_mode == "uncertainty"
                           else [np.nan, np.nan])

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
            "best_val_task_loss": self.best_val_task_loss,
            "best_val_auxiliary_loss": self.best_val_auxiliary_loss,
            "best_epoch": self.best_epoch,
            "task_log_vars": task_log_vars,
            "task_precisions": task_precisions,
        }

    def save_metrics(self, metrics: Dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
