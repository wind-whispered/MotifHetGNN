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
    ):
        self.model = model
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.patience = patience

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _compute_loss(
        self,
        cls_logits: torch.Tensor,
        reg_out: torch.Tensor,
        goal_diff: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute combined classification + regression loss."""
        cls_labels = goal_diff_to_class(goal_diff.squeeze())
        loss_cls = self.cls_criterion(cls_logits, cls_labels)
        loss_reg = self.reg_criterion(reg_out.squeeze(), goal_diff.squeeze().float())
        loss = self.weight_cls * loss_cls + self.weight_reg * loss_reg
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
        all_preds_cls, all_labels_cls = [], []
        all_preds_reg, all_labels_reg = [], []
        n_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            cls_logits, reg_out = self.model(batch)
            goal_diff = batch.y.to(self.device)

            loss, _, _ = self._compute_loss(cls_logits, reg_out, goal_diff)
            total_loss += float(loss)

            cls_preds = cls_logits.argmax(dim=-1).cpu().numpy()
            cls_labels = goal_diff_to_class(goal_diff.squeeze()).cpu().numpy()
            all_preds_cls.extend(cls_preds.tolist())
            all_labels_cls.extend(cls_labels.tolist())

            all_preds_reg.extend(reg_out.squeeze().cpu().numpy().tolist())
            all_labels_reg.extend(goal_diff.squeeze().cpu().numpy().tolist())

            n_batches += 1

        acc = np.mean(np.array(all_preds_cls) == np.array(all_labels_cls))
        mae = np.mean(np.abs(np.array(all_preds_reg) - np.array(all_labels_reg)))

        return {
            "loss": total_loss / max(n_batches, 1),
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

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_val_loss": self.best_val_loss,
        }

    def save_metrics(self, metrics: Dict, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
