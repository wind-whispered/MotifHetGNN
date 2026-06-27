"""
Task 9 - Part D: Evaluation metrics on test set.
"""
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .model import HeteroFootballGNN, goal_diff_to_class

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_on_test(
    model: HeteroFootballGNN,
    test_loader,
    device: str = "cpu",
) -> Dict:
    """
    Full evaluation on test set.
    Returns dict with accuracy, MAE, RMSE, and per-class metrics.
    """
    model.eval()
    model = model.to(device)

    all_cls_preds, all_cls_labels = [], []
    all_reg_preds, all_reg_labels = [], []
    all_match_ids = []

    for batch in test_loader:
        batch = batch.to(device)
        cls_logits, reg_out = model(batch)
        goal_diff = batch.y.squeeze()

        cls_preds = cls_logits.argmax(dim=-1).cpu().numpy()
        cls_labels = goal_diff_to_class(goal_diff).cpu().numpy()
        reg_preds = reg_out.squeeze().cpu().numpy()
        reg_labels = goal_diff.cpu().numpy()

        all_cls_preds.extend(cls_preds.tolist())
        all_cls_labels.extend(cls_labels.tolist())
        all_reg_preds.extend(reg_preds.tolist() if reg_preds.ndim > 0 else [float(reg_preds)])
        all_reg_labels.extend(reg_labels.tolist() if reg_labels.ndim > 0 else [float(reg_labels)])

        if hasattr(batch, "match_id"):
            mids = batch.match_id
            if isinstance(mids, torch.Tensor):
                all_match_ids.extend(mids.tolist())
            else:
                all_match_ids.append(mids)

    cls_arr = np.array(all_cls_labels)
    pred_arr = np.array(all_cls_preds)
    reg_arr = np.array(all_reg_labels, dtype=float)
    reg_pred_arr = np.array(all_reg_preds, dtype=float)

    accuracy = float(np.mean(cls_arr == pred_arr))
    mae = float(np.mean(np.abs(reg_arr - reg_pred_arr)))
    rmse = float(np.sqrt(np.mean((reg_arr - reg_pred_arr) ** 2)))

    cls_report = classification_report(
        cls_arr, pred_arr,
        labels=[0, 1, 2],
        target_names=["Home Win", "Draw", "Away Win"],
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(cls_arr, pred_arr, labels=[0, 1, 2]).tolist()

    # Per-match predictions DataFrame
    pred_df = pd.DataFrame({
        "match_id": all_match_ids if all_match_ids else list(range(len(all_cls_labels))),
        "true_class": all_cls_labels,
        "pred_class": all_cls_preds,
        "true_goal_diff": all_reg_labels,
        "pred_goal_diff": all_reg_preds,
    })

    return {
        "accuracy": accuracy,
        "mae": mae,
        "rmse": rmse,
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        "predictions_df": pred_df,
    }


def compute_baseline_accuracy(match_meta_df: pd.DataFrame) -> float:
    """
    Naive baseline: always predict the most frequent outcome.
    Used as sanity check for GNN performance.
    """
    goal_diff = match_meta_df["goal_diff"]
    labels = pd.cut(goal_diff, bins=[-np.inf, -0.5, 0.5, np.inf], labels=[2, 1, 0])
    most_common = labels.value_counts().index[0]
    return float((labels == most_common).mean())
