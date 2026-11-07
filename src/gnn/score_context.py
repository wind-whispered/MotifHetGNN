"""Shared, lossless encoding of cutoff-observable scoreboard context."""
from __future__ import annotations

import torch


SCORE_CONTEXT_REVISION = "exact_scaled_unclipped_v2"


def cutoff_score_features(data, device: torch.device,
                          model_name: str) -> torch.Tensor:
    """Return exact scaled home, away and difference scores plus level flag.

    Scaling stabilises optimisation but deliberately does not clip rare high
    scores, so every score-aware neural model is nested with the score-only
    controls. The persisted goal difference is checked against both scores.
    """
    required = ("current_home_score", "current_away_score",
                "current_goal_diff")
    if not all(hasattr(data, name) for name in required):
        raise ValueError(
            f"score-aware {model_name} requires cutoff score attributes")
    home = torch.as_tensor(
        data.current_home_score, device=device).float().reshape(-1)
    away = torch.as_tensor(
        data.current_away_score, device=device).float().reshape(-1)
    reported_diff = torch.as_tensor(
        data.current_goal_diff, device=device).float().reshape(-1)
    if not (home.numel() == away.numel() == reported_diff.numel()):
        raise ValueError("cutoff score attributes have inconsistent batch sizes")
    diff = home - away
    if not torch.allclose(diff, reported_diff, atol=1e-6, rtol=0.0):
        raise ValueError("current_goal_diff disagrees with cutoff scores")
    return torch.stack([
        home / 5.0,
        away / 5.0,
        diff / 5.0,
        (diff == 0).float(),
    ], dim=-1)
