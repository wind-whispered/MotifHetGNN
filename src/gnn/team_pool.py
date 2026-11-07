"""Edge-free team-pooling baseline for graph-signal attribution.

The model applies the same-depth residual MLP independently to every player,
mean-pools home and away players, and uses the same dual prediction heads as
the graph models.  It deliberately never reads ``edge_index`` or edge
attributes, so it is a genuine node-feature-only control rather than a
constant-feature topology model.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_mean_pool

from .score_context import cutoff_score_features


class TeamPoolMLP(nn.Module):
    """Permutation-invariant DeepSets-style home/away node baseline."""

    NODE_TYPES = ("home_player", "away_player")

    def __init__(
        self,
        node_feature_dim: int = 39,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        n_cls: int = 3,
        use_current_score: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_current_score = use_current_score
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(node_feature_dim, hidden_dim)
            for node_type in self.NODE_TYPES
        })
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.Linear(hidden_dim, hidden_dim)
                for node_type in self.NODE_TYPES
            })
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in self.NODE_TYPES
            })
            for _ in range(num_layers)
        ])
        graph_dim = 2 * hidden_dim + (4 if use_current_score else 0)
        self.cls_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _batch_vector(data: HeteroData, node_type: str,
                      features: torch.Tensor) -> torch.Tensor:
        batch = getattr(data[node_type], "batch", None)
        if batch is None:
            return torch.zeros(features.size(0), dtype=torch.long,
                               device=features.device)
        return batch

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = {
            node_type: self.input_proj[node_type](data[node_type].x)
            for node_type in self.NODE_TYPES
        }
        for layer, norm in zip(self.layers, self.norms):
            for node_type in self.NODE_TYPES:
                residual = embeddings[node_type]
                update = layer[node_type](residual)
                update = F.relu(norm[node_type](update))
                update = F.dropout(update, p=self.dropout,
                                   training=self.training)
                embeddings[node_type] = residual + update

        pooled = []
        for node_type in self.NODE_TYPES:
            values = embeddings[node_type]
            pooled.append(global_mean_pool(
                values, self._batch_vector(data, node_type, values)))
        graph_embedding = torch.cat(pooled, dim=-1)
        if self.use_current_score:
            score_features = cutoff_score_features(
                data, graph_embedding.device, "TeamPool-MLP")
            graph_embedding = torch.cat(
                [graph_embedding, score_features], dim=-1)
        return self.cls_head(graph_embedding), self.reg_head(graph_embedding)
