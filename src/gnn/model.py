"""
Task 9 - Part B: Heterogeneous GNN architecture.
HeteroConv with per-edge-type message passing, dual output heads.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, Linear, global_mean_pool
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class HeteroFootballGNN(nn.Module):
    """
    Heterogeneous GNN for football match graph analysis.

    Architecture:
        - Input projection per node type
        - L layers of HeteroConv (SAGEConv per edge type)
        - Global mean pooling (separately for home and away)
        - Concatenated graph embedding
        - Dual output heads: classification (win/draw/loss) + regression (goal_diff)
    """

    NODE_TYPES = ["home_player", "away_player"]
    EDGE_TYPES = [
        ("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player"),
        ("home_player", "adversarial", "away_player"),
        ("away_player", "adversarial", "home_player"),
        ("home_player", "turnover", "away_player"),
        ("away_player", "turnover", "home_player"),
    ]

    def __init__(
        self,
        node_feature_dim: int = 39,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        n_cls: int = 3,  # win / draw / loss
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("torch_geometric required")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection per node type
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(node_feature_dim, hidden_dim)
            for nt in self.NODE_TYPES
        })

        # HeteroConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in self.EDGE_TYPES:
                conv_dict[et] = SAGEConv(hidden_dim, hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Batch normalization per layer per node type
        self.bns = nn.ModuleList([
            nn.ModuleDict({
                nt: nn.BatchNorm1d(hidden_dim)
                for nt in self.NODE_TYPES
            })
            for _ in range(num_layers)
        ])

        # Graph-level embedding: concat home + away pooled embeddings
        graph_emb_dim = hidden_dim * 2

        # Classification head: win / draw / loss
        self.cls_head = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls),
        )

        # Regression head: goal_diff
        self.reg_head = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: "HeteroData") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            cls_logits: (batch, 3) classification logits
            reg_out: (batch, 1) goal_diff prediction
        """
        # Initial node embeddings
        x_dict = {}
        for nt in self.NODE_TYPES:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                x = data[nt].x
                x_dict[nt] = self.input_proj[nt](x)
            else:
                x_dict[nt] = torch.zeros(0, self.hidden_dim, device=next(self.parameters()).device)

        # Edge index dict (filter to existing edges)
        edge_index_dict = {}
        for et in self.EDGE_TYPES:
            if hasattr(data[et], "edge_index"):
                edge_index_dict[et] = data[et].edge_index

        # Message passing layers
        for layer_idx, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            # Apply BN + residual + activation
            for nt in self.NODE_TYPES:
                if nt in x_dict_new and x_dict_new[nt].shape[0] > 0:
                    h = self.bns[layer_idx][nt](x_dict_new[nt])
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    # Residual connection
                    if x_dict[nt].shape == h.shape:
                        h = h + x_dict[nt]
                    x_dict[nt] = h

        # Global mean pooling per node type
        # Need batch vectors for pooling
        home_x = x_dict.get("home_player", torch.zeros(0, self.hidden_dim))
        away_x = x_dict.get("away_player", torch.zeros(0, self.hidden_dim))

        # Handle batched graphs
        if hasattr(data["home_player"], "batch"):
            home_batch = data["home_player"].batch
            away_batch = data["away_player"].batch
        else:
            home_batch = torch.zeros(home_x.shape[0], dtype=torch.long, device=home_x.device)
            away_batch = torch.zeros(away_x.shape[0], dtype=torch.long, device=away_x.device)

        home_emb = global_mean_pool(home_x, home_batch) if home_x.shape[0] > 0 else torch.zeros(1, self.hidden_dim)
        away_emb = global_mean_pool(away_x, away_batch) if away_x.shape[0] > 0 else torch.zeros(1, self.hidden_dim)

        # Graph embedding
        graph_emb = torch.cat([home_emb, away_emb], dim=-1)

        # Output heads
        cls_logits = self.cls_head(graph_emb)
        reg_out = self.reg_head(graph_emb)

        return cls_logits, reg_out

    def get_node_embeddings(self, data: "HeteroData") -> Dict[str, torch.Tensor]:
        """
        Extract final node embeddings (used for gradient attribution).
        Same as forward but returns x_dict instead of pooled output.
        """
        x_dict = {}
        for nt in self.NODE_TYPES:
            if hasattr(data[nt], "x") and data[nt].x is not None:
                x_dict[nt] = self.input_proj[nt](data[nt].x)
            else:
                x_dict[nt] = torch.zeros(0, self.hidden_dim)

        edge_index_dict = {}
        for et in self.EDGE_TYPES:
            if hasattr(data[et], "edge_index"):
                edge_index_dict[et] = data[et].edge_index

        for layer_idx, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            for nt in self.NODE_TYPES:
                if nt in x_dict_new and x_dict_new[nt].shape[0] > 0:
                    h = self.bns[layer_idx][nt](x_dict_new[nt])
                    h = F.relu(h)
                    if x_dict[nt].shape == h.shape:
                        h = h + x_dict[nt]
                    x_dict[nt] = h

        return x_dict


def goal_diff_to_class(goal_diff: torch.Tensor) -> torch.Tensor:
    """Convert goal_diff to class label: 0=home_win, 1=draw, 2=away_win."""
    labels = torch.zeros_like(goal_diff, dtype=torch.long)
    labels[goal_diff > 0] = 0  # home win
    labels[goal_diff == 0] = 1  # draw
    labels[goal_diff < 0] = 2  # away win
    return labels
