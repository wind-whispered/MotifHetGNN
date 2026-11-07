"""Development-only root-preserving update; not the frozen manuscript model.

Unlike normalising the complete state, the update-only normalisation leaves
the root path intact. All relation availability and assist-redaction contracts
are inherited unchanged. The final availability mask prevents BatchNorm's
offset from inventing a message for an isolated node.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .mcar_model import MCARFootballGNN


class RootPreservingSPARC(MCARFootballGNN):
    def __init__(self, *args, update_norm="batch", **kwargs):
        super().__init__(*args, **kwargs)
        if update_norm not in {"batch", "layer"}:
            raise ValueError(update_norm)
        for layer in self.norms:
            for nt in self.NODE_TYPES:
                layer[nt] = (nn.BatchNorm1d(self.hidden_dim)
                             if update_norm == "batch" else nn.LayerNorm(self.hidden_dim))

    def _relation_messages(self, layer, x_dict, data):
        messages, availability = super()._relation_messages(layer, x_dict, data)
        self._update_support = [torch.stack(list(availability[nt].values()), 1).any(1)
                                for nt in self.NODE_TYPES]
        self._update_cursor = 0
        return messages, availability

    def _residual_update(self, previous, fused, norm):
        support = self._update_support[self._update_cursor]
        self._update_cursor += 1
        update = F.dropout(F.relu(norm(fused)), p=self.dropout, training=self.training)
        return previous + update * support[:, None].to(update.dtype)
