"""Experimental edge-value branch; not the frozen manuscript architecture.

PyG GATv2 uses edge attributes in its attention coefficients, whereas its
message values are transformed source-node vectors. This wrapper supplements
those values with an attention-weighted edge projection. Zero initialisation
preserves the wrapped encoder output at construction. Assist redaction and
observed-relation masks remain the responsibility of the parent SPARC model.
"""
import torch
from torch import nn


class EdgeValueEncoder(nn.Module):
    def __init__(self, encoder, edge_dim, hidden_dim):
        super().__init__()
        if encoder.add_self_loops or encoder.heads != 1:
            raise ValueError("Candidate requires the manuscript's loop-free, single-head encoder")
        self.encoder = encoder
        self.edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.edge_value.weight)

    def forward(self, x, edge_index, edge_attr):
        out, (used_edges, alpha) = self.encoder(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        # The returned coefficients include the same training-time dropout
        # mask as the original node-value stream. No second attention is fit.
        values = self.edge_value(edge_attr) * alpha
        update = torch.zeros_like(out)
        update.index_add_(0, used_edges[1], values)
        return out + update


def add_edge_values(model):
    """Modify only relation encoders in an isolated experimental instance."""
    for layer in model.layers:
        for src, relation, dst in model.EDGE_TYPES:
            key = "__".join((src, relation, dst))
            layer[key] = EdgeValueEncoder(layer[key], model.EDGE_DIMS[relation],
                                          model.hidden_dim)
    return model
