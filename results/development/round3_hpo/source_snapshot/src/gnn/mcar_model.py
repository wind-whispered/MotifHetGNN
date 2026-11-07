"""Conflict-aware shared/private relation learning for football graphs."""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, RGCNConv, global_mean_pool
from src.networks.edge_features import PASS_EDGE_DIM, ADV_EDGE_DIM, TURN_EDGE_DIM
from .score_context import cutoff_score_features


SPARC_REVISION = "availability_masked_messages_v3"


def masked_relation_softmax(
    logits: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Normalise fusion mass over observed relations only."""
    has_relation = availability.any(dim=1, keepdim=True)
    masked = logits.masked_fill(~availability, -1e9)
    weights = F.softmax(masked, dim=-1)
    return torch.where(has_relation, weights, torch.zeros_like(weights))


def masked_pairwise_conflict(
    messages: torch.Tensor,
    availability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return pairwise cosine disagreement only where both relations exist.

    ``messages`` has shape ``[N, 3, d]`` and ``availability`` has shape
    ``[N, 3]``. PyTorch assigns zero cosine similarity to a zero vector; without
    this mask an absent relation would therefore be mislabelled as conflict 1.
    """
    pairs = ((0, 1), (0, 2), (1, 2))
    pair_available = torch.stack(
        [availability[:, left] & availability[:, right]
         for left, right in pairs], dim=1)
    disagreement = 1.0 - torch.stack([
        F.cosine_similarity(messages[:, left], messages[:, right], dim=-1)
        for left, right in pairs
    ], dim=1)
    return disagreement * pair_available.to(disagreement.dtype), pair_available


def observed_relation_mean(
    messages: torch.Tensor,
    availability: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask absent messages and average only over observed relations.

    Relation convolutions contain trainable biases, so a target node with no
    incoming edge of a given relation can otherwise receive a non-zero vector.
    The explicit mask closes that bypass before shared/private decomposition.
    """
    available = availability.unsqueeze(-1).to(messages.dtype)
    masked = messages * available
    count = available.sum(dim=1).clamp_min(1.0)
    mean = masked.sum(dim=1) / count
    has_relation = availability.any(dim=1, keepdim=True)
    return masked, mean, has_relation


class MCARFootballGNN(nn.Module):
    """Edge-aware HGNN with motif- and conflict-conditioned relation gates.

    Relation messages are computed independently.  Instead of HeteroConv's
    fixed sum, a gate conditioned on typed-triad participation and pairwise
    disagreement assigns a node-specific weight to pass, adversarial, and
    turnover information.  An optional shared/private decomposition separates
    relation-invariant and relation-specific evidence.
    """

    NODE_TYPES = ("home_player", "away_player")
    EDGE_TYPES = (
        ("home_player", "pass", "home_player"),
        ("away_player", "pass", "away_player"),
        ("home_player", "adversarial", "away_player"),
        ("away_player", "adversarial", "home_player"),
        ("home_player", "turnover", "away_player"),
        ("away_player", "turnover", "home_player"),
    )
    EDGE_DIMS = {
        "pass": PASS_EDGE_DIM,
        "adversarial": ADV_EDGE_DIM,
        "turnover": TURN_EDGE_DIM,
    }

    def __init__(self, node_feature_dim=39, motif_dim=7, hidden_dim=64,
                 num_layers=3, dropout=0.3, n_cls=3, use_edge_attr=True,
                 use_motif_context=True, adaptive_gate=True,
                 use_info_decomposition=False, use_conflict_gate=True,
                 use_availability_gate=True, use_current_score=False,
                 use_assist_flags=False,
                 mask_unavailable_relations=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.use_motif_context = use_motif_context
        self.adaptive_gate = adaptive_gate
        self.use_info_decomposition = use_info_decomposition
        self.use_conflict_gate = use_conflict_gate
        self.use_availability_gate = use_availability_gate
        self.use_current_score = use_current_score
        self.use_assist_flags = use_assist_flags
        self.mask_unavailable_relations = mask_unavailable_relations
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(node_feature_dim, hidden_dim) for nt in self.NODE_TYPES
        })
        self.motif_proj = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(motif_dim, hidden_dim), nn.ReLU())
            for nt in self.NODE_TYPES
        })

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.shared_projs = nn.ModuleList()
        self.private_projs = nn.ModuleList()
        self.conflict_projs = nn.ModuleList()
        self.availability_projs = nn.ModuleList()
        for _ in range(num_layers):
            relation_convs = nn.ModuleDict()
            for src, rel, dst in self.EDGE_TYPES:
                key = "__".join((src, rel, dst))
                relation_convs[key] = GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, heads=1,
                    concat=False, edge_dim=self.EDGE_DIMS[rel],
                    add_self_loops=False, dropout=dropout,
                )
            self.layers.append(relation_convs)
            self.gates.append(nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                ) for nt in self.NODE_TYPES
            }))
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in self.NODE_TYPES
            }))
            if use_info_decomposition:
                self.shared_projs.append(nn.ModuleDict({
                    nt: nn.Linear(hidden_dim, hidden_dim, bias=False)
                    for nt in self.NODE_TYPES
                }))
                self.private_projs.append(nn.ModuleDict({
                    nt: nn.ModuleList([nn.Linear(
                        hidden_dim, hidden_dim, bias=False)
                                       for _ in range(3)])
                    for nt in self.NODE_TYPES
                }))
                self.conflict_projs.append(nn.ModuleDict({
                    nt: nn.Linear(3, hidden_dim, bias=False)
                    for nt in self.NODE_TYPES
                }) if use_conflict_gate else nn.ModuleDict())
                self.availability_projs.append(nn.ModuleDict({
                    nt: nn.Linear(3, hidden_dim, bias=False)
                    for nt in self.NODE_TYPES
                }) if use_availability_gate else nn.ModuleDict())
            else:
                self.shared_projs.append(nn.ModuleDict())
                self.private_projs.append(nn.ModuleDict())
                self.conflict_projs.append(nn.ModuleDict())
                self.availability_projs.append(nn.ModuleDict())

        # A score-aware temporal variant receives only information observable
        # at the event cutoff.  Explicit lead/level indicators make the
        # CurrentScore comparison a strong nested baseline rather than asking
        # the MLP to discover equality from a single scalar.
        graph_dim = hidden_dim * 2 + (4 if use_current_score else 0)
        self.cls_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.last_gate_weights: Dict[str, torch.Tensor] = {}
        self.last_decomposition: Dict[str, Dict[str, torch.Tensor]] = {}
        self._auxiliary_loss = torch.tensor(0.0)

    def _prepare_edge_attributes(self, relation, edge_attr):
        edge_attr = edge_attr.float()
        if not self.use_edge_attr:
            return torch.zeros_like(edge_attr)
        if relation == "pass" and not self.use_assist_flags:
            edge_attr = edge_attr.clone()
            edge_attr[:, 10:12] = 0.0
        return edge_attr

    def _relation_messages(self, layer, x_dict, data):
        by_target = {nt: {r: [] for r in ("pass", "adversarial", "turnover")}
                     for nt in self.NODE_TYPES}
        availability = {
            nt: {rel: torch.zeros(x_dict[nt].size(0), dtype=torch.bool,
                                  device=x_dict[nt].device)
                 for rel in ("pass", "adversarial", "turnover")}
            for nt in self.NODE_TYPES
        }
        for et in self.EDGE_TYPES:
            src, rel, dst = et
            store = data[et]
            if not hasattr(store, "edge_index") or store.edge_index.numel() == 0:
                continue
            key = "__".join(et)
            edge_attr = self._prepare_edge_attributes(rel, store.edge_attr)
            msg = layer[key]((x_dict[src], x_dict[dst]), store.edge_index,
                             edge_attr=edge_attr)
            by_target[dst][rel].append(msg)
            availability[dst][rel][store.edge_index[1].unique()] = True
        out = {nt: {} for nt in self.NODE_TYPES}
        for nt in self.NODE_TYPES:
            for rel in ("pass", "adversarial", "turnover"):
                messages = by_target[nt][rel]
                out[nt][rel] = (torch.stack(messages).sum(0) if messages
                                else torch.zeros_like(x_dict[nt]))
        return out, availability

    def encode(self, data):
        x_dict = {nt: self.input_proj[nt](data[nt].x.float())
                  for nt in self.NODE_TYPES}
        motif_dict = {}
        for nt in self.NODE_TYPES:
            motif_x = getattr(data[nt], "motif_x", None)
            if motif_x is None and self.use_motif_context:
                raise ValueError("MCARFootballGNN requires node-level motif_x")
            if motif_x is None or not self.use_motif_context:
                motif_x = torch.zeros((data[nt].x.size(0), 7),
                                      device=data[nt].x.device)
            motif_dict[nt] = self.motif_proj[nt](motif_x.float())

        auxiliary_terms = []
        for (layer, gate, norm, shared_proj, private_proj, conflict_proj,
             availability_proj) in zip(
                self.layers, self.gates, self.norms, self.shared_projs,
                self.private_projs, self.conflict_projs,
                self.availability_projs):
            messages, availability_dict = self._relation_messages(
                layer, x_dict, data)
            new_x = {}
            for nt in self.NODE_TYPES:
                stacked = torch.stack([
                    messages[nt]["pass"], messages[nt]["adversarial"],
                    messages[nt]["turnover"]
                ], dim=1)
                availability = torch.stack([
                    availability_dict[nt]["pass"],
                    availability_dict[nt]["adversarial"],
                    availability_dict[nt]["turnover"],
                ], dim=1)
                relation_summary = stacked.mean(dim=1)
                if self.use_info_decomposition:
                    stacked, relation_mean, has_relation = observed_relation_mean(
                        stacked, availability)
                    relation_summary = relation_mean
                    conflict, pair_available = masked_pairwise_conflict(
                        stacked, availability)
                    if self.use_availability_gate:
                        relation_summary = (
                            relation_summary
                            + availability_proj[nt](availability.float()))
                    if self.use_conflict_gate:
                        relation_summary = (relation_summary
                                            + conflict_proj[nt](conflict))
                elif self.mask_unavailable_relations:
                    stacked, relation_summary, _ = observed_relation_mean(
                        stacked, availability)
                gate_input = torch.cat([
                    x_dict[nt], motif_dict[nt], relation_summary
                ], dim=-1)
                if self.adaptive_gate:
                    logits = gate[nt](gate_input)
                    if (self.use_info_decomposition
                            or self.mask_unavailable_relations):
                        # A relation without an incoming message cannot receive
                        # fusion mass. Nodes with no relation messages retain a
                        # zero update rather than a fabricated uniform mixture.
                        weights = masked_relation_softmax(
                            logits, availability)
                    else:
                        weights = F.softmax(logits, dim=-1)
                else:
                    if self.mask_unavailable_relations:
                        denominator = availability.sum(
                            dim=1, keepdim=True).clamp_min(1)
                        weights = availability.to(stacked.dtype) / denominator
                    else:
                        weights = torch.full(
                            (stacked.size(0), 3), 1.0 / 3.0,
                            device=stacked.device)
                if self.use_info_decomposition:
                    shared = shared_proj[nt](relation_mean)
                    shared = shared * has_relation.to(shared.dtype)
                    private = torch.stack([
                        private_proj[nt][i](stacked[:, i] - relation_mean)
                        for i in range(3)
                    ], dim=1)
                    private = private * availability.unsqueeze(-1).to(
                        private.dtype)
                    fused = shared + (private * weights.unsqueeze(-1)).sum(dim=1)

                    # Shared/private orthogonality discourages the private
                    # channels from simply copying the common component.
                    shared_unit = F.normalize(shared, dim=-1)
                    private_unit = F.normalize(private, dim=-1)
                    orth = (private_unit * shared_unit.unsqueeze(1)).sum(-1)
                    observed = availability.to(orth.dtype)
                    auxiliary_terms.append(
                        (orth.square() * observed).sum()
                        / observed.sum().clamp_min(1.0))

                    # Pairwise disagreement is retained as an interpretable
                    # conflict score; high values indicate relation-specific
                    # evidence pointing in different directions.
                    private_conflict, _ = masked_pairwise_conflict(
                        private, availability)
                    self.last_decomposition[nt] = {
                        "shared_norm": shared.norm(dim=-1).detach(),
                        "private_norm": private.norm(dim=-1).detach(),
                        "availability": availability.detach(),
                        "pair_available": pair_available.detach(),
                        "conflict": conflict.detach(),
                        "private_conflict": private_conflict.detach(),
                    }
                else:
                    fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
                new_x[nt] = self._residual_update(x_dict[nt], fused, norm[nt])
                self.last_gate_weights[nt] = weights.detach()
            x_dict = new_x
        device = next(self.parameters()).device
        self._auxiliary_loss = (torch.stack(auxiliary_terms).mean()
                                if auxiliary_terms else torch.zeros((), device=device))
        return x_dict

    def _residual_update(self, previous, fused, norm):
        """Frozen post-residual update; development variants override this hook."""
        return norm(previous + F.dropout(
            F.relu(fused), p=self.dropout, training=self.training))

    def auxiliary_loss(self) -> torch.Tensor:
        """Regularisation used only by the shared/private decomposition."""
        return self._auxiliary_loss

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dict = self.encode(data)
        pooled = []
        for nt in self.NODE_TYPES:
            batch = getattr(data[nt], "batch", None)
            if batch is None:
                batch = torch.zeros(x_dict[nt].size(0), dtype=torch.long,
                                    device=x_dict[nt].device)
            pooled.append(global_mean_pool(x_dict[nt], batch))
        graph_emb = torch.cat(pooled, dim=-1)
        if self.use_current_score:
            score_features = cutoff_score_features(
                data, graph_emb.device, "SPARC-GNN")
            graph_emb = torch.cat([graph_emb, score_features], dim=-1)
        return self.cls_head(graph_emb), self.reg_head(graph_emb)


class _DualHead(nn.Module):
    def _init_heads(self, hidden_dim, dropout, n_cls=3):
        graph_dim = hidden_dim * 2
        self.cls_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls))
        self.reg_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def _pool(self, x_dict, data):
        pooled = []
        for nt in MCARFootballGNN.NODE_TYPES:
            batch = getattr(data[nt], "batch", None)
            if batch is None:
                batch = torch.zeros(x_dict[nt].size(0), dtype=torch.long,
                                    device=x_dict[nt].device)
            pooled.append(global_mean_pool(x_dict[nt], batch))
        emb = torch.cat(pooled, dim=-1)
        return self.cls_head(emb), self.reg_head(emb)


class HGTFootballGNN(_DualHead):
    """Strong heterogeneous Transformer baseline."""
    def __init__(self, node_feature_dim=39, hidden_dim=64, num_layers=3,
                 dropout=0.3, n_cls=3):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_feature_dim, hidden_dim)
                                   for nt in MCARFootballGNN.NODE_TYPES})
        metadata = (list(MCARFootballGNN.NODE_TYPES),
                    list(MCARFootballGNN.EDGE_TYPES))
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim)
                           for nt in MCARFootballGNN.NODE_TYPES})
            for _ in range(num_layers)
        ])
        self.dropout = dropout
        self._init_heads(hidden_dim, dropout, n_cls)

    def forward(self, data):
        x = {nt: self.proj[nt](data[nt].x.float())
             for nt in MCARFootballGNN.NODE_TYPES}
        for conv, norms in zip(self.convs, self.norms):
            update = conv(x, data.edge_index_dict)
            x = {nt: norms[nt](x[nt] + F.dropout(F.relu(update[nt]),
                    p=self.dropout, training=self.training))
                 for nt in MCARFootballGNN.NODE_TYPES}
        return self._pool(x, data)


class RGCNFootballGNN(_DualHead):
    """Relational GCN baseline using the six directed relation types."""
    def __init__(self, node_feature_dim=39, hidden_dim=64, num_layers=3,
                 dropout=0.3, n_cls=3):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_feature_dim, hidden_dim)
                                   for nt in MCARFootballGNN.NODE_TYPES})
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim,
                     num_relations=len(MCARFootballGNN.EDGE_TYPES), num_bases=4)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                    for _ in range(num_layers)])
        self.dropout = dropout
        self._init_heads(hidden_dim, dropout, n_cls)

    def forward(self, data):
        home = self.proj["home_player"](data["home_player"].x.float())
        away = self.proj["away_player"](data["away_player"].x.float())
        n_home = home.size(0)
        x = torch.cat([home, away], dim=0)
        edges, types = [], []
        for rid, (src, rel, dst) in enumerate(MCARFootballGNN.EDGE_TYPES):
            if (src, rel, dst) not in data.edge_types:
                continue
            ei = data[(src, rel, dst)].edge_index
            if ei is None:
                continue
            if ei.numel() == 0:
                continue
            ei = ei.clone()
            if src == "away_player":
                ei[0] += n_home
            if dst == "away_player":
                ei[1] += n_home
            edges.append(ei)
            types.append(torch.full((ei.size(1),), rid, dtype=torch.long,
                                    device=ei.device))
        if edges:
            edge_index = torch.cat(edges, dim=1)
            edge_type = torch.cat(types)
        else:
            device = x.device
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + F.dropout(F.relu(conv(x, edge_index, edge_type)),
                    p=self.dropout, training=self.training))
        x_dict = {"home_player": x[:n_home], "away_player": x[n_home:]}
        return self._pool(x_dict, data)
