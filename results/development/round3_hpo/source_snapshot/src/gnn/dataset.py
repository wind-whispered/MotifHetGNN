"""
Task 9 - Part A: PyTorch Geometric Dataset for heterogeneous football graphs.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Dataset, HeteroData

logger = logging.getLogger(__name__)

CHRONOLOGICAL_SPLIT_REVISION = "complete_date_boundaries_v2"


def _split_at_complete_dates(meta, ratios):
    """Split a date-sorted frame without placing one date in two partitions."""
    if len(ratios) < 2 or abs(float(sum(ratios)) - 1.0) > 1e-8:
        raise ValueError("date-partition ratios must sum to one")
    dates = meta["match_date"].dt.normalize()
    if bool(dates.isna().any()):
        raise ValueError("strict chronological splitting requires every match date")
    sizes = dates.groupby(dates, sort=True).size()
    if len(sizes) < len(ratios):
        raise ValueError("not enough distinct dates for strict partitions")
    cumulative = sizes.cumsum().to_numpy()
    targets = np.cumsum(ratios)[:-1] * len(meta)
    boundary_positions = []
    previous = -1
    n_groups = len(sizes)
    for boundary_number, target in enumerate(targets):
        remaining_partitions = len(ratios) - boundary_number - 1
        first = previous + 1
        last = n_groups - remaining_partitions - 1
        candidates = np.arange(first, last + 1)
        position = int(candidates[np.argmin(
            np.abs(cumulative[candidates] - target))])
        boundary_positions.append(position)
        previous = position

    boundaries = [sizes.index[position] for position in boundary_positions]
    partitions = []
    lower = None
    for boundary in boundaries:
        mask = dates <= boundary
        if lower is not None:
            mask &= dates > lower
        partitions.append(meta.loc[mask, "match_id"].astype(int).tolist())
        lower = boundary
    partitions.append(meta.loc[dates > lower, "match_id"].astype(int).tolist())
    return tuple(partitions)


def typed_triad_context(data: HeteroData) -> Dict[str, torch.Tensor]:
    """Return per-node participation in connected typed triads.

    Seven channels encode which non-empty subset of {pass, adversarial,
    turnover} occurs inside a connected induced triad.  Counts are transformed
    with log1p and normalised per graph.  This is deliberately independent of
    the outcome label and can therefore be cached before dataset splitting.
    """
    node_types = ("home_player", "away_player")
    offsets = {}
    n_total = 0
    for nt in node_types:
        offsets[nt] = n_total
        n_total += int(data[nt].num_nodes)

    relation_bit = {"pass": 1, "adversarial": 2, "turnover": 4}
    pair_mask = torch.zeros((n_total, n_total), dtype=torch.uint8)
    for src_t, rel, dst_t in data.edge_types:
        if rel not in relation_bit:
            continue
        edge_index = data[(src_t, rel, dst_t)].edge_index.cpu()
        if edge_index.numel() == 0:
            continue
        src = edge_index[0] + offsets[src_t]
        dst = edge_index[1] + offsets[dst_t]
        bit = relation_bit[rel]
        pair_mask[src, dst] |= bit
        pair_mask[dst, src] |= bit

    counts = torch.zeros((n_total, 7), dtype=torch.float32)
    for i in range(n_total - 2):
        for j in range(i + 1, n_total - 1):
            ij = int(pair_mask[i, j])
            for k in range(j + 1, n_total):
                ik = int(pair_mask[i, k])
                jk = int(pair_mask[j, k])
                if (ij > 0) + (ik > 0) + (jk > 0) < 2:
                    continue
                mask = ij | ik | jk
                channel = mask - 1
                counts[i, channel] += 1
                counts[j, channel] += 1
                counts[k, channel] += 1

    counts = torch.log1p(counts)
    scale = counts.amax(dim=0, keepdim=True).clamp_min(1.0)
    counts = counts / scale
    return {
        nt: counts[offsets[nt]: offsets[nt] + int(data[nt].num_nodes)]
        for nt in node_types
    }


class FootballHeteroDataset(Dataset):
    """
    PyTorch Geometric Dataset that loads pre-built heterogeneous graphs
    from data/networks/heterogeneous/{match_id}.pt

    Each graph has:
        - data['home_player'].x, data['away_player'].x  (node features)
        - data[edge_type].edge_index, edge_attr          (edges)
        - data.y  (goal_diff, shape [1])
        - data.match_id
    """

    def __init__(
        self,
        root: str,
        match_ids: Optional[List[int]] = None,
        use_motif_context: bool = False,
        motif_cache_dir: Optional[str] = None,
        required_graph_revision: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.graph_dir = Path(root) / "data" / "networks" / "heterogeneous"
        self._match_ids = match_ids
        self.use_motif_context = use_motif_context
        self.motif_cache_dir = Path(motif_cache_dir) if motif_cache_dir else Path(root) / "data" / "gnn" / "motif_context"
        self.required_graph_revision = required_graph_revision
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        if self._match_ids is not None:
            return [f"{mid}.pt" for mid in self._match_ids]
        return [p.name for p in self.graph_dir.glob("*.pt")]

    @property
    def processed_file_names(self) -> List[str]:
        # Already processed: just return the same files
        return self.raw_file_names

    def download(self):
        pass  # graphs already built by Task 3

    def process(self):
        pass  # graphs already built by Task 3

    def len(self) -> int:
        return len(self.raw_file_names)

    def get(self, idx: int) -> HeteroData:
        fname = self.raw_file_names[idx]
        path = self.graph_dir / fname
        data = torch.load(path, weights_only=False)
        if (self.required_graph_revision is not None
                and getattr(data, "graph_revision", None)
                != self.required_graph_revision):
            raise RuntimeError(
                f"graph revision mismatch for {path}: "
                f"{getattr(data, 'graph_revision', None)!r} != "
                f"{self.required_graph_revision!r}")
        if self.use_motif_context:
            cache_path = self.motif_cache_dir / fname
            if cache_path.exists():
                context = torch.load(cache_path, weights_only=True)
            else:
                context = typed_triad_context(data)
            for nt, value in context.items():
                data[nt].motif_x = value
        return data

    @property
    def match_ids(self) -> List[int]:
        if self._match_ids is not None:
            return self._match_ids
        return [int(Path(f).stem) for f in self.raw_file_names]


def split_dataset(
    dataset: FootballHeteroDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Split dataset into train/val/test by match_id (not by time).
    Returns (train_ids, val_ids, test_ids).
    """
    import random
    random.seed(seed)

    all_ids = list(dataset.match_ids)
    random.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def split_dataset_by_metadata(
    dataset: FootballHeteroDataset,
    metadata_path: str,
    strategy: str = "chronological",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    holdout_competition: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """Leakage-aware match splits used by the Information Sciences revision.

    Strategies:
      chronological: oldest matches train, middle validation, newest test,
        with every calendar date assigned wholly to one partition;
      competition: one named competition is test, remaining matches are split
        chronologically into train/validation;
      team_grouped: connected components of the team co-occurrence graph are
        assigned as indivisible groups.
    """
    import random
    import pandas as pd

    ratios = train_ratio + val_ratio + test_ratio
    if abs(ratios - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to one")
    ids = set(dataset.match_ids)
    meta = pd.read_parquet(metadata_path)
    meta = meta[meta["match_id"].isin(ids)].copy()
    meta["match_date"] = pd.to_datetime(meta["match_date"], errors="coerce")
    meta = meta.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    if strategy == "chronological":
        return _split_at_complete_dates(
            meta, (train_ratio, val_ratio, test_ratio))

    if strategy == "competition":
        if not holdout_competition:
            raise ValueError("holdout_competition is required")
        test = meta.loc[meta["competition_name"] == holdout_competition, "match_id"].astype(int).tolist()
        rest = meta.loc[meta["competition_name"] != holdout_competition, "match_id"].astype(int).tolist()
        if not test:
            raise ValueError(f"competition not found: {holdout_competition}")
        train_share = train_ratio / (train_ratio + val_ratio)
        train, validation = _split_at_complete_dates(
            meta.loc[meta["competition_name"] != holdout_competition],
            (train_share, 1.0 - train_share))
        return train, validation, test

    if strategy == "team_grouped":
        import networkx as nx
        graph = nx.Graph()
        graph.add_edges_from(zip(meta["home_team_id"], meta["away_team_id"]))
        components = list(nx.connected_components(graph))
        component_matches = []
        for teams in components:
            mask = meta["home_team_id"].isin(teams) & meta["away_team_id"].isin(teams)
            component_matches.append(meta.loc[mask, "match_id"].astype(int).tolist())
        # Largest components first; assign each to the partition with the
        # greatest remaining capacity. Entire connected components stay
        # together, guaranteeing strict team identity separation.
        rng = random.Random(seed)
        rng.shuffle(component_matches)
        component_matches.sort(key=len, reverse=True)
        targets = np.array([train_ratio, val_ratio, test_ratio]) * len(meta)
        assigned = [[], [], []]
        sizes = np.zeros(3, dtype=float)
        for matches in component_matches:
            remaining = targets - sizes
            slot = int(np.argmax(remaining / np.maximum(targets, 1)))
            assigned[slot].extend(matches)
            sizes[slot] += len(matches)
        if min(map(len, assigned)) == 0:
            raise ValueError("team_grouped split produced an empty partition")
        return tuple(assigned)

    raise ValueError(f"unknown split strategy: {strategy}")


def create_data_loaders(
    root: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 32,
    use_motif_context: bool = False,
):
    """Create PyG DataLoaders for train/val/test splits."""
    from torch_geometric.loader import DataLoader

    kwargs = {"use_motif_context": use_motif_context}
    train_dataset = FootballHeteroDataset(root=root, match_ids=train_ids, **kwargs)
    val_dataset = FootballHeteroDataset(root=root, match_ids=val_ids, **kwargs)
    test_dataset = FootballHeteroDataset(root=root, match_ids=test_ids, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
