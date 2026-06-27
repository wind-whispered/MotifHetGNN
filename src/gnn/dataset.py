"""
Task 9 - Part A: PyTorch Geometric Dataset for heterogeneous football graphs.
"""
from typing import Callable, List, Optional
import logging
from pathlib import Path

import torch
from torch_geometric.data import Dataset, HeteroData

logger = logging.getLogger(__name__)


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
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.graph_dir = Path(root) / "data" / "networks" / "heterogeneous"
        self._match_ids = match_ids
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


def create_data_loaders(
    root: str,
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    batch_size: int = 32,
):
    """Create PyG DataLoaders for train/val/test splits."""
    from torch_geometric.loader import DataLoader

    train_dataset = FootballHeteroDataset(root=root, match_ids=train_ids)
    val_dataset = FootballHeteroDataset(root=root, match_ids=val_ids)
    test_dataset = FootballHeteroDataset(root=root, match_ids=test_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
