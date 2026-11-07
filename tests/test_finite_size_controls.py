from collections import Counter
import pickle
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import networkx as nx

from scripts.run_finite_size_motif_analysis import (
    connected_counts, load_graph_edges, weak_projection,
)
from scripts.run_task11_highorder import load_A
from src.motifs.randomization import (
    directed_simple_edge_swap, undirected_simple_edge_swap,
)


def _undirected_degrees(edges):
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return degree


class FiniteSizeControlTests(unittest.TestCase):
    def test_analysis_loaders_use_canonical_graph_not_gtrie_header(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([1001, 1002, 1003])
        graph.add_edge(1001, 1002)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.gpickle"
            with path.open("wb") as handle:
                pickle.dump(graph, handle)

            n_nodes, edges = load_graph_edges(path)
            adjacency = load_A(path)

        self.assertEqual(n_nodes, 3)
        self.assertEqual(edges, {(0, 1)})
        self.assertEqual(adjacency.shape, (3, 3))
        self.assertEqual(int(adjacency.sum()), 1)

    def test_undirected_swap_preserves_edge_count_and_each_degree(self):
        edges = {
            (0, 1), (0, 2), (0, 3), (1, 2),
            (1, 4), (2, 5), (3, 4), (4, 5),
        }
        rewired, successful, attempts = undirected_simple_edge_swap(
            edges, target_swaps=20, rng=random.Random(42))
        self.assertEqual(len(rewired), len(edges))
        self.assertEqual(_undirected_degrees(rewired),
                         _undirected_degrees(edges))
        self.assertTrue(all(u < v for u, v in rewired))
        self.assertGreater(successful, 0)
        self.assertGreaterEqual(attempts, successful)

    def test_directed_swap_preserves_edge_count_and_in_out_degrees(self):
        edges = {
            (0, 1), (0, 2), (1, 2), (1, 3),
            (2, 3), (2, 4), (3, 4), (3, 5),
            (4, 0), (4, 5), (5, 0), (5, 1),
        }
        rewired, successful, attempts = directed_simple_edge_swap(
            edges, target_swaps=30, rng=random.Random(42))
        self.assertEqual(len(rewired), len(edges))
        self.assertEqual(Counter(u for u, _ in rewired),
                         Counter(u for u, _ in edges))
        self.assertEqual(Counter(v for _, v in rewired),
                         Counter(v for _, v in edges))
        self.assertTrue(all(u != v for u, v in rewired))
        self.assertGreater(successful, 0)
        self.assertGreaterEqual(attempts, successful)

    def test_connected_counts_uses_the_weak_projection(self):
        forward_path = {(0, 1), (1, 2)}
        reverse_path = {(1, 0), (2, 1)}
        self.assertEqual(connected_counts(3, forward_path, k_max=3),
                         {2: 2, 3: 1})
        self.assertEqual(connected_counts(3, reverse_path, k_max=3),
                         {2: 2, 3: 1})

    def test_weak_projection_collapses_reciprocal_dyads(self):
        directed = {(0, 1), (1, 0), (1, 2)}
        self.assertEqual(weak_projection(directed), {(0, 1), (1, 2)})


if __name__ == "__main__":
    unittest.main()
