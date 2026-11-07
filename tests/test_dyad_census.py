import networkx as nx
from scripts.build_dyad_census import dyad_counts


def test_dyads_ignore_weights_loops_and_parallel_events():
    graph = nx.MultiDiGraph()
    graph.add_edges_from([(1, 2), (1, 2), (2, 1), (2, 3), (3, 3)])
    assert dyad_counts(graph) == dict(n_recip=1, n_pairs=2, n_asym=1)


def test_edgeless_has_zero_counts():
    assert dyad_counts(nx.empty_graph(4, create_using=nx.DiGraph)) == dict(n_recip=0, n_pairs=0, n_asym=0)
