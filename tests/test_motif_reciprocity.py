from fractions import Fraction
from itertools import permutations
import networkx as nx
import numpy as np
import pytest
from src.motifs.reciprocity import (
    TRIAD_IDS, TRIAD_RECIPROCITY, HIGH_RECIPROCITY_TRIADS, ZERO_RECIPROCITY_TRIADS,
    motif_reciprocity, reciprocal_link_fraction,
)
from src.motifs.enumerator import canonical_motif_id


@pytest.mark.parametrize("mid", TRIAD_IDS)
def test_adjacency_reciprocity_matches_networkx_and_canonical_label(mid):
    edges = [(u, v) for u in range(3) for v in range(3) if mid & (1 << (8-u*3-v))]
    assert canonical_motif_id(tuple(edges), 3) == mid
    assert float(motif_reciprocity(mid)) == nx.reciprocity(nx.DiGraph(edges))
    for perm in permutations(range(3)):
        label = sum(1 << (8-perm[u]*3-perm[v]) for u, v in edges)
        assert motif_reciprocity(label) == motif_reciprocity(mid)


def test_reciprocity_groups_use_semantics_not_column_positions():
    assert HIGH_RECIPROCITY_TRIADS == {14, 74, 78, 110, 238}
    assert ZERO_RECIPROCITY_TRIADS == {6, 12, 36, 38, 98}
    assert motif_reciprocity(102) == Fraction(1, 2)
    assert motif_reciprocity(110) == Fraction(4, 5)
    assert TRIAD_RECIPROCITY[78] == 1


def test_dyad_to_arc_conversion_is_monotone_but_not_identity():
    values = reciprocal_link_fraction(np.array([10, 10, 10]), np.array([0, 5, 10]))
    np.testing.assert_allclose(values, [0, 2/3, 1])
    assert np.all(np.diff(values) > 0)


def test_undefined_or_invalid_template_rejected():
    for mid in [0, -1, 512, 256]:
        with pytest.raises(ValueError):
            motif_reciprocity(mid)
