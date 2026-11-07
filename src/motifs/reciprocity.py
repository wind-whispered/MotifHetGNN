"""Reciprocity derived from adjacency bits, never a hand-labelled motif table."""
from fractions import Fraction

TRIAD_IDS = (6, 12, 14, 36, 38, 46, 74, 78, 98, 102, 108, 110, 238)
# Preserve the existing feature-column order. This is NOT reciprocity order.
TRIAD_DISPLAY_ORDER = (98, 12, 38, 102, 6, 36, 108, 46, 110, 74, 14, 238, 78)


def motif_reciprocity(motif_id: int, order: int = 3) -> Fraction:
    """Fraction of directed links whose reverse exists in a loop-free template."""
    if order < 1 or not 0 <= motif_id < 1 << (order * order):
        raise ValueError("Invalid adjacency integer or order")
    edges = {(u, v) for u in range(order) for v in range(order)
             if motif_id & (1 << (order * order - 1 - (u * order + v)))}
    if any(u == v for u, v in edges):
        raise ValueError("Motif templates must be loop-free")
    if not edges:
        raise ValueError("Link reciprocity is undefined for an edgeless graph")
    return Fraction(sum((v, u) in edges for u, v in edges), len(edges))


TRIAD_RECIPROCITY = {mid: float(motif_reciprocity(mid)) for mid in TRIAD_IDS}
HIGH_RECIPROCITY_TRIADS = frozenset(mid for mid in TRIAD_IDS
                                  if motif_reciprocity(mid) >= Fraction(2, 3))
ZERO_RECIPROCITY_TRIADS = frozenset(mid for mid in TRIAD_IDS if motif_reciprocity(mid) == 0)


def reciprocal_link_fraction(connected_dyads, mutual_dyads):
    """2 mutual / (asymmetric + 2 mutual); scalars or nonempty pandas arrays.

    connected_dyads includes both asymmetric and mutual pairs. The old ratio
    mutual/connected measures a different quantity, not directed-link reciprocity.
    Inputs here are nonempty dyad groups; empty networks have no defined ratio.
    """
    return 2 * mutual_dyads / (connected_dyads + mutual_dyads)
