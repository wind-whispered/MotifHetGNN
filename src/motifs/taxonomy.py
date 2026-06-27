"""
Motif taxonomy: classification of motifs into tactical categories,
and definition of known heterogeneous motif patterns.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Homogeneous motif IDs from original paper (Milo labeling for 3-motifs)
# ---------------------------------------------------------------------------
# Motifs with positive z-score (over-represented) in original paper:
POSITIVE_ZSCORE_MOTIFS_3 = {78, 238, 14, 74}
# Motifs with negative z-score (under-represented):
NEGATIVE_ZSCORE_MOTIFS_3 = {38, 98, 102, 12, 6, 108, 46, 36}

# Motifs significantly correlated with goal_diff (original paper Table 5):
GOAL_CORR_MOTIFS = {
    12: "positive_home",
    38: "negative_home",
    238: "negative_home",
    14: "positive_home",
    110: "negative_home",
    78: "positive_away",
}

# Motifs with bidirectional links (passback tendency indicators)
BIDIRECTIONAL_MOTIFS_3 = {14, 74, 78, 110, 238}


# ---------------------------------------------------------------------------
# Tactical semantic labels for known motif types
# ---------------------------------------------------------------------------
TACTICAL_SEMANTICS = {
    # Homogeneous 3-motifs
    "homo_3_12": {
        "label": "One-directional attack path",
        "description": "Linear forward pass chain, no passback",
        "zone_preference": "attacking",
        "goal_diff_relation": "positive_home",
    },
    "homo_3_38": {
        "label": "Blocked pass redirect",
        "description": "Original path blocked, rerouted via third player",
        "zone_preference": "middle",
        "goal_diff_relation": "negative_home",
    },
    "homo_3_78": {
        "label": "Triangle with hub",
        "description": "Central hub player distributes without direct exchange between satellites",
        "zone_preference": "middle",
        "goal_diff_relation": "positive_away",
    },
    "homo_3_238": {
        "label": "Fully connected triangle",
        "description": "All players pass to all others; unclear attack path",
        "zone_preference": "defensive",
        "goal_diff_relation": "negative_home",
    },
    # Heterogeneous motifs
    "hetero_counterpress": {
        "label": "Counterpress trigger",
        "description": "Turnover within 5 seconds triggers adversarial action",
        "zone_preference": "attacking",
        "goal_diff_relation": "positive",
        "statsbomb_fields": ["counterpress=True", "E_adv -> E_pass"],
    },
    "hetero_under_pressure_chain": {
        "label": "Under-pressure pass chain",
        "description": "Multiple successive passes made under defensive pressure",
        "zone_preference": "defensive",
        "goal_diff_relation": "negative",
        "statsbomb_fields": ["under_pressure=True x multiple edges"],
    },
    "hetero_interception_counter": {
        "label": "Interception-to-counter",
        "description": "Interception event followed by counter-attack pass sequence",
        "zone_preference": "middle",
        "goal_diff_relation": "positive",
        "statsbomb_fields": ["Interception", "play_pattern=From Counter"],
    },
    "hetero_dispossessed_reorg": {
        "label": "Dispossession defensive reorganization",
        "description": "After losing ball, defending team forms compact adversarial structure",
        "zone_preference": "middle",
        "goal_diff_relation": "negative_opponent",
        "statsbomb_fields": ["Dispossessed", "E_adv connections"],
    },
}


def label_motif_record(row: pd.Series) -> str:
    """Assign tactical label to a motif record row."""
    k = row.get("motif_order_k", 3)
    motif_id = row.get("motif_id")
    motif_type = row.get("motif_type", "cooperative")

    if motif_type == "cooperative" and k == 3 and motif_id:
        key = f"homo_3_{motif_id}"
        if key in TACTICAL_SEMANTICS:
            return TACTICAL_SEMANTICS[key]["label"]

    # Generic labels
    if motif_type == "adversarial":
        return f"Adversarial-k{k}"
    if motif_type == "mixed":
        return f"Mixed-k{k}"
    return f"Cooperative-k{k}-m{motif_id}"


def build_table6_semantic_df(
    zscore_df: pd.DataFrame,
    regression_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Table 6: tactical semantics description table.
    Merges z-score significance with regression correlation info.
    """
    records = []
    for sem_key, sem_info in TACTICAL_SEMANTICS.items():
        records.append({
            "motif_id": sem_key,
            "motif_type": "heterogeneous" if sem_key.startswith("hetero") else "homogeneous",
            "tactical_label": sem_info["label"],
            "description": sem_info["description"],
            "zone_preference": sem_info.get("zone_preference", ""),
            "goal_diff_relation": sem_info.get("goal_diff_relation", ""),
            "statsbomb_fields": "; ".join(sem_info.get("statsbomb_fields", [])),
        })
    return pd.DataFrame(records)


def assign_motif_order_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'is_significant' flag and 'tactical_label' to motif DataFrame."""
    df = df.copy()
    if "z" in df.columns:
        df["is_significant"] = df["z"].abs() > 1.96
    if "motif_id" in df.columns:
        df["tactical_label"] = df.apply(label_motif_record, axis=1)
    return df
