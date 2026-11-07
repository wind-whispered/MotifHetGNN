import pandas as pd

from scripts.analyze_temporal_controlled_effects import (
    CUTOFFS, EXPECTED_SEEDS, FIXED_MODELS, METRICS, _contrasts,
    controlled_effects,
)


def test_controlled_temporal_effect_direction_pairing_and_coverage():
    contrasts = _contrasts()
    models = {name for _, target, reference, _ in contrasts
              for name in (target, reference)}
    ranks = {
        "Score+Strength": 0.0,
        "CausalMatchState-HGB": 0.2,
        "TeamPool-MLP+Score": 1.0,
        "TeamPool-MLP+Score+ValBlend": 2.0,
        "TeamPool-MLP+Score+ContextBlend": 6.0,
        "HeteroSAGE-sum+Score": 1.2,
        "HeteroSAGE-sum+Score+ValBlend": 2.2,
        "HeteroSAGE-sum+Score+ContextBlend": 7.0,
        "SPARC-GNN+Score": 1.4,
        "SPARC-GNN+Score+ValBlend": 2.4,
        "SPARC-GNN+Score+ContextBlend": 8.0,
    }
    rows = []
    for cutoff in CUTOFFS:
        for model in models:
            seeds = [-1] if model in FIXED_MODELS else EXPECTED_SEEDS
            for seed in seeds:
                rank = ranks[model]
                rows.append({
                    "cutoff": cutoff,
                    "model": model,
                    "seed": seed,
                    "accuracy": .30 + .03 * rank,
                    "macro_f1": .25 + .03 * rank,
                    "draw_f1": .15 + .03 * rank,
                    "mae": 1.60 - .05 * rank,
                    "r2": .02 + .04 * rank,
                })
    indexed = pd.DataFrame(rows).set_index(
        ["cutoff", "model", "seed"]).sort_index()
    effects = controlled_effects(indexed, repeats=1000)

    assert len(effects) == len(CUTOFFS) * len(contrasts) * len(METRICS)
    assert (effects["mean_improvement"] > 0).all()
    assert set(effects["n_target_seeds"]) == {5}
    assert set(effects["n_reference_seeds"]) == {1, 5}
    assert set(effects.loc[effects.metric == "mae", "direction"]) == {
        "reference_minus_target"}
    assert set(effects.loc[effects.metric != "mae", "direction"]) == {
        "target_minus_reference"}
