import pandas as pd

from scripts.analyze_temporal_paired_effects import (
    CONTRASTS, CUTOFFS, EXPECTED_SEEDS, METRICS, paired_effects,
)


def test_temporal_effect_direction_pairing_and_coverage():
    rows = []
    models = {name for _, target, reference in CONTRASTS
              for name in (target, reference)}
    for cutoff in CUTOFFS:
        for model in models:
            seeds = [-1] if model == "CurrentScore-Calibrated" else EXPECTED_SEEDS
            for seed in seeds:
                rank = {
                    "CurrentScore-Calibrated": 0,
                    "TeamPool-MLP": 1,
                    "EdgeAware-Fixed": 1,
                    "HeteroSAGE-sum": 2,
                    "SPARC-GNN": 3,
                    "TeamPool-MLP+Score": 4,
                    "HeteroSAGE-sum+Score": 5,
                    "SPARC-GNN+Score": 6,
                }[model]
                rows.append({
                    "cutoff": cutoff, "model": model, "seed": seed,
                    "accuracy": .4 + .05 * rank,
                    "macro_f1": .3 + .05 * rank,
                    "draw_f1": .2 + .05 * rank,
                    "mae": 1.4 - .1 * rank,
                    "r2": .05 * rank,
                })
    effects = paired_effects(pd.DataFrame(rows), repeats=1000)
    assert len(effects) == len(CUTOFFS) * len(CONTRASTS) * len(METRICS)
    assert (effects["mean_improvement"] > 0).all()
    assert set(effects["n_target_seeds"]) == {5}
    assert set(effects["n_reference_seeds"]) == {1, 5}
