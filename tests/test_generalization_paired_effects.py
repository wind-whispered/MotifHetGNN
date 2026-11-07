import pandas as pd

from scripts.analyze_generalization_paired_effects import (
    BASELINES, EXPECTED_SEEDS, METRICS, PROTOCOLS, TARGET, paired_effects,
)


def test_generalization_effect_direction_and_coverage():
    rows = []
    for protocol in PROTOCOLS:
        for model in (TARGET, *BASELINES):
            for seed in EXPECTED_SEEDS:
                target = model == TARGET
                rows.append({
                    "protocol": protocol, "model": model, "seed": seed,
                    "accuracy": .6 if target else .5,
                    "macro_f1": .5 if target else .4,
                    "draw_f1": .3 if target else .2,
                    "mae": 1.0 if target else 1.2,
                    "r2": .2 if target else .1,
                })
    effects = paired_effects(pd.DataFrame(rows), repeats=1000)
    assert len(effects) == len(PROTOCOLS) * len(BASELINES) * len(METRICS)
    assert set(effects["n_seeds"]) == {5}
    assert (effects["mean_improvement"] > 0).all()
    assert (effects["target_better_seeds"] == 5).all()
