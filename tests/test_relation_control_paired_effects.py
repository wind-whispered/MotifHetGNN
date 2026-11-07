import pandas as pd

from scripts.analyze_relation_control_paired_effects import (
    CONTRASTS, EXPECTED_SEEDS, METRICS, paired_effects,
)


def test_relation_control_effect_direction_and_coverage():
    rows = []
    conditions = [("all", "all")] + [
        (condition, relation) for _, condition, relation in CONTRASTS]
    for condition, relation in conditions:
        for seed in EXPECTED_SEEDS:
            is_reference = condition == "all"
            rows.append({
                "condition": condition,
                "relation": relation,
                "seed": seed,
                "accuracy": .6 if is_reference else .5,
                "macro_f1": .5 if is_reference else .4,
                "draw_f1": .3 if is_reference else .2,
                "mae": 1.0 if is_reference else 1.2,
                "r2": .2 if is_reference else .1,
            })
    effects = paired_effects(pd.DataFrame(rows), repeats=1000)
    assert len(effects) == len(CONTRASTS) * len(METRICS)
    assert set(effects["n_seeds"]) == {5}
    assert (effects["mean_improvement"] > 0).all()
    assert (effects["reference_better_seeds"] == 5).all()
