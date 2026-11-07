import numpy as np

from scripts.analyze_ablation_paired_effects import (
    _exact_two_sided_sign_flip,
)


def test_five_seed_exact_sign_flip_resolution():
    effects = np.ones(5, dtype=float)
    assert _exact_two_sided_sign_flip(effects) == 0.0625


def test_exact_sign_flip_is_invariant_to_effect_direction():
    effects = np.asarray([0.2, 0.1, -0.05, 0.3, 0.4])
    assert _exact_two_sided_sign_flip(effects) == (
        _exact_two_sided_sign_flip(-effects))
