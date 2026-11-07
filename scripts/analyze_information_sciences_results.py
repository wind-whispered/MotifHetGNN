"""Aggregate multi-seed results and paired match-level bootstrap intervals."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "information_sciences"

IS_BOOTSTRAP = os.environ.get("IS_BOOTSTRAP", "1").lower() not in {"0", "false", "no"}
BOOTSTRAP_REPEATS = max(100, int(os.environ.get("IS_BOOTSTRAP_REPEATS", "2000")))
HIERARCHICAL_REPEATS = max(
    100, int(os.environ.get("IS_HIERARCHICAL_REPEATS", "5000")))
MODEL_ALIASES = {"MCAR-HGNN": "MotifGate", "MCAR-Decomp": "SPARC-GNN"}
INVALID_FULL_MATCH_MODELS = {"CurrentScore", "CurrentScore-Calibrated"}
METRICS = ("accuracy", "macro_f1", "draw_f1", "mae", "r2")


def metric(frame, name):
    if name == "accuracy":
        return accuracy_score(frame.true_class, frame.pred_class)
    if name == "macro_f1":
        return f1_score(frame.true_class, frame.pred_class, average="macro",
                        zero_division=0)
    if name == "draw_f1":
        return f1_score(frame.true_class, frame.pred_class, labels=[1],
                        average="macro", zero_division=0)
    if name == "mae":
        return mean_absolute_error(frame.true_goal_diff, frame.pred_goal_diff)
    if name == "r2":
        return r2_score(frame.true_goal_diff, frame.pred_goal_diff)
    raise ValueError(name)


def _metric_arrays(true_class, pred_class, true_goal_diff, pred_goal_diff,
                   name):
    if name not in METRICS:
        raise ValueError(name)
    return _all_metric_arrays(
        true_class, pred_class, true_goal_diff, pred_goal_diff)[
            METRICS.index(name)]


def _all_metric_arrays(true_class, pred_class, true_goal_diff,
                       pred_goal_diff):
    """Compute all five metrics without repeated sklearn dispatch overhead."""
    true_class = np.asarray(true_class, dtype=np.int8)
    pred_class = np.asarray(pred_class, dtype=np.int8)
    true_goal_diff = np.asarray(true_goal_diff, dtype=float)
    pred_goal_diff = np.asarray(pred_goal_diff, dtype=float)
    accuracy = float(np.mean(true_class == pred_class))
    f1_values = []
    for label in (0, 1, 2):
        true_label = true_class == label
        pred_label = pred_class == label
        tp = int(np.count_nonzero(true_label & pred_label))
        fp = int(np.count_nonzero(~true_label & pred_label))
        fn = int(np.count_nonzero(true_label & ~pred_label))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    macro_f1 = float(np.mean(f1_values))
    draw_f1 = float(f1_values[1])
    residual_ss = float(np.square(true_goal_diff - pred_goal_diff).sum())
    total_ss = float(np.square(
        true_goal_diff - true_goal_diff.mean()).sum())
    if total_ss == 0.0:
        r2 = 1.0 if residual_ss == 0.0 else 0.0
    else:
        r2 = 1.0 - residual_ss / total_ss
    return np.asarray([
        accuracy, macro_f1, draw_f1,
        float(np.mean(np.abs(true_goal_diff - pred_goal_diff))), r2,
    ], dtype=float)


def hierarchical_paired_bootstrap(predictions, target, baseline,
                                  repeats=HIERARCHICAL_REPEATS):
    """Resample common test matches and initialisation seeds jointly."""
    seed_values = sorted(
        set(predictions.loc[predictions.model == target, "seed"])
        & set(predictions.loc[predictions.model == baseline, "seed"])
    )
    paired = {}
    common_matches = None
    for seed in seed_values:
        target_rows = predictions[
            (predictions.model == target) & (predictions.seed == seed)
        ].set_index("match_id").sort_index()
        baseline_rows = predictions[
            (predictions.model == baseline) & (predictions.seed == seed)
        ].set_index("match_id").sort_index()
        matches = target_rows.index.intersection(baseline_rows.index)
        common_matches = matches if common_matches is None else (
            common_matches.intersection(matches))
        paired[seed] = (target_rows, baseline_rows)
    if not seed_values or common_matches is None or len(common_matches) == 0:
        return []

    common_matches = common_matches.sort_values()
    columns = ["true_class", "pred_class", "true_goal_diff", "pred_goal_diff"]
    arrays = {}
    for seed, (target_rows, baseline_rows) in paired.items():
        target_values = target_rows.loc[common_matches, columns].to_numpy()
        baseline_values = baseline_rows.loc[common_matches, columns].to_numpy()
        if not np.array_equal(target_values[:, 0], baseline_values[:, 0]) \
                or not np.array_equal(target_values[:, 2], baseline_values[:, 2]):
            raise ValueError(
                f"Target labels disagree for {baseline}, seed={seed}")
        arrays[seed] = (target_values, baseline_values)

    observed = {name: [] for name in METRICS}
    for seed in seed_values:
        target_values, baseline_values = arrays[seed]
        target_metrics = _all_metric_arrays(
            target_values[:, 0], target_values[:, 1],
            target_values[:, 2], target_values[:, 3])
        baseline_metrics = _all_metric_arrays(
            baseline_values[:, 0], baseline_values[:, 1],
            baseline_values[:, 2], baseline_values[:, 3])
        for index, name in enumerate(METRICS):
            observed[name].append(target_metrics[index] -
                                  baseline_metrics[index])

    rng = np.random.default_rng(
        42 + sum((i + 1) * ord(char) for i, char in enumerate(baseline)))
    bootstrap = np.empty((repeats, len(METRICS)), dtype=float)
    n_matches = len(common_matches)
    for repeat in range(repeats):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values),
                                   replace=True)
        sampled_matches = rng.integers(0, n_matches, size=n_matches)
        deltas = []
        for seed in sampled_seeds:
            target_values, baseline_values = arrays[int(seed)]
            target_sample = target_values[sampled_matches]
            baseline_sample = baseline_values[sampled_matches]
            deltas.append(
                _all_metric_arrays(
                    target_sample[:, 0], target_sample[:, 1],
                    target_sample[:, 2], target_sample[:, 3])
                - _all_metric_arrays(
                    baseline_sample[:, 0], baseline_sample[:, 1],
                    baseline_sample[:, 2], baseline_sample[:, 3]))
        bootstrap[repeat] = np.mean(deltas, axis=0)

    rows = []
    for index, name in enumerate(METRICS):
        distribution = bootstrap[:, index]
        probability_nonpositive = float(np.mean(distribution <= 0))
        probability_nonnegative = float(np.mean(distribution >= 0))
        rows.append({
            "target": target,
            "baseline": baseline,
            "metric": name,
            "delta_target_minus_baseline": float(np.mean(observed[name])),
            "ci_low": float(np.quantile(distribution, 0.025)),
            "ci_high": float(np.quantile(distribution, 0.975)),
            "two_sided_tail_probability": min(
                1.0, 2 * min(probability_nonpositive,
                             probability_nonnegative)),
            "n_seeds": len(seed_values),
            "n_matches": n_matches,
            "repeats": repeats,
        })
    return rows


def resolve_target_model(results):
    for candidate in ("SPARC-GNN", "MCAR-HGNN", "MotifGate"):
        if candidate in set(results["model"]):
            return candidate
    raise ValueError("No comparable target model found in model_results")


def _clean_results(frame):
    """Coerce identifiers and reject complete-match score leakage."""
    frame = frame.copy()
    frame["model"] = frame["model"].where(frame["model"].notna(), other=pd.NA)
    if frame["model"].dtype != "string":
        frame["model"] = frame["model"].astype(str)
    frame["model"] = frame["model"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    frame = frame.dropna(subset=["model"]).copy()
    frame["model"] = frame["model"].replace(MODEL_ALIASES)
    invalid = sorted(set(frame["model"]).intersection(INVALID_FULL_MATCH_MODELS))
    if invalid:
        raise ValueError(
            "Full-time score rows are label leakage in the complete-match "
            f"comparison: {invalid}. Run finalize_information_sciences_results.py."
        )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["seed"]).copy()
    frame["seed"] = frame["seed"].astype(int)
    return frame


def main():
    results = _clean_results(pd.read_parquet(OUT / "model_results.parquet"))
    predictions = _clean_results(pd.read_parquet(OUT / "predictions.parquet"))
    summary = results.groupby("model").agg(
        seeds=("seed", "nunique"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        draw_f1_mean=("draw_f1", "mean"), draw_f1_std=("draw_f1", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    ).reset_index()
    summary.to_csv(OUT / "model_results_compact_summary.csv", index=False)

    if IS_BOOTSTRAP:
        rng = np.random.default_rng(42)
        rows = []
        target = resolve_target_model(results)
        target_seed_set = set(results.loc[results["model"] == target, "seed"])
        baseline_models = sorted(set(predictions["model"]) - {target})
        for baseline in baseline_models:
            base_rows = predictions[predictions["model"] == baseline]
            for seed in sorted(set(base_rows["seed"]).intersection(target_seed_set)):
                a = predictions[(predictions["model"] == target) &
                               (predictions["seed"] == seed)].sort_values("match_id")
                b = base_rows[base_rows["seed"] == seed].sort_values("match_id")
                if len(a) == 0 or len(b) == 0:
                    continue
                if len(a) != len(b) or not np.array_equal(a["match_id"].to_numpy(),
                                                          b["match_id"].to_numpy()):
                    # Guard against partial reruns where prediction ids are misaligned.
                    continue
                n = len(a)
                a_values = a[["true_class", "pred_class", "true_goal_diff",
                              "pred_goal_diff"]].to_numpy()
                b_values = b[["true_class", "pred_class", "true_goal_diff",
                              "pred_goal_diff"]].to_numpy()
                observed = (_all_metric_arrays(
                    a_values[:, 0], a_values[:, 1], a_values[:, 2], a_values[:, 3])
                    - _all_metric_arrays(
                        b_values[:, 0], b_values[:, 1], b_values[:, 2], b_values[:, 3]))
                deltas = np.empty((BOOTSTRAP_REPEATS, len(METRICS)), dtype=float)
                for repeat in range(BOOTSTRAP_REPEATS):
                    idx = rng.integers(0, n, n)
                    a_sample, b_sample = a_values[idx], b_values[idx]
                    deltas[repeat] = (
                        _all_metric_arrays(
                            a_sample[:, 0], a_sample[:, 1],
                            a_sample[:, 2], a_sample[:, 3])
                        - _all_metric_arrays(
                            b_sample[:, 0], b_sample[:, 1],
                            b_sample[:, 2], b_sample[:, 3]))
                for metric_index, metric_name in enumerate(METRICS):
                    rows.append({
                        "baseline": baseline, "seed": seed, "metric": metric_name,
                        "delta": observed[metric_index],
                        "ci_low": np.quantile(deltas[:, metric_index], .025),
                        "ci_high": np.quantile(deltas[:, metric_index], .975),
                    })
        pd.DataFrame(rows).to_csv(OUT / "paired_bootstrap.csv", index=False)
        print(f"paired_bootstrap rows={len(rows)} (repeats={BOOTSTRAP_REPEATS})")

        hierarchical_rows = []
        for baseline in baseline_models:
            hierarchical_rows.extend(hierarchical_paired_bootstrap(
                predictions, target, baseline))
        pd.DataFrame(hierarchical_rows).to_csv(
            OUT / "hierarchical_paired_bootstrap.csv", index=False)
        print(f"hierarchical_paired_bootstrap rows={len(hierarchical_rows)} "
              f"(repeats={HIERARCHICAL_REPEATS})")
    else:
        print("bootstrap disabled by IS_BOOTSTRAP=0")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
