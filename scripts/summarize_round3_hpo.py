"""Verify saved development predictions/checkpoints; never evaluate test graphs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    out = ROOT / "data/information_sciences/round3_hpo"
    protocol = json.loads((out/"protocol.json").read_text())
    split_path = ROOT / "data/information_sciences/chronological_split.json"
    split = json.loads(split_path.read_text())
    assert digest(split_path) == protocol["split_sha256"]
    for file, checksum in protocol["sources"].items():
        assert digest(ROOT/file) == checksum, file
        snapshot = out/"source_snapshot"/file
        assert snapshot.exists() and digest(snapshot) == checksum, file
    for file, checksum in protocol["protected_hashes"].items():
        assert digest(out.parent/file) == checksum, file
    meta = pd.read_parquet(ROOT/"data/processed/matches_meta.parquet").set_index("match_id")
    y_train = (meta.loc[split["train"], "home_score"]-meta.loc[split["train"], "away_score"]).to_numpy(np.float32)
    variance = np.var(y_train)
    cls_train = np.where(y_train > 0, 0, np.where(y_train == 0, 1, 2))
    weights = np.bincount(cls_train, minlength=3).astype(float)**-.5
    weights /= weights.mean()
    frame = pd.read_csv(out/"validation_results.csv")
    assert not frame.duplicated(["model", "config", "seed"]).any()
    errors = {}
    for row in frame.to_dict("records"):
        stem = f"{row['model']}_{row['config']}_seed{row['seed']}"
        assert digest(out/f"{stem}.pt") == row["checkpoint_sha256"], stem
        pred = pd.read_parquet(out/f"{stem}_validation.parquet")
        assert pred.match_id.tolist() == split["validation"]
        assert not set(pred.match_id) & set(split["test"])
        true = pred.true_goal_diff.to_numpy()
        expected = meta.loc[pred.match_id, "home_score"].to_numpy()-meta.loc[pred.match_id, "away_score"].to_numpy()
        np.testing.assert_array_equal(true, expected)
        cls = np.where(true > 0, 0, np.where(true == 0, 1, 2))
        np.testing.assert_array_equal(cls, pred.true_class)
        prob = pred[["prob_home", "prob_draw", "prob_away"]].to_numpy()
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-6)
        guessed, reg = prob.argmax(1), pred.pred_goal_diff.to_numpy()
        ce = -(weights[cls]*np.log(np.maximum(prob[np.arange(len(cls)), cls], 1e-30))).sum()/weights[cls].sum()
        mse = np.mean((reg-true)**2)
        metrics = dict(accuracy=accuracy_score(cls, guessed), macro_f1=f1_score(cls, guessed, average="macro", zero_division=0),
            draw_f1=f1_score(cls, guessed, labels=[1], average="macro", zero_division=0),
            mae=mean_absolute_error(true, reg), r2=r2_score(true, reg), mse=mse, ce=ce,
            selection_score=.5*ce+.5*mse/variance)
        for metric, value in metrics.items():
            delta = abs(float(value)-row[metric])
            errors[metric] = max(errors.get(metric, 0), delta)
            assert delta < 2e-6, (stem, metric, value, row[metric])
        history = pd.read_csv(out/f"{stem}_history.csv")
        assert len(history) == row["epochs_run"]
        best = history.loc[history.selection_score.idxmin()]
        assert int(best.epoch) == row["best_epoch"]
        assert abs(best.selection_score-row["selection_score"]) < 2e-6
        assert row["test_evaluated"] is False and row["split"] == "validation"
    complete = (out/"frozen_validation_selection.json").exists()
    if not args.allow_incomplete:
        assert complete and len(frame) == 84
    report = dict(status="passed", complete=complete, completed_fits=len(frame), expected_fits=84,
        verified_predictions=len(frame)*len(split["validation"]), maximum_metric_errors=errors,
        original_primary_hashes_unchanged=True, test_predictions_used=False,
        interpretation="Development validation only; not Table 3 and not a fresh-holdout/SOTA comparison.")
    if complete:
        selected = json.loads((out/"frozen_validation_selection.json").read_text())["selections"]
        report["selected"] = selected
        screen = frame[frame.phase == "screen"]
        expected_screen = {(model, config["name"], 7)
                           for model in protocol["models"] for config in protocol["grid"]}
        assert set(map(tuple, screen[["model", "config", "seed"]].to_numpy())) == expected_screen
        finalists = json.loads((out/"frozen_finalists.json").read_text())["finalists"]
        expected_finalists = (screen.sort_values(["model", "selection_score", "config"])
                              .groupby("model").head(2)[["model", "config"]].to_dict("records"))
        assert finalists == expected_finalists
        replicates = frame[frame.phase == "replicate"]
        expected_replicates = {(entry["model"], entry["config"], seed)
                               for entry in finalists for seed in [13, 21]}
        assert set(map(tuple, replicates[["model", "config", "seed"]].to_numpy())) == expected_replicates
        aggregate = frame.groupby(["model", "config"]).agg(n=("seed", "count"),
            score=("selection_score", "mean"), f1=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
            r2=("r2", "mean"), r2_std=("r2", "std"), mae=("mae", "mean")).reset_index()
        winners = (aggregate[aggregate.n == 3].sort_values(["model", "score", "config"])
                   .groupby("model").head(1).to_dict("records"))
        assert len(winners) == len(selected) == 7
        for expected, saved in zip(winners, selected):
            assert (expected["model"], expected["config"]) == (saved["model"], saved["config"])
            for key in ["n", "score", "f1", "f1_std", "r2", "r2_std", "mae"]:
                assert abs(expected[key]-saved[key]) < 1e-12, (key, expected, saved)
        for entry in selected:
            subset = frame[(frame.model == entry["model"]) & (frame.config == entry["config"])]
            assert set(subset.seed) == {7, 13, 21}
            assert abs(subset.macro_f1.mean()-entry["f1"]) < 1e-12
        report["duplicated_native_control_finalists"] = [model for model in protocol["models"]
            if model not in {"SPARC-GNN", "EdgeAware-Fixed", "MotifGate-ObservedOnly"}
            and {e["config"] for e in finalists if e["model"] == model} == {"post_control", "root_control"}]
    (out/"verification.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
