"""Fresh reproduction cannot require the original machine's repair archives."""
import json
import pandas as pd
from scripts import audit_information_sciences_completion as audit


def test_fresh_flag_skips_archives_but_not_feature_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "OUT", tmp_path)
    payload = dict(feature_revision=audit.ASSIST_PROXY_FEATURE_REVISION,
        graph_revision=audit.HETEROGENEOUS_GRAPH_REVISION,
        split_sizes=dict(train=2426, validation=346, test=692),
        assist_only_calibrated=dict(macro_f1=.72, r2=.75),
        test_correlations=dict(goal_assist_pair_difference=.9))
    path = tmp_path/"full_match_assist_proxy_audit.json"
    path.write_text(json.dumps(payload))
    assert audit.audit_feature_leakage()[0]["status"] == "FAIL"
    assert audit.audit_feature_leakage(require_archives=False)[0]["status"] == "PASS"
    payload["feature_revision"] = "wrong"
    path.write_text(json.dumps(payload))
    assert audit.audit_feature_leakage(require_archives=False)[0]["status"] == "FAIL"


def test_fresh_flag_does_not_hide_missing_or_invalid_census(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "audit_census", lambda: dict(status="PASS", counts={}))
    assert [c["status"] for c in audit.audit_highorder()] == ["PASS", "FAIL"]
    assert [c["status"] for c in audit.audit_highorder(require_repair=False)] == ["PASS"]
    monkeypatch.setattr(audit, "audit_census", lambda: dict(status="FAIL", counts={}))
    assert audit.audit_highorder(require_repair=False)[0]["status"] == "FAIL"


def test_fresh_flag_keeps_event_clock_check_without_legacy_plot_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "OUT", tmp_path/"information")
    directory = tmp_path/"data/processed"
    directory.mkdir(parents=True)
    frame = pd.DataFrame(dict(minute=[47], second=[30], continuous_minute=[47.5]))
    frame.to_parquet(directory/"events_pass.parquet")
    assert [r["status"] for r in audit.audit_event_clock()] == ["PASS", "FAIL"]
    assert [r["status"] for r in audit.audit_event_clock(require_legacy_growth=False)] == ["PASS"]
    frame["continuous_minute"] = 92.5
    frame.to_parquet(directory/"events_pass.parquet")
    assert audit.audit_event_clock(require_legacy_growth=False)[0]["status"] == "FAIL"
