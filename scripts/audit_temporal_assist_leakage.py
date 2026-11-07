"""Audit retrospective assist annotations at strict temporal cutoffs.

StatsBomb stores ``shot_assist`` and ``goal_assist`` on the pass event, even
though the assisted shot occurs later.  A pass-time prefix can therefore admit
an annotation whose referenced shot is outside the prefix.  This audit joins
each flagged pass to ``pass.assisted_shot_id`` in the raw event stream and
counts those boundary crossings exactly.
"""
from collections import Counter
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (15, 30, 45, 60, 75)


def elapsed(event):
    return float(event.get("minute", 0)) + float(event.get("second", 0)) / 60.0


def audit():
    columns = [
        "match_id", "event_id", "minute", "second", "goal_assist",
        "shot_assist",
    ]
    passes = pd.read_parquet(
        ROOT / "data" / "processed" / "events_pass.parquet", columns=columns)
    passes = passes[passes["goal_assist"] | passes["shot_assist"]].copy()

    counters = Counter()
    by_cutoff = {cutoff: Counter() for cutoff in CUTOFFS}
    examples = []
    raw_events = ROOT / "StatsBomb" / "data" / "events"

    for match_id, match_passes in passes.groupby("match_id", sort=False):
        path = raw_events / f"{int(match_id)}.json"
        if not path.exists():
            counters["missing_raw_match"] += 1
            continue
        events = json.loads(path.read_text(encoding="utf-8"))
        event_by_id = {event.get("id"): event for event in events}

        for row in match_passes.itertuples(index=False):
            counters["flagged_passes"] += 1
            pass_event = event_by_id.get(row.event_id)
            if pass_event is None:
                counters["missing_raw_pass"] += 1
                continue
            assisted_shot_id = (pass_event.get("pass", {}) or {}).get(
                "assisted_shot_id")
            if not assisted_shot_id:
                counters["missing_assisted_shot_id"] += 1
                continue
            shot_event = event_by_id.get(assisted_shot_id)
            if shot_event is None:
                counters["missing_assisted_shot_event"] += 1
                continue

            counters["linked_assists"] += 1
            pass_time = elapsed(pass_event)
            shot_time = elapsed(shot_event)
            if shot_time < pass_time:
                counters["negative_time_order"] += 1

            crossed = False
            for cutoff in CUTOFFS:
                if pass_time < cutoff <= shot_time:
                    crossed = True
                    bucket = by_cutoff[cutoff]
                    bucket["all"] += 1
                    bucket["goal_assist"] += int(bool(row.goal_assist))
                    bucket["shot_assist"] += int(bool(row.shot_assist))
                    if len(examples) < 25:
                        examples.append({
                            "match_id": int(match_id),
                            "pass_event_id": str(row.event_id),
                            "shot_event_id": str(assisted_shot_id),
                            "cutoff": cutoff,
                            "pass_time": pass_time,
                            "shot_time": shot_time,
                            "goal_assist": bool(row.goal_assist),
                            "shot_assist": bool(row.shot_assist),
                        })
            counters["cross_boundary_passes"] += int(crossed)

    payload = {
        "cutoffs": list(CUTOFFS),
        "strict_rule": "pass_time < cutoff and shot_time >= cutoff",
        "source": "StatsBomb pass.assisted_shot_id",
        "counts": dict(counters),
        "by_cutoff": {
            str(cutoff): dict(by_cutoff[cutoff]) for cutoff in CUTOFFS
        },
        "examples": examples,
        "interpretation": (
            "Pass-level assist flags are retrospective annotations and must be "
            "masked in event-prefix prediction graphs."
        ),
    }
    output = ROOT / "data" / "information_sciences" / (
        "temporal_assist_leakage_audit.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output


def main():
    payload, output = audit()
    print(json.dumps(payload["counts"], indent=2))
    print(json.dumps(payload["by_cutoff"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
