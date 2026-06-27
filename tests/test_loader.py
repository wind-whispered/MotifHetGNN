"""Tests for data loading utilities."""
import json
import os
import tempfile
import pytest
from src.data.loader import load_competitions, load_all_match_ids


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_load_competitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        comp_data = [
            {"competition_id": 1, "season_id": 1, "competition_name": "Test League",
             "competition_gender": "male", "country_name": "Testland",
             "season_name": "2020/21", "match_updated": "", "match_available": ""}
        ]
        _write_json(os.path.join(tmpdir, "competitions.json"), comp_data)
        comps = load_competitions(tmpdir)
        assert len(comps) == 1
        assert comps[0]["competition_id"] == 1


def test_load_match_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        match_data = [{"match_id": 42}, {"match_id": 43}]
        _write_json(os.path.join(tmpdir, "matches", "1", "1.json"), match_data)
        match_ids = load_all_match_ids(tmpdir)
        assert 42 in match_ids
        assert 43 in match_ids
