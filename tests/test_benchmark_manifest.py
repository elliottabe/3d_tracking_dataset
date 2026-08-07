"""Tests for scripts/benchmark/manifest.py (benchmark manifest + path resolution)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest

MANIFEST = {
    "benchmark_root": "/tmp/bench",
    "proximity_threshold_bl": 2.0,
    "bouts": [
        {"run_key": "courtship_S1_recA", "layout": "canonical",
         "source_root": "/data/processed/courtship/Session1/recA/pose",
         "recording_cfg": "session1", "session_dir": "/data/video/recA",
         "assay": "courtship", "bout": 3, "flies": [0, 1],
         "tags": ["close"], "render_frames": [5]},
        {"run_key": "freerun_S11_recB", "layout": "flat",
         "source_root": "/data/free_running/Session11_recB_bouts",
         "recording_cfg": "free_running_session11", "session_dir": "/data/video/recB",
         "assay": "free_running", "bout": 1, "flies": [0],
         "tags": [], "render_frames": [5, 9]},
    ],
}


def write_manifest(tmp_path, data=MANIFEST) -> Path:
    p = tmp_path / "bouts.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_and_entries(tmp_path):
    m = load_manifest(write_manifest(tmp_path))
    es = entries(m)
    assert len(es) == 2
    assert es[0]["run_key"] == "courtship_S1_recA"
    assert m["proximity_threshold_bl"] == 2.0


def test_bout_fly_dir_canonical_source():
    e = MANIFEST["bouts"][0]
    assert bout_fly_dir({**e, "fly": 0}) == Path(
        "/data/processed/courtship/Session1/recA/pose/bouts/bout_00003/fly0")


def test_bout_fly_dir_flat_source():
    e = MANIFEST["bouts"][1]
    assert bout_fly_dir({**e, "fly": 0}) == Path(
        "/data/free_running/Session11_recB_bouts/bouts/bout_00001/fly0")


def test_bout_fly_dir_under_root():
    e = MANIFEST["bouts"][0]
    assert bout_fly_dir({**e, "fly": 1}, root=Path("/tmp/var1")) == Path(
        "/tmp/var1/courtship_S1_recA/bouts/bout_00003/fly1")


def test_load_manifest_rejects_duplicate_run_key_bout(tmp_path):
    bad = dict(MANIFEST, bouts=[MANIFEST["bouts"][0], MANIFEST["bouts"][0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(write_manifest(tmp_path, bad))


def test_load_manifest_rejects_bad_layout(tmp_path):
    bad = dict(MANIFEST, bouts=[{**MANIFEST["bouts"][0], "layout": "weird"}])
    with pytest.raises(ValueError, match="layout"):
        load_manifest(write_manifest(tmp_path, bad))
