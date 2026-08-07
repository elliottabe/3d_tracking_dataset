"""Tests for scripts/benchmark/freeze_inputs.py (synthetic source trees)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.freeze_inputs import freeze, verify


def make_source_bout(root: Path, bout: int, flies=(0,), with_filt=True,
                     sex_json=None) -> Path:
    bdir = root / "bouts" / f"bout_{bout:05d}"
    for f in flies:
        d = bdir / f"fly{f}"
        d.mkdir(parents=True)
        np.savez(d / "kp2d.npz", kp2d=np.zeros((4, 2, 3, 2)), conf=np.ones((4, 2, 3)))
        np.savez(d / "kp3d.npz", kp3d=np.zeros((4, 3, 3)), conf3d=np.ones((4, 3)))
        if with_filt:
            np.savez(d / "kp3d_filt.npz", kp3d=np.zeros((4, 3, 3)),
                     conf3d=np.ones((4, 3)))
    if sex_json is not None:
        (bdir / "sex.json").write_text(json.dumps(sex_json))
    return bdir


def manifest_for(src: Path, run_key="courtship_r", assay="courtship",
                 flies=(0, 1), bout=1):
    return {"benchmark_root": "", "proximity_threshold_bl": 2.0, "bouts": [{
        "run_key": run_key, "layout": "canonical", "source_root": str(src),
        "recording_cfg": "session1", "session_dir": "/nope", "assay": assay,
        "bout": bout, "flies": list(flies), "tags": [], "render_frames": []}]}


def test_freeze_copies_inputs_and_checksums(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0, 1), sex_json={"male_fly": 1})
    sums = freeze(manifest_for(src), dest)
    base = dest / "frozen" / "courtship_r" / "bouts" / "bout_00001"
    for f in (0, 1):
        for name in ("kp2d.npz", "kp3d.npz", "kp3d_filt.npz"):
            assert (base / f"fly{f}" / name).exists()
    assert (base / "sex.json").exists()
    assert (dest / "frozen" / "checksums.json").exists()
    assert any(k.endswith("kp2d.npz") for k in sums)
    assert verify(dest) == []


def test_freeze_missing_artifact_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    bdir = make_source_bout(src, 1, flies=(0,))
    (bdir / "fly0" / "kp3d.npz").unlink()
    with pytest.raises(FileNotFoundError, match="kp3d.npz"):
        freeze(manifest_for(src, assay="free_running", flies=(0,)), dest)


def test_freeze_optional_filt_and_sex(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 2, flies=(0,), with_filt=False)   # no filt, no sex.json
    freeze(manifest_for(src, run_key="fr", assay="free_running",
                        flies=(0,), bout=2), dest)
    base = dest / "frozen" / "fr" / "bouts" / "bout_00002"
    assert not (base / "fly0" / "kp3d_filt.npz").exists()


def test_verify_detects_tamper(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0,))
    freeze(manifest_for(src, run_key="fr", assay="free_running", flies=(0,)), dest)
    victim = dest / "frozen" / "fr" / "bouts" / "bout_00001" / "fly0" / "kp3d.npz"
    victim.write_bytes(b"corrupt")
    bad = verify(dest)
    assert len(bad) == 1 and bad[0].endswith("kp3d.npz")
