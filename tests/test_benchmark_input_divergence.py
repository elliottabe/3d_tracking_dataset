"""Tests for scripts/benchmark/input_divergence.py.

Quantifies whether the four A/B variants actually shared identical 2D
(ViTPose kp2d) inputs, or whether the staleness cascade in run_bout.py
(masks_are_stale -> kp2d/kp3d/kp3d_filt deleted) caused some bout-flies to
regenerate their 2D independently per variant. Synthetic-only: no GPU, no
real benchmark tree.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.benchmark.input_divergence import (
    VARIANTS, kp2d_stats, main, scan_divergence, summarize,
)


def _write_kp2d(path: Path, kp2d: np.ndarray, conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kp2d=kp2d, conf=conf)


# ---------------------------------------------------------------- kp2d_stats

def test_kp2d_stats_identical(tmp_path):
    rng = np.random.default_rng(0)
    kp2d = rng.random((3, 2, 4, 2)).astype(np.float64)
    conf = rng.random((3, 2, 4)).astype(np.float64)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, kp2d, conf)
    _write_kp2d(b, kp2d.copy(), conf.copy())

    stats = kp2d_stats(a, b)

    assert stats["identical"] is True
    assert stats["shape_mismatch"] is False
    assert stats["median_px"] == 0.0
    assert stats["p99_px"] == 0.0
    assert stats["max_px"] == 0.0
    assert stats["frac_gt_half_px"] == 0.0
    assert stats["max_conf_diff"] == 0.0


def test_kp2d_stats_small_perturbation(tmp_path):
    kp2d = np.zeros((2, 1, 3, 2), dtype=np.float64)
    kp2d_other = kp2d.copy()
    kp2d_other[0, 0, 0, 0] += 0.001  # tiny sub-pixel move on one coord
    conf = np.ones((2, 1, 3), dtype=np.float64)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, kp2d, conf)
    _write_kp2d(b, kp2d_other, conf.copy())

    stats = kp2d_stats(a, b)

    assert stats["shape_mismatch"] is False
    assert stats["identical"] is False
    assert stats["max_px"] == pytest.approx(0.001, abs=1e-9)
    # only 1 of 6 (t,c,k) points differs -> median over all points is 0
    assert stats["median_px"] == pytest.approx(0.0, abs=1e-9)
    assert stats["frac_gt_half_px"] == 0.0


def test_kp2d_stats_large_move(tmp_path):
    kp2d = np.zeros((2, 1, 3, 2), dtype=np.float64)
    kp2d_other = kp2d.copy()
    kp2d_other[1, 0, 2, 0] += 300.0  # one coordinate moved 300px
    conf = np.ones((2, 1, 3), dtype=np.float64)
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, kp2d, conf)
    _write_kp2d(b, kp2d_other, conf.copy())

    stats = kp2d_stats(a, b)

    assert stats["identical"] is False
    assert stats["max_px"] == pytest.approx(300.0)
    assert stats["frac_gt_half_px"] > 0.0


def test_kp2d_stats_nan_same_position_treated_equal(tmp_path):
    kp2d = np.zeros((2, 1, 2, 2), dtype=np.float64)
    kp2d[0, 0, 1, :] = np.nan
    conf = np.ones((2, 1, 2), dtype=np.float64)
    conf[1, 0, 0] = np.nan
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, kp2d, conf)
    _write_kp2d(b, kp2d.copy(), conf.copy())

    stats = kp2d_stats(a, b)

    assert stats["identical"] is True
    assert stats["max_px"] == 0.0
    assert stats["max_conf_diff"] == 0.0


def test_kp2d_stats_shape_mismatch_does_not_raise(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, np.zeros((2, 1, 3, 2)), np.ones((2, 1, 3)))
    _write_kp2d(b, np.zeros((5, 1, 3, 2)), np.ones((5, 1, 3)))

    stats = kp2d_stats(a, b)

    assert stats["shape_mismatch"] is True
    assert stats["identical"] is False


def test_kp2d_stats_conf_shape_mismatch_does_not_raise(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    _write_kp2d(a, np.zeros((2, 1, 3, 2)), np.ones((2, 1, 3)))
    _write_kp2d(b, np.zeros((2, 1, 3, 2)), np.ones((2, 1, 4)))

    stats = kp2d_stats(a, b)

    assert stats["shape_mismatch"] is True
    assert stats["identical"] is False


# ------------------------------------------------------------- scan_divergence

def _rel(run_key: str, bout: int, fly: int) -> Path:
    return Path(run_key) / "bouts" / f"bout_{bout:05d}" / f"fly{fly}" / "kp2d.npz"


def _rand_kp2d(seed: int):
    rng = np.random.default_rng(seed)
    kp2d = rng.random((2, 1, 2, 2)).astype(np.float64)
    conf = rng.random((2, 1, 2)).astype(np.float64)
    return kp2d, conf


def _manifest(run_key: str, bouts: list[int], benchmark_root: Path) -> dict:
    return {
        "benchmark_root": str(benchmark_root),
        "proximity_threshold_bl": 2.0,
        "bouts": [
            {"run_key": run_key, "layout": "flat", "source_root": "/nope",
             "recording_cfg": "c", "session_dir": "/nope",
             "assay": "free_running", "bout": b, "flies": [0],
             "tags": [], "render_frames": []}
            for b in bouts
        ],
    }


def test_scan_divergence_clean_divergent_incomplete(tmp_path):
    frozen_root = tmp_path / "frozen"
    variants_root = tmp_path / "variants"
    run_key = "rk"
    manifest = _manifest(run_key, [1, 2, 3], tmp_path)

    # bout 1: CLEAN -- hardlinked into every variant.
    kp2d1, conf1 = _rand_kp2d(1)
    rel1 = _rel(run_key, 1, 0)
    fpath1 = frozen_root / rel1
    _write_kp2d(fpath1, kp2d1, conf1)
    for v in VARIANTS:
        vp = variants_root / v / rel1
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath1, vp)

    # bout 2: DIVERGENT -- regenerated (different content) in 2 of 4 variants.
    kp2d2, conf2 = _rand_kp2d(2)
    rel2 = _rel(run_key, 2, 0)
    fpath2 = frozen_root / rel2
    _write_kp2d(fpath2, kp2d2, conf2)
    for v in VARIANTS[:2]:
        vp = variants_root / v / rel2
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath2, vp)
    for v in VARIANTS[2:]:
        vp = variants_root / v / rel2
        bad = kp2d2.copy()
        bad[0, 0, 0, 0] += 5.0
        _write_kp2d(vp, bad, conf2)

    # bout 3: INCOMPLETE -- missing entirely from one variant, hardlinked
    # (clean) into the rest.
    kp2d3, conf3 = _rand_kp2d(3)
    rel3 = _rel(run_key, 3, 0)
    fpath3 = frozen_root / rel3
    _write_kp2d(fpath3, kp2d3, conf3)
    for v in VARIANTS[:3]:
        vp = variants_root / v / rel3
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath3, vp)
    # VARIANTS[3] deliberately absent for bout 3.

    rows = scan_divergence(manifest, variants_root, frozen_root)
    assert len(rows) == 3
    by_bout = {r["bout"]: r for r in rows}

    r1 = by_bout[1]
    assert r1["run_key"] == run_key and r1["fly"] == 0
    assert r1["missing"] == []
    assert r1["regenerated"] == []
    assert set(r1["frozen_identical"]) == set(VARIANTS)
    assert r1["worst"] is None
    assert r1["status"] == "clean"
    assert r1["shape_mismatch"] is False

    r2 = by_bout[2]
    assert set(r2["regenerated"]) == set(VARIANTS[2:])
    assert set(r2["frozen_identical"]) == set(VARIANTS[:2])
    assert r2["missing"] == []
    assert r2["worst"] == pytest.approx(5.0)
    for v in VARIANTS[2:]:
        assert r2["stats"][v]["identical"] is False
    assert r2["status"] == "divergent"
    assert r2["shape_mismatch"] is False

    r3 = by_bout[3]
    assert r3["missing"] == [VARIANTS[3]]
    assert r3["regenerated"] == []
    assert set(r3["present"]) == set(VARIANTS[:3])
    # Missing from a variant is NOT evidence of a controlled comparison --
    # INCOMPLETE takes precedence over the present variants being clean.
    assert r3["status"] == "incomplete"

    summary = summarize(rows)
    assert summary["n_bout_flies"] == 3
    assert summary["n_clean"] == 1       # bout 1 only
    assert summary["n_divergent"] == 1   # bout 2
    assert summary["n_incomplete"] == 1  # bout 3
    assert summary["worst_max_px"] == pytest.approx(5.0)
    assert summary["n_shape_mismatch"] == 0
    # Partition invariant: every row falls into exactly one bucket.
    assert (summary["n_clean"] + summary["n_divergent"]
            + summary["n_incomplete"]) == summary["n_bout_flies"]


def test_scan_divergence_all_clean_summary(tmp_path):
    frozen_root = tmp_path / "frozen"
    variants_root = tmp_path / "variants"
    run_key = "rk2"
    manifest = _manifest(run_key, [1], tmp_path)
    kp2d1, conf1 = _rand_kp2d(11)
    rel1 = _rel(run_key, 1, 0)
    fpath1 = frozen_root / rel1
    _write_kp2d(fpath1, kp2d1, conf1)
    for v in VARIANTS:
        vp = variants_root / v / rel1
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath1, vp)

    rows = scan_divergence(manifest, variants_root, frozen_root)
    assert rows[0]["status"] == "clean"
    summary = summarize(rows)
    assert summary == {
        "n_bout_flies": 1, "n_clean": 1, "n_divergent": 0,
        "n_incomplete": 0, "n_shape_mismatch": 0, "worst_max_px": None,
    }


def test_scan_divergence_missing_and_regenerated_counts_once_as_incomplete(tmp_path):
    """A bout-fly that is BOTH missing from one variant AND regenerated
    (differs from frozen) in another must still count once, as incomplete --
    missing takes precedence over divergent."""
    frozen_root = tmp_path / "frozen"
    variants_root = tmp_path / "variants"
    run_key = "rk4"
    manifest = _manifest(run_key, [1], tmp_path)

    kp2d1, conf1 = _rand_kp2d(41)
    rel1 = _rel(run_key, 1, 0)
    fpath1 = frozen_root / rel1
    _write_kp2d(fpath1, kp2d1, conf1)
    # VARIANTS[0]: hardlinked (frozen-identical).
    vp0 = variants_root / VARIANTS[0] / rel1
    vp0.parent.mkdir(parents=True, exist_ok=True)
    os.link(fpath1, vp0)
    # VARIANTS[1]: regenerated (differs from frozen).
    vp1 = variants_root / VARIANTS[1] / rel1
    bad = kp2d1.copy()
    bad[0, 0, 0, 0] += 2.0
    _write_kp2d(vp1, bad, conf1)
    # VARIANTS[2]: missing entirely.
    # VARIANTS[3]: hardlinked (frozen-identical).
    vp3 = variants_root / VARIANTS[3] / rel1
    vp3.parent.mkdir(parents=True, exist_ok=True)
    os.link(fpath1, vp3)

    rows = scan_divergence(manifest, variants_root, frozen_root)
    assert len(rows) == 1
    r = rows[0]
    assert r["missing"] == [VARIANTS[2]]
    assert r["regenerated"] == [VARIANTS[1]]
    assert r["status"] == "incomplete"

    summary = summarize(rows)
    assert summary["n_bout_flies"] == 1
    assert summary["n_incomplete"] == 1
    assert summary["n_clean"] == 0
    assert summary["n_divergent"] == 0
    assert (summary["n_clean"] + summary["n_divergent"]
            + summary["n_incomplete"]) == summary["n_bout_flies"]


def test_scan_divergence_shape_mismatch_surfaced_in_row_and_summary(tmp_path):
    frozen_root = tmp_path / "frozen"
    variants_root = tmp_path / "variants"
    run_key = "rk5"
    manifest = _manifest(run_key, [1], tmp_path)

    kp2d1, conf1 = _rand_kp2d(51)
    rel1 = _rel(run_key, 1, 0)
    fpath1 = frozen_root / rel1
    _write_kp2d(fpath1, kp2d1, conf1)
    for v in VARIANTS[:3]:
        vp = variants_root / v / rel1
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath1, vp)
    # Last variant regenerated with a DIFFERENT shape (e.g. different frame
    # count) -- kp2d_stats can't produce a numeric max_px for this one, but
    # it must still be visible as the most severe kind of divergence.
    vp_bad = variants_root / VARIANTS[3] / rel1
    bad_kp2d = np.concatenate([kp2d1, kp2d1[:1]], axis=0)  # extra frame
    bad_conf = np.concatenate([conf1, conf1[:1]], axis=0)
    _write_kp2d(vp_bad, bad_kp2d, bad_conf)

    rows = scan_divergence(manifest, variants_root, frozen_root)
    assert len(rows) == 1
    r = rows[0]
    assert r["regenerated"] == [VARIANTS[3]]
    assert r["stats"][VARIANTS[3]]["shape_mismatch"] is True
    assert r["shape_mismatch"] is True
    assert r["status"] == "divergent"

    summary = summarize(rows)
    assert summary["n_shape_mismatch"] == 1
    assert summary["n_divergent"] == 1
    assert summary["n_incomplete"] == 0


# --------------------------------------------------------------------- main

def test_main_writes_json_and_prints_table(tmp_path, capsys):
    frozen_root = tmp_path / "bench" / "frozen"
    variants_root = tmp_path / "bench" / "variants"
    run_key = "rk3"
    manifest_dict = _manifest(run_key, [1], tmp_path / "bench")
    manifest_path = tmp_path / "bouts.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_dict))

    kp2d1, conf1 = _rand_kp2d(21)
    rel1 = _rel(run_key, 1, 0)
    fpath1 = frozen_root / rel1
    _write_kp2d(fpath1, kp2d1, conf1)
    for v in VARIANTS:
        vp = variants_root / v / rel1
        vp.parent.mkdir(parents=True, exist_ok=True)
        os.link(fpath1, vp)

    out_path = tmp_path / "divergence.json"
    main(["--manifest", str(manifest_path), "--out", str(out_path)])

    captured = capsys.readouterr()
    assert run_key in captured.out
    assert "CLEAN" in captured.out

    payload = json.loads(out_path.read_text())
    assert "rows" in payload and "summary" in payload
    assert payload["summary"]["n_bout_flies"] == 1
    assert payload["summary"]["n_clean"] == 1
