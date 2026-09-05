"""Per-fly STAC marker-offset sample: gates, stratification, path resolution.

Why this exists: `offsets.h5` used to be fit ONCE per run root on whichever
bout-fly reached the stage first and shared with both flies. Measured on
Session0/2025_10_20_13_20_04 bout 28 the sample was fly0 (the female) frames
0..499, so the male ran IK with her marker offsets (his abdomen fitted 1.10x
too long, wings 1.01-1.06x). The replacement pools every triangulated bout of
ONE fly, keeps only high-confidence, physically plausible frames, and takes a
stratified sample so no bout dominates. Synthetic fixtures throughout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.offsets_sample import (
    load_fly_bouts,
    offsets_fit_cfg,
    resolve_offsets_path,
    select_offsets_sample,
)

K = 5


def _bout(n, conf=0.99, seed=0):
    rng = np.random.default_rng(seed)
    kp = rng.normal(size=(n, K, 3))
    c = np.full((n, K), conf, dtype=float)
    return kp, c


def test_gates_reject_nonfinite_lowconf_and_scale_outliers_and_record_counts():
    kp, c = _bout(10)
    kp[0, 2, 1] = np.nan                 # non-finite frame
    c[1, 3] = 0.2                        # one low-confidence keypoint
    scale = np.full(10, 1.0)
    scale[2] = 5.0                       # implausible implied body scale
    scale[3:] += np.linspace(-0.01, 0.01, 7)   # a little spread so MAD > 0
    s = select_offsets_sample({7: kp}, {7: c}, n_frames=100, min_conf=0.7,
                              scale_by_bout={7: scale}, mad_k=3.0)
    frames = [f for _, f in s.frames]
    assert 0 not in frames and 1 not in frames and 2 not in frames
    assert len(frames) == 7
    b = s.provenance["bouts"]["7"]
    assert (b["nonfinite"], b["low_conf"], b["scale_outlier"], b["selected"]) == (1, 1, 1, 7)
    assert s.provenance["scale_gate"]["applied"] is True
    assert s.kp3d.shape == (7, K, 3)


def test_min_conf_is_per_frame_minimum_over_keypoints():
    kp, c = _bout(4)
    c[:, 0] = [0.95, 0.7, 0.69, 0.1]     # frame passes only if EVERY kp >= min_conf
    s = select_offsets_sample({1: kp}, {1: c}, n_frames=10, min_conf=0.7)
    assert sorted(f for _, f in s.frames) == [0, 1]


def test_missing_confidence_skips_that_gate_and_says_so():
    kp, _ = _bout(6)
    s = select_offsets_sample({1: kp}, {1: None}, n_frames=10, min_conf=0.7)
    assert len(s.frames) == 6
    assert s.provenance["conf_gate_applied"] is False


def test_stratified_round_robin_caps_total_and_balances_bouts():
    bouts = {1: _bout(100, seed=1), 2: _bout(100, seed=2), 3: _bout(100, seed=3)}
    s = select_offsets_sample({b: k for b, (k, _) in bouts.items()},
                              {b: c for b, (_, c) in bouts.items()},
                              n_frames=30, min_conf=0.7)
    assert len(s.frames) == 30
    per_bout = {b: sum(1 for bb, _ in s.frames if bb == b) for b in bouts}
    assert per_bout == {1: 10, 2: 10, 3: 10}
    # a short bout is exhausted and the others fill the remainder
    bouts[3] = _bout(4, seed=3)
    s = select_offsets_sample({b: k for b, (k, _) in bouts.items()},
                              {b: c for b, (_, c) in bouts.items()},
                              n_frames=30, min_conf=0.7)
    per_bout = {b: sum(1 for bb, _ in s.frames if bb == b) for b in bouts}
    assert per_bout[3] == 4 and per_bout[1] + per_bout[2] == 26
    assert abs(per_bout[1] - per_bout[2]) <= 1


def test_within_bout_highest_confidence_frames_are_taken_first():
    kp, c = _bout(6)
    c[:, 1] = [0.90, 0.99, 0.80, 0.95, 0.85, 0.75]
    s = select_offsets_sample({1: kp}, {1: c}, n_frames=3, min_conf=0.7)
    assert [f for _, f in s.frames] == [1, 3, 0]
    # kp3d rows follow the same order as `frames`
    np.testing.assert_array_equal(s.kp3d[0], kp[1])


def test_scale_gate_with_zero_spread_rejects_nothing():
    kp, c = _bout(5)
    s = select_offsets_sample({1: kp}, {1: c}, n_frames=10, min_conf=0.7,
                              scale_by_bout={1: np.ones(5)})
    assert len(s.frames) == 5
    assert s.provenance["bouts"]["1"]["scale_outlier"] == 0


def test_no_qualifying_frames_raises_with_the_gate_counts():
    kp, c = _bout(3, conf=0.1)
    with pytest.raises(ValueError, match="low_conf"):
        select_offsets_sample({1: kp}, {1: c}, n_frames=10, min_conf=0.7)


def _write_npz(p: Path, kp, conf=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    if conf is None:
        np.savez(p, kp3d=kp)
    else:
        np.savez(p, kp3d=kp, conf3d=conf)


def test_load_fly_bouts_prefers_filt_reads_conf_and_only_this_fly(tmp_path):
    run = tmp_path / "run"
    kf, cf = _bout(3, seed=1)
    kr, cr = _bout(3, seed=2)
    _write_npz(run / "bouts/bout_00002/fly1/kp3d_filt.npz", kf, cf)
    _write_npz(run / "bouts/bout_00002/fly1/kp3d.npz", kr, cr)
    _write_npz(run / "bouts/bout_00005/fly1/kp3d.npz", kr)           # no conf3d
    _write_npz(run / "bouts/bout_00003/fly0/kp3d_filt.npz", kr, cr)  # other fly
    got = load_fly_bouts(run, 1)
    assert sorted(got) == [2, 5]
    np.testing.assert_array_equal(got[2][0], kf)
    np.testing.assert_array_equal(got[2][1], cf)
    assert got[5][1] is None


def test_resolve_offsets_path_is_per_fly_only_under_canonical_identity(tmp_path):
    p, mode = resolve_offsets_path(tmp_path, 1, "canonical", allow_shared=False, reason="ok")
    assert (Path(p).name, mode) == ("offsets_fly1.h5", "per_fly")
    with pytest.raises(RuntimeError, match="allow_shared_offsets"):
        resolve_offsets_path(tmp_path, 1, "unknown", allow_shared=False, reason="2/9 bouts lack sex.json")
    p, mode = resolve_offsets_path(tmp_path, 0, "unknown", allow_shared=True, reason="x")
    assert (Path(p).name, mode) == ("offsets.h5", "shared")


def test_offsets_fit_cfg_zeroes_temporal_smoothing_without_touching_the_original():
    cfg = OmegaConf.create({"anatomy": {"model": {"JAXLS_SMOOTH_WEIGHT": 0.005,
                                                  "JAXLS_SMOOTH_Q_MULT": {"wing_roll_left": 5.0}}},
                            "model": "${anatomy.model}", "stac": {"n_fit_frames": 500}})
    fit = offsets_fit_cfg(cfg)
    assert fit.model.JAXLS_SMOOTH_WEIGHT == 0.0
    assert fit.model.JAXLS_SMOOTH_Q_MULT is None
    # independent frames go to JaxlsBatchSolver's vmapped per-frame path in
    # one call; chunking would only add loop overhead
    assert fit.model.JAXLS_CHUNK_SIZE == 0
    assert "JAXLS_CHUNK_SIZE" not in cfg.anatomy.model
    assert cfg.model.JAXLS_SMOOTH_WEIGHT == 0.005
    assert cfg.anatomy.model.JAXLS_SMOOTH_Q_MULT == {"wing_roll_left": 5.0}


def test_provenance_is_json_serialisable():
    kp, c = _bout(4)
    s = select_offsets_sample({1: kp}, {1: c}, n_frames=2, min_conf=0.7,
                              scale_by_bout={1: np.ones(4)})
    json.dumps(s.provenance)
