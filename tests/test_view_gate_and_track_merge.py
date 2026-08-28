"""Safeguards for the failure that made bout_00028's identities appear to switch.

The female went edge-on against the wall; her mask fell to 2-5% of its area on
five cameras and left the frame entirely on two. The detector then labelled the
MALE as her on 87-100% of frames -- including cameras where her mask was 100%
valid -- and the DLT dragged her 3D track onto his.

Neither existing gate caught it. `masks.min_views` counts views by mask
VALIDITY, and she had a median of 5 valid views throughout, so it fired on 0
frames. `detector.view_conf_thresh` looks at confidence, and views where she
had NO mask at all still scored a median 0.856 -- above the 0.6 gate.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from scripts.run_bout import (check_track_merge, gate_low_coverage_frames,
                              mask_areas_per_view, view_mask_agreement)


# ---------------------------------------------------------------- agreement

def _views(dist_px, area_px=400.0, n_cam=3, T=4):
    """kp2d whose per-view centroid sits `dist_px` from that view's mask."""
    centroids = np.zeros((T, n_cam, 2), float)
    kp2d = np.zeros((T, n_cam, 5, 2), float)
    dd = np.asarray(dist_px, float)
    kp2d[..., 0] = dd.reshape(-1, 1, 1) if dd.ndim else dd
    areas = np.full((T, n_cam), area_px)
    valid = np.ones((T, n_cam), bool)
    return kp2d, centroids, valid, areas


def test_none_is_a_strict_no_op():
    kp2d, cen, valid, areas = _views(10_000.0)
    out = view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=None)
    assert out.dtype == bool and out.all(), "None must leave every valid view agreeing"


def test_a_prediction_on_its_own_mask_agrees():
    kp2d, cen, valid, areas = _views(5.0, area_px=400.0)   # 5px, fly size 20px
    assert view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=3.0).all()


def test_a_prediction_on_the_other_fly_is_rejected():
    # fly size sqrt(400)=20px, so 3 fly-lengths = 60px; 250px is the real
    # inter-fly distance measured on the exemplar bout.
    kp2d, cen, valid, areas = _views(250.0, area_px=400.0)
    assert not view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=3.0).any()


def test_the_threshold_scales_with_the_flys_own_size():
    """The point of sqrt(area): the same pixel distance is fine for a big fly
    and wrong for a small one, so one threshold transfers across cameras."""
    d = 100.0
    big = view_mask_agreement(*_views(d, area_px=10_000.0), max_fly_lengths=3.0)   # 100px fly
    small = view_mask_agreement(*_views(d, area_px=400.0), max_fly_lengths=3.0)    # 20px fly
    assert big.all(), "100px offset is within 3 lengths of a 100px fly"
    assert not small.any(), "100px offset is 5 lengths from a 20px fly"


def test_a_view_with_no_mask_can_never_agree():
    """Measured: maskless views still emitted keypoints at conf 0.856 and
    entered the DLT. Nothing says where the animal is, so they are not evidence."""
    kp2d, cen, valid, areas = _views(0.0)
    valid[:, 1] = False
    out = view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=3.0)
    assert not out[:, 1].any()
    assert out[:, 0].all() and out[:, 2].all()


def test_nan_keypoints_do_not_agree():
    kp2d, cen, valid, areas = _views(0.0)
    kp2d[2, 1] = np.nan
    assert not view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=3.0)[2, 1]


def test_agreement_feeds_the_existing_min_views_gate():
    """The two compose: disagreeing views shrink the per-frame count, which is
    what makes min_views finally fire on this failure."""
    kp2d, cen, valid, areas = _views(np.array([0.0, 0.0, 500.0, 500.0]), n_cam=5)
    agree = view_mask_agreement(kp2d, cen, valid, areas, max_fly_lengths=3.0)
    per_frame = agree.sum(axis=1)
    assert list(per_frame) == [5, 5, 0, 0]
    kp3d = np.ones((4, 6, 3))
    gated, n = gate_low_coverage_frames(kp3d, per_frame, 4)
    assert n == 2 and np.isnan(gated[2]).all() and np.isfinite(gated[:2]).all()


def test_mask_areas_per_view_counts_pixels():
    m = np.zeros((3, 2, 5, 5), bool)
    m[0, 0, :2, :3] = True          # 6 px
    m[1, 1] = True                  # 25 px
    a = mask_areas_per_view(m)
    assert a.shape == (3, 2) and a[0, 0] == 6 and a[1, 1] == 25 and a[2, 0] == 0


# --------------------------------------------------------------- track merge

def _write_bout(tmp_path, sep, body=10.0, T=100):
    """Two flies `sep` apart (scalar or (T,)), each with a `body`-long spread."""
    d = tmp_path / "bout_00000"
    sep = np.broadcast_to(np.asarray(sep, float), (T,))
    for fly, base in ((0, 0.0), (1, None)):
        k = np.zeros((T, 4, 3))
        off = np.zeros(T) if fly == 0 else sep
        k[:, 0, 0] = off                     # Scutellum
        k[:, 1, 0] = off + body              # furthest keypoint
        k[:, 2, 0] = off + body / 2
        k[:, 3, 0] = off + body / 3
        p = d / f"fly{fly}"; p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "kp3d.npz", kp3d=k)
    return str(d)


def test_well_separated_flies_report_ok(tmp_path):
    r = check_track_merge(_write_bout(tmp_path, sep=25.0, body=10.0))
    assert r["status"] == "ok" and r["n_merged"] == 0
    assert r["body_length"] == pytest.approx(10.0)


def test_collapsed_tracks_are_flagged(tmp_path):
    r = check_track_merge(_write_bout(tmp_path, sep=0.5, body=10.0))
    assert r["status"] == "merged"
    assert r["frac_merged"] == pytest.approx(1.0)
    assert r["min_separation"] == pytest.approx(0.5)


def test_a_brief_close_pass_is_not_called_a_merge(tmp_path):
    """Courting flies genuinely touch; only a sustained collapse is a merge."""
    sep = np.full(100, 25.0); sep[:1] = 0.5          # 1% of frames
    r = check_track_merge(_write_bout(tmp_path, sep=sep, body=10.0))
    assert r["status"] == "ok", r


def test_threshold_scales_with_body_length(tmp_path):
    """Same separation, different body size -> different verdict."""
    assert check_track_merge(_write_bout(tmp_path / "a", sep=3.0, body=10.0))["status"] == "merged"
    assert check_track_merge(_write_bout(tmp_path / "b", sep=3.0, body=2.0))["status"] == "ok"


def test_missing_or_all_nan_input_is_unknown_not_a_false_ok(tmp_path):
    assert check_track_merge(str(tmp_path / "nope"))["status"] == "unknown"
    d = _write_bout(tmp_path, sep=25.0)
    import os
    k = np.full((100, 4, 3), np.nan)
    np.savez(os.path.join(d, "fly0", "kp3d.npz"), kp3d=k)
    assert check_track_merge(d)["status"] == "unknown"


# ------------------------------------------------- unsolvable-fly handling

def test_an_unsolvable_fly_is_recorded_not_raised():
    """A fly whose keypoints were all gated away has no pose to fit. That is a
    recorded outcome, not a crash: raising took out the SLURM array task, the
    OTHER fly, and the dependent aggregate job (Session0 bouts 8/19/26).

    The contract that matters downstream: NO DONE marker and NO stac_ik.h5, so
    combine_ik_outputs skips the fly rather than treating it as processed.
    """
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "scripts" / "run_bout.py"
    tree = ast.parse(src.read_text())
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "process_bout_fly"][0]
    body = ast.get_source_segment(src.read_text(), fn)
    i = body.index("elif not _segs:")
    block = body[i:i + 1600]
    assert "unsolvable.json" in block, "the reason must be persisted"
    assert "return" in block, "must return rather than raise"
    assert "raise RuntimeError" not in block.split("return")[0], \
        "the no-segment path must not raise"


def test_the_frozen_pose_guard_still_raises():
    """Recording an unsolvable fly must NOT have softened the frozen-pose gate:
    a solve that returned a rigid pose is a different failure and still fails."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_bout.py").read_text()
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "process_bout_fly"][0]
    body = ast.get_source_segment(src, fn)
    assert "joints_frozen(q)" in body
    after = body[body.index("joints_frozen(q)"):]
    assert "raise RuntimeError" in after[:400], "the frozen-pose gate must still raise"
