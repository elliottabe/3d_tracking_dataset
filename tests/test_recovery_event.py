import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io, recovery_event as ev

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason="source clip not present")


@pytest.fixture
def restore_sys_path():
    """Undo `sys.path` edits an import performs, for the rest of the session.

    Importing `recovery_clip` pulls in `kp_colors`, which prepends
    `third_party/JARVIS-HybridNet` to `sys.path` at import time (and JARVIS in
    turn prepends its own `jarvis/` subdirectory). That subdirectory contains a
    `utils` package, so after this import a plain `import utils` in the SAME
    process resolves to JARVIS's utils instead of this repo's -- which breaks
    `tests/test_ik_explainer_triangulate.py`'s keypoint filter
    (`utils.keypoint_filter`) when both run in one pytest session. Restoring
    the path keeps this test from deciding what a later, unrelated one imports.
    """
    saved = list(sys.path)
    yield
    sys.path[:] = saved


def test_event_constants_match_the_spec():
    assert ev.EVENT["kp"] == "T1R_TaTip"
    assert ev.EVENT["cam"] == "Cam2012853"
    assert (ev.EVENT["t0"], ev.EVENT["t1"]) == (420, 465)
    assert ev.EVENT["peak_2d"] == 441
    assert ev.EVENT["peak_3d"] == 443


def test_load_tracks_shapes_and_frames():
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    n = e["t1"] - e["t0"]
    for k in ("raw", "filt", "ik"):
        assert t[k].shape == (n, 3), f"{k} has shape {t[k].shape}"
    assert t["det2d"].shape == (n, 2)
    assert t["rep2d"].shape == (n, 2)
    assert t["conf"].shape == (n,)
    assert np.array_equal(t["frames"], np.arange(e["t0"], e["t1"]))


def test_the_detector_really_fails_at_the_peak_frame():
    """The clip's whole premise: at frame 441 this camera is ~122 px off the
    consensus and reports low confidence. If this stops holding, the event
    moved and the clip would be showing nothing."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    i = e["peak_2d"] - e["t0"]
    gap = np.linalg.norm(t["det2d"][i] - t["rep2d"][i])
    assert gap > 80.0, f"detector-vs-reprojection gap only {gap:.0f} px"
    assert t["conf"][i] < 0.7, f"confidence {t['conf'][i]:.2f} not low"


def test_cam_disagree_matches_a_direct_recomputation():
    """Every camera's detector-vs-raw-triangulation distance, recomputed here
    from the on-disk arrays by an independent path."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    kp_names = clip_io.model_kp_names()
    k = kp_names.index(e["kp"])
    d = clip_io.out_dirs(CLIP)
    z2 = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    cam_names = [str(c) for c in z2["cam_names"]]
    raw = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)["kp3d"]
    mats, names = clip_io.load_dlt(str(Path(CLIP) / "calibration"))
    assert names == cam_names
    want = np.stack([
        np.linalg.norm(z2["kp2d"][f, :, k] - clip_io.project(mats, raw[f])[:, k],
                       axis=-1)
        for f in range(e["t0"], e["t1"])])
    assert list(t["cam_names"]) == cam_names
    assert t["cam_disagree"].shape == (e["t1"] - e["t0"], len(cam_names))
    assert np.allclose(t["cam_disagree"], want, equal_nan=True)


def test_the_caveat_states_the_numbers_it_measured(restore_sys_path):
    """The one figure the clip puts on screen must be the one in the data.
    Recomputed from the loaded arrays -- NOT compared to a fixed string."""
    from scripts.viz.ik_explainer import recovery_clip as rc

    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    lines = rc.caveat_lines(t, e)

    i = e["peak_2d"] - e["t0"]
    cams = list(t["cam_names"])
    others = [j for j, n in enumerate(cams) if n != e["cam"]]
    d = t["cam_disagree"][i, others]
    text = " ".join(lines)
    assert f"{len(others)} cameras" in text
    assert f"{d.min():.0f}-{d.max():.0f} px" in text, \
        f"caveat {text!r} does not state the measured range"
    # and no stale literal survived from the design doc
    assert "83-121" not in text
    # the qualitative claim must be true of those numbers: the failure is not
    # confined to the one camera on screen.
    assert d.max() > 5.0, "other cameras agree with the consensus after all"


def test_recovery_is_present_and_ordered():
    """raw spikes; filtered and IK do not. This is the claim the clip makes."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    ar, af, ai = (ev.accel(t[k]) for k in ("raw", "filt", "ik"))
    assert ar.max() > 1.0, f"raw peak accel {ar.max():.3f} — expected a spike"
    assert af.max() < 0.2, f"filtered peak accel {af.max():.3f} — not smooth"
    assert ai.max() < 0.2, f"IK peak accel {ai.max():.3f} — not smooth"
    assert ar.max() / af.max() > 10.0


def test_shared_scale_is_read_from_disk_not_hardcoded():
    """The IK trace's mm conversion must follow the scale the IK was solved at.
    A stale constant would render the IK trace in wrong mm while it still
    looked perfectly smooth -- the clip's claim would survive and be false."""
    scale = ev.load_shared_scale(CLIP)
    with np.load(clip_io.out_dirs(CLIP)["predictions"] / "06_stages.npz",
                 allow_pickle=True) as z:
        on_disk = float(z["shared_scale"])
    assert scale == on_disk, "load_shared_scale did not return the on-disk value"
    assert scale != ev.SHARED_SCALE_NOMINAL, \
        "on-disk scale coincidentally equals the bound -- test cannot tell them apart"
    assert abs(scale - ev.SHARED_SCALE_NOMINAL) / ev.SHARED_SCALE_NOMINAL \
        <= ev.SHARED_SCALE_TOL_FRAC


def test_a_different_body_scale_fails_loudly(tmp_path, monkeypatch):
    """Re-running IK at another scale must raise, not quietly rescale mm."""
    real = clip_io.out_dirs

    def fake_out_dirs(clip=CLIP):
        d = dict(real(clip))
        pred = tmp_path / "predictions"
        pred.mkdir(exist_ok=True)
        with np.load(d["predictions"] / "06_stages.npz", allow_pickle=True) as z:
            payload = {k: z[k] for k in z.files}
        payload["shared_scale"] = np.array(ev.SHARED_SCALE_NOMINAL * 1.5)
        np.savez(pred / "06_stages.npz", **payload)
        d["predictions"] = pred
        return d

    monkeypatch.setattr(ev.clip_io, "out_dirs", fake_out_dirs)
    with pytest.raises(ValueError, match="different body scale"):
        ev.load_shared_scale(CLIP)


def test_accel_is_zero_for_constant_velocity():
    t = np.arange(10)[:, None] * np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(ev.accel(t), 0.0, atol=1e-9)


def test_worst_axis_picks_the_largest_excursion():
    n = 20
    a = np.zeros((n, 3))
    a[10, 1] = 5.0                      # a big kick on y only
    assert ev.worst_axis(a) == 1


# ---------------------------------------------------------------------------
# Multi-camera panel stack
# ---------------------------------------------------------------------------

def test_per_camera_arrays_are_consistent_with_the_single_camera_ones():
    """The event camera's panel must be a SLICE of the all-camera arrays, not a
    second computation -- otherwise the top panel and the extra panels could
    silently disagree about the same camera."""
    t = ev.load_tracks()
    n, c = len(t["frames"]), len(t["cam_names"])
    assert t["det2d_all"].shape == (n, c, 2)
    assert t["rep2d_all"].shape == (n, c, 2)
    assert t["conf_all"].shape == (n, c)
    j = t["cam_names"].index(ev.EVENT["cam"])
    np.testing.assert_array_equal(t["det2d"], t["det2d_all"][:, j])
    np.testing.assert_allclose(t["rep2d"], t["rep2d_all"][:, j])
    np.testing.assert_array_equal(t["conf"], t["conf_all"][:, j])


def test_select_panel_cams_brackets_the_caveat():
    """Event camera first, then the WORST and the CLEANEST of the rest.

    Both halves matter: the worst shows the failure is not confined to one
    view, the cleanest shows some views see it correctly. A selection that
    returned the two worst would illustrate only half the on-screen claim.
    """
    t = ev.load_tracks()
    picks = ev.select_panel_cams(t, n_extra=2)
    assert len(picks) == len(set(picks)) == 3
    assert picks[0] == ev.EVENT["cam"]
    assert all(p in t["cam_names"] for p in picks)

    peak = np.nanmax(t["cam_disagree"], axis=0)
    others = {n: peak[t["cam_names"].index(n)]
              for n in t["cam_names"] if n != ev.EVENT["cam"]}
    assert picks[1] == max(others, key=others.get)
    assert picks[2] == min(others, key=others.get)
    # The bracket must be a real spread, else the stack shows nothing new.
    assert others[picks[1]] > 2.0 * others[picks[2]]


def test_select_panel_cams_rejects_impossible_requests():
    t = ev.load_tracks()
    with pytest.raises(ValueError, match="only"):
        ev.select_panel_cams(t, n_extra=len(t["cam_names"]))


def test_all_camera_panels_share_one_zoom(restore_sys_path):
    """Every panel must be at the SAME px/mm.

    Sizing each panel to its own markers zoomed the cleanest camera to ~4x
    against the event camera's ~1x, which would have drawn its ~20 px miss
    LARGER on screen than the event camera's 122 px one -- a figure that
    inverts the comparison it exists to make. This is the honesty property of
    the stack, so it is asserted rather than left to inspection.
    """
    from scripts.viz.ik_explainer import recovery_clip as rc

    t = ev.load_tracks()
    cams = ev.select_panel_cams(t, n_extra=rc.N_EXTRA_CAMS)
    idx = [t["cam_names"].index(c) for c in cams]
    extents = [rc._crop_extent(t["det2d_all"][:, j], t["rep2d_all"][:, j],
                               rc.PANEL_W, rc.SUB_H) for j in idx]
    cw, ch = rc._shared_crop_size(extents, 1936, 448)

    # One size for all, and big enough that no camera's markers are cropped out.
    for (cx, cy, w, h), j, name in zip(extents, idx, cams):
        assert w <= cw + 1 and h <= ch + 1, f"{name} needs a bigger crop than shared"
        x0, y0 = rc._crop_origin(cx, cy, cw, ch, 1936, 448)
        for key in ("det2d_all", "rep2d_all"):
            p = np.asarray(t[key][:, j], np.float64) - [x0, y0]
            assert (p[:, 0] >= 0).all() and (p[:, 0] < cw).all(), f"{name} {key} x out of crop"
            assert (p[:, 1] >= 0).all() and (p[:, 1] < ch).all(), f"{name} {key} y out of crop"

    sx, sy = rc.PANEL_W / float(cw), rc.SUB_H / float(ch)
    assert abs(sx - sy) / max(sx, sy) <= 0.01
