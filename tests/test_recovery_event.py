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


def test_recovery_is_present_and_ordered():
    """raw spikes; filtered and IK do not. This is the claim the clip makes."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    ar, af, ai = (ev.accel(t[k]) for k in ("raw", "filt", "ik"))
    assert ar.max() > 1.0, f"raw peak accel {ar.max():.3f} — expected a spike"
    assert af.max() < 0.2, f"filtered peak accel {af.max():.3f} — not smooth"
    assert ai.max() < 0.2, f"IK peak accel {ai.max():.3f} — not smooth"
    assert ar.max() / af.max() > 10.0


def test_accel_is_zero_for_constant_velocity():
    t = np.arange(10)[:, None] * np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(ev.accel(t), 0.0, atol=1e-9)


def test_worst_axis_picks_the_largest_excursion():
    n = 20
    a = np.zeros((n, 3))
    a[10, 1] = 5.0                      # a big kick on y only
    assert ev.worst_axis(a) == 1
