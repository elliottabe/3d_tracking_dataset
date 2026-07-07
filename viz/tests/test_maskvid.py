import os, pytest

# viz.core.io -> stac_mjx transitively imports mujoco, which resolves its GL
# backend once at import time, so pin it up front (same rationale as
# test_sidebyside.py). Importing viz.views.maskvid itself is cheap -- its only
# heavy dependency (bout_start_frame) is imported lazily inside run().
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from viz.views import maskvid


def test_maskvid_importable_and_callable():
    assert callable(maskvid.run)


def test_maskvid_registered_in_cli():
    from viz import cli
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    assert "maskvid" in sub.choices
    ns = p.parse_args(["maskvid", "--run", "/r", "--bout", "2", "--n", "40",
                       "--n-cams", "2", "--cams", "CamA", "CamB", "--panel-h", "256"])
    assert ns.func == "_maskvid"
    assert ns.run == "/r" and ns.bout == 2 and ns.n == 40
    assert ns.n_cams == 2 and ns.cams == ["CamA", "CamB"] and ns.panel_h == 256


def test_maskvid_missing_run_raises(tmp_path):
    # Exercises run()'s early path (recording resolve -> load_masks) with a bogus
    # predictions dir: both flies' sam3_masks.npz are absent, so run() must fail
    # cleanly with FileNotFoundError before ever needing bout_start_frame or any
    # video decode.
    class A: pass
    a = A()
    a.run = str(tmp_path / "no_such_predictions"); a.bout = 1
    a.n = 2; a.n_cams = 3; a.cams = None; a.panel_h = 240; a.fps = 30
    a.out = str(tmp_path / "o.mp4")
    with pytest.raises(FileNotFoundError):
        maskvid.run(a)


# --- full render: raw session mp4s + a real bout's sam3_masks.npz. No GPU
# needed (masks are 2D; no jax/mujoco render). Runs only when the fixture
# predictions dir and its bout masks are actually present on the node.
RUN = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/"
       "Session0/2025_10_20_13_20_04/Predictions_3D_36233268_fixed")
BOUT = 1
_MASKS = os.path.join(RUN, f"bout_{BOUT:05d}", "sam3_masks.npz")

_skip_render = pytest.mark.skipif(
    not os.path.isfile(_MASKS),
    reason="maskvid render needs a real bout sam3_masks.npz fixture",
)


@_skip_render
def test_maskvid_writes_mp4(tmp_path):
    class A: pass
    a = A()
    a.run = RUN; a.bout = BOUT
    a.n = 8; a.n_cams = 2; a.cams = None; a.panel_h = 160; a.fps = 30
    a.out = str(tmp_path / "maskvid.mp4")
    assert maskvid.run(a) == 0
    assert os.path.exists(a.out) and os.path.getsize(a.out) > 0
    still = os.path.splitext(a.out)[0] + "_still.png"
    assert os.path.exists(still) and os.path.getsize(still) > 0
