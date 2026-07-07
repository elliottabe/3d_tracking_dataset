import os, pytest

# Must be set before the FIRST `import mujoco` anywhere in this process --
# mujoco resolves its GL backend once, at import time (same rationale as
# test_views_smoke.py). Importing viz.views.sidebyside is itself cheap (its
# Stac / bout_start_frame imports are lazy inside run()), but viz.core.io ->
# stac_mjx transitively imports mujoco, so pin the backend up front.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from viz.views import sidebyside


def test_sidebyside_importable_and_callable():
    assert callable(sidebyside.run)


def test_sidebyside_missing_run_raises(tmp_path):
    # Exercises run()'s early path (recording resolve -> load_kp2d) with a
    # bogus run root: it must fail cleanly on the missing artifact, never
    # reaching (or needing) the heavy jax/mujoco render path.
    class A: pass
    a = A()
    a.run = str(tmp_path / "no_such_run"); a.bout = 1; a.fly = 0
    a.start = 0; a.n = 2; a.cams = None; a.camera = None
    a.conf = 0.3; a.panel_h = 480; a.fps = 30; a.out = str(tmp_path / "o.mp4")
    with pytest.raises((FileNotFoundError, OSError)):
        sidebyside.run(a)


# --- full render: MuJoCo EGL render on GPU + raw session mp4s. Only run when a
# GPU is actually visible to JAX AND a run with both kp2d.npz and stac_ik.h5
# for the fixture (bout, fly) is present. Mirrors test_views_smoke.py's
# _skip_fit_check guard so CI (no GPU / no data) skips gracefully.
RUN = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07062026"
FLY_DIR = os.path.join(RUN, "bouts", "bout_00001", "fly0")


def _jax_has_gpu():
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


_skip_render = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(FLY_DIR, "kp2d.npz"))
         and os.path.isfile(os.path.join(FLY_DIR, "stac_ik.h5"))
         and _jax_has_gpu()),
    reason="sidebyside render needs kp2d.npz + stac_ik.h5 fixtures and a JAX-visible GPU",
)


@_skip_render
def test_sidebyside_writes_mp4(tmp_path):
    class A: pass
    a = A()
    a.run = RUN; a.bout = 1; a.fly = 0
    a.start = 0; a.n = 2; a.cams = None; a.camera = "track1"
    a.conf = 0.3; a.panel_h = 240; a.fps = 30
    a.out = str(tmp_path / "sbs.mp4")
    assert sidebyside.run(a) == 0
    assert os.path.exists(a.out) and os.path.getsize(a.out) > 0
    still = os.path.splitext(a.out)[0] + "_still.png"
    assert os.path.exists(still) and os.path.getsize(still) > 0
