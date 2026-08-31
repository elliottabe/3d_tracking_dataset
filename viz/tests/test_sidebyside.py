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


def test_run_default_path_unchanged_when_views_absent(tmp_path, monkeypatch):
    # --views is opt-in: an args object with no `views` attribute at all (the
    # shape every existing caller uses today, incl. the test above) must take
    # the ORIGINAL single-view branch, never the new multi-view one. Prove it
    # by making the multi-view path raise if it is ever reached, then check
    # the SAME exception/behaviour as test_sidebyside_missing_run_raises.
    def _boom(*a, **k):
        raise AssertionError("_run_multiview must not be called when --views is absent")
    monkeypatch.setattr(sidebyside, "_run_multiview", _boom)

    class A: pass
    a = A()
    a.run = str(tmp_path / "no_such_run"); a.bout = 1; a.fly = 0
    a.start = 0; a.n = 2; a.cams = None; a.camera = None
    a.conf = 0.3; a.panel_h = 480; a.fps = 30; a.out = str(tmp_path / "o.mp4")
    with pytest.raises((FileNotFoundError, OSError)):
        sidebyside.run(a)


def test_run_default_path_unchanged_when_views_falsy(tmp_path, monkeypatch):
    # Same guarantee for callers that DO set `.views` but leave it empty
    # (argparse default=None, or an explicit "") -- still must not dispatch.
    def _boom(*a, **k):
        raise AssertionError("_run_multiview must not be called when --views is falsy")
    monkeypatch.setattr(sidebyside, "_run_multiview", _boom)

    class A: pass
    for falsy in (None, ""):
        a = A()
        a.run = str(tmp_path / "no_such_run"); a.bout = 1; a.fly = 0
        a.start = 0; a.n = 2; a.cams = None; a.camera = None
        a.conf = 0.3; a.panel_h = 480; a.fps = 30; a.out = str(tmp_path / "o.mp4")
        a.views = falsy
        with pytest.raises((FileNotFoundError, OSError)):
            sidebyside.run(a)


def test_run_dispatches_to_multiview_only_when_views_set(tmp_path, monkeypatch):
    calls = []

    def _fake_multiview(args, views_arg):
        calls.append((args, views_arg))
        return 0

    monkeypatch.setattr(sidebyside, "_run_multiview", _fake_multiview)

    class A: pass
    a = A()
    a.run = str(tmp_path / "no_such_run"); a.bout = 1; a.fly = 0
    a.views = "left,top,right"
    assert sidebyside.run(a) == 0
    assert calls == [(a, "left,top,right")]


def test_resolve_multiview_cameras_accepts_roles_and_explicit_names(monkeypatch):
    from viz.core import rigviews
    RV = rigviews.RigView

    def _fake_classify(calib_dir, elevation_target=0.5):
        return {
            "top": RV("Cam0", (0, 0, -1), 90.0),
            "left": RV("Cam1", (0, -0.85, -0.52), 31.0),
            "right": RV("Cam2", (0, 0.88, -0.48), 29.0),
        }

    def _fake_view_directions(calib_dir):
        return {"Cam3": (0.0, -1.0, 0.0)}

    monkeypatch.setattr(rigviews, "classify_views", _fake_classify)
    monkeypatch.setattr(rigviews, "view_directions", _fake_view_directions)

    all_cameras = ["Cam0", "Cam1", "Cam2", "Cam3"]
    resolved = sidebyside._resolve_multiview_cameras("ignored", all_cameras, "left, top,Cam3")
    names = [n for n, _, _ in resolved]
    roles = [r for _, r, _ in resolved]
    assert names == ["Cam1", "Cam0", "Cam3"]
    assert roles == ["left", "top", "view"]


def test_resolve_multiview_cameras_rejects_unknown_token(monkeypatch):
    from viz.core import rigviews
    RV = rigviews.RigView
    monkeypatch.setattr(rigviews, "classify_views",
                        lambda calib_dir, elevation_target=0.5: {
                            "top": RV("Cam0", (0, 0, -1), 90.0),
                            "left": RV("Cam1", (0, -1, 0), 0.0),
                            "right": RV("Cam2", (0, 1, 0), 0.0),
                        })
    with pytest.raises(ValueError, match="neither a role"):
        sidebyside._resolve_multiview_cameras("ignored", ["Cam0", "Cam1", "Cam2"], "bogus")


def test_run_multiview_refuses_non_rigcam_right(tmp_path):
    # '--right rigcam' is the only view-matched multi-camera mode; 'mujoco'
    # renders an unrelated model-space camera and 'reproj' has no MuJoCo
    # render at all, so either would be misleading per-row. Must refuse
    # BEFORE touching any run/recording data (this call passes a bogus run).
    class A: pass
    a = A()
    a.run = str(tmp_path / "no_such_run"); a.bout = 1; a.fly = 0
    a.views = "left,top,right"; a.right = "mujoco"
    with pytest.raises(ValueError, match="requires --right rigcam"):
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
