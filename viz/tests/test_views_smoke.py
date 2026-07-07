import os, numpy as np, pytest

# Must be set before the FIRST `import mujoco` anywhere in this process --
# mujoco resolves its GL backend once, at import time. `from viz.views import
# overlay, legskel` below transitively imports mujoco (viz.core.io ->
# stac_mjx.io_dict_to_hdf5 -> stac_mjx/__init__.py -> stac_mjx.viz), which
# would otherwise beat fit_check.py's own module-top env-var-setdefault (that
# module is only imported lazily, inside test_fit_check_writes_mp4) and lock
# in the wrong (non-EGL) backend for the whole test process.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from viz.views import overlay, legskel

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
skip = pytest.mark.skipif(not os.path.isdir(ROOT), reason="courtship run not present")

@skip
def test_overlay_writes_png(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.show="mesh,kp,mask,axis"; a.compare=None; a.bodyalign=False; a.out=str(tmp_path/"o.png")
    assert overlay.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


@skip
def test_overlay_out_of_range_frame_no_crash(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=999999; a.cams=None
    a.show="mesh,kp,mask,axis"; a.compare=None; a.bodyalign=False; a.out=str(tmp_path/"o_oor.png")
    assert overlay.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


@skip
def test_overlay_bodyalign_no_crash(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.show="mesh,kp,mask,axis"; a.compare=None; a.bodyalign=True; a.out=str(tmp_path/"o_bodyalign.png")
    assert overlay.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


@skip
def test_legskel_writes_png(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.compare=None; a.out=str(tmp_path/"l.png")
    assert legskel.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


@skip
def test_legskel_out_of_range_frame_no_crash(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=999999; a.cams=None
    a.compare=None; a.out=str(tmp_path/"l_oor.png")
    assert legskel.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


@skip
def test_legskel_compare_no_crash(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.compare=ROOT; a.out=str(tmp_path/"l_compare.png")
    assert legskel.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0


# --- reproj-video: reads a bout's per-fly dense 3D CSV + raw session mp4s
# (each ~9GB) and writes one reprojected mp4 per camera. To keep this test
# fast we render a single camera and cap at 2 frames; the core.io unit tests
# (test_io.py::test_load_data3d_csv / test_write_video_writes_nonzero_mp4)
# are the primary coverage for the new core functions this view is built on.
REPROJ_SESSION = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04"
REPROJ_PRED = os.path.join(REPROJ_SESSION, "Predictions_3D_36233268")
REPROJ_CAM = "Cam2012630"

_skip_reproj = pytest.mark.skipif(
    not (os.path.isdir(REPROJ_SESSION)
         and os.path.isfile(os.path.join(REPROJ_PRED, "bout_00001", "fly0.csv"))
         and os.path.isfile(os.path.join(REPROJ_SESSION, f"{REPROJ_CAM}.mp4"))),
    reason="reproj-video fixture (session mp4s + per-bout fly CSVs) not present",
)


@_skip_reproj
def test_reproj_video_writes_mp4(tmp_path):
    from viz.views import reproj_video
    class A: pass
    a = A()
    a.session_dir = REPROJ_SESSION; a.pred_dir = REPROJ_PRED; a.bout = 1
    a.cameras = [REPROJ_CAM]; a.with_masks = False; a.out = str(tmp_path)
    a.max_frames = 2
    assert reproj_video.run(a) == 0
    mp4s = list(tmp_path.glob("reproj_bout1_*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].stat().st_size > 0


@_skip_reproj
def test_reproj_video_with_masks_subset(tmp_path):
    """Exercises the previously-buggy path: rendering a ONE-camera subset of
    the full calibration camera list with --with-masks. Fix 1 requires masks
    to be loaded against the full calibration camera order and indexed by
    each camera's native (calibration) index, not its position within this
    single-camera rendered subset -- a subset of size 1 is exactly the case
    where "position in subset" (always 0) and "native index" (whatever
    REPROJ_CAM's real calibration slot is) are most likely to diverge. This
    can't assert visual mask correctness from here, but it must not crash
    (e.g. an out-of-range native index into the mask array) and must produce
    a real mp4.
    """
    from viz.views import reproj_video
    class A: pass
    a = A()
    a.session_dir = REPROJ_SESSION; a.pred_dir = REPROJ_PRED; a.bout = 1
    a.cameras = [REPROJ_CAM]; a.with_masks = True; a.out = str(tmp_path)
    a.max_frames = 2
    assert reproj_video.run(a) == 0
    mp4s = list(tmp_path.glob("reproj_bout1_*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].stat().st_size > 0


# --- clip: cut|stack|render, all pure cv2/numpy (no jax/mujoco), so these
# run on CPU regardless of GPU visibility. cut/render read the full session
# mp4s (~9GB each) but only a couple frames per camera; stack reads
# already-small pre-cut clips.
CLIP_SESSION = REPROJ_SESSION  # same session dir used by reproj-video fixtures
CLIP_STACK_DIR = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1/"
                   "2026_04_02_12_11_50/clips_125664_126200")

_skip_clip_cut = pytest.mark.skipif(
    not (os.path.isdir(CLIP_SESSION)
         and os.path.isfile(os.path.join(CLIP_SESSION, "Cam2012630.mp4"))),
    reason="clip cut fixture (session mp4) not present",
)
_skip_clip_stack = pytest.mark.skipif(
    not os.path.isdir(CLIP_STACK_DIR),
    reason="clip stack fixture (pre-cut clips dir) not present",
)
_skip_clip_render = pytest.mark.skipif(
    not (os.path.isdir(CLIP_SESSION)
         and os.path.isfile(os.path.join(CLIP_SESSION, "Cam2012630.mp4"))),
    reason="clip render fixture (session mp4) not present",
)


@_skip_clip_cut
def test_clip_cut(tmp_path):
    from viz.views import clip
    class A: pass
    a = A()
    a.mode = "cut"; a.session_dir = CLIP_SESSION; a.start = 0; a.end = 1
    a.cameras = ["Cam2012630"]; a.bout_dir = None; a.out = str(tmp_path); a.fps = 30
    assert clip.run(a) == 0
    mp4s = list(tmp_path.glob("Cam2012630_frames_*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].stat().st_size > 0


@_skip_clip_cut
def test_clip_cut_native_fps(tmp_path):
    """Regression test for the fix requiring `cut` (no --fps given) to write
    each camera's clip at ITS OWN native fps, probed from the source mp4 via
    cv2.CAP_PROP_FPS -- not a hardcoded 30. This rig captures at ~hundreds of
    fps, so a hardcoded 30 badly mislabels the cut clip's duration/speed."""
    import cv2
    from viz.views import clip
    class A: pass
    a = A()
    a.mode = "cut"; a.session_dir = CLIP_SESSION; a.start = 0; a.end = 1
    a.cameras = ["Cam2012630"]; a.bout_dir = None; a.out = str(tmp_path); a.fps = None
    assert clip.run(a) == 0
    mp4s = list(tmp_path.glob("Cam2012630_frames_*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].stat().st_size > 0

    src_cap = cv2.VideoCapture(os.path.join(CLIP_SESSION, "Cam2012630.mp4"))
    src_fps = src_cap.get(cv2.CAP_PROP_FPS)
    src_cap.release()
    out_cap = cv2.VideoCapture(str(mp4s[0]))
    out_fps = out_cap.get(cv2.CAP_PROP_FPS)
    out_cap.release()
    print(f"[test_clip_cut_native_fps] source fps={src_fps!r} output fps={out_fps!r}")
    assert src_fps > 0
    assert abs(out_fps - src_fps) <= 1.0


@_skip_clip_stack
def test_clip_stack(tmp_path):
    from viz.views import clip
    class A: pass
    a = A()
    a.mode = "stack"; a.session_dir = CLIP_STACK_DIR; a.cameras = None
    a.out = str(tmp_path / "stack.mp4"); a.fps = 30
    a.start = None; a.end = None; a.bout_dir = None
    assert clip.run(a) == 0
    out = tmp_path / "stack.mp4"
    assert out.exists() and out.stat().st_size > 0


@_skip_clip_render
def test_clip_render(tmp_path):
    from viz.views import clip
    class A: pass
    a = A()
    a.mode = "render"; a.session_dir = CLIP_SESSION; a.start = 0; a.end = 2
    a.cameras = ["Cam2012630", "Cam2012631"]; a.bout_dir = None
    a.out = str(tmp_path / "render.mp4"); a.fps = 30
    assert clip.run(a) == 0
    out = tmp_path / "render.mp4"
    assert out.exists() and out.stat().st_size > 0


# --- kp-qc: matplotlib + ViTPose model eval. Slow on CPU (JAX_PLATFORMS=cpu
# hides the GPU during the fast core suite), so only run it when a GPU is
# actually visible to JAX AND the checkpoint + data-root are present.
CKPT_RUN = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v4_kp_maskaware_fc"
DATA_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"


def _jax_has_gpu():
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


_skip_kp_qc = pytest.mark.skipif(
    not (os.path.isdir(CKPT_RUN) and os.path.isdir(DATA_ROOT) and _jax_has_gpu()),
    reason="kp-qc needs the v4 ckpt run, the V3 data-root, and a JAX-visible GPU",
)


@_skip_kp_qc
def test_kp_qc_writes_pngs(tmp_path):
    from viz.views import kp_qc
    class A: pass
    a = A()
    a.run_dir = CKPT_RUN; a.ckpt = None; a.recording = None
    a.n = 1; a.female_vs_male = False; a.out = str(tmp_path)
    a.data_root = DATA_ROOT
    assert kp_qc.run(a) == 0
    for name in ("viz_frames.png", "viz_perkp.png"):
        f = tmp_path / name
        assert f.exists() and f.stat().st_size > 0


# --- fit-check: mujoco EGL render on GPU. Only run it when a GPU is
# actually visible to JAX AND the fixture stac_ik.h5 is present.
IK_H5 = ("/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026/"
         "bouts/bout_00001/fly1/stac_ik.h5")

_skip_fit_check = pytest.mark.skipif(
    not (os.path.isfile(IK_H5) and _jax_has_gpu()),
    reason="fit-check needs the stac_ik.h5 fixture and a JAX-visible GPU",
)


@_skip_fit_check
def test_fit_check_writes_mp4(tmp_path):
    from viz.views import fit_check
    class A: pass
    a = A()
    a.ik_h5 = IK_H5; a.start = 0; a.n = 1; a.camera = "track1"; a.out = str(tmp_path)
    a.body_model_dir = "/home/eabe/Research/MyRepos/fruitfly_body_models"
    a.no_error = False; a.n_stills = 1
    assert fit_check.run(a) == 0
    mp4s = list(tmp_path.glob("fit_check_*.mp4"))
    assert len(mp4s) == 1 and mp4s[0].stat().st_size > 0
