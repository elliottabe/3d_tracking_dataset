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
