import os, numpy as np, pytest
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
