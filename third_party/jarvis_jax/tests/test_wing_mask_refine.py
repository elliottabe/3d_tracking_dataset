"""Tests for the Adam refinement of WING PITCH against the SAM masks.

The fixtures build the REAL MuJoCo model, rasterise the model's own silhouette
at a KNOWN wing pitch to serve as the "mask", and feed an identity model->mm
bridge, so `test_it_recovers_a_known_pitch_offset` is a genuine round trip:
truth -> masks -> perturb -> refine -> back toward truth.

GPU-ONLY, deliberately. `mjx.kinematics` for this 87-joint / 68-body model is
pathologically slow to compile under the XLA CPU backend (measured: still
compiling after 20 min, vs 6.4 s on an L40S), so under `JAX_PLATFORMS=cpu` the
whole module skips rather than hanging a CPU test run. Every other module in
this feature is pure JAX and stays CPU-runnable; this one FKs a real model.
"""
from __future__ import annotations

import contextlib
import functools
import os
import types

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytestmark = pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="mjx.kinematics compile under the XLA CPU backend takes >20 min for "
           "this model (6.4 s on an L40S); run this module on a GPU node.")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy import ndimage

from jarvis_jax.tracking.appendage_dof import appendage_vertex_indices
from jarvis_jax.tracking.fk import load_anatomy, make_fk_repose
from jarvis_jax.tracking.mask_sdf import sdf_stack_from_masks
from jarvis_jax.tracking.wing_coverage import wing_target_points

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"

# configs/outputs/default.yaml uses a `basename` resolver; register it or compose fails.
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)


# --------------------------------------------------------------------------
# fixture machinery
# --------------------------------------------------------------------------
# Synthetic AFFINE cameras (the real rig's DLT matrices have 3rd row [0,0,0,1],
# verified, so the pipeline's projection really is uv = M @ X + t). Three views
# so wing pitch -- a rotation about the model's y axis -- is observed: the "top"
# view (x=X, y=Y) barely sees it, "side" (x=X, y=Z) and "front" (x=Y, y=Z) do.
_IMG_HW = (320, 320)
_PXU = 350.0                       # pixels per model unit
_CTR = 160.0
_CAM_MS = np.array([
    [[_PXU, 0.0, 0.0], [0.0, _PXU, 0.0]],     # top
    [[_PXU, 0.0, 0.0], [0.0, 0.0, _PXU]],     # side
    [[0.0, _PXU, 0.0], [0.0, 0.0, _PXU]],     # front
], np.float32)
_CAM_TS = np.full((3, 2), _CTR, np.float32)


@functools.lru_cache(maxsize=1)
def _anatomy():
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="pipeline", overrides=["paths=hyak"])
    anat = load_anatomy(cfg.anatomy.mjcf_path, cfg.anatomy.cse_mesh_npz)
    return cfg, anat, make_fk_repose(anat)


_CLOSE_PX = 4


def _rasterise(verts_mm, M, t):
    """SOLID silhouette of a vertex cloud under an affine camera.

    Closing must be generous. Seen face-on, the wing's 8004 vertices spread over
    ~5000 px, and a 2px closing left the blade SPECKLED -- measured 45% of the
    wing footprint inside the "mask", which is nothing like a SAM mask and made
    every coverage-target diagnostic unreadable. dilate(4)/fill/erode(4) gives
    100% on all three views.
    """
    uv = np.asarray(verts_mm) @ np.asarray(M).T + np.asarray(t)
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    ok = (x >= 0) & (x < _IMG_HW[1]) & (y >= 0) & (y < _IMG_HW[0])
    img = np.zeros(_IMG_HW, bool)
    img[y[ok], x[ok]] = True
    img = ndimage.binary_dilation(img, iterations=_CLOSE_PX)
    img = ndimage.binary_fill_holes(img)
    return ndimage.binary_erosion(img, iterations=_CLOSE_PX)


@functools.lru_cache(maxsize=4)
def _problem_arrays(n_frames, pitch_offset_deg, base_pitch):
    """Expensive, cacheable half of the fixture: q_true/q_pert + masks + SDF."""
    from jarvis_jax.tracking.wing_mask_refine import wing_pitch_dof_mask

    cfg, anat, fk = _anatomy()
    opt_mask = wing_pitch_dof_mask(anat["m"])
    pitch_adr = np.flatnonzero(opt_mask)          # never hardcode 9 / 12

    q_true = np.tile(np.asarray(anat["qpos0"], np.float32), (n_frames, 1))
    wobble = np.linspace(-0.05, 0.05, n_frames).astype(np.float32)
    q_true[:, pitch_adr[0]] = base_pitch + wobble
    q_true[:, pitch_adr[1]] = base_pitch - wobble

    C = _CAM_MS.shape[0]
    masks = np.zeros((n_frames, C) + _IMG_HW, bool)
    for t in range(n_frames):
        verts = np.asarray(fk(jnp.asarray(q_true[t])))         # all 139 353
        for c in range(C):
            masks[t, c] = _rasterise(verts, _CAM_MS[c], _CAM_TS[c])
    valid = np.ones((n_frames, C), bool)
    sdf, gs, go, present = sdf_stack_from_masks(masks, valid, out_hw=(128, 128))

    q_pert = q_true.copy()
    q_pert[:, pitch_adr] += np.float32(np.deg2rad(pitch_offset_deg))
    return q_true, q_pert, masks, sdf, gs, go, present


def _build_problem(*, n_frames=5, pitch_offset_deg=25.0, base_pitch=0.35,
                   n_steps=60, frame_chunk=None, lr=2e-2):
    from jarvis_jax.tracking.wing_mask_refine import (body_vertex_indices,
                                                      qpos_limits,
                                                      wing_pitch_dof_mask)
    cfg, anat, fk = _anatomy()
    mesh = cfg.anatomy.cse_mesh_npz
    q_true, q_pert, masks, sdf, gs, go, present = _problem_arrays(
        n_frames, pitch_offset_deg, base_pitch)
    lb, ub = qpos_limits(anat["m"])
    T = n_frames
    kw = dict(
        fk_repose=fk,
        wing_vert_idx=appendage_vertex_indices(mesh, subset="fps_300", include=("wing",)),
        body_vert_idx=body_vertex_indices(mesh, stride=40),
        cam_Ms=_CAM_MS, cam_ts=_CAM_TS,
        sdf=sdf, grid_scale=gs, grid_offset=go, present=present, masks=masks,
        bridge_s=np.ones(T, np.float32),
        bridge_R=np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy(),
        bridge_t=np.zeros((T, 3), np.float32),
        opt_mask=wing_pitch_dof_mask(anat["m"]), lb=lb, ub=ub,
        n_steps=n_steps, lr=lr, frame_chunk=frame_chunk or T,
    )
    return q_true, q_pert, kw


def _tiny_problem(n_frames=5, **kw):
    _, q_pert, kwargs = _build_problem(n_frames=n_frames, **kw)
    return q_pert, kwargs


def _synthetic_from_model(pitch_offset_deg=25.0):
    return _build_problem(n_frames=3, pitch_offset_deg=pitch_offset_deg,
                          n_steps=200, lr=1e-2)


@contextlib.contextmanager
def _count_traces(mod, name):
    """Count Python-level invocations of `mod.name`. Under `jax.jit` that is
    exactly the number of TRACES: jit runs the Python body once per new input
    signature and replays the compiled executable otherwise."""
    orig = getattr(mod, name)
    box = types.SimpleNamespace(value=0)

    @functools.wraps(orig)
    def counting(*a, **k):
        box.value += 1
        return orig(*a, **k)

    setattr(mod, name, counting)
    try:
        yield box
    finally:
        setattr(mod, name, orig)


# --------------------------------------------------------------------------
# the load-bearing guarantees
# --------------------------------------------------------------------------
def test_only_the_masked_dofs_move():
    """With opt_mask limited to wing pitch, every other qpos entry must come
    back bit-identical."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    moved = np.abs(q1 - q0).max(axis=0) > 1e-9
    assert moved[kw["opt_mask"]].any(), "the optimised DOFs should move"
    assert not moved[~np.asarray(kw["opt_mask"])].any(), "everything else must be frozen"


def test_joint_limits_are_respected():
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    m = np.asarray(kw["opt_mask"])
    assert (q1[:, m] >= np.asarray(kw["lb"])[m] - 1e-6).all()
    assert (q1[:, m] <= np.asarray(kw["ub"])[m] + 1e-6).all()


def test_a_frame_with_no_present_camera_is_left_at_q_init():
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    kw["present"] = np.zeros_like(kw["present"])
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    assert np.allclose(q1, q0), "no evidence must mean no change"


def test_it_recovers_a_known_pitch_offset():
    """Synthesise masks FROM the model at a known pitch, perturb pitch, and
    check the refinement moves back toward the truth. The only test that shows
    the objective points the right way."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0)
    q1 = np.asarray(refine_wing_pitch(q_pert, **kw))
    m = np.asarray(kw["opt_mask"])
    err0 = np.abs(q_pert[:, m] - q_true[:, m]).mean()
    err1 = np.abs(q1[:, m] - q_true[:, m]).mean()
    assert err1 < 0.5 * err0, f"pitch error {err0:.3f} -> {err1:.3f} rad"


# --------------------------------------------------------------------------
# structural properties that stand in for a (flaky) wall-clock assert
# --------------------------------------------------------------------------
def test_the_step_loop_compiles_once_not_per_chunk():
    """A re-trace per FRAME chunk is the difference between 40 s and 20 min."""
    from jarvis_jax.tracking import wing_mask_refine as R
    q0, kw = _tiny_problem(n_frames=40, n_steps=3)   # 3 chunks at frame_chunk=16
    kw["frame_chunk"] = 16
    with _count_traces(R, "_refine_chunk") as n:
        R.refine_wing_pitch(q0, **kw)
    assert n.value == 1, f"retraced {n.value}x -- pad chunks to a constant shape"


def test_only_the_wing_vertices_are_fk_d():
    """FK over all 139 353 vertices per step per frame is a ~1400x waste."""
    _, kw = _tiny_problem()
    assert len(kw["wing_vert_idx"]) < 400, "wing selection must be the fps subset"


def test_chunked_and_unchunked_agree():
    """Frame chunking is a memory device, not a change of objective: the padded
    last chunk and the one-frame smoothness overlap must reproduce the
    single-chunk answer."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem(n_frames=5, n_steps=20)
    whole = np.asarray(refine_wing_pitch(q0, **kw))
    kw["frame_chunk"] = 2                            # 2 + 2 + 1(padded)
    chunked = np.asarray(refine_wing_pitch(q0, **kw))
    np.testing.assert_allclose(chunked, whole, atol=2e-3)


# --------------------------------------------------------------------------
# the index-space traps this repo keeps getting bitten by
# --------------------------------------------------------------------------
def test_the_pitch_mask_is_built_by_joint_name_and_is_pitch_only():
    """qposadr 9/12 are wing_pitch_left/right in THIS model, but the mask must
    be derived from the names -- yaw carries the courtship song and roll is the
    best-observed wing DOF; neither may move."""
    import mujoco
    from jarvis_jax.tracking.wing_mask_refine import wing_pitch_dof_mask
    _, anat, _ = _anatomy()
    mj = anat["m"]
    m = np.asarray(wing_pitch_dof_mask(mj))
    sel = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(mj.njnt)
           if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
           and m[int(mj.jnt_qposadr[j])]]
    assert sorted(sel) == ["wing_pitch_left", "wing_pitch_right"], sel
    assert int(m.sum()) == 2
    assert not m[:7].any(), "the free root joint must never be selected"


def test_a_model_without_the_pitch_joints_raises_instead_of_silently_freezing():
    """A renamed or swapped body model must fail loudly, not quietly refine
    nothing -- an all-False opt_mask returns q_init and every downstream metric
    reads 'no change' rather than 'broken'."""
    import mujoco
    from jarvis_jax.tracking.wing_mask_refine import wing_pitch_dof_mask
    mj = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body>"
        "<joint name='elbow' type='hinge'/><geom type='sphere' size='.1'/>"
        "</body></worldbody></mujoco>")
    with pytest.raises(ValueError, match="wing_pitch"):
        wing_pitch_dof_mask(mj)


def test_camera_matrices_are_selected_by_name_not_by_glob_position():
    """THE camera-order trap. kp2d/masks are canonicalised BY NAME by
    load_bout_masks(expected_cameras=...); the camera matrices must be selected
    by the SAME names, or one camera's wing is scored against another camera's
    mask and the residual still looks plausible."""
    from jarvis_jax.tracking.wing_mask_refine import affine_cameras_by_name
    cfg, _, _ = _anatomy()
    names = list(cfg.recording.cameras)
    Ms, ts = affine_cameras_by_name(cfg.recording.calib_dir, names)
    assert Ms.shape == (len(names), 2, 3) and ts.shape == (len(names), 2)

    shuffled = list(reversed(names))
    Ms_s, ts_s = affine_cameras_by_name(cfg.recording.calib_dir, shuffled)
    np.testing.assert_array_equal(Ms_s, Ms[::-1])
    np.testing.assert_array_equal(ts_s, ts[::-1])

    with pytest.raises(ValueError, match="CamNotReal"):
        affine_cameras_by_name(cfg.recording.calib_dir, names[:2] + ["CamNotReal"])


# --------------------------------------------------------------------------
# the target sampler's call-site contract
# --------------------------------------------------------------------------
def test_cropped_target_sampling_matches_the_full_frame_call():
    """Targets are sampled on the mask's bbox crop (padded by dilate_px) purely
    for speed -- 448x1936 full frames would cost ~10x. It must be the same
    answer, in ORIGINAL pixels."""
    from jarvis_jax.tracking.wing_mask_refine import _frame_camera_targets
    rng = np.random.default_rng(0)
    mask = np.zeros((200, 400), bool)
    mask[60:130, 150:300] = True
    body_uv = np.stack([rng.uniform(140, 250, 200), rng.uniform(55, 125, 200)], 1)
    got = _frame_camera_targets(mask, body_uv, n_points=64, dilate_px=3, seed=7)
    want = wing_target_points(mask, body_uv, n_points=64, dilate_px=3, seed=7)
    np.testing.assert_array_equal(got, want)
    fin = np.isfinite(got).all(1)
    assert fin.sum() > 0
    assert mask[got[fin, 1].astype(int), got[fin, 0].astype(int)].all()


def test_nonfinite_body_vertices_are_dropped_before_sampling():
    """Real projections produce NaN for occluded / behind-camera points, and
    `np.round(NaN).astype(int64)` is implementation-defined. Prefilter, and if
    EVERY body vertex is non-finite treat the whole mask as unexplained."""
    from jarvis_jax.tracking.wing_mask_refine import _frame_camera_targets
    mask = np.zeros((80, 80), bool)
    mask[20:60, 20:60] = True
    body = np.array([[30.0, 30.0], [np.nan, np.nan], [40.0, 40.0]])
    clean = np.array([[30.0, 30.0], [40.0, 40.0]])
    np.testing.assert_array_equal(
        _frame_camera_targets(mask, body, n_points=32, dilate_px=2, seed=1),
        _frame_camera_targets(mask, clean, n_points=32, dilate_px=2, seed=1))

    allnan = np.full((4, 2), np.nan)
    got = _frame_camera_targets(mask, allnan, n_points=32, dilate_px=2, seed=1)
    fin = np.isfinite(got).all(1)
    assert fin.all(), "no body evidence -> every mask pixel is unexplained"
    assert mask[got[:, 1].astype(int), got[:, 0].astype(int)].all()


# --------------------------------------------------------------------------
# NaN frames -- found by running on the real bout, not by the synthetic fixture
# --------------------------------------------------------------------------
def test_a_nan_frame_does_not_poison_the_rest_of_its_chunk():
    """Session0 bout 28 fly0 has 483 of 2007 qpos rows ENTIRELY non-finite (the
    frames STAC could not solve; `bridge_ok` false marks them). They share a
    frame chunk with good frames, and a single NaN in the summed objective
    makes the gradient NaN for every frame in that chunk -- measured: the whole
    bout came back NaN. The NaN rows must be excluded from the cost and handed
    back untouched."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem(n_frames=5, n_steps=20)
    q0 = q0.copy()
    q0[2] = np.nan                                   # mid-chunk, as in real data
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    good = [0, 1, 3, 4]
    assert np.isfinite(q1[good]).all(), "a NaN frame poisoned its neighbours"
    m = np.asarray(kw["opt_mask"])
    assert (np.abs(q1[good][:, m] - q0[good][:, m]).max() > 1e-9), \
        "the good frames should still be refined"
    assert np.isnan(q1[2]).all(), "the NaN frame must come back as NaN, not invented"


def test_a_nonfinite_bridge_frame_is_excluded_too():
    """`bridge_ok=False` frames can carry a non-finite bridge; the model->mm map
    is applied before projection, so one NaN there is the same poison."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem(n_frames=5, n_steps=20)
    kw["bridge_t"] = np.array(kw["bridge_t"])
    kw["bridge_t"][1] = np.nan
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    good = [0, 2, 3, 4]
    assert np.isfinite(q1[good]).all()
    np.testing.assert_array_equal(q1[1], q0[1])      # left exactly at q_init
    m = np.asarray(kw["opt_mask"])
    assert np.abs(q1[good][:, m] - q0[good][:, m]).max() > 1e-9
