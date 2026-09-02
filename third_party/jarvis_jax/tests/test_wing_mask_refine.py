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


@functools.lru_cache(maxsize=6)
def _problem_arrays(n_frames, pitch_offset_deg, base_pitch, rest):
    """Expensive, cacheable half of the fixture: q_true/q_pert + masks + SDF.

    `rest=True` puts wing YAW and ROLL at the model's spring reference
    (`qpos_spring`: yaw +1.5 == the joint's upper stop, roll +0.7) instead of
    qpos0's zeros. That is the regime the defect actually lives in -- wings
    folded back over the abdomen, 48-80% of the blade hidden inside the body
    silhouette, so pitch swings it THROUGH the body. At qpos0 the wings are
    extended and pitch merely twists a blade that already sticks out, which is
    the flattering case.
    """
    import mujoco
    from jarvis_jax.tracking.wing_mask_refine import wing_pitch_dof_mask

    cfg, anat, fk = _anatomy()
    opt_mask = wing_pitch_dof_mask(anat["m"])
    pitch_adr = np.flatnonzero(opt_mask)          # never hardcode 9 / 12

    q_true = np.tile(np.asarray(anat["qpos0"], np.float32), (n_frames, 1))
    if rest:
        mj = anat["m"]
        for j in range(mj.njnt):
            nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm and ("wing_yaw" in nm or "wing_roll" in nm):
                a = int(mj.jnt_qposadr[j])
                q_true[:, a] = mj.qpos_spring[a]
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


#: wing pitch at the model's spring reference, -1.0 rad = -57.3 deg -- the
#: "rest (-57 deg)" figure the plan's acceptance criterion 5 names.
_REST_PITCH = -1.0


def _build_problem(*, n_frames=5, pitch_offset_deg=25.0, base_pitch=0.35,
                   n_steps=60, frame_chunk=None, lr=2e-2, rest=False):
    from jarvis_jax.tracking.wing_mask_refine import (body_vertex_indices,
                                                      qpos_limits,
                                                      wing_pitch_dof_mask)
    cfg, anat, fk = _anatomy()
    mesh = cfg.anatomy.cse_mesh_npz
    if rest:
        base_pitch = _REST_PITCH
    q_true, q_pert, masks, sdf, gs, go, present = _problem_arrays(
        n_frames, pitch_offset_deg, base_pitch, rest)
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


def _synthetic_from_model(pitch_offset_deg=25.0, rest=False):
    return _build_problem(n_frames=3, pitch_offset_deg=pitch_offset_deg,
                          n_steps=200, lr=1e-2, rest=rest)


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
    """Includes a frame whose STAC pitch is OUTSIDE the joint range. The final
    hard clamp must be gated on having evidence too, not on opt_mask alone --
    otherwise a no-evidence frame is silently pulled to the joint stop, which
    is a change, and "no evidence must mean no change" is violated by the one
    line that runs after the optimiser."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    q0 = q0.copy()
    ub = np.asarray(kw["ub"])[np.asarray(kw["opt_mask"])]
    q0[1, np.flatnonzero(kw["opt_mask"])] = ub + 0.6      # out of range, no evidence
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


def test_the_coverage_term_alone_points_the_right_way():
    """The combined test above passes at ratio 0.25 while coverage does net
    harm, so it would keep passing if coverage pointed almost anywhere. Pin the
    coverage term's DIRECTION on its own: containment off, coverage must still
    move pitch most of the way back to truth."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0)
    q1 = np.asarray(refine_wing_pitch(q_pert, **dict(kw, containment_weight=0.0)))
    m = np.asarray(kw["opt_mask"])
    err0 = np.abs(q_pert[:, m] - q_true[:, m]).mean()
    err1 = np.abs(q1[:, m] - q_true[:, m]).mean()
    assert err1 < 0.5 * err0, f"coverage-only pitch error {err0:.3f} -> {err1:.3f} rad"


def test_it_recovers_a_known_pitch_offset_at_the_springref_rest_attitude():
    """The HARD regime, and the one the defect lives in: wings folded back at
    the spring reference, blade largely hidden inside the body silhouette. The
    extended-wing fixture cannot exercise the containment degeneracy at all."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0, rest=True)
    q1 = np.asarray(refine_wing_pitch(q_pert, **kw))
    m = np.asarray(kw["opt_mask"])
    err0 = np.abs(q_pert[:, m] - q_true[:, m]).mean()
    err1 = np.abs(q1[:, m] - q_true[:, m]).mean()
    assert err1 < 0.5 * err0, f"rest-attitude pitch error {err0:.3f} -> {err1:.3f} rad"


def test_at_the_rest_attitude_the_coverage_term_earns_its_place():
    """The cell the sweep left unpinned, and the ONLY measured justification for
    shipping `coverage_weight > 0` at all.

    At the springref REST attitude 48-80% of the blade is hidden inside the body
    silhouette. That is where containment's documented degeneracy (minimise it
    by tucking the wing further IN) and the coverage term's bias meet, so it is
    the cell that decides whether the second term is worth its cost.

    MEASURED 2026-09-01 at the SHIPPED `huber_delta=8.0`, from a 25 deg
    perturbation (docs/benchmark/2026-09-01-wing-mask-fit/):

        both terms         1.14 deg
        containment only   2.49 deg
        coverage only     43.49 deg   <- ALONE, 18 deg WORSE than doing nothing

    So the term is neither "insurance" nor "a tie": it is a COMPLEMENT that
    halves the error here, and it must never be relied on by itself. Only that
    load-bearing half is asserted. It matters because on the real bout the term
    is INERT -- Session0 bout 28 fly0 and fly1 give the same pose to three
    significant figures at coverage 0.0 and 0.3 -- so without this cell there is
    no measurement anywhere that distinguishes the two.

    `huber_delta` is passed explicitly: the module default is 0.0, at which the
    chamfer is an unrobustified L2 and this comparison is 1.63 vs 2.49 rather
    than 1.14 vs 2.49.
    """
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0, rest=True)
    m = np.asarray(kw["opt_mask"])
    err = {}
    for tag, over in (("both", dict(huber_delta=8.0)),
                      ("containment", dict(huber_delta=8.0, coverage_weight=0.0))):
        q1 = np.asarray(refine_wing_pitch(q_pert, **dict(kw, **over)))
        err[tag] = float(np.abs(q1[:, m] - q_true[:, m]).mean())
    assert err["both"] < 0.8 * err["containment"], (
        f"at the springref rest attitude the coverage term must IMPROVE on "
        f"containment alone -- that is the whole reason it ships at a non-zero "
        f"weight, since on the real bout it changes nothing: "
        f"both {np.degrees(err['both']):.2f} deg vs containment-only "
        f"{np.degrees(err['containment']):.2f} deg")


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


def test_the_cost_loop_fks_only_the_wing_vertex_subset():
    """Actually exercises refine_wing_pitch, unlike the fixture-only assertion
    this replaces. FK inside the Adam loop must be over the wing subset, and the
    full 139 353-vertex array must never be reached. NOTE the saving is the
    gather and the projection, NOT the kinematics: fk_repose runs full
    mjx.kinematics whatever `indices` says."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem(n_frames=3, n_steps=2)
    inner = kw["fk_repose"]
    seen = []

    def recording(qpos, scale=1.0, indices=None):
        seen.append(None if indices is None else int(np.shape(indices)[0]))
        return inner(qpos, scale, indices)

    kw["fk_repose"] = recording
    refine_wing_pitch(q0, **kw)
    assert seen, "refine_wing_pitch never called fk_repose"
    assert None not in seen, "FK was run over the FULL vertex array"
    assert set(seen) == {len(kw["wing_vert_idx"]), len(kw["body_vert_idx"])}, seen


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


# --------------------------------------------------------------------------
# param_mode: the SMOOTH-CORRECTION redesign
# --------------------------------------------------------------------------
# Measured on Session0 bout 28 fly1 (spec section 8.2): the marker-only control
# carries bilaterally phase-locked courtship song in wing PITCH -- L/R coherence
# 0.852 against a 0.35 significance floor on epoch 2, a clean periodic
# cross-correlation at a 6.4-FRAME period -- and the free per-frame fit replaces
# it with its own mask noise (coherence 0.038-0.139, correlation with the
# control's own hp(pitch) 0.017). `smooth_weight` cannot fix that: it is one
# scalar against a per-frame-independent objective, and both ends of the
# measured sweep destroy the lock.
#
# The fix is to stop the stage carrying high-frequency content AT ALL, and the
# tests below pin that BY CONSTRUCTION rather than by a threshold:
#   1. the knot basis has no power in the song band  (pure numpy, no fit), and
#   2. the correction a spline-mode fit produces lies in that basis' span.
# Together those two are the proof; neither alone is.
_SONG_PERIOD_FRAMES = 6.4          # measured; fs-independent form of f0
_HP_NYQ = 0.1                      # the scorer's own high pass: 40 Hz at fs 800


def _hp_fraction(x, wn=_HP_NYQ):
    """Fraction of a signal's variance above the scorer's 40 Hz high pass.

    Same 2nd-order Butterworth `wing_mask_fit_ab.hp_filt` uses, so a number here
    means the same thing it means in the acceptance report.
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(2, wn, btype="high")
    x = np.asarray(x, float)
    x = x - x.mean()
    return float(np.var(filtfilt(b, a, x)) / max(np.var(x), 1e-30))


def test_the_knot_basis_cannot_represent_the_song():
    """HALF ONE of the by-construction proof, and it needs no optimiser.

    A piecewise-linear basis with knots every 32 frames is an order of magnitude
    below the 6.4-frame song period in frequency, so NO knot vector -- however
    the optimiser chooses it -- can put power in the song band. The K = 2
    control is what shows the test measures the basis and not the random draw.
    """
    from jarvis_jax.tracking.wing_mask_refine import knot_basis

    rng = np.random.default_rng(0)
    F = 512
    frac = {}
    for K in (32, 2):
        B = knot_basis(F, K)
        assert B.shape[0] == F
        np.testing.assert_allclose(B.sum(axis=1), 1.0, atol=1e-6)   # partition of unity
        frac[K] = float(np.median([
            _hp_fraction(B @ rng.standard_normal(B.shape[1])) for _ in range(64)]))
    assert frac[32] < 1e-3, (
        f"a 32-frame knot basis puts {frac[32]:.2e} of its variance above the "
        f"40 Hz high pass -- it can carry the {_SONG_PERIOD_FRAMES}-frame song")
    assert frac[2] > 100 * frac[32], (
        f"the K=2 control must be able to carry it: {frac[2]:.2e} vs {frac[32]:.2e}")


def test_spline_mode_confines_the_correction_to_a_GLOBAL_knot_basis():
    """HALF TWO, and it also pins chunk-to-chunk CONTINUITY.

    The claim is not "each chunk is smooth" -- three independently-splined
    chunks stitched end to end have a STEP at every boundary, and a step is
    broadband, so it would put the song band straight back. The claim is that
    the correction over the WHOLE trajectory lies in the span of ONE global
    piecewise-linear basis with knots every K frames. That holds only if each
    chunk's first knot is anchored to the previous chunk's last.

    The free arm is run on the same problem as the control: without it, a
    fixture whose truth happens to be linear in t would let free mode pass too.
    """
    from jarvis_jax.tracking.wing_mask_refine import knot_basis, refine_wing_pitch
    T, K = 24, 4
    q0, kw = _tiny_problem(n_frames=T, n_steps=30)
    kw["frame_chunk"] = 8                       # 3 chunks, 2 segments each
    m = np.flatnonzero(np.asarray(kw["opt_mask"]))

    Bg = knot_basis(T, K)                       # knots at 0,4,...,24 -- global
    P = Bg @ np.linalg.pinv(Bg)                 # projector onto the basis span

    resid = {}
    for mode in ("spline", "free"):
        q1 = np.asarray(refine_wing_pitch(q0, param_mode=mode, knot_spacing=K, **kw))
        d = (q1 - q0)[:, m]
        assert np.abs(d).max() > 1e-3, f"{mode}: the fit did not move pitch at all"
        resid[mode] = float(np.abs(d - P @ d).max() / np.abs(d).max())
    assert resid["spline"] < 1e-4, (
        f"the spline correction leaves the {K}-frame knot span by "
        f"{resid['spline']:.2e} of its own amplitude -- either the basis is not "
        f"applied or the chunk boundaries are not anchored")
    assert resid["free"] > 100 * resid["spline"], (
        f"free mode must NOT lie in the span, or this test proves nothing: "
        f"free {resid['free']:.2e} vs spline {resid['spline']:.2e}")


def test_spline_mode_still_recovers_a_known_pitch_offset():
    """PLACEMENT must survive the reparameterisation, at BOTH attitudes.

    "Preserves the song but no longer fixes the wings" is a failure, not a
    success, so the round trip that proves the objective points the right way is
    re-run in the new mode -- including the springref REST attitude, the regime
    the defect actually lives in.
    """
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    for rest in (False, True):
        q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0, rest=rest)
        m = np.asarray(kw["opt_mask"])
        q1 = np.asarray(refine_wing_pitch(
            q_pert, param_mode="spline", knot_spacing=len(q_pert), **kw))
        err0 = np.abs(q_pert[:, m] - q_true[:, m]).mean()
        err1 = np.abs(q1[:, m] - q_true[:, m]).mean()
        assert err1 < 0.5 * err0, (
            f"spline mode at the {'rest' if rest else 'extended'} attitude: "
            f"pitch error {np.degrees(err0):.2f} -> {np.degrees(err1):.2f} deg")


def test_lowpass_mode_leaves_q_inits_high_frequency_content_alone():
    """Option B, the honest control for A: `q_out = q_init + lowpass(q_fit-q_init)`.

    Whatever the free solve does, the high-frequency content of the RESULT is
    the high-frequency content of the STAC pose it started from. Exact for an
    ideal filter; for the shipped 4th-order Butterworth at a cutoff 3.2x below
    the scorer's high pass the leak is asserted here at < 2% of the song's own
    amplitude -- which is what "by construction" is worth in practice.
    """
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    from scipy.signal import butter, filtfilt
    T = 64
    q0, kw = _tiny_problem(n_frames=T, n_steps=20)
    m = np.flatnonzero(np.asarray(kw["opt_mask"]))
    # a synthetic "song": 6.4-frame period, 1 deg, on the pitch DOFs of q_init
    t = np.arange(T)
    q0 = q0.copy()
    q0[:, m] += np.deg2rad(1.0) * np.sin(2 * np.pi * t / _SONG_PERIOD_FRAMES)[:, None]

    q1 = np.asarray(refine_wing_pitch(q0, param_mode="lowpass", knot_spacing=32, **kw))
    b, a = butter(2, _HP_NYQ, btype="high")
    hp0 = filtfilt(b, a, q0[:, m], axis=0)
    hp1 = filtfilt(b, a, q1[:, m], axis=0)
    leak = float(np.abs(hp1 - hp0).max() / np.abs(hp0).max())
    assert np.abs(q1[:, m] - q0[:, m]).max() > 1e-3, "the fit did not move pitch"
    assert leak < 0.02, (
        f"lowpass mode changed q_init's high-frequency content by {leak:.3f} of "
        f"its amplitude -- the correction is not band-limited")


def test_free_is_the_default_and_the_new_knobs_do_not_touch_it():
    """The negative result was measured on `free`; it must stay reproducible.

    BIT-IDENTITY IS NOT AVAILABLE and asserting it would be a false claim: two
    IDENTICAL calls to `refine_wing_pitch` on this L40S already differ by
    5.96e-08 rad on 2 of 558 entries (XLA reduction non-determinism in float32).
    So the guard is "no further from the default than the default is from
    itself" -- which is still five orders of magnitude tighter than any real
    change of parameterisation, whose corrections are ~1e-2 rad.
    """
    import inspect
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    sig = inspect.signature(refine_wing_pitch)
    assert sig.parameters["param_mode"].default == "free"
    q0, kw = _tiny_problem(n_frames=6, n_steps=10)
    a = np.asarray(refine_wing_pitch(q0, **kw))
    a2 = np.asarray(refine_wing_pitch(q0, **kw))          # the noise floor itself
    b = np.asarray(refine_wing_pitch(q0, param_mode="free", knot_spacing=3, **kw))
    noise = float(np.abs(a - a2).max())
    delta = float(np.abs(a - b).max())
    assert delta <= max(noise, 1e-6), (
        f"param_mode='free' moved the fit by {delta:.3e} rad against a "
        f"run-to-run float32 noise floor of {noise:.3e} -- the arm the measured "
        f"negative result was taken on is no longer reproducible")


def test_an_unknown_param_mode_raises():
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem(n_frames=4, n_steps=2)
    with pytest.raises(ValueError, match="param_mode"):
        refine_wing_pitch(q0, param_mode="smooth", **kw)


def test_the_spline_step_loop_also_compiles_once():
    """Same guarantee as free mode: a re-trace per chunk is 40 s vs 20 min."""
    from jarvis_jax.tracking import wing_mask_refine as R
    q0, kw = _tiny_problem(n_frames=40, n_steps=3)
    kw["frame_chunk"] = 16
    with _count_traces(R, "_refine_chunk") as n:
        R.refine_wing_pitch(q0, param_mode="spline", knot_spacing=8, **kw)
    assert n.value == 1, f"retraced {n.value}x -- pad chunks to a constant shape"


# --------------------------------------------------------------------------
# Task 10: the validity gate -- skipping badly-tracked frames is a FEATURE
# --------------------------------------------------------------------------
def test_the_frame_gate_HOLDS_skipped_frames_at_q_init_in_a_BAND_LIMITED_mode():
    """THE reason `frame_keep` exists, and it is not the same as `present`.

    Zeroing a frame's `present` row freezes it only in `free` mode, where the
    frame gate lives on the parameter UPDATE. In `spline`/`lowpass` one knot
    spans many frames, so the knots INTERPOLATE across a no-evidence frame and
    the low-pass smears across it -- measured, that is exactly what happens on
    Session0 bout 28 fly0's collapsed masks, where every band-limited arm turns
    a sliver-driven error into a smooth 130 deg swing.

    The `present`-only arm below is the control: if it also came back at q_init
    this test would prove nothing, because the gate would be redundant.
    """
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    T, K = 24, 4
    q0, kw = _tiny_problem(n_frames=T, n_steps=30)
    kw["frame_chunk"] = 8
    m = np.flatnonzero(np.asarray(kw["opt_mask"]))
    skip = np.zeros(T, bool)
    skip[10:14] = True                       # a 4-frame "bad tracking" stretch

    kw_ng = dict(kw)
    kw_ng["present"] = np.asarray(kw["present"], bool).copy()
    kw_ng["present"][skip] = False           # the remedy that CANNOT work alone
    q_present_only = np.asarray(refine_wing_pitch(
        q0, param_mode="spline", knot_spacing=K, **kw_ng))
    interp = np.abs(q_present_only[skip][:, m] - q0[skip][:, m]).max()
    assert interp > 1e-3, (
        f"the `present`-only control moved the gated frames by only {interp:.2e} "
        f"rad -- if zeroing `present` already froze them the gate would be "
        f"redundant and this test would prove nothing")

    q_gated = np.asarray(refine_wing_pitch(
        q0, param_mode="spline", knot_spacing=K, frame_keep=~skip, **kw_ng))
    assert np.array_equal(q_gated[skip], q0[skip]), (
        "a gated frame must come back at EXACTLY its STAC pose, not decayed "
        "toward it")
    assert np.abs(q_gated[~skip][:, m] - q0[~skip][:, m]).max() > 1e-3, \
        "the kept frames must still be fitted"


def test_the_gate_envelope_ramps_rather_than_stepping_in_a_band_limited_mode():
    """A hard 0/1 gate is BROADBAND -- exactly the content `spline`/`lowpass`
    exist to keep out of the 6.4-frame song band. The envelope returns to 1 over
    one knot spacing, so the fastest slope the gate can introduce is 1/K of the
    correction's amplitude per frame.

    THIS IS ASSERTED AS THE ENVELOPE'S SHAPE, not as a step bound, and the
    difference was found by mutation: with the gate deleted entirely, a
    `max |diff| <= amp/K` assertion still PASSED, because a K=8 spline
    correction is already that smooth on its own. What only the envelope can
    produce is a correction that is exactly 0 at the skipped frame and
    ATTENUATED in proportion to the distance from it -- 1/8 of the local
    amplitude one frame out, 1/2 four frames out, back to full at K.
    """
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    T, K = 48, 8
    q0, kw = _tiny_problem(n_frames=T, n_steps=20)
    kw["frame_chunk"] = 16
    m = np.flatnonzero(np.asarray(kw["opt_mask"]))
    t0 = 24
    skip = np.zeros(T, bool)
    skip[t0] = True

    q1 = np.asarray(refine_wing_pitch(q0, param_mode="spline", knot_spacing=K,
                                      frame_keep=~skip, **kw))
    d = np.abs((q1 - q0)[:, m]).mean(axis=1)          # (T,) amplitude per frame
    full = d[t0 + K]
    assert full > 1e-3, "the fit did not move pitch"
    assert d[t0] == 0.0
    step = np.abs(np.diff(d)).max()
    assert step <= d.max() / K * 1.05 + 1e-6, (
        f"the gated correction steps by {step:.3e} rad in one frame against an "
        f"amplitude of {d.max():.3e} over a {K}-frame ramp")
    # the ramp itself: env(t0+j) = j/K, and the underlying spline correction is
    # near-constant over 8 frames, so the RATIO to the full value tracks it
    for j, want in ((1, 1 / K), (4, 4 / K)):
        got = d[t0 + j] / full
        assert abs(got - want) < 0.25, (
            f"{j} frames from the skipped frame the correction is {got:.3f} of "
            f"its full value; a {K}-frame linear ramp wants {want:.3f} "
            f"(a hard gate would give 1.0, no gate at all would give ~1.0)")


def test_a_hard_dpitch_bound_no_evidence_can_override():
    """The model's own joint limits are NOT a bound: -72.8..+167.3 deg leaves the
    optimiser free to travel most of that range legally, and on bout 28 fly0 it
    did -- a 130 deg swing driven by a collapsed mask. `max_dpitch_deg` is the
    physiological bound that makes such an answer unrepresentable."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0)
    m = np.flatnonzero(np.asarray(kw["opt_mask"]))
    free = np.asarray(refine_wing_pitch(q_pert, **kw))
    d_free = np.abs(free[:, m] - q_pert[:, m]).max()
    assert d_free > np.deg2rad(5.0), (
        f"the unbounded fit moves only {np.degrees(d_free):.2f} deg -- the bound "
        f"below would not bite and the test would prove nothing")

    bound = np.asarray(refine_wing_pitch(q_pert, max_dpitch_deg=5.0, **kw))
    d = np.abs(bound[:, m] - q_pert[:, m]).max()
    assert d <= np.deg2rad(5.0) + 1e-6, \
        f"max_dpitch_deg=5 let the correction reach {np.degrees(d):.3f} deg"


def test_the_gate_and_the_bound_are_no_ops_at_their_defaults():
    """`free` + no gate is the arm every measured number in
    docs/benchmark/2026-09-01-wing-mask-fit/ was taken on, and it must stay
    reproducible: passing an all-True `frame_keep` or a bound wider than the
    correction may not move the fit further than it moves run to run."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    import inspect
    sig = inspect.signature(refine_wing_pitch)
    assert sig.parameters["frame_keep"].default is None
    assert sig.parameters["max_dpitch_deg"].default is None
    q0, kw = _tiny_problem(n_frames=6, n_steps=10)
    a = np.asarray(refine_wing_pitch(q0, **kw))
    noise = float(np.abs(a - np.asarray(refine_wing_pitch(q0, **kw))).max())
    b = np.asarray(refine_wing_pitch(q0, frame_keep=np.ones(6, bool),
                                     max_dpitch_deg=180.0, **kw))
    delta = float(np.abs(a - b).max())
    assert delta <= max(noise, 1e-6), (
        f"the gate at its no-op settings moved the fit by {delta:.3e} rad "
        f"against a run-to-run noise floor of {noise:.3e}")
