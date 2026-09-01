"""Adam refinement of WING PITCH against the SAM masks.

WHY ONLY PITCH. Each wing body carries just two markers, and `WingX_base` maps
to the THORAX, so wing pitch is 99.6% of the marker Jacobian's null direction
(per-column sensitivity: yaw 0.271, roll 0.344, pitch 0.042). Unconstrained by
the keypoints, the STAC solve rotates the blade inward through the abdomen. The
masks DO contain wing pixels, so pitch -- and only pitch -- is recovered here.
Yaw carries the courtship song and roll is the best-observed wing DOF; both are
left to the marker solve, along with root, thorax, abdomen and legs.

THE OBJECTIVE is the two opposing mask terms, evaluated on the ~100 wing
vertices of the `fps_300` subset:

  containment (`mask_containment.containment_residual`) -- one-sided
      `relu(sdf + margin)`: a wing vertex that projects OUTSIDE the fly's mask
      is pulled back in; one already inside costs nothing.
  coverage (`wing_coverage.coverage_residual`) -- softmin distance from each
      mask pixel the BODY does not explain to the nearest projected wing
      vertex: the wing is pulled OUT to cover mask area nothing else accounts
      for.

Containment alone is minimised perfectly by tucking the wing inside the body
silhouette -- i.e. by the very bug this exists to fix -- so the pair is what
makes the problem well posed. Plus temporal smoothness on the optimised joints,
and a soft joint-limit barrier.

COVERAGE IS NORMALISED PER TARGET (`coverage_normalize=True`), containment is
not, and the asymmetry is deliberate. Containment contributes ONE residual per
wing VERTEX -- a fixed geometric quantity, ~100 of them, most of them exactly
zero because the vertex is already inside the mask. Coverage contributes one
residual per sampled TARGET, and that count is `n_target_points`, a free knob.
Summing both left `coverage_weight` meaningless on its own and the two terms
about two orders of magnitude apart: at the nominal
`containment_weight == coverage_weight == 0.3` the objective was effectively
coverage-only, and on Session0 bout 28 fly0 the break-even against containment
sat near `coverage_weight = 0.03`. Taking the MEAN over the finite targets makes
`coverage_weight` invariant to `n_target_points` and puts the two terms on one
scale (verified: `coverage_weight=3.0` normalised reproduces the old unnormalised
0.3 to within 0.2 of a percentage point on both real-bout metrics).

RELATEDLY, `huber_delta` matters more than a robustness nicety here. At 0.0 the
coverage term is a plain L2 on the softmin distance, whose gradient grows with
distance, so the FARTHEST unexplained pixels dominate -- SAM halo, crescents
where the mesh body does not register on the imaged body, an occluding second
fly. Measured on the real bout, `huber_delta=8` improves the fit at EVERY
coverage weight tried. (It was also unusable until 2026-09-01: `_huber_sqrt`
returned correct values with an all-NaN gradient.)

WHY ADAM, not the jaxls Gauss-Newton/LM used elsewhere: jaxls damps with a
non-scale-invariant `lambda*I`, and the pixel-scale silhouette Jacobian has
`|diag(JtJ)| ~ 1e6`, so every GN step overshoots and is rejected -- measured, a
plain gradient step cut the cost ~45% while LM rejected all steps. Adam's
per-coordinate normalisation also makes the step size ~`lr` radians regardless
of the cost scale, which is what lets a fixed `n_steps` budget close a known
pitch offset.

Derived from the deleted `silhouette_refine.py` (recovered at `0bc36fe^`),
which already had the right shape: FK once per frame with vertex selection,
bridge model->mm, cameras by `lax.scan`, frames by `vmap`, Adam steps by
`lax.scan` (one trace per call, loss history for free), and `opt_mask` gating
BOTH the gradient and the parameter update. What changed: the chamfer term is
fed wing-coverage targets instead of mask-BOUNDARY points, the DOF set is wing
pitch only, and -- the genuinely new work -- the trajectory is now processed in
FRAME CHUNKS padded to a constant shape, because the whole `(T, nq)` used to go
in one call and at T=2007 x 7 cameras x 128x128 f32 the SDF stack alone is
~920 MB.

TWO CHUNK PARAMETERS, DO NOT CONFLATE THEM. `chunk_size` is
`coverage_residual`'s memory chunk over TARGET POINTS (`jax.lax.map(...,
batch_size=chunk_size)`, peak `O(chunk_size * n_wing_verts)`); `frame_chunk` is
the number of FRAMES per device call.

CAMERA ORDER. `cam_Ms`/`cam_ts` and the `sdf`/`masks` camera axis must be in
the SAME order. Build the former with `affine_cameras_by_name(calib_dir,
cfg.recording.cameras)` and the latter with `load_bout_masks(...,
expected_cameras=cfg.recording.cameras)`: both then index BY NAME off one
canonical list. Positional construction from `ReprojectionTool._camera_list`
against an un-reordered mask npz projects one camera's wing onto another
camera's mask, and the residual still looks entirely plausible.

float32 throughout: this is a pixel-scale cost, not a reprojection identity.
"""
from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax
import jax.numpy as jnp
import optax

from jarvis_jax.tracking.mask_containment import containment_residual
from jarvis_jax.tracking.wing_coverage import coverage_residual, wing_target_points

#: The only two DOFs this module ever moves. Addressed BY NAME -- they sit at
#: qposadr 9 and 12 in fruitfly_v1, but hardcoding that is exactly the
#: integer-index trap that has produced confident, self-consistent, completely
#: wrong numbers in this pipeline before.
WING_PITCH_JOINTS = ("wing_pitch_left", "wing_pitch_right")


# ---------------------------------------------------------------------------
# model / mesh / camera selection -- all BY NAME
# ---------------------------------------------------------------------------
def wing_pitch_dof_mask(model):
    """(nq,) bool, True only at the qpos addresses of `WING_PITCH_JOINTS`.

    Raises ValueError if the model does not have both joints, so a renamed or
    swapped body model fails loudly instead of silently refining nothing.
    """
    import mujoco
    mask = np.zeros(int(model.nq), bool)
    found = []
    for j in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name not in WING_PITCH_JOINTS:
            continue
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"joint {name!r} is not a hinge; wing_pitch must be "
                             "a single-qpos hinge DOF")
        mask[int(model.jnt_qposadr[j])] = True
        found.append(name)
    missing = [n for n in WING_PITCH_JOINTS if n not in found]
    if missing:
        raise ValueError(f"model has no joint(s) named {missing}; "
                         f"wing_pitch refinement cannot address its DOFs by name")
    return mask


def qpos_limits(model):
    """(lb, ub) (nq,) float32 joint limits, addressed by qposadr.

    +-inf for unlimited joints and for every qpos of the free root / ball
    joints, which have no scalar range.
    """
    import mujoco
    nq = int(model.nq)
    lb = np.full(nq, -np.inf, np.float32)
    ub = np.full(nq, np.inf, np.float32)
    scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) not in scalar or not bool(model.jnt_limited[j]):
            continue
        a = int(model.jnt_qposadr[j])
        lb[a], ub[a] = np.asarray(model.jnt_range[j], np.float32)
    return lb, ub


def body_vertex_indices(mesh_npz, *, stride=20):
    """Full-array indices of the NON-wing (body) vertices, every `stride`-th.

    These are the vertices `wing_target_points` rasterises to decide which mask
    pixels the body already explains, so they must be DENSE enough that their
    dilated footprint covers the body silhouette. A sparse set (e.g. the 200
    non-wing vertices of `fps_300`) leaves body pixels looking unexplained, and
    the coverage term then pulls the wing ONTO the body -- the failure this
    module exists to undo. They are FK'd once per frame from the frozen
    `q_init`, not per Adam step, so density is cheap.
    """
    z = np.load(mesh_npz, allow_pickle=True)
    seg_ids = np.asarray(z["seg_ids"])
    seg_names = [str(s) for s in z["seg_names"]]
    wing_ids = [int(i) for i, n in zip(seg_ids, seg_names) if "wing" in n.lower()]
    if not wing_ids:
        raise ValueError(f"{mesh_npz}: no segment name contains 'wing'")
    vseg = np.asarray(z["vertex_segment"])
    idx = np.flatnonzero(~np.isin(vseg, wing_ids))
    return idx[::int(stride)].astype(np.int32)


def affine_cameras_by_name(calib_dir, cameras):
    """(cam_Ms (C,2,3), cam_ts (C,2)) float32 for `cameras`, selected BY NAME.

    The rig's DLT matrices are affine (3rd row `[0,0,0,1]`, verified on the
    Session0 calibration), so `uv = X @ M.T + t` exactly. Pass the SAME
    canonical name list used for `load_bout_masks(expected_cameras=...)`; that
    is what keeps the camera axes of the projection and of the masks aligned.
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    rt = ReprojectionTool(calib_dir)
    names = [str(c) for c in cameras]
    missing = [c for c in names if c not in rt.cameras]
    if missing:
        raise ValueError(f"calibration {calib_dir!r} has no camera(s) {missing}; "
                         f"it has {sorted(rt.cameras)}")
    P = np.stack([np.asarray(rt.cameras[c].cameraMatrix, np.float64) for c in names])
    bad = [c for c, p in zip(names, P) if not np.allclose(p[2], (0.0, 0.0, 0.0, 1.0))]
    if bad:
        raise ValueError(f"camera(s) {bad} are not affine (DLT row 2 != [0,0,0,1]); "
                         "the mm->px projection in this module is affine only")
    return P[:, :2, :3].astype(np.float32), P[:, :2, 3].astype(np.float32)


# ---------------------------------------------------------------------------
# coverage targets (host side; the body pose is frozen so these are constant)
# ---------------------------------------------------------------------------
def _frame_camera_targets(mask, body_uv, *, n_points, dilate_px=3, seed=0):
    """`wing_target_points` on the mask's bbox CROP, returned in ORIGINAL px.

    Identical to calling `wing_target_points` on the full frame -- the crop is
    grown by `dilate_px + 1`, so no body vertex outside it can dilate onto a
    mask pixel inside it, and the surviving pixels keep their row-major order,
    so the seeded `rng.choice` draws the same ones -- but O(bbox) instead of
    O(H*W). On the real 448x1936 frames that is a ~10x saving on 14 049
    (frame, camera) pairs.

    `body_uv` is PREFILTERED to finite rows here: real projections produce NaN
    for occluded / behind-camera points, and `wing_target_points` casts with
    `np.round(...).astype(np.int64)`, which is implementation-defined on NaN
    (on x86 it lands on INT64_MIN and only survives by accident of the
    subsequent bounds check). If EVERY body vertex is non-finite the body
    explains nothing, and the whole mask is treated as wing-attributable --
    the correct reading, made deliberate.
    """
    mask = np.asarray(mask, bool)
    body_uv = np.asarray(body_uv, np.float64).reshape(-1, 2)
    body_uv = body_uv[np.isfinite(body_uv).all(axis=1)]

    rows = mask.any(axis=1)
    if not rows.any():
        return np.full((n_points, 2), np.nan, np.float32)
    cols = mask.any(axis=0)
    pad = int(dilate_px) + 1
    H, W = mask.shape
    y0 = max(0, int(np.argmax(rows)) - pad)
    y1 = min(H, H - int(np.argmax(rows[::-1])) + pad)
    x0 = max(0, int(np.argmax(cols)) - pad)
    x1 = min(W, W - int(np.argmax(cols[::-1])) + pad)

    tgt = wing_target_points(mask[y0:y1, x0:x1], body_uv - (x0, y0),
                             n_points=n_points, dilate_px=dilate_px, seed=seed)
    tgt[:, 0] += x0          # NaN padding rows stay NaN
    tgt[:, 1] += y0
    return tgt


def _body_uv_chunk(q_c, brs_c, brR_c, brt_c, *, fk_repose, body_idx, cam_Ms, cam_ts):
    """(F, C, V, 2) projected BODY vertices, original px."""
    def one(q, s, R, t):
        v = fk_repose(q, 1.0, body_idx)
        vmm = s * (v @ R.T) + t
        return jnp.einsum("cij,vj->cvi", cam_Ms, vmm) + cam_ts[:, None, :]
    return jax.vmap(one)(q_c, brs_c, brR_c, brt_c)


# ---------------------------------------------------------------------------
# the cost
# ---------------------------------------------------------------------------
def _frame_cost(q_t, br_s, br_R, br_t, sdf_c, gsc_c, goff_c, pres_c, tgt_c, *,
                fk_repose, wing_idx, cam_Ms, cam_ts, containment_weight,
                coverage_weight, beta, huber_delta, margin, chunk_size,
                coverage_normalize):
    """containment + wing coverage for ONE frame, summed over cameras.

    The wing vertices are FK'd ONCE in model frame, mapped to mm by the
    per-frame bridge, and only then projected per camera (`jax.lax.scan`).
    """
    verts = fk_repose(q_t, 1.0, wing_idx)              # (M,3) model frame
    vmm = br_s * (verts @ br_R.T) + br_t               # -> mm

    def scan_body(carry, cam):
        M, tt, sdf_i, gs_i, go_i, pr_i, tgt_i = cam
        proj = vmm @ M.T + tt                          # (M,2) original px
        r_cont = containment_residual(proj, sdf_i, gs_i, go_i, 1.0,
                                      margin=margin, present=pr_i)
        r_cov = coverage_residual(tgt_i, proj, beta=beta, huber_delta=huber_delta,
                                  chunk_size=chunk_size)
        r_cov = jnp.where(pr_i, r_cov, 0.0)
        cov = jnp.sum((coverage_weight * r_cov) ** 2)
        if coverage_normalize:
            # MEAN over the finite targets, not a sum. Static Python `if`, like
            # coverage_residual's huber branch. See the module docstring.
            n_fin = jnp.maximum(jnp.sum(jnp.isfinite(tgt_i).all(axis=-1)), 1.0)
            cov = cov / n_fin
        c = jnp.sum((containment_weight * r_cont) ** 2) + cov
        return carry + c, None

    total, _ = jax.lax.scan(
        scan_body, jnp.float32(0.0),
        (cam_Ms, cam_ts, sdf_c, gsc_c, goff_c, pres_c, tgt_c))
    return total


def _refine_chunk(q0_c, sdf_c, gsc_c, goff_c, pres_c, tgt_c, brs_c, brR_c, brt_c,
                  real_c, q_prev, has_prev, *,
                  fk_repose, wing_idx, cam_Ms, cam_ts, opt_mask, lb_row, ub_row,
                  containment_weight, coverage_weight, smooth_weight,
                  limit_weight, beta, huber_delta, margin, n_steps, lr, chunk_size,
                  coverage_normalize):
    """Adam-refine ONE padded frame chunk. Returns (q_ref (F,nq), history).

    `real_c` marks the frames that are not padding; `q_prev`/`has_prev` carry
    the previous chunk's last refined pose so the smoothness term overlaps
    chunk boundaries by one frame. Every chunk has the same shapes, so this
    traces exactly once per `refine_wing_pitch` call.
    """
    fc = functools.partial(
        _frame_cost, fk_repose=fk_repose, wing_idx=wing_idx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, containment_weight=containment_weight,
        coverage_weight=coverage_weight, beta=beta, huber_delta=huber_delta,
        margin=margin, chunk_size=chunk_size,
        coverage_normalize=coverage_normalize)

    # A frame with no present camera carries NO mask evidence, and a padded row
    # carries none by construction: both are frozen at q_init rather than being
    # dragged around by the smoothness term alone.
    frame_ok = real_c & jnp.any(pres_c, axis=1)                      # (F,)
    upd = opt_mask[None, :] & frame_ok[:, None]                      # (F,nq)
    real_f = real_c.astype(jnp.float32)

    def make_q(dq):
        return q0_c + jnp.where(upd, dq, 0.0)

    def objective(dq):
        q = make_q(dq)
        mask_cost = jnp.sum(jax.vmap(fc)(q, brs_c, brR_c, brt_c,
                                         sdf_c, gsc_c, goff_c, pres_c, tgt_c))
        dj = (q[1:] - q[:-1]) * opt_mask[None, :]
        pair = (real_f[1:] * real_f[:-1])[:, None]
        d0 = (q[0] - q_prev) * opt_mask * has_prev * real_f[0]
        smooth = (smooth_weight ** 2) * (jnp.sum(pair * dj ** 2) + jnp.sum(d0 ** 2))
        over = jax.nn.relu(q - ub_row[None, :])
        under = jax.nn.relu(lb_row[None, :] - q)
        limit = (limit_weight ** 2) * jnp.sum(real_f[:, None] * (over ** 2 + under ** 2))
        return mask_cost + smooth + limit

    opt = optax.adam(lr)
    dq0 = jnp.zeros_like(q0_c)
    val_and_grad = jax.value_and_grad(objective)

    def step(carry, _):
        dq, st = carry
        v, g = val_and_grad(dq)
        g = jnp.where(upd, g, 0.0)               # never move a frozen DOF or frame
        updates, st = opt.update(g, st, dq)
        return (optax.apply_updates(dq, updates), st), v

    (dq_final, _), hist = jax.lax.scan(step, (dq0, opt.init(dq0)), None, length=n_steps)
    hist = jnp.concatenate([hist, objective(dq_final)[None]])

    q_ref = make_q(dq_final)
    # Gated on `upd`, NOT on opt_mask: a frame with no evidence was never
    # optimised, so clamping it would silently move a STAC pitch that happens to
    # sit outside the joint range -- "no evidence must mean no change".
    q_ref = jnp.where(upd, jnp.clip(q_ref, lb_row[None, :], ub_row[None, :]), q_ref)
    return q_ref, hist


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _pad_repeat(a, F):
    a = np.asarray(a)
    if a.shape[0] == F:
        return a
    return np.concatenate([a, np.repeat(a[-1:], F - a.shape[0], axis=0)], axis=0)


def _pad_zero(a, F):
    a = np.asarray(a)
    if a.shape[0] == F:
        return a
    return np.concatenate(
        [a, np.zeros((F - a.shape[0],) + a.shape[1:], a.dtype)], axis=0)


def refine_wing_pitch(
    q_init, *,
    fk_repose, wing_vert_idx, body_vert_idx, cam_Ms, cam_ts,
    sdf, grid_scale, grid_offset, present, masks,
    bridge_s, bridge_R, bridge_t, opt_mask, lb, ub,
    containment_weight=0.3, coverage_weight=0.3, smooth_weight=0.005,
    limit_weight=10.0, beta=8.0, huber_delta=0.0, margin=0.0,
    coverage_normalize=True,
    n_target_points=128, dilate_px=3, target_seed=0,
    n_steps=300, lr=1e-2, chunk_size=32, frame_chunk=64, prefetch=True,
    return_history=False,
):
    """Refine the wing-pitch DOFs of `q_init` (T, nq) against the SAM masks.

    Args:
        q_init: (T, nq) float32 STAC/keypoint pose. Everything `opt_mask` does
            not select comes back bit-identical.
        fk_repose: `fk.make_fk_repose(anat)` -- `(qpos, scale, indices) -> (K,3)`.
        wing_vert_idx: FULL-array vertex indices of the wing (the ~100 wing
            vertices of `fps_300`); the only vertices FK'd inside the loop.
        body_vert_idx: FULL-array vertex indices of the body, for deciding
            which mask pixels the body already explains. See
            `body_vertex_indices` for the density requirement.
        cam_Ms, cam_ts: (C,2,3) / (C,2) affine mm->px projection, in the SAME
            camera order as `sdf`/`masks`. Use `affine_cameras_by_name`.
        sdf, grid_scale, grid_offset, present: `mask_sdf.sdf_stack_from_masks`
            output, (T,C,H,W) / (T,C,2) / (T,C,2) / (T,C).
        masks: (T,C,H,W) bool full-frame masks (anything indexable as
            `masks[t][c]`), used only to sample the coverage targets.
        bridge_s, bridge_R, bridge_t: (T,) / (T,3,3) / (T,3) per-frame
            model->mm similarity.
        opt_mask: (nq,) bool -- use `wing_pitch_dof_mask(model)`.
        lb, ub: (nq,) joint limits -- use `qpos_limits(model)`.
        n_target_points: coverage targets sampled per (frame, camera).
        coverage_normalize: divide the coverage sum by the number of finite
            targets, making it a mean. See the module docstring -- without it
            `coverage_weight` is not comparable to `containment_weight` and
            depends on `n_target_points`.
        chunk_size: `coverage_residual`'s memory chunk over TARGET POINTS.
        frame_chunk: FRAMES per device call. Chunks are padded to this constant
            shape so the step loop traces once.
        prefetch: build the next chunk's coverage targets on a worker thread
            while the current chunk optimises on the GPU.
        return_history: also return the (n_chunks, n_steps+1) objective history.

    Returns:
        (T, nq) float32 numpy, or `(q_refined, history)` if `return_history`.
    """
    q0 = np.asarray(q_init, np.float32)
    if q0.ndim != 2:
        raise ValueError(f"q_init must be (T, nq), got {q0.shape}")
    T, nq = q0.shape

    opt_mask_np = np.asarray(opt_mask, bool)
    if opt_mask_np.shape != (nq,):
        raise ValueError(f"opt_mask must be ({nq},), got {opt_mask_np.shape}")
    cam_Ms_np = np.asarray(cam_Ms, np.float32)
    cam_ts_np = np.asarray(cam_ts, np.float32)
    C = cam_Ms_np.shape[0]
    sdf = np.asarray(sdf, np.float32)
    present = np.asarray(present, bool)
    if sdf.shape[:2] != (T, C) or present.shape != (T, C):
        raise ValueError(
            f"sdf {sdf.shape} / present {present.shape} disagree with T={T}, "
            f"C={C} (cam_Ms and the mask camera axis must both be canonical)")
    # `masks` may be a streaming object without .shape; check it when it has one.
    mshape = getattr(masks, "shape", None)
    if mshape is not None and tuple(mshape[:2]) != (T, C):
        raise ValueError(f"masks {tuple(mshape)} disagree with T={T}, C={C}")

    lb_row = np.where(opt_mask_np, np.asarray(lb, np.float32), -np.inf).astype(np.float32)
    ub_row = np.where(opt_mask_np, np.asarray(ub, np.float32), np.inf).astype(np.float32)

    grid_scale = np.asarray(grid_scale, np.float32)
    grid_offset = np.asarray(grid_offset, np.float32)
    bridge_s = np.asarray(bridge_s, np.float32)
    bridge_R = np.asarray(bridge_R, np.float32)
    bridge_t = np.asarray(bridge_t, np.float32)

    # NON-FINITE FRAMES. Session0 bout 28 fly0 has 483 of 2007 qpos rows
    # entirely NaN -- the frames STAC could not solve, which Stage D marks with
    # bridge_ok=False. The objective SUMS over the frames in a chunk, so one NaN
    # makes the gradient NaN for every frame that shares the chunk and the whole
    # bout comes back NaN (measured, before this guard). Such frames are
    # substituted with a finite stand-in pose for the cost, frozen via `present`,
    # and handed back exactly as they arrived.
    finite_frame = (np.isfinite(q0).all(axis=1)
                    & np.isfinite(bridge_s)
                    & np.isfinite(bridge_R).all(axis=(1, 2))
                    & np.isfinite(bridge_t).all(axis=1))
    if not finite_frame.any():
        out = q0.copy()
        return (out, np.zeros((0, int(n_steps) + 1), np.float32)) if return_history else out
    if not finite_frame.all():
        stand_in = int(np.argmax(finite_frame))
        q_safe = q0.copy()
        q_safe[~finite_frame] = q0[stand_in]
        bridge_s = np.where(finite_frame, bridge_s, bridge_s[stand_in])
        bridge_R = np.where(finite_frame[:, None, None], bridge_R, bridge_R[stand_in])
        bridge_t = np.where(finite_frame[:, None], bridge_t, bridge_t[stand_in])
        present = present & finite_frame[:, None]
    else:
        q_safe = q0

    F = int(frame_chunk) if frame_chunk else T
    F = max(1, min(F, T))
    n_chunks = int(np.ceil(T / F))

    cam_Ms_j = jnp.asarray(cam_Ms_np)
    cam_ts_j = jnp.asarray(cam_ts_np)
    wing_idx_j = jnp.asarray(np.asarray(wing_vert_idx, np.int32))
    body_idx_j = jnp.asarray(np.asarray(body_vert_idx, np.int32))

    body_uv_fn = jax.jit(functools.partial(
        _body_uv_chunk, fk_repose=fk_repose, body_idx=body_idx_j,
        cam_Ms=cam_Ms_j, cam_ts=cam_ts_j))

    step_fn = jax.jit(functools.partial(
        _refine_chunk, fk_repose=fk_repose, wing_idx=wing_idx_j,
        cam_Ms=cam_Ms_j, cam_ts=cam_ts_j, opt_mask=jnp.asarray(opt_mask_np),
        lb_row=jnp.asarray(lb_row), ub_row=jnp.asarray(ub_row),
        containment_weight=float(containment_weight),
        coverage_weight=float(coverage_weight),
        smooth_weight=float(smooth_weight),
        limit_weight=float(limit_weight), beta=float(beta),
        huber_delta=float(huber_delta), margin=float(margin),
        n_steps=int(n_steps), lr=float(lr), chunk_size=int(chunk_size),
        coverage_normalize=bool(coverage_normalize)))

    def prepare(k):
        s, e = k * F, min(T, (k + 1) * F)
        n = e - s
        qc = _pad_repeat(q_safe[s:e], F)
        brs = _pad_repeat(bridge_s[s:e], F)
        brR = _pad_repeat(bridge_R[s:e], F)
        brt = _pad_repeat(bridge_t[s:e], F)
        pres = _pad_zero(present[s:e], F)
        real = np.zeros(F, bool)
        real[:n] = True
        # The body pose is FROZEN by opt_mask, so these projections -- and hence
        # the coverage targets -- do not change as the wing pitch moves.
        buv = np.asarray(body_uv_fn(jnp.asarray(qc), jnp.asarray(brs),
                                    jnp.asarray(brR), jnp.asarray(brt)))
        tgt = np.full((F, C, int(n_target_points), 2), np.nan, np.float32)
        for i in range(n):
            mrow = masks[s + i]
            for c in range(C):
                if not present[s + i, c]:
                    continue
                tgt[i, c] = _frame_camera_targets(
                    mrow[c], buv[i, c], n_points=int(n_target_points),
                    dilate_px=int(dilate_px),
                    seed=int(target_seed) + (s + i) * C + c)
        return dict(
            s=s, e=e, n=n,
            args=(jnp.asarray(qc),
                  jnp.asarray(_pad_zero(sdf[s:e], F)),
                  jnp.asarray(_pad_repeat(grid_scale[s:e], F)),
                  jnp.asarray(_pad_repeat(grid_offset[s:e], F)),
                  jnp.asarray(pres), jnp.asarray(tgt),
                  jnp.asarray(brs), jnp.asarray(brR), jnp.asarray(brt),
                  jnp.asarray(real)))

    q_out = q_safe.copy()
    hist_all = []
    q_prev = jnp.asarray(q_safe[0])
    has_prev = np.float32(0.0)

    pool = ThreadPoolExecutor(1) if (prefetch and n_chunks > 1) else None
    try:
        pending = pool.submit(prepare, 0) if pool else None
        for k in range(n_chunks):
            ch = pending.result() if pool else prepare(k)
            if pool and k + 1 < n_chunks:
                pending = pool.submit(prepare, k + 1)
            q_ref, hist = step_fn(*ch["args"], q_prev, jnp.asarray(has_prev))
            q_ref = np.asarray(q_ref)
            q_out[ch["s"]:ch["e"]] = q_ref[:ch["n"]]
            hist_all.append(np.asarray(hist))
            # a non-finite last frame is not a usable smoothness anchor
            q_prev = jnp.asarray(q_out[ch["e"] - 1])
            has_prev = np.float32(finite_frame[ch["e"] - 1])
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    q_out[~finite_frame] = q0[~finite_frame]     # hand the NaN rows back untouched
    if return_history:
        return q_out, np.stack(hist_all)
    return q_out
