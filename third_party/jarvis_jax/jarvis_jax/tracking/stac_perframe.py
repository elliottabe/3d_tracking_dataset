"""Per-frame multi-start STAC IK -- the Stage C solver when ``stac.solver: per_frame``.

Replaces the jaxls whole-trajectory ("batch") solve. Why (bout 28, 2026-09-05,
`docs/benchmark/2026-09-04-wing-ik-spike/notes.md`): the batch solve starts
every hinge at 0 and stalls the folded wing's blade pitch against the yaw stop
(~-20 deg, marker optimum ~-45..-55), and its aggregate termination leaves the
whole body under-converged (female all-keypoint residual 0.73 mm where a
per-frame solve reaches 0.23). Its smoothness term at the shipped weight is
numerically negligible, so nothing is lost by dropping the temporal coupling;
the per-frame solve is also faster.

Per frame:
  1. production warm start: root xyz from the root keypoint, root orientation
     from the four trunk keypoints, hinges at 0 (the batch solver's own start);
  2. the "zero" start is solved (vmapped single-frame LM, per-frame termination);
  3. the rest starts are CHAINED from that solution -- same pose with the left,
     right, or both wing pitches reset to the model's spring rest (-57.3 deg) --
     so they converge in few iterations (user step 3);
  4. candidates are ranked per frame by marker cost with a Viterbi switch
     penalty on pose jumps, wing DOFs weighted extra (`switch_wing_weight`), so
     a basin flip costs more than ordinary leg motion.
Output has the stac_ik.h5 schema the rest of the pipeline reads, plus
`ik_start_idx`, `ik_iterations` (per frame; watch the fraction at n_iter before
lowering the cap on other bouts) and, if `save_candidates`, `qpos_candidates`
and `candidate_costs` for offline tuning of the switch weights.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Sequence

import numpy as np

from jarvis_jax.tracking.stac_polish import (
    dof_mask,
    frame_costs_np,
    viterbi_select,
)

STAC_H5_KEYS = ("config", "kp_data", "kp_names", "marker_sites", "names_qpos", "names_xpos",
                "offsets", "qpos", "qvel", "xpos", "xquat")


def warm_start(kp_flat: np.ndarray, kp_names: Sequence[str], model_cfg, q_base: np.ndarray) -> np.ndarray:
    """(T, nq) production warm start: hinges from `q_base` (the model's qpos0),
    root xyz from ROOT_OPTIMIZATION_KEYPOINT, orientation from
    JAXLS_ORIENTATION_KEYPOINTS. Mirrors compute_stac._pose_optimization_jaxls."""
    import jax.numpy as jnp
    from stac_mjx.compute_stac import _estimate_orientation_from_keypoints
    names = list(kp_names)
    T = kp_flat.shape[0]
    q = np.tile(np.asarray(q_base, np.float64), (T, 1))
    q[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    ir = names.index(str(model_cfg.ROOT_OPTIMIZATION_KEYPOINT))
    q[:, :3] = kp_flat[:, ir * 3:ir * 3 + 3]
    ok = model_cfg.JAXLS_ORIENTATION_KEYPOINTS
    quats = _estimate_orientation_from_keypoints(
        jnp.asarray(kp_flat), names.index(str(ok["rear"])), names.index(str(ok["left"])),
        names.index(str(ok["right"])), names.index(str(ok["front"])) if "front" in ok else -1)
    q[:, 3:7] = np.asarray(quats)
    return q


def chained_starts(q_solved: np.ndarray, names_qpos: Sequence[str], qpos_spring: np.ndarray,
                   starts: Sequence[str]) -> Dict[str, np.ndarray]:
    """Rest starts built FROM the zero-start solution: everything as solved,
    only the named wing pitch(es) reset to spring rest. Known: wing_rest_left,
    wing_rest_right, wing_rest_both."""
    names = list(names_qpos)
    ipl, ipr = names.index("wing_pitch_left"), names.index("wing_pitch_right")
    out = {}
    for s in starts:
        q = np.array(q_solved, dtype=np.float64, copy=True)
        if s == "wing_rest_left":
            q[:, ipl] = qpos_spring[ipl]
        elif s == "wing_rest_right":
            q[:, ipr] = qpos_spring[ipr]
        elif s == "wing_rest_both":
            q[:, ipl] = qpos_spring[ipl]; q[:, ipr] = qpos_spring[ipr]
        else:
            raise ValueError(f"stac.per_frame.starts: unknown start {s!r} (zero is implicit)")
        out[s] = q
    return out


def wing_dof_weights(names_qpos: Sequence[str], wing_weight: float) -> np.ndarray:
    """Per-DOF multiplier for the Viterbi jump penalty: `wing_weight` on wing_* DOFs, 1 elsewhere."""
    w = np.ones(len(names_qpos))
    w[dof_mask(names_qpos, ["wing_*"])] = float(wing_weight)
    return w


def qvel_from_qpos(q: np.ndarray, dt: float, freejoint: bool = True, max_qvel: float = 20.0) -> np.ndarray:
    """Vectorised numpy twin of stac_mjx.utils.compute_velocity_from_kinematics
    (same convention: forward difference with the last frame repeated,
    free-joint angular velocity as the axis-angle of q_t^-1 q_{t+1} over dt,
    joint velocities clipped to +-max_qvel). The original loops over frames in
    Python with small JAX ops -- 118 s for 1500 frames (profiled 2026-09-05).
    NaN frames propagate NaN."""
    q = np.asarray(q, np.float64)
    qp = np.concatenate([q, q[-1:]], axis=0)
    if not freejoint:
        return np.clip((qp[1:] - qp[:-1]) / dt, -max_qvel, max_qvel)
    v_joint = (qp[1:, 7:] - qp[:-1, 7:]) / dt
    v_trans = (qp[1:, :3] - qp[:-1, :3]) / dt
    a, b = qp[:-1, 3:7], qp[1:, 3:7]
    # conj(a) * b, quaternions as [w, x, y, z]
    aw, ax, ay, az = a[:, 0], -a[:, 1], -a[:, 2], -a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    dq = np.stack([aw * bw - ax * bx - ay * by - az * bz,
                   aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx,
                   aw * bz + ax * by - ay * bx + az * bw], axis=1)
    dq = dq / np.linalg.norm(dq, axis=1, keepdims=True)
    ang = 2.0 * np.arccos(np.clip(dq[:, 0], -1.0, 1.0))
    sin_half = np.sqrt(np.maximum(1.0 - dq[:, 0] ** 2, 0.0))
    ang = (ang + np.pi) % (2 * np.pi) - np.pi                   # wrap to (-pi, pi], as the original does
    with np.errstate(invalid="ignore", divide="ignore"):
        axis = np.where(sin_half[:, None] > 1e-7, dq[:, 1:] / sin_half[:, None], 0.0)
    v_gyro = axis * ang[:, None] / dt
    out = np.concatenate([v_trans, v_gyro, v_joint], axis=1)
    out[:, 6:] = np.clip(out[:, 6:], -max_qvel, max_qvel)
    return out


def write_stac_h5(path: str, *, cfg, kp_names, names_qpos, names_xpos, kp_data, marker_sites,
                  offsets, qpos, xpos, xquat, qvel, extra: dict | None = None, attrs: dict | None = None):
    """stac_ik.h5 with the schema stac_mjx.io.save_data_to_h5 produces (float32
    arrays, |S names, config as YAML), plus optional extra datasets/attrs."""
    import h5py
    from omegaconf import OmegaConf
    with h5py.File(path, "w") as f:
        f.create_dataset("config", data=np.bytes_(OmegaConf.to_yaml(cfg)))
        for k, v in (("kp_names", kp_names), ("names_qpos", names_qpos), ("names_xpos", names_xpos)):
            f.create_dataset(k, data=np.array([str(n) for n in v], dtype="S"))
        for k, v in (("kp_data", kp_data), ("marker_sites", marker_sites), ("offsets", offsets),
                     ("qpos", qpos), ("xpos", xpos), ("xquat", xquat), ("qvel", qvel)):
            f.create_dataset(k, data=np.asarray(v, np.float32))
        for k, v in (extra or {}).items():
            f.create_dataset(k, data=np.asarray(v))
        for k, v in (attrs or {}).items():
            f.attrs[k] = v


def solve_per_frame_ik(cfg, kp3d_scaled: np.ndarray, kp_names: List[str], *, xml_path: str,
                       offsets_h5: str, out_h5: str, solve_mask: np.ndarray | None = None,
                       log_prefix: str = "[ik-perframe]") -> dict:
    """Solve every frame of `kp3d_scaled` ((T,K,3) or (T,K*3), MODEL units,
    NaN allowed) independently; write `out_h5` in the stac_ik.h5 schema.
    Frames with `solve_mask` False (or non-finite warm-start inputs) are NaN."""
    import h5py
    import jax
    import jax.numpy as jnp
    from stac_mjx import utils
    from stac_mjx.stac import Stac
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from mujoco import mjx

    import os as _os
    _prof = _os.environ.get("STAC_PERFRAME_PROFILE") == "1"
    _marks = [("start", time.time())]
    def _mark(name):
        if _prof: _marks.append((name, time.time()))
    pc = cfg.stac.per_frame
    kp = np.asarray(kp3d_scaled, np.float64)
    T = kp.shape[0]
    kp_flat = kp.reshape(T, -1)
    with h5py.File(offsets_h5, "r") as f:
        offsets = np.asarray(f["offsets"][:], np.float64)
        off_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["kp_names"][:]]
    if off_names != list(kp_names):
        raise ValueError(f"{offsets_h5}: kp_names differ from the pipeline's KP_NAMES")

    t0 = time.time()
    stac = Stac(str(xml_path), cfg, list(kp_names))
    mj = stac._mj_model
    site_idxs = np.asarray(stac._body_site_idxs)
    mj.site_pos[site_idxs] = offsets
    mjx_model, mjx_data = utils.mjx_load(mj)
    mjx_model = utils.set_site_pos(mjx_model, jnp.asarray(offsets), jnp.asarray(site_idxs))
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)
    nq = int(mj.nq)
    names_qpos = list(stac._part_names)                 # per-qpos joint names, as stac_mjx writes them
    assert len(names_qpos) == nq
    kp_w = np.asarray(stac._kp_weights, np.float64)
    _mark("model+mjx setup")

    # which frames can be started: all warm-start keypoints finite (+ caller's mask)
    M = cfg.model
    names = list(kp_names)
    need = [names.index(str(M.ROOT_OPTIMIZATION_KEYPOINT))] + [
        names.index(str(M.JAXLS_ORIENTATION_KEYPOINTS[k])) for k in ("rear", "left", "right", "front")
        if k in M.JAXLS_ORIENTATION_KEYPOINTS]
    solvable = np.all([np.isfinite(kp[:, i]).all(axis=1) for i in need], axis=0)
    if solve_mask is not None:
        solvable &= np.asarray(solve_mask, bool)
    idx = np.flatnonzero(solvable)
    if idx.size == 0:
        raise ValueError("per-frame IK: no frame has finite root/orientation keypoints")
    kp_s = kp_flat[idx]
    q_init = warm_start(kp_s, names, M, np.asarray(mjx_data.qpos))

    solver = JaxlsBatchSolver(
        n_iter=int(pc.n_iter), smooth_weight=0.0, independent_frames=True,
        lambda_initial=float(pc.lambda_initial), lambda_min=float(pc.lambda_min),
        cost_tolerance=float(pc.cost_tolerance), gradient_tolerance=float(pc.gradient_tolerance),
        parameter_tolerance=float(pc.parameter_tolerance), independent_batch=int(pc.get("batch", 512)))
    common = dict(mjx_model=mjx_model, mjx_data_template=mjx_data, kp_data=jnp.asarray(kp_s),
                  qs_to_opt=jnp.ones(nq, bool), kps_to_opt=jnp.asarray(kp_w), lb=stac._lb, ub=stac._ub,
                  site_idxs=jnp.asarray(site_idxs), q_reg_weights=jnp.zeros(nq))
    cand_names, cand_q, cand_it, t_solve = [], [], [], {}
    ts = time.time()
    q_zero = np.asarray(solver.solve_trajectory(q_init=jnp.asarray(q_init), **common), np.float64)
    cand_names.append("zero"); cand_q.append(q_zero); cand_it.append(np.asarray(solver.last_iterations))
    t_solve["zero"] = round(time.time() - ts, 1)
    for name, q0 in chained_starts(q_zero, names_qpos, np.asarray(mj.qpos_spring), list(pc.starts)).items():
        ts = time.time()
        cand_q.append(np.asarray(solver.solve_trajectory(q_init=jnp.asarray(q0), **common), np.float64))
        cand_names.append(name); cand_it.append(np.asarray(solver.last_iterations)); t_solve[name] = round(time.time() - ts, 1)
    _mark("solves")
    costs = np.stack([frame_costs_np(mj, site_idxs, q, kp_s, kp_w) for q in cand_q])        # (S, Ts)
    _mark("candidate costs (CPU FK)")
    mask_all = np.ones(nq, bool)
    choice = viterbi_select(costs, cand_q, mask_all, float(pc.switch_weight),
                            dof_weights=wing_dof_weights(names_qpos, float(pc.get("switch_wing_weight", 1.0))))
    q_sel = np.stack([cand_q[choice[t]][t] for t in range(len(idx))])
    it_sel = np.stack([cand_it[choice[t]][t] for t in range(len(idx))])
    _mark("viterbi")

    # full-length arrays (unsolved frames NaN), FK-derived outputs, qvel
    q_full = np.full((T, nq), np.nan); q_full[idx] = q_sel
    # Output FK on the CPU: a jax.vmap of mjx kinematics over T frames spent
    # 128 s in XLA compilation for 1500 frames (profiled 2026-09-05) where
    # MuJoCo's C kinematics does the same in ~0.1 s.
    import mujoco
    dcpu = mujoco.MjData(mj)
    xpos = np.full((T, mj.nbody, 3), np.nan); xquat = np.full((T, mj.nbody, 4), np.nan); msites = np.full((T, len(site_idxs), 3), np.nan)
    for t in idx:
        dcpu.qpos[:] = q_full[t]
        mujoco.mj_kinematics(mj, dcpu)
        xpos[t] = dcpu.xpos; xquat[t] = dcpu.xquat; msites[t] = dcpu.site_xpos[site_idxs]
    _mark("FK (CPU)")
    qvel = qvel_from_qpos(np.nan_to_num(q_full), dt=mj.opt.timestep, freejoint=stac._freejoint)
    qvel[~solvable] = np.nan
    _mark("qvel")
    names_xpos = list(stac._body_names)
    start_idx = np.full(T, -1, np.int32); start_idx[idx] = choice
    iters = np.full(T, -1, np.int32); iters[idx] = it_sel
    extra = {"ik_start_idx": start_idx, "ik_iterations": iters}
    if bool(pc.get("save_candidates", True)):
        qc = np.full((len(cand_q), T, nq), np.nan, np.float32); qc[:, idx] = np.stack(cand_q).astype(np.float32)
        cc = np.full((len(cand_q), T), np.nan, np.float32); cc[:, idx] = costs
        extra["qpos_candidates"] = qc; extra["candidate_costs"] = cc
    write_stac_h5(out_h5, cfg=cfg, kp_names=names, names_qpos=names_qpos, names_xpos=names_xpos,
                  kp_data=kp_flat, marker_sites=msites, offsets=offsets, qpos=q_full, xpos=xpos, xquat=xquat, qvel=qvel,
                  extra=extra, attrs={"ik_solver": "per_frame", "ik_candidates": json.dumps(cand_names),
                                       "ik_switch_weight": float(pc.switch_weight),
                                       "ik_switch_wing_weight": float(pc.get("switch_wing_weight", 1.0))})
    _mark("h5 write")
    if _prof:
        print(f"{log_prefix} profile: " + ", ".join(f"{n} {t1 - t0:.1f}s" for (n0, t0), (n, t1) in zip(_marks[:-1], _marks[1:])), flush=True)
    n_cap = int(np.sum(it_sel >= int(pc.n_iter)))
    frac = {n: round(float(np.mean(choice == i)), 3) for i, n in enumerate(cand_names)}
    summary = dict(seconds=round(time.time() - t0, 1), solve_seconds=t_solve, n_frames=int(T), n_solved=int(idx.size),
                   chosen_fraction=frac, switches=int(np.sum(choice[1:] != choice[:-1])),
                   iterations_median=float(np.median(it_sel)), iterations_p95=float(np.percentile(it_sel, 95)),
                   frames_at_iteration_cap=n_cap, cost_median=float(np.nanmedian(costs[choice, np.arange(len(idx))])))
    print(f"{log_prefix} {idx.size}/{T} frames solved in {summary['seconds']} s (solves {t_solve}); chosen {frac}; "
          f"{summary['switches']} switches; LM iterations median {summary['iterations_median']:.0f} p95 "
          f"{summary['iterations_p95']:.0f}, {n_cap} frames at the cap of {int(pc.n_iter)}; marker cost median "
          f"{summary['cost_median']:.3e}", flush=True)
    return summary
