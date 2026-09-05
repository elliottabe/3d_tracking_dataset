"""Per-frame weak-DOF polish after the STAC batch solve.

Why this exists (measured 2026-09-04/05, Session0/2025_10_20_13_20_04 bout 28
fly1, `docs/benchmark/2026-09-04-wing-ik-spike/notes.md`): the jaxls batch
solve initialises every hinge at 0, descends the folded wing's blade pitch to
about -20 deg, and stalls there with the wing yaw pinned on its joint stop --
the constrained LM step cannot follow the curved valley toward the marker
optimum near -45 deg, so the wing renders edge-on. Tolerances do not help
(same final cost at 1e-5 and 1e-12). Started from the model's REST pitch
instead, the same solver with the same bounds reaches the optimum in ~70
iterations (wing-tip residual 1.15 -> 0.56 mm); but re-initialising an
EXTENDED wing at rest drops it into a wrong basin. So: several starts per
frame, choose by marker cost, per frame.

The polish uses `JaxlsBatchSolver`'s vmapped independent-frame path (no
temporal term, dense per-frame factorisation, per-frame termination, seconds
for a bout), and writes back ONLY the DOFs named by `stac.polish.dofs`
(wings and abdomen by default); root and legs keep the smoothed batch
solution. The batch qpos is kept in the h5 as `qpos_batch`, with the chosen
start and the per-frame cost before/after, so the change is auditable.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from typing import Dict, List, Sequence

import numpy as np

POLISH_ATTR = "polish_sig"
BATCH_KEY = "qpos_batch"
CANDIDATE_NONE = "none"          # the untouched batch pose, always a candidate


def polish_signature(pcfg) -> str:
    """Stable hash of every polish setting that changes the result."""
    keys = ("dofs", "starts", "lambda_initial", "lambda_min", "cost_tolerance",
            "gradient_tolerance", "parameter_tolerance", "n_iter", "switch_weight")
    payload = {k: (list(pcfg[k]) if isinstance(pcfg.get(k), (list, tuple)) or
                   type(pcfg.get(k)).__name__ == "ListConfig" else pcfg.get(k))
               for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def dof_mask(names_qpos: Sequence[str], patterns: Sequence[str]) -> np.ndarray:
    """Boolean (nq,) mask of qpos entries whose joint name matches any glob."""
    return np.array([any(fnmatch.fnmatch(n, p) for p in patterns) for n in names_qpos], bool)


def build_starts(q_batch: np.ndarray, names_qpos: Sequence[str], qpos_spring: np.ndarray,
                 starts: Sequence[str]) -> Dict[str, np.ndarray]:
    """{start name: (T, nq) initial qpos}. Known starts:
      batch            -- the batch solution itself
      wing_rest_left   -- batch, with wing_pitch_left set to its spring rest
      wing_rest_right  -- same for the right wing
      wing_rest_both   -- both wings' pitch at rest
    The rest value is the model's springref (`qpos_spring`), NOT zero: for the
    fruitfly the folded-wing rest pitch is -57.3 deg and pulling toward 0
    would unfold the wing."""
    names = list(names_qpos)
    ipl, ipr = names.index("wing_pitch_left"), names.index("wing_pitch_right")
    out = {}
    for s in starts:
        q = np.array(q_batch, dtype=np.float64, copy=True)
        if s == "batch":
            pass
        elif s == "wing_rest_left":
            q[:, ipl] = qpos_spring[ipl]
        elif s == "wing_rest_right":
            q[:, ipr] = qpos_spring[ipr]
        elif s == "wing_rest_both":
            q[:, ipl] = qpos_spring[ipl]; q[:, ipr] = qpos_spring[ipr]
        else:
            raise ValueError(f"stac.polish.starts: unknown start {s!r}")
        out[s] = q
    return out


def select_by_cost(costs: np.ndarray) -> np.ndarray:
    """(S, T) per-candidate per-frame costs -> (T,) index of the cheapest
    candidate. NaN costs never win; a frame where every candidate is NaN gets
    index 0 (the caller places the untouched batch pose at index 0)."""
    c = np.where(np.isfinite(costs), costs, np.inf)
    idx = np.argmin(c, axis=0)
    idx[~np.isfinite(c).any(axis=0)] = 0
    return idx


def viterbi_select(costs: np.ndarray, merged: Sequence[np.ndarray], mask: np.ndarray,
                   switch_weight: float, dof_weights: np.ndarray | None = None) -> np.ndarray:
    """Temporally consistent candidate choice: minimise
        sum_t cost[c_t, t] + switch_weight * sum_t ||q_t(c_t) - q_{t-1}(c_{t-1})||^2 over masked DOFs.
    With switch_weight == 0 this is `select_by_cost`. Why: for a fly whose two
    wing-pitch basins have near-equal marker cost, a per-frame argmin flips
    between them on noise (measured 2026-09-05: female wing flat/edge-on
    alternating frame to frame, pitch jitter x2.4). A genuine ~1 deg/frame motion
    costs ~3e-4 rad^2 * w; a 30 deg basin jump costs 0.27 rad^2 * w per DOF.
    Non-finite costs are treated as +inf; non-finite poses as no-jump (NaN frames
    are unsolved and keep candidate 0 downstream)."""
    S, T = costs.shape
    if switch_weight <= 0 or T == 1:
        return select_by_cost(costs)
    unary = np.where(np.isfinite(costs), costs, np.inf)
    Qm = np.stack([np.where(np.isfinite(q[:, mask]), q[:, mask], 0.0) for q in merged])   # (S, T, M)
    # Optional per-DOF weights on the jump penalty (length nq, masked here):
    # a basin flip on a wing DOF can be made expensive while ordinary leg
    # motion between candidates stays cheap.
    w = np.ones(int(mask.sum())) if dof_weights is None else np.asarray(dof_weights, float)[mask]
    D = np.zeros((T, S, S))
    for t in range(1, T):
        diff = Qm[:, t, None, :] - Qm[None, :, t - 1, :]                                   # (S_t, S_{t-1}, M)
        D[t] = switch_weight * np.sum(w * diff * diff, axis=-1)
    score = unary[:, 0].copy(); back = np.zeros((T, S), np.int32)
    for t in range(1, T):
        tot = score[None, :] + D[t]                       # (S_t, S_{t-1})
        back[t] = np.argmin(tot, axis=1)
        score = unary[:, t] + tot[np.arange(S), back[t]]
    idx = np.empty(T, np.int32); idx[-1] = int(np.argmin(score))
    for t in range(T - 1, 0, -1):
        idx[t - 1] = back[t, idx[t]]
    idx[~np.isfinite(unary).any(axis=0)] = 0
    return idx


def merge_dofs(q_batch: np.ndarray, q_chosen: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Batch pose everywhere except the masked DOFs, which come from the
    chosen polished pose. Frames whose chosen pose is non-finite keep batch."""
    out = np.array(q_batch, dtype=np.float64, copy=True)
    ok = np.isfinite(q_chosen).all(axis=1)
    out[np.ix_(ok, mask)] = q_chosen[np.ix_(ok, mask)]
    return out


def merged_candidates(q_batch: np.ndarray, cand_q: Sequence[np.ndarray], mask: np.ndarray) -> List[np.ndarray]:
    """Each candidate grafted onto the batch pose on the masked DOFs only.
    Candidate 0 is the untouched batch pose by construction."""
    return [merge_dofs(q_batch, q, mask) for q in cand_q]


def frame_costs_np(mj_model, site_idxs, q: np.ndarray, kp: np.ndarray, kp_w: np.ndarray) -> np.ndarray:
    """(T,) sum over finite keypoint coords of ((site - kp) * w)^2, MuJoCo CPU FK.
    Same residual the solver minimises (NaN keypoints contribute nothing)."""
    import mujoco
    d = mujoco.MjData(mj_model)
    T = q.shape[0]
    out = np.full(T, np.nan)
    kp3 = kp.reshape(T, -1, 3)
    w3 = kp_w.reshape(-1, 3)
    for t in range(T):
        if not np.isfinite(q[t]).all():
            continue
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(mj_model, d)
        r = (d.site_xpos[site_idxs] - kp3[t]) * w3
        r = np.where(np.isfinite(r), r, 0.0)
        out[t] = float(np.sum(r * r))
    return out


def polish_stac_h5(cfg, stac_h5_path: str, kp_names: List[str], *, xml_path: str,
                   log_prefix: str = "[polish]") -> dict:
    """Polish the weak DOFs of an existing stac_ik.h5 IN PLACE. Idempotent on
    the polish signature: a file already carrying this signature is left alone.
    Returns a summary dict (also printed)."""
    import h5py
    import jax.numpy as jnp
    import mujoco
    from omegaconf import OmegaConf
    from stac_mjx import utils
    from stac_mjx.stac import Stac
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver

    pcfg = cfg.stac.polish
    sig = polish_signature(pcfg)
    with h5py.File(stac_h5_path, "r") as f:
        if f.attrs.get(POLISH_ATTR) == sig:
            print(f"{log_prefix} {stac_h5_path}: already polished (sig {sig}), skipping", flush=True)
            return {"skipped": True, "sig": sig}
        q_batch = np.asarray(f[BATCH_KEY][:] if BATCH_KEY in f else f["qpos"][:], np.float64)
        kp_data = np.asarray(f["kp_data"][:], np.float64)
        offsets = np.asarray(f["offsets"][:], np.float64)
        names_qpos = [n.decode() if isinstance(n, bytes) else str(n) for n in f["names_qpos"][:]]
        h5_kp_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["kp_names"][:]]
        has_qvel = "qvel" in f
    if h5_kp_names != list(kp_names):
        raise ValueError(f"{stac_h5_path}: kp_names differ from the pipeline's KP_NAMES")
    T, nq = q_batch.shape

    t0 = time.time()
    stac = Stac(str(xml_path), cfg, list(kp_names))
    mj = stac._mj_model
    site_idxs = np.asarray(stac._body_site_idxs)
    mj.site_pos[site_idxs] = offsets                    # the run's fitted offsets
    mjx_model, mjx_data = utils.mjx_load(mj)
    mjx_model = utils.set_site_pos(mjx_model, jnp.asarray(offsets), jnp.asarray(site_idxs))
    from mujoco import mjx
    mjx_data = mjx.kinematics(mjx_model, mjx_data)
    mjx_data = mjx.com_pos(mjx_model, mjx_data)
    kp_w = np.asarray(stac._kp_weights, np.float64)
    qpos_spring = np.asarray(mj.qpos_spring, np.float64)

    solver = JaxlsBatchSolver(
        n_iter=int(pcfg.n_iter), smooth_weight=0.0, independent_frames=True,
        lambda_initial=float(pcfg.lambda_initial), lambda_min=float(pcfg.lambda_min),
        cost_tolerance=float(pcfg.cost_tolerance), gradient_tolerance=float(pcfg.gradient_tolerance),
        parameter_tolerance=float(pcfg.parameter_tolerance),
        independent_batch=int(pcfg.get("batch", 512)))
    solved = np.isfinite(q_batch).all(axis=1)          # unsolved (NaN) frames stay NaN
    starts = build_starts(q_batch[solved], names_qpos, qpos_spring, list(pcfg.starts))
    common = dict(mjx_model=mjx_model, mjx_data_template=mjx_data,
                  kp_data=jnp.asarray(kp_data[solved]), qs_to_opt=jnp.ones(nq, bool),
                  kps_to_opt=jnp.asarray(kp_w), lb=stac._lb, ub=stac._ub,
                  site_idxs=jnp.asarray(site_idxs), q_reg_weights=jnp.zeros(nq))
    mask = dof_mask(names_qpos, list(pcfg.dofs))
    cand_names = [CANDIDATE_NONE]
    cand_q = [q_batch[solved]]
    for name, q0 in starts.items():
        cand_names.append(name)
        cand_q.append(np.asarray(solver.solve_trajectory(q_init=jnp.asarray(q0), **common), np.float64))
    # Rank candidates by the cost of what would actually be WRITTEN: the
    # candidate's masked DOFs grafted onto the batch root/legs. Ranking the
    # full polished pose instead let a graft lose against the untouched batch
    # (measured: median cost UP 6.07e-3 -> 6.63e-3 on the female, 2026-09-05).
    merged = merged_candidates(q_batch[solved], cand_q, mask)
    costs = np.stack([frame_costs_np(mj, site_idxs, q, kp_data[solved], kp_w) for q in merged])   # (S, Ts)
    choice = viterbi_select(costs, merged, mask, float(pcfg.get("switch_weight", 0.0)))
    q_new_solved = np.stack([merged[choice[t]][t] for t in range(int(solved.sum()))])
    cost_after = costs[choice, np.arange(len(choice))]
    q_new = np.array(q_batch, copy=True); q_new[solved] = q_new_solved
    choice_full = np.full(T, -1, np.int32); choice_full[solved] = choice
    cb_full = np.full(T, np.nan); cb_full[solved] = costs[0]
    ca_full = np.full(T, np.nan); ca_full[solved] = cost_after

    # FK for the derived arrays, same convention as compute_stac
    import jax
    def fk(q):
        data = mjx_data.replace(qpos=q)
        data = utils.kinematics(mjx_model, data)
        data = utils.com_pos(mjx_model, data)
        return data.xpos, data.xquat, utils.get_site_xpos(data, jnp.asarray(site_idxs))
    xpos, xquat, msites = jax.vmap(fk)(jnp.asarray(np.where(np.isfinite(q_new), q_new, 0.0)))
    xpos, xquat, msites = (np.array(a, dtype=np.float64) for a in (xpos, xquat, msites))   # writable copies
    for a in (xpos, xquat, msites):
        a[~solved] = np.nan
    qvel = None
    if has_qvel:
        qv = utils.compute_velocity_from_kinematics(
            jnp.asarray(np.where(np.isfinite(q_new), q_new, 0.0)), dt=mj.opt.timestep, freejoint=stac._freejoint)
        qvel = np.array(qv, dtype=np.float64); qvel[~solved] = np.nan

    with h5py.File(stac_h5_path, "a") as f:
        if BATCH_KEY not in f:
            f.create_dataset(BATCH_KEY, data=q_batch.astype(np.float32))
        for k, v in (("qpos", q_new), ("xpos", xpos), ("xquat", xquat), ("marker_sites", msites)):
            f[k][...] = v.astype(f[k].dtype)
        if qvel is not None:
            f["qvel"][...] = qvel.astype(f["qvel"].dtype)
        for k, v in (("polish_start_idx", choice_full), ("polish_cost_before", cb_full), ("polish_cost_after", ca_full)):
            if k in f: del f[k]
            f.create_dataset(k, data=v)
        f.attrs[POLISH_ATTR] = sig
        f.attrs["polish_candidates"] = json.dumps(cand_names)
        f.attrs["polish_dofs"] = json.dumps([n for n, m in zip(names_qpos, mask) if m])

    frac = {n: float(np.mean(choice == i)) for i, n in enumerate(cand_names)}
    n_switch = int(np.sum(choice[1:] != choice[:-1]))
    ipl, ipr = names_qpos.index("wing_pitch_left"), names_qpos.index("wing_pitch_right")
    summary = {
        "sig": sig, "seconds": round(time.time() - t0, 1), "n_frames": int(T), "n_solved": int(solved.sum()),
        "chosen_fraction": frac,
        "cost_median_before": float(np.nanmedian(costs[0])), "cost_median_after": float(np.nanmedian(cost_after)),
        "frames_improved_fraction": float(np.mean(cost_after < costs[0] - 1e-12)),
        "wing_pitch_left_med_deg": [float(np.degrees(np.nanmedian(q_batch[:, ipl]))), float(np.degrees(np.nanmedian(q_new[:, ipl])))],
        "wing_pitch_right_med_deg": [float(np.degrees(np.nanmedian(q_batch[:, ipr]))), float(np.degrees(np.nanmedian(q_new[:, ipr])))],
        "dofs_written": int(mask.sum()),
        "candidate_switches": n_switch,
    }
    print(f"{log_prefix} {stac_h5_path}: {summary['n_solved']}/{T} frames, {summary['seconds']} s; "
          f"chosen {frac}; marker cost median {summary['cost_median_before']:.3e} -> {summary['cost_median_after']:.3e} "
          f"({100*summary['frames_improved_fraction']:.0f}% frames improved); wing_pitch L/R median deg "
          f"{summary['wing_pitch_left_med_deg'][0]:.1f}->{summary['wing_pitch_left_med_deg'][1]:.1f} / "
          f"{summary['wing_pitch_right_med_deg'][0]:.1f}->{summary['wing_pitch_right_med_deg'][1]:.1f}; "
          f"{mask.sum()} DOFs written; {n_switch} candidate switches", flush=True)
    return summary
