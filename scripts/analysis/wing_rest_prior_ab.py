"""A/B the wing-blade rest prior: does it stop the wing passing through the abdomen?

THE DEFECT. The IK has no contact model. On bout_00001 fly0 the fitted wings
interpenetrate the abdomen in 100% of frames at a median depth of 0.046 model
units (~20% of a wing length), while the model's OWN spring-rest pose only
grazes at -0.0013. So the body model is fine and the fit is not. The wing is
folded correctly -- yaw within 0.2 deg of rest -- but the BLADE ORIENTATION is
off: roll +15.8 deg and pitch +41.6 deg from rest, correlating with penetration
depth at r=0.80 and r=0.75. Three near-collinear wing keypoints fix where the
wing points, not its rotation about that axis, so the blade drifts into the body.

THE PRIOR. A weak L2 pull of wing_roll/wing_pitch toward the model's springref
(NOT toward zero -- rest is yaw +85.94, roll +40.11, pitch -57.30 deg).

EXPECTATION, read the figure against this:
  A  penetration depth falls toward the ~0.0013 the model shows at its own rest
     pose. Depth UNCHANGED means the prior is too weak to matter; depth driven
     NEGATIVE (a gap opening up) means it is too strong and is lifting the wing
     off an abdomen it should be resting on.
  B  marker residual barely moves. The blade DOFs are near-unobservable, so
     constraining them should cost almost nothing in fit. A clear rise means
     the prior is fighting real data rather than filling in an unobserved DOF.
  C  the SINGING MALE's extended wing is untouched: yaw trajectory overlays
     baseline and the wing still leaves rest by tens of degrees. This is the
     failure that would matter most -- an unconditional pull toward the FOLDED
     rest pose could drag an extended wing back toward the body and erase the
     courtship signal. If C degrades, the prior needs gating on wing extension.
  D  wing direction error vs the measured keypoints does not worsen.

A prior that fixes A while ruining C is not usable, so both clips are required:
bout_00001 fly0 has folded wings (the defect) and bout_00003 fly1 is the singer
with one wing extended (the thing to protect).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "third_party/jarvis_jax"), os.path.join(REPO, "stac-mjx")):
    sys.path.insert(0, _p)

WING_JOINTS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
               "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def penetration(m, d, wing_geoms, abd_geoms):
    """True min gap per wing from MuJoCo. Negative = interpenetrating."""
    import mujoco
    return [min(mujoco.mj_geomDistance(m, d, gw, ga, 1.0, None)
                for gw in wing_geoms[s] for ga in abd_geoms) for s in "LR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--weights", default="0,1e-3,1e-2,1e-1")
    ap.add_argument("--joints", default="wing_roll_left,wing_pitch_left,"
                                        "wing_roll_right,wing_pitch_right")
    ap.add_argument("--gate", action="store_true",
                    help="gate the prior on wing extension (roll/pitch driven by that wing's yaw)")
    ap.add_argument("--gate-sigma-deg", type=float, default=25.0)
    ap.add_argument("--tag", default="clip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py
    import mujoco
    from omegaconf import OmegaConf
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from stac_mjx.stac import _resolve_rest_prior, _resolve_reg_gate
    from viz.config import resolve_body_model_xml

    h5p = os.path.join(args.dir, "stac_ik.h5")
    dd = ioh5.load(h5p)
    kpn = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(dd["kp_names"])]
    with h5py.File(h5p, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
        names_q = [x.decode() for x in f["names_qpos"][()]]
    XML = resolve_body_model_xml(cfg.model.MJCF_PATH)

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    abd_geoms = [g for g in range(m.ngeom)
                 if bn[m.geom_bodyid[g]] and "abd" in bn[m.geom_bodyid[g]].lower()]
    wing_geoms = {s: [g for g in range(m.ngeom)
                      if m.geom_bodyid[g] == mujoco.mj_name2id(
                          m, mujoco.mjtObj.mjOBJ_BODY, f"wing_{'left' if s == 'L' else 'right'}")]
                  for s in "LR"}
    col = {n: names_q.index(n) for n in WING_JOINTS if n in names_q}
    rest = {n: float(np.degrees(m.qpos_spring[
        int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])]))
        for n in WING_JOINTS}

    inp = build_solver_inputs(h5p, XML)
    T_all = np.asarray(inp["q_init"]).shape[0]
    t0 = max(0, min(args.t0, T_all - 1))
    nt = min(args.nt, T_all - t0)
    sl = slice(t0, t0 + nt)
    q_init = np.asarray(inp["q_init"])[sl]
    kp = np.asarray(inp["kp_data"])[sl]
    print(f"[{args.tag}] {args.dir}\n  frames {t0}..{t0+nt} of {T_all}, nq={q_init.shape[1]}")
    print(f"  rest pose: " + ", ".join(f"{k.replace('wing_','')}={v:.1f}" for k, v in rest.items()))

    # FK rig with the FITTED marker offsets, for residuals
    sites = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    off = np.asarray(dd["offsets"], float)
    for i, n in enumerate(kpn):
        sn = f"tracking[{n}]"
        if sn in sites:
            m.site_pos[sites.index(sn)] = off[i]
    kp_used = [n for n in kpn if f"tracking[{n}]" in sites]
    idxs = [sites.index(f"tracking[{n}]") for n in kp_used]
    kp_cols = [kpn.index(n) for n in kp_used]
    wing_kp = [i for i, n in enumerate(kp_used) if "wing" in n.lower()]

    joints = args.joints.split(",")
    weights = [float(x) for x in args.weights.split(",")]
    rep = {"dir": args.dir, "tag": args.tag, "rest_deg": rest, "joints": joints,
           "weights": weights, "gated": bool(args.gate),
           "gate_sigma_deg": args.gate_sigma_deg, "results": {}}
    store = {}

    for w in weights:
        spec = None if w == 0 else {j: w for j in joints}
        qrw, qref = _resolve_rest_prior(m, spec)
        gate = None
        if args.gate and spec:
            gate = _resolve_reg_gate(m, {j: ("wing_yaw_left" if j.endswith("_left")
                                             else "wing_yaw_right") for j in joints},
                                     sigma_deg=args.gate_sigma_deg)
        solver = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1, use_se3_root=True,
                                  q_ref=qref, reg_gate=gate)
        q = np.asarray(solver.solve_trajectory(
            q_init=q_init, mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
            kp_data=kp, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
            lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
            q_reg_weights=(inp["q_reg_weights"] if qrw is None else qrw)))

        gaps, R = [], np.empty((len(q), len(idxs)))
        for t in range(len(q)):
            d.qpos[:] = q[t]
            mujoco.mj_forward(m, d)
            gaps.append(penetration(m, d, wing_geoms, abd_geoms))
            R[t] = np.linalg.norm(d.site_xpos[idxs] - kp[t].reshape(-1, 3)[kp_cols], axis=1)
        gaps = np.asarray(gaps)                     # (T,2)
        store[w] = dict(q=q, gaps=gaps, R=R)

        pen = -gaps
        rec = dict(
            pen_median=float(np.median(pen)), pen_max=float(pen.max()),
            pct_frames_penetrating=float(100 * np.mean(gaps.min(1) < 0)),
            resid_all=float(np.nanmedian(R)),
            resid_wing=float(np.nanmedian(R[:, wing_kp])) if wing_kp else float("nan"),
        )
        for n, c in col.items():
            x = np.degrees(q[:, c])
            rec[f"{n}_median"] = float(np.median(x))
            rec[f"{n}_dev_from_rest"] = float(np.median(x) - rest[n])
        rep["results"][str(w)] = rec
        print(f"  solved w={w:g}")

    np.savez_compressed(os.path.join(args.out, f"rest_ab_{args.tag}.npz"),
                        weights=np.array(weights), kp_used=np.array(kp_used),
                        **{f"q_{w}": store[w]["q"] for w in weights},
                        **{f"gaps_{w}": store[w]["gaps"] for w in weights},
                        **{f"R_{w}": store[w]["R"] for w in weights})

    b = rep["results"][str(weights[0])]
    print(f"\n{'weight':>9}{'pen median':>12}{'pen max':>10}{'% frames':>10}"
          f"{'resid all':>11}{'d%':>8}{'resid wing':>12}{'d%':>8}")
    for w in weights:
        r = rep["results"][str(w)]
        print(f"{w:9g}{r['pen_median']:12.4f}{r['pen_max']:10.4f}"
              f"{r['pct_frames_penetrating']:10.1f}{r['resid_all']:11.5f}"
              f"{100*(r['resid_all']-b['resid_all'])/b['resid_all']:8.2f}"
              f"{r['resid_wing']:12.5f}"
              f"{100*(r['resid_wing']-b['resid_wing'])/b['resid_wing']:8.2f}")
    print(f"\n{'weight':>9}" + "".join(f"{n.replace('wing_','').replace('_left','L').replace('_right','R'):>12}"
                                       for n in col))
    print(f"{'rest':>9}" + "".join(f"{rest[n]:12.1f}" for n in col))
    for w in weights:
        r = rep["results"][str(w)]
        print(f"{w:9g}" + "".join(f"{r[f'{n}_median']:12.1f}" for n in col))

    with open(os.path.join(args.out, f"rest_ab_{args.tag}.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {args.out}/rest_ab_{args.tag}.json")


if __name__ == "__main__":
    main()
