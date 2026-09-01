#!/usr/bin/env python3
"""Render the wing PITCH rest prior: control vs treatment, on real solved poses.

WHAT IS WRONG TODAY. Each wing body carries only two markers (WingX_base maps to
the THORAX), so two points give an axis, not an orientation, and the leftover
rotation is pitch: per-column marker sensitivity is yaw 0.271 / roll 0.344 /
pitch 0.042, and the Jacobian's null direction is 99.6% pitch. Unconstrained,
the solver rolls the blade into the abdomen -- the folded wing sits ~58 deg off
its rest pitch and the wing/abdomen geoms interpenetrate.

EXPECTATION -- read the panels against this, and note the SONG guard is the one
that matters, because a smoothness prior on ROLL was already tried and destroyed
it (roll is the STRONGEST-observed wing DOF, which is why):
  * TREATMENT (right): both wing blades lie flat along the abdomen, following
    the body's dorsal surface, and no grey wing surface passes through the
    orange abdomen.
  * CONTROL (left): at least the NON-SINGING wing should visibly cut into the
    abdomen. If control already looks flat on the frames shown, the frames were
    badly chosen -- pick frames by penetration depth, not at random.
  * THE SINGING WING MUST NOT CHANGE. The male sings by unilateral EXTENSION,
    which lives in yaw. If the extended wing's angle differs between panels,
    the prior is stealing real motion and the weight is too high -- that is a
    reject regardless of how flat the resting wing looks.
  * The dorsal view is the one that shows "flat on the back"; the posterior view
    is the one that shows penetration. Both are included because either alone
    can hide the failure.

Frames are chosen as the WORST control penetration frames, so the panel shows the
defect rather than a flattering moment.
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _m in [k for k in list(sys.modules) if k == "viz" or k.startswith("viz.")]:
    del sys.modules[_m]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

WING = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
        "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def solve(inp, mj, q_init, kp, weight):
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from stac_mjx.stac import _resolve_rest_prior
    spec = None if weight == 0 else {"wing_pitch_left": weight,
                                     "wing_pitch_right": weight}
    qw, qref = _resolve_rest_prior(mj, spec)
    s = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1, use_se3_root=True, q_ref=qref)
    return np.asarray(s.solve_trajectory(
        q_init=q_init, mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
        kp_data=kp, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
        lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
        q_reg_weights=(inp["q_reg_weights"] if qw is None else qw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help=".../bouts/bout_000NN/flyF")
    ap.add_argument("--weight", type=float, default=0.01)
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--views", default="dorsal,posterior",
                    help="dorsal shows FLAT-ON-THE-BACK; posterior/lateral show "
                         "penetration. More than one, because either alone can "
                         "hide the failure.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import mujoco, h5py, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from omegaconf import OmegaConf
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.tracking.ik_solve import build_solver_inputs
    from viz.config import resolve_body_model_xml

    h5p = os.path.join(args.dir, "stac_ik.h5")
    d = ioh5.load(h5p)
    with h5py.File(h5p, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
    XML = resolve_body_model_xml(cfg.model.MJCF_PATH)

    spec = mujoco.MjSpec.from_file(XML)
    spec.visual.global_.offwidth = int(args.width)
    spec.visual.global_.offheight = int(args.height)
    # fovy is the zoom: the default 45 deg puts the fly at a few percent of the
    # frame and the wing/abdomen relationship becomes unreadable.
    for nm, pos in (("dorsal", [0.0, 0.0, 0.55]),
                    ("posterior", [-0.60, 0.0, 0.10]),
                    ("lateral", [0.0, -0.60, 0.10])):
        c = spec.worldbody.add_camera()
        c.name = nm
        c.pos = pos
        c.fovy = 9.0
        c.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    mj = spec.compile()
    dat = mujoco.MjData(mj)
    # aim both cameras at the thorax
    tb = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "thorax")
    for nm in ("dorsal", "posterior", "lateral"):
        cid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_CAMERA, nm)
        mj.cam_targetbodyid[cid] = tb

    adr = {n: int(mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)])
           for n in WING}
    rest = np.asarray(mj.qpos_spring, float)
    wg = {sd: [g for g in range(mj.ngeom)
               if mj.geom_bodyid[g] == mujoco.mj_name2id(
                   mj, mujoco.mjtObj.mjOBJ_BODY, f"wing_{sd}")]
          for sd in ("left", "right")}
    abd = [g for g in range(mj.ngeom)
           if "abdomen" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY,
                                              int(mj.geom_bodyid[g])) or "")]

    inp = build_solver_inputs(h5p, XML)
    nt = min(args.nt, np.asarray(inp["q_init"]).shape[0])
    q_init = np.asarray(inp["q_init"])[:nt]
    kp = np.asarray(inp["kp_data"])[:nt]
    print(f"solving {nt} frames: control and weight={args.weight}")
    qc = solve(inp, mj, q_init, kp, 0.0)
    qt = solve(inp, mj, q_init, kp, args.weight)

    def pen(q, t):
        dat.qpos[:] = q[t]; mujoco.mj_forward(mj, dat)
        return {sd: min(mujoco.mj_geomDistance(mj, dat, g1, g2, 1.0, None)
                        for g1 in wg[sd] for g2 in abd) for sd in ("left", "right")}
    depth = np.array([min(pen(qc, t).values()) for t in range(nt)])
    picks = list(np.argsort(depth)[:args.n_frames])       # most negative = worst
    print(f"  worst control penetration frames: {picks} "
          f"(depths {np.round(depth[picks],5).tolist()})")

    rend = mujoco.Renderer(mj, height=int(args.height), width=int(args.width))
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    rows = [(t, v) for t in picks for v in views]
    fig, axes = plt.subplots(len(rows), 2,
                             figsize=(5.2 * 2, 4.2 * len(rows)), squeeze=False)
    for r, (t, view) in enumerate(rows):
        for c, (label, q) in enumerate((("CONTROL (no prior)", qc),
                                        (f"TREATMENT (pitch prior w={args.weight})", qt))):
            dat.qpos[:] = q[t]; mujoco.mj_forward(mj, dat)
            rend.update_scene(dat, camera=view)
            axes[r][c].imshow(rend.render())
            p = pen(q, t)
            po = {n: np.degrees(q[t, adr[n]] - rest[adr[n]]) for n in WING}
            yl = np.degrees(q[t, adr["wing_yaw_left"]])
            yr = np.degrees(q[t, adr["wing_yaw_right"]])
            axes[r][c].set_title(
                f"{label}   f{t} {view}\n"
                f"penetration  L {p['left']:+.4f}   R {p['right']:+.4f}\n"
                f"pitch off rest  L {po['wing_pitch_left']:+.0f}\u00b0  "
                f"R {po['wing_pitch_right']:+.0f}\u00b0\n"
                f"SONG |yawL-yawR| = {abs(yl-yr):.0f}\u00b0",
                fontsize=8)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    fig.suptitle("Wing pitch rest prior -- wings should lie FLAT along the abdomen, "
                 "not pass through it.\nNegative penetration = blade inside the body. "
                 "The SINGING wing's yaw must be unchanged between panels.",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(f"{args.out}.png", dpi=130)
    print(f"wrote {args.out}.png")
    for t in picks:
        pc, pt = pen(qc, t), pen(qt, t)
        print(f"  frame {t}: penetration L {pc['left']:+.5f} -> {pt['left']:+.5f}   "
              f"R {pc['right']:+.5f} -> {pt['right']:+.5f}")


if __name__ == "__main__":
    main()
