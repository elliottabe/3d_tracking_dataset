"""Build and validate a differentiable wing/abdomen penetration proxy.

The IK has no contact model, so nothing stops the solver driving the wing blade
through the abdomen. Measured on bout_00001 fly0: the fitted poses interpenetrate
in 100% of frames at a median depth of 0.043 model units (~20% of a wing length),
while the model's OWN spring-rest pose (wings folded) only grazes at -0.0013.
So the geometry is fine and the fit is not; the blade orientation DOFs (roll off
rest by 16 deg, pitch by 42 deg, r=0.80/0.75 against penetration depth) are
near-unobservable from three near-collinear wing keypoints and drift into the body.

The proxy has to be cheap and differentiable inside the jaxls cost, so:
  * the wing is SAMPLED AS POINTS (its geoms are near-flat -- semi-axes
    0.0032 x 0.0758 x 0.1564 -- so a sphere approximation would be useless);
  * each abdomen body becomes ONE sphere.

EXPECTATION, and the reason this file validates before anything is wired into
the solver: the proxy must agree with MuJoCo's own mj_geomDistance on the two
poses we already measured -- near zero at spring rest (the folded wing is
anatomically correct and the prior must NOT fight it) and clearly positive on
the fitted poses. A proxy that flags the rest pose would push every folded wing
off the abdomen and corrupt exactly the poses that are currently right.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np


def _geom_local_extent(m, g):
    """(lo, hi) of geom g in ITS BODY's frame, mesh vertices included."""
    import mujoco

    gp = m.geom_pos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n].reshape(-1, 3)
        # geom frame -> body frame
        R = m.geom_quat[g]
        import mujoco as _mj
        Rm = np.zeros(9)
        _mj.mju_quat2Mat(Rm, R)
        v = v @ Rm.reshape(3, 3).T
        return v.min(0) + gp, v.max(0) + gp
    return gp - m.geom_size[g], gp + m.geom_size[g]


def build_spec(xml, *, wing_bodies=("wing_left", "wing_right"),
               abd_pattern=r"^abdomen(_\d+)?$", n_wing_pts=10):
    """Wing sample points (body-local) + one sphere per abdomen body."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]

    # ---- wing sample points, from the visual mesh vertices ----
    wing_body_ids, wing_pts = [], []
    for wb in wing_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, wb)
        V = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V.append(m.mesh_vert[a:a + n].reshape(-1, 3) + m.geom_pos[g])
        V = np.concatenate(V, 0)
        # spread the samples along the wing's long axis (y) so the tip, which is
        # what swings through the abdomen, is represented rather than averaged away
        order = np.argsort(V[:, 1])
        idx = np.linspace(0, len(order) - 1, n_wing_pts).astype(int)
        wing_body_ids.append(bid)
        wing_pts.append(V[order[idx]])

    # ---- one sphere per abdomen body ----
    abd_body_ids, abd_centers, abd_radii = [], [], []
    for i, n in enumerate(bname):
        if not n or not re.match(abd_pattern, n):
            continue
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        found = False
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != i:
                continue
            e0, e1 = _geom_local_extent(m, g)
            lo, hi, found = np.minimum(lo, e0), np.maximum(hi, e1), True
        if not found:
            continue
        c = 0.5 * (lo + hi)
        abd_body_ids.append(i)
        abd_centers.append(c)
        # inscribed-ish: the half-extent of the two SMALLER axes. The full
        # half-diagonal would swallow the whole segment and shove the wing far
        # off the body; abdomen segments are short so this tracks the surface.
        abd_radii.append(float(np.mean(np.sort(0.5 * (hi - lo))[:2])))

    return dict(
        wing_body_ids=np.array(wing_body_ids, np.int32),
        wing_pts=np.array(wing_pts, np.float64),            # (W, P, 3)
        abd_body_ids=np.array(abd_body_ids, np.int32),
        abd_centers=np.array(abd_centers, np.float64),      # (A, 3)
        abd_radii=np.array(abd_radii, np.float64),          # (A,)
        body_names=[bname[i] for i in abd_body_ids],
    )


def penetration_numpy(m, d, spec, margin=0.0):
    """Max penetration depth per wing, using the same math the cost will use."""
    out = []
    for w, bid in enumerate(spec["wing_body_ids"]):
        R = d.xmat[bid].reshape(3, 3)
        pw = spec["wing_pts"][w] @ R.T + d.xpos[bid]              # (P,3)
        c = np.stack([d.xmat[b].reshape(3, 3) @ spec["abd_centers"][a] + d.xpos[b]
                      for a, b in enumerate(spec["abd_body_ids"])])   # (A,3)
        dist = np.linalg.norm(pw[:, None, :] - c[None, :, :], axis=-1)  # (P,A)
        pen = np.maximum(0.0, (spec["abd_radii"] + margin)[None, :] - dist)
        out.append(float(pen.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="stac_ik.h5 with fitted qpos")
    ap.add_argument("--n-pts", type=int, default=10)
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    import h5py
    import mujoco

    with h5py.File(args.h5, "r") as f:
        q = f["qpos"][()]
        cfg = f["config"][()].decode()
    xml = re.search(r"MJCF_PATH:\s*(\S+)", cfg).group(1)

    spec = build_spec(xml, n_wing_pts=args.n_pts)
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    print(f"wing sample points: {spec['wing_pts'].shape}")
    print(f"abdomen spheres:    {len(spec['abd_radii'])}  {spec['body_names']}")
    print(f"  radii: {np.round(spec['abd_radii'], 4)}")

    # --- truth from MuJoCo, for the two reference poses ---
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    ag = [g for g in range(m.ngeom)
          if bn[m.geom_bodyid[g]] and "abd" in bn[m.geom_bodyid[g]].lower()]
    wg = {s: [g for g in range(m.ngeom)
              if m.geom_bodyid[g] == mujoco.mj_name2id(
                  m, mujoco.mjtObj.mjOBJ_BODY, f"wing_{'left' if s == 'L' else 'right'}")]
          for s in "LR"}

    def truth():
        return [min(mujoco.mj_geomDistance(m, d, gw, ga, 1.0, None)
                    for gw in wg[s] for ga in ag) for s in "LR"]

    J = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
         for n in ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                   "wing_yaw_right", "wing_roll_right", "wing_pitch_right")}

    print(f"\n{'pose':26}{'mj_geomDistance gap':>24}{'proxy penetration':>22}")
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    print(f"{'qpos0':26}{str(np.round(truth(), 4)):>24}"
          f"{str(np.round(penetration_numpy(m, d, spec), 4)):>22}")

    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos0
    for n, a in J.items():
        d.qpos[a] = m.qpos_spring[a]
    mujoco.mj_forward(m, d)
    rest_truth, rest_proxy = truth(), penetration_numpy(m, d, spec)
    print(f"{'SPRING REST (folded)':26}{str(np.round(rest_truth, 4)):>24}"
          f"{str(np.round(rest_proxy, 4)):>22}   <-- prior must not fight this")

    T = min(len(q), args.frames)
    step = max(1, T // 100)
    tv, pv = [], []
    for t in range(0, T, step):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        tv.append(min(truth()))
        pv.append(max(penetration_numpy(m, d, spec)))
    tv, pv = np.array(tv), np.array(pv)
    print(f"{'FITTED (median)':26}{np.median(tv):>24.4f}{np.median(pv):>22.4f}")

    from scipy import stats
    r, p = stats.pearsonr(-tv, pv)
    print(f"\nproxy vs truth over {len(tv)} fitted frames: r = {r:+.3f} (p={p:.1g})")
    print(f"suggested margin so SPRING REST is feasible: {-max(rest_proxy):.4f} "
          f"(i.e. subtract {max(rest_proxy):.4f} from the radii)")
    print(json.dumps({"rest_proxy_max": float(max(rest_proxy)),
                      "fitted_proxy_median": float(np.median(pv)),
                      "truth_fitted_median_depth": float(-np.median(tv)),
                      "proxy_truth_r": float(r)}, indent=2))


if __name__ == "__main__":
    main()
