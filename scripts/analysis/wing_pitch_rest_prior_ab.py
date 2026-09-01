#!/usr/bin/env python3
"""A/B a REST prior on wing PITCH -- the provably unobservable wing DOF.

WHY PITCH, and why this is not the attempt that already failed:
  * measured at the spring-rest pose with the run's own fitted offsets, the
    Jacobian d[V12;V13]/d[yaw,roll,pitch] has per-column sensitivity
    yaw 0.271 / roll 0.344 / pitch 0.042, and its NULL direction is
    [0.035, 0.088, -0.996] -- 99.6% pitch, 13x weaker than the best direction.
    Each wing body carries only TWO markers (WingX_base maps to the THORAX), so
    two points give an axis, not an orientation, and the leftover rotation is
    pitch.
  * the earlier rejected attempt was a SMOOTHNESS prior (JAXLS_SMOOTH_Q_MULT)
    on ROLL. Roll is the STRONGEST-observed wing DOF here, which is exactly why
    it destroyed the extended wing's signal. Different mechanism, different DOF.
  * consequence in the fit: the folded wing sits 41.6 deg off rest in pitch (vs
    15.8 in roll), the blade/abdomen interpenetrates on 100% of frames at ~20%
    of a wing length, and the measured-vs-fitted blade normal is 13-34 deg off.

EXPECTATION -- read the output against this, and note the song guard comes FIRST
because that is what killed the last attempt:
  1. SONG PRESERVED. The male sings by unilateral wing EXTENSION, which lives in
     yaw: |yaw_L - yaw_R| and its pulse statistics (hp_rms, kurtosis, peak count)
     must stay at baseline. If they drop, the prior is stealing real motion and
     the weight is too high -- STOP, whatever the pitch numbers say.
  2. PITCH MOVES TOWARD REST, and by a lot, since the markers barely constrain
     it: |pitch - rest| should fall from ~40 deg toward single digits.
  3. MARKER FIT BARELY CHANGES. Per-keypoint residual on the wing markers should
     rise only slightly -- the prior acts in the null space, so if residuals jump
     it is fighting real signal.
  4. ROLL IS COLLATERAL. roll.pitch column correlation is 0.74, so watch roll:
     it should move much less than pitch. If roll follows pitch closely, the
     prior is not as targeted as the SVD suggests.

Run:
  python scripts/analysis/wing_pitch_rest_prior_ab.py \
      --dir <.../bouts/bout_00028/fly1> --weights 0,0.01,0.1,1.0 --out figures/<topic>
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


def pulse_stats(x, fs, hp_hz=40.0):
    """High-passed RMS, kurtosis and peak count -- pulse song is a sparse train
    of brief transients, so narrowband power at a carrier does NOT measure it."""
    from scipy.signal import butter, filtfilt, find_peaks
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return dict(hp_rms=np.nan, kurtosis=np.nan, n_peaks=0)
    b, a = butter(2, hp_hz / (fs / 2.0), btype="high")
    y = filtfilt(b, a, x - x.mean())
    s = float(np.std(y))
    k = float(np.mean(((y - y.mean()) / (s + 1e-12)) ** 4))
    pk, _ = find_peaks(np.abs(y), height=2.0 * s)
    return dict(hp_rms=s, kurtosis=k, n_peaks=int(pk.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help=".../bouts/bout_000NN/flyF")
    ap.add_argument("--weights", default="0,0.01,0.1,1.0")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--nt", type=int, default=600)
    ap.add_argument("--fs", type=float, default=800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco, h5py, json
    from omegaconf import OmegaConf
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.tracking.ik_solve import build_solver_inputs
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from stac_mjx.stac import _resolve_rest_prior
    from viz.config import resolve_body_model_xml

    h5p = os.path.join(args.dir, "stac_ik.h5")
    d = ioh5.load(h5p)
    kpn = [x.decode() if isinstance(x, bytes) else str(x)
           for x in np.asarray(d["kp_names"])]
    with h5py.File(h5p, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
    XML = resolve_body_model_xml(cfg.model.MJCF_PATH)
    mj = mujoco.MjModel.from_xml_path(XML)
    adr, rng = {}, {}
    for n in WING:
        j = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)
        adr[n] = int(mj.jnt_qposadr[j]); rng[n] = mj.jnt_range[j].copy()
    rest = np.asarray(mj.qpos_spring, float)

    inp = build_solver_inputs(h5p, XML)
    T_all = np.asarray(inp["q_init"]).shape[0]
    t0 = max(0, min(args.t0, T_all - 1)); nt = min(args.nt, T_all - t0)
    sl = slice(t0, t0 + nt)
    q_init = np.asarray(inp["q_init"])[sl]; kp = np.asarray(inp["kp_data"])[sl]
    print(f"{args.dir}\n  frames {t0}..{t0+nt} of {T_all}  fs={args.fs}")

    # fitted offsets onto the model sites, for honest per-keypoint residuals
    dat = mujoco.MjData(mj)
    sites = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mj.nsite)]
    off = np.asarray(d["offsets"], float)
    for i, n in enumerate(kpn):
        sn = f"tracking[{n}]"
        if sn in sites:
            mj.site_pos[sites.index(sn)] = off[i]
    wing_kp = [n for n in kpn if "Wing" in n and f"tracking[{n}]" in sites]
    widx = [sites.index(f"tracking[{n}]") for n in wing_kp]
    wrow = [kpn.index(n) for n in wing_kp]

    res = {}
    for w in [float(x) for x in args.weights.split(",")]:
        spec = None if w == 0 else {"wing_pitch_left": w, "wing_pitch_right": w}
        qw, qref = _resolve_rest_prior(mj, spec)
        s = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1, use_se3_root=True,
                            q_ref=qref)
        q = np.asarray(s.solve_trajectory(
            q_init=q_init, mjx_model=inp["mjx_model"],
            mjx_data_template=inp["mjx_data"], kp_data=kp,
            qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
            lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
            q_reg_weights=(inp["q_reg_weights"] if qw is None else qw)))
        Rw = np.empty((len(q), len(widx)))
        for t in range(len(q)):
            dat.qpos[:] = q[t]; mujoco.mj_forward(mj, dat)
            Rw[t] = np.linalg.norm(dat.site_xpos[widx] - kp[t].reshape(-1, 3)[wrow], axis=1)
        # blade plane + wing/abdomen penetration -- the two numbers that
        # correspond to what a viewer actually sees. Penetration uses MuJoCo's
        # own mj_geomDistance (negative = interpenetrating), so it is not a
        # proxy; the model at its OWN rest pose only grazes at -0.0013, so any
        # clearly negative value is the blade inside the body.
        wg = {sd: [g for g in range(mj.ngeom)
                   if mj.geom_bodyid[g] == mujoco.mj_name2id(
                       mj, mujoco.mjtObj.mjOBJ_BODY, f"wing_{sd}")]
              for sd in ("left", "right")}
        abd = [g for g in range(mj.ngeom)
               if "abdomen" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY,
                                                  int(mj.geom_bodyid[g])) or "")]
        def norm_ang(side):
            """angle between the MEASURED and FITTED base-V12-V13 plane normals"""
            nm_, nf_ = [], []
            for t in range(len(q)):
                dat.qpos[:] = q[t]; mujoco.mj_forward(mj, dat)
                P = []
                M = []
                for qn in ("base", "V12", "V13"):
                    n = f"Wing{side}_{qn}"
                    if n not in kpn or f"tracking[{n}]" not in sites:
                        return np.nan
                    P.append(dat.site_xpos[sites.index(f"tracking[{n}]")].copy())
                    M.append(kp[t].reshape(-1, 3)[kpn.index(n)])
                for acc, X in ((nf_, P), (nm_, M)):
                    v = np.cross(np.asarray(X[1]) - X[0], np.asarray(X[2]) - X[0])
                    L = np.linalg.norm(v)
                    acc.append(v / L if L > 1e-12 else np.full(3, np.nan))
            nm_, nf_ = np.asarray(nm_), np.asarray(nf_)
            ok = np.isfinite(nm_).all(-1) & np.isfinite(nf_).all(-1)
            if not ok.any():
                return np.nan
            c = np.abs(np.sum(nm_[ok] * nf_[ok], axis=-1))
            return float(np.median(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        pen = {}
        if abd:
            for sd in ("left", "right"):
                worst = []
                for t in range(0, len(q), 10):          # every 10th frame is plenty
                    dat.qpos[:] = q[t]; mujoco.mj_forward(mj, dat)
                    dmin = min(mujoco.mj_geomDistance(mj, dat, g1, g2, 1.0, None)
                               for g1 in wg[sd] for g2 in abd)
                    worst.append(dmin)
                pen[sd] = float(np.median(worst))
        deg = lambda a: np.degrees(a)
        song = deg(q[:, adr["wing_yaw_left"]] - q[:, adr["wing_yaw_right"]])
        row = dict(weight=w,
                   wing_resid_median=float(np.nanmedian(Rw)),
                   song_absmean_deg=float(np.nanmean(np.abs(song))),
                   song=pulse_stats(song, args.fs),
                   penetration_median=pen,
                   blade_normal_deg={sd: norm_ang("L" if sd == "left" else "R")
                                     for sd in ("left", "right")})
        for n in WING:
            row[n] = dict(off_rest_deg=float(np.nanmedian(
                              np.abs(deg(q[:, adr[n]] - rest[adr[n]])))),
                          pulse=pulse_stats(deg(q[:, adr[n]]), args.fs))
        res[w] = row
        print(f"  solved weight={w}")

    print(f"\n{'weight':>8}{'wing resid':>12}{'|yawL-yawR|':>13}{'song hp_rms':>13}"
          f"{'song kurt':>11}{'song peaks':>12}{'pitchL off rest':>17}{'rollL off rest':>16}")
    for w, r in res.items():
        print(f"{w:>8}{r['wing_resid_median']:>12.4f}{r['song_absmean_deg']:>13.2f}"
              f"{r['song']['hp_rms']:>13.4f}{r['song']['kurtosis']:>11.2f}"
              f"{r['song']['n_peaks']:>12d}"
              f"{r['wing_pitch_left']['off_rest_deg']:>17.1f}"
              f"{r['wing_roll_left']['off_rest_deg']:>16.1f}")
    print(f"\n{'weight':>8}{'penetration L':>16}{'penetration R':>16}"
          f"{'blade normal L':>16}{'blade normal R':>16}")
    for w, r in res.items():
        pl=r['penetration_median'].get('left'); pr=r['penetration_median'].get('right')
        print(f"{w:>8}{(f'{pl:.5f}' if pl is not None else 'n/a'):>16}"
              f"{(f'{pr:.5f}' if pr is not None else 'n/a'):>16}"
              f"{r['blade_normal_deg']['left']:>15.1f}d"
              f"{r['blade_normal_deg']['right']:>15.1f}d")
    print("  (penetration: MuJoCo mj_geomDistance, NEGATIVE = blade inside the "
          "abdomen; the model's own rest pose grazes at -0.0013)")
    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(args.out, "pitch_prior_ab.json"), "w"), indent=2)
    print(f"\nwrote {args.out}/pitch_prior_ab.json")


if __name__ == "__main__":
    main()
