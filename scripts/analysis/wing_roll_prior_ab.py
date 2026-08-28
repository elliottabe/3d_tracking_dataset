"""A/B the wing blade-roll smoothness prior on a real courtship clip.

WHY: wing roll (blade twist) is near-unobservable from the three near-collinear
wing keypoints (7.3 deg / 10.8 deg apart; the solver's own note records a
Jacobian column ~14x weaker than the strong wing DOFs). It therefore wanders on
measurement noise and parks against its joint stop (wing_yaw_right saturated on
22.9% of bout-18 frames). The prior damps roll's frame-to-frame change ONLY,
via JAXLS_SMOOTH_Q_MULT, leaving yaw/pitch -- the DOFs that carry wing
DIRECTION, and hence the ~193 Hz courtship song -- at the global weight.

EXPECTATION IF THE PRIOR IS CORRECT (read the figure against this):
  Panel A  roll(t): the damped traces are visibly smoother than baseline and
           spend less time flat against the +85.9 / -57.3 deg stops.
  Panel B  yaw(t) and pitch(t): damped traces overlay baseline almost exactly.
           If they change SHAPE, the prior is stealing real wing motion.
  Panel C  pulse-song transients in yaw/pitch survive: hp_rms, kurtosis, pulse
           count and pulse amplitude all near their baseline values. Pulse song
           is a sparse train of brief wing transients, so narrowband power at a
           carrier frequency does NOT measure it -- a smoothness prior can smear
           every pulse while leaving band power almost unchanged. Falling
           kurtosis or a falling pulse count means the prior is erasing song,
           which is the failure we most care about.

RUN THIS ON THE SINGING MALE. In bout_00003 that is fly1, confirmed by
unilateral wing extension (|yawL-yawR| median 24 deg, 25% of frames >30 deg,
against 5.4 deg for fly0). Running it on the non-singing fly measures nothing.
  Panel D  per-keypoint marker residual, wing keypoints highlighted: flat or
           slightly better. A rise means roll was absorbing real fit error.

A result where roll is unchanged means the multiplier never reached the cost
(check _resolve_smooth_q_mult and the solver cache key), not that roll is stiff.
"""
import os, sys, json, argparse
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (REPO, os.path.join(REPO, "third_party/jarvis_jax"), os.path.join(REPO, "stac-mjx")):
    sys.path.insert(0, p)

WING_DOFS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
             "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def band_power(x, fs, lo, hi):
    """Power of x in [lo, hi] Hz, via Welch. x is detrended by welch itself."""
    from scipy.signal import welch
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return np.nan
    nper = min(512, x.size)
    f, P = welch(x, fs=fs, nperseg=nper, detrend="linear")
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(P[m], f[m])) if m.any() else np.nan


def pulse_stats(x, fs, hp_hz=40.0, min_ipi_s=0.02):
    """Transient statistics of a wing-angle trace, for PULSE song.

    Pulse song is a sparse train of brief wing transients, not a sustained
    tone, so narrowband power at a sine-song carrier does not measure it. A
    smoothness prior smears transients while barely moving band power, so the
    quantities that matter are impulsiveness and the pulse events themselves:

      hp_rms    RMS of the >hp_hz component -- how much fast motion survives
      kurtosis  impulsiveness; a sparse pulse train is strongly leptokurtic,
                and smoothing drives this toward the Gaussian value of 0
      n_peaks   transients above 4x MAD, separated by at least min_ipi_s
      peak_amp  median height of those transients (deg)
    """
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.stats import kurtosis
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return dict(hp_rms=np.nan, kurtosis=np.nan, n_peaks=0, peak_amp=np.nan)
    b, a = butter(4, hp_hz / (fs / 2.0), btype="high")
    hp = filtfilt(b, a, x)
    mad = float(np.median(np.abs(hp - np.median(hp)))) or 1e-12
    pk, props = find_peaks(np.abs(hp), height=4.0 * mad,
                           distance=max(1, int(min_ipi_s * fs)))
    return dict(hp_rms=float(np.std(hp)),
                kurtosis=float(kurtosis(hp, fisher=True)),
                n_peaks=int(pk.size),
                peak_amp=float(np.median(props["peak_heights"])) if pk.size else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="bout fly dir containing stac_ik.h5")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--nt", type=int, default=1024)
    ap.add_argument("--mults", default="1,5,20", help="roll smoothness multipliers")
    ap.add_argument("--fs", type=float, default=800.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco, h5py
    from omegaconf import OmegaConf
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs
    from stac_mjx.stac_core_jaxls import JaxlsBatchSolver
    from stac_mjx.stac import _resolve_smooth_q_mult
    from viz.config import resolve_body_model_xml

    h5p = os.path.join(args.dir, "stac_ik.h5")
    d = ioh5.load(h5p)
    kpn = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(d["kp_names"])]
    with h5py.File(h5p, "r") as f:
        cfg = OmegaConf.create(f["config"][()].decode())
    XML = resolve_body_model_xml(cfg.model.MJCF_PATH)

    mj = mujoco.MjModel.from_xml_path(XML)
    dof = {}
    for n in WING_DOFS:
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)
        dof[n] = (int(mj.jnt_qposadr[jid]), mj.jnt_range[jid].copy())

    inp = build_solver_inputs(h5p, XML)
    T_all = np.asarray(inp["q_init"]).shape[0]
    t0 = max(0, min(args.t0, T_all - 1))
    nt = min(args.nt, T_all - t0)
    sl = slice(t0, t0 + nt)
    q_init = np.asarray(inp["q_init"])[sl]
    kp = np.asarray(inp["kp_data"])[sl]
    print(f"clip {args.dir}\n  frames {t0}..{t0+nt} of {T_all}  nq={q_init.shape[1]}  fs={args.fs}")

    # FK rig for per-keypoint residuals, with the FITTED marker offsets
    dat = mujoco.MjData(mj)
    sites = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mj.nsite)]
    off = np.asarray(d["offsets"], float)
    for i, n in enumerate(kpn):
        sn = f"tracking[{n}]"
        if sn in sites:
            mj.site_pos[sites.index(sn)] = off[i]
    kp_used = [n for n in kpn if f"tracking[{n}]" in sites]
    idxs = [sites.index(f"tracking[{n}]") for n in kp_used]

    mults = [float(x) for x in args.mults.split(",")]
    out = {}
    for mv in mults:
        spec = None if mv == 1.0 else {"wing_roll_left": mv, "wing_roll_right": mv}
        sqm = _resolve_smooth_q_mult(mj, spec)
        s = JaxlsBatchSolver(n_iter=50, smooth_weight=0.1, use_se3_root=True,
                             smooth_q_mult=sqm)
        q = np.asarray(s.solve_trajectory(
            q_init=q_init, mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
            kp_data=kp, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
            lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
            q_reg_weights=inp["q_reg_weights"]))
        R = np.empty((len(q), len(idxs)))
        for t in range(len(q)):
            dat.qpos[:] = q[t]
            mujoco.mj_forward(mj, dat)
            R[t] = np.linalg.norm(dat.site_xpos[idxs] - kp[t].reshape(-1, 3)[
                [kpn.index(n) for n in kp_used]], axis=1)
        out[mv] = dict(q=q, R=R)
        print(f"  solved mult={mv}")

    np.savez_compressed(
        os.path.join(args.out, "roll_ab.npz"),
        kp_used=np.array(kp_used), mults=np.array(mults),
        **{f"q_{mv}": out[mv]["q"] for mv in mults},
        **{f"R_{mv}": out[mv]["R"] for mv in mults})

    # ---------------- report ----------------
    rep = {"dir": args.dir, "t0": t0, "nt": nt, "fs": args.fs, "mults": mults, "dofs": {}}
    base = out[1.0]
    wing_kp = [i for i, n in enumerate(kp_used) if "wing" in n.lower()]
    print(f"\nwing keypoints: {[kp_used[i] for i in wing_kp]}")
    print(f"\n{'DOF':>18}{'mult':>6}{'med|dq| deg':>13}{'sat %':>8}"
          f"{'hp RMS deg':>12}{'kurtosis':>10}{'n pulses':>10}{'pulse amp':>11}"
          f"{'sineBP':>10}{'pulseBP':>10}")
    for n in WING_DOFS:
        a, (lo, hi) = dof[n]
        rep["dofs"][n] = {}
        for mv in mults:
            x = np.degrees(out[mv]["q"][:, a])
            dq = float(np.nanmedian(np.abs(np.diff(x))))
            tol = 0.01 * np.degrees(hi - lo)
            sat = float(np.mean((x <= np.degrees(lo) + tol) | (x >= np.degrees(hi) - tol)) * 100)
            ps = pulse_stats(x, args.fs)
            # sine song ~150-200 Hz; pulse-song carrier ~200-350 Hz
            sine_bp = band_power(np.radians(x), args.fs, 150.0, 200.0)
            pulse_bp = band_power(np.radians(x), args.fs, 200.0, 350.0)
            rep["dofs"][n][str(mv)] = dict(med_abs_dq_deg=dq, saturation_pct=sat,
                                           sine_bandpower=sine_bp, pulse_bandpower=pulse_bp, **ps)
            print(f"{n:>18}{mv:6.0f}{dq:13.4f}{sat:8.2f}{ps['hp_rms']:12.4f}"
                  f"{ps['kurtosis']:10.2f}{ps['n_peaks']:10d}"
                  f"{ps['peak_amp'] if np.isfinite(ps['peak_amp']) else float('nan'):11.4f}"
                  f"{sine_bp:10.2e}{pulse_bp:10.2e}")

    print(f"\n{'mult':>6}{'resid all':>12}{'resid wing':>12}{'vs base all':>13}{'vs base wing':>14}")
    b_all = float(np.nanmedian(base["R"]))
    b_wing = float(np.nanmedian(base["R"][:, wing_kp])) if wing_kp else np.nan
    rep["resid"] = {}
    for mv in mults:
        R = out[mv]["R"]
        ra = float(np.nanmedian(R))
        rw = float(np.nanmedian(R[:, wing_kp])) if wing_kp else np.nan
        rep["resid"][str(mv)] = dict(all=ra, wing=rw,
                                     d_all_pct=100 * (ra - b_all) / b_all,
                                     d_wing_pct=100 * (rw - b_wing) / b_wing if wing_kp else None)
        print(f"{mv:6.0f}{ra:12.5f}{rw:12.5f}{100*(ra-b_all)/b_all:12.2f}%"
              f"{(100*(rw-b_wing)/b_wing if wing_kp else float('nan')):13.2f}%")

    with open(os.path.join(args.out, "roll_ab.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nwrote {args.out}/roll_ab.json + roll_ab.npz")


if __name__ == "__main__":
    main()
