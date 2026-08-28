"""Figure for the wing blade-roll prior A/B, on the SINGING MALE.

EXPECTATION IF THE PRIOR IS SAFE (read the figure against this):
  A  wing_yaw_right leaves the folded-rest line (springref = +85.94 deg, which
     is also the joint stop) -- that departure IS the wing extension. The
     damped traces should follow the baseline yaw exactly, since only roll is
     damped.
  B  the high-passed roll of the EXTENDED wing should keep its transients
     under damping. Flattened transients mean the prior is erasing fast wing
     motion. Whether that motion is SONG is a separate question -- answered by
     panel E, not by this panel.
  E  the control that decides it, on a BREAKDOWN-FREE window: the same DOF on
     the NON-SINGING fly (wings folded), plus the singer's own NON-extended
     wing. A signal specific to the extended wing is wing-driven; one present
     in all three is solver artefact. Do not score this over the whole bout --
     a late-bout tracking breakdown in either fly dominates the statistic and
     inverts the answer.
  F  spectrum over the same window. A narrowband peak present only in the
     singer's extended wing is the signal the prior would destroy.
  C  hp-RMS per DOF: roll falls (intended) and nothing else rises. Pitch or
     yaw rising means the motion was displaced, not removed -- the same
     trajectory re-expressed through different joints.
  D  wing-keypoint residual should be flat. A rise means the damped roll was
     carrying real fit.

Identity matters: in bout_00003 the singer is fly1 (unilateral extension,
|yawL-yawR| median 24 deg vs 5.4 deg for fly0). The same A/B run on fly0 looks
harmless because that fly's wings are folded at rest and barely move.
"""
import os, sys, json, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAW_REST_DEG = 85.94   # springref == upper joint stop == wings folded back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--h5", required=True, help="stac_ik.h5 for names_qpos")
    ap.add_argument("--control-h5", default=None,
                    help="stac_ik.h5 of the NON-singing fly, for the panel-E control")
    ap.add_argument("--fs", type=float, default=800.0)
    ap.add_argument("--win", default="0.2,1.1",
                    help="breakdown-free window (s) for the panel E/F control")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt, find_peaks

    z = np.load(args.npz, allow_pickle=True)
    rep = json.load(open(args.json))
    mults = [float(m) for m in z["mults"]]
    kp_used = [x.decode() if isinstance(x, bytes) else str(x) for x in z["kp_used"]]
    with h5py.File(args.h5, "r") as f:
        names = [x.decode() for x in f["names_qpos"][()]]
    col = {n: names.index(n) for n in
           ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
            "wing_yaw_right", "wing_roll_right", "wing_pitch_right")}

    Q = {m: z[f"q_{m}"] for m in mults}
    R = {m: z[f"R_{m}"] for m in mults}
    T = Q[mults[0]].shape[0]
    tt = np.arange(T) / args.fs
    shades = {mults[0]: ("#222222", 1.6), **{m: (c, 1.1) for m, c in
              zip(mults[1:], ["#E8871A", "#C44E52"])}}

    def hp(x):
        b, a = butter(4, 40.0 / (args.fs / 2), btype="high")
        return filtfilt(b, a, x)

    fig, ax = plt.subplots(2, 3, figsize=(20, 9))

    a = ax[0, 0]
    for m in mults:
        c, lw = shades[m]
        a.plot(tt, np.degrees(Q[m][:, col["wing_yaw_right"]]), color=c, lw=lw,
               label=f"yaw_right  ×{m:g}")
        a.plot(tt, np.degrees(Q[m][:, col["wing_roll_right"]]), color=c, lw=lw,
               ls="--", alpha=.7, label=f"roll_right  ×{m:g}")
    a.axhline(YAW_REST_DEG, color="#4C72B0", ls=":", lw=1.5)
    a.text(tt[-1], YAW_REST_DEG + 1.5, "folded rest / joint stop (85.9°)",
           ha="right", fontsize=8, color="#4C72B0")
    a.set_xlabel("time (s, 800 Hz)"); a.set_ylabel("joint angle (deg)")
    a.set_title("A. Extended (right) wing of the singing male\n"
                "departure from 85.9° = wing extension", fontsize=11)
    a.legend(fontsize=7, ncol=len(mults))

    # pick the busiest 150 ms of baseline roll for the zoom
    base_roll = np.degrees(Q[mults[0]][:, col["wing_roll_right"]])
    w = int(0.15 * args.fs)
    e = np.array([np.std(hp(base_roll)[i:i + w]) for i in range(0, T - w, 20)])
    i0 = int(np.argmax(e) * 20)
    sl = slice(i0, i0 + w)

    a = ax[0, 1]
    for m in mults:
        c, lw = shades[m]
        h = hp(np.degrees(Q[m][:, col["wing_roll_right"]]))
        a.plot(tt[sl], h[sl], color=c, lw=lw, label=f"×{m:g}")
        if m == mults[0]:
            mad = np.median(np.abs(h - np.median(h)))
            pk, _ = find_peaks(np.abs(h), height=4 * mad, distance=int(.02 * args.fs))
            pk = pk[(pk >= sl.start) & (pk < sl.stop)]
            a.plot(tt[pk], h[pk], "v", color="#55A868", ms=7,
                   label="detected transient (baseline)")
    a.set_xlabel("time (s)"); a.set_ylabel("wing_roll_right, >40 Hz (deg)")
    a.set_title("B. Fast roll motion on the extended wing\n"
                "(busiest 150 ms; see E/F for whether it is wing-driven)", fontsize=11)
    a.legend(fontsize=8)

    a = ax[0, 2]
    dofs = list(col.keys())
    x = np.arange(len(dofs)); wdt = 0.8 / len(mults)
    for k, m in enumerate(mults):
        v = [rep["dofs"][n][str(m)]["hp_rms"] for n in dofs]
        a.bar(x + k * wdt, v, wdt, label=f"×{m:g}", color=shades[m][0], alpha=.85)
    a.set_xticks(x + wdt * (len(mults) - 1) / 2)
    a.set_xticklabels([d.replace("wing_", "") for d in dofs], rotation=20, fontsize=9)
    a.set_ylabel("RMS of >40 Hz component (deg)")
    a.set_title("C. Where the fast wing motion goes\n"
                "(roll falls — does anything else rise?)", fontsize=11)
    a.legend(fontsize=8)

    a = ax[1, 0]
    wing_kp = [i for i, n in enumerate(kp_used) if "wing" in n.lower()]
    b_all = np.nanmedian(R[mults[0]]); b_w = np.nanmedian(R[mults[0]][:, wing_kp])
    da = [100 * (np.nanmedian(R[m]) - b_all) / b_all for m in mults]
    dw = [100 * (np.nanmedian(R[m][:, wing_kp]) - b_w) / b_w for m in mults]
    a.plot(mults, da, "o-", color="#4C72B0", label="all keypoints")
    a.plot(mults, dw, "s-", color="#C44E52", label="wing keypoints")
    a.axhline(0, color="k", lw=.8)
    a.set_xscale("log"); a.set_xticks(mults); a.set_xticklabels([f"×{m:g}" for m in mults])
    a.set_xlabel("wing_roll smoothness multiplier")
    a.set_ylabel("change in median marker residual (%)")
    a.set_title("D. Cost of the prior in fit quality\n(positive = worse)", fontsize=11)
    a.legend(fontsize=8)

    # ---- panel E/F: the control. Is this motion wing-driven, or artefact? ----
    if args.control_h5:
        from scipy.signal import welch
        with h5py.File(args.control_h5, "r") as f:
            names0 = [x.decode() for x in f["names_qpos"][()]]
            q0 = f["qpos"][()][:T]
        c0 = {n: names0.index(n) for n in col}
        w0, w1 = (float(v) for v in args.win.split(","))
        W = slice(int(w0 * args.fs), int(w1 * args.fs))
        series = [
            ("fly1 SINGER, EXTENDED wing", np.degrees(Q[mults[0]][:, col["wing_roll_right"]]), "#C44E52"),
            ("fly1 SINGER, folded wing",   np.degrees(Q[mults[0]][:, col["wing_roll_left"]]),  "#E8871A"),
            ("fly0 non-singer, folded",    np.degrees(q0[:, c0["wing_roll_right"]]),           "#4C72B0"),
        ]
        a = ax[1, 1]
        for lbl, x, cl in series:
            h = hp(x)
            a.plot(np.arange(W.start, W.stop) / args.fs, h[W], color=cl, lw=.8, alpha=.85,
                   label=f"{lbl}\n  hp-RMS {np.std(h[W]):.2f}°")
        a.set_xlabel("time (s, 800 Hz)")
        a.set_ylabel("wing_roll, >40 Hz (deg)")
        a.set_title(f"E. CONTROL, breakdown-free window {w0:g}-{w1:g}s\n"
                    "is the motion specific to the EXTENDED wing?", fontsize=11)
        a.legend(fontsize=7)

        a = ax[1, 2]
        for lbl, x, cl in series:
            f_, P = welch(x[W], fs=args.fs, nperseg=512, detrend="linear")
            m = (f_ >= 20) & (f_ <= 400)
            a.semilogy(f_[m], P[m], color=cl, lw=1.2, label=lbl)
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("PSD of wing_roll (deg²/Hz)")
        a.set_title("F. Spectrum over the same window\n"
                    "peak only in the extended wing = wing-driven signal", fontsize=11)
        a.legend(fontsize=7)

    fig.suptitle("Wing blade-roll prior on the SINGING MALE "
                 "(bout_00003 fly1; fly0 = non-singing control)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(args.out, "roll_prior_fly1.png")
    fig.savefig(png, dpi=130)
    print("wrote", png)
    for m in mults:
        print(f"  ×{m:g}: roll_right hp_RMS {rep['dofs']['wing_roll_right'][str(m)]['hp_rms']:.4f}° "
              f"pitch_right hp_RMS {rep['dofs']['wing_pitch_right'][str(m)]['hp_rms']:.4f}° "
              f"wing resid {dw[mults.index(m)]:+.2f}%")


if __name__ == "__main__":
    main()
