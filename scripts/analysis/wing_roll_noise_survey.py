"""Dataset-wide check of the two claims the wing blade-roll prior rests on.

CLAIM 1  "roll wanders on noise while yaw/pitch track real motion", supported
         earlier by lag-1 autocorrelation 0.34-0.37 (roll) vs 0.55-0.78
         (yaw/pitch) measured on a SINGLE clip.
CLAIM 2  implicit: the wings are otherwise fit well enough that damping roll
         is the remaining problem.

EXPECTATION IF CLAIM 1 HOLDS (read the figure against this):
  Panel A  roll lag-1 AC plotted against yaw/pitch lag-1 AC should sit clearly
           BELOW the y=x line -- roll less autocorrelated, i.e. noisier.
  Panel B  the med|d roll| histogram should sit to the RIGHT of med|d yaw/pitch|
           -- bigger frame-to-frame steps for the same underlying motion.
  Points on the diagonal in A and overlapping histograms in B mean roll is no
  noisier than the DOFs beside it, and the prior has nothing to damp.

EXPECTATION FOR PANEL C (claim 2): a courting male extends a wing to roughly
  90 deg. If the fitted wing excursion tops out far below that, the wings are
  globally under-articulated and damping a wing DOF further is the wrong
  direction regardless of what A and B show.

Identity note: this pools BOTH flies in every bout, so the conclusion does not
depend on the 22 bouts whose sex labels are still pending a swap.
"""
import os, sys, re, json, glob, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOFS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
        "wing_yaw_right", "wing_roll_right", "wing_pitch_right")
MALE_EXTENSION_DEG = 90.0   # approximate wing extension of a singing male


def lag1_ac(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return np.nan
    x = x - x.mean()
    den = np.dot(x, x)
    return float(np.dot(x[1:], x[:-1]) / den) if den > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/gscratch/portia/eabe/data/Johnson_lab/processed/"
                                      "courtship/Session*/*/pose/bouts/bout_*/fly*/stac_ik.h5")
    ap.add_argument("--example", default="/gscratch/portia/eabe/data/Johnson_lab/processed/"
                                         "courtship/Session0/2025_10_20_13_20_04/pose/bouts/"
                                         "bout_00003/fly0/stac_ik.h5")
    ap.add_argument("--xml", default=os.path.join(REPO, "..", "fruitfly_body_models",
                                                  "fruitfly_v2_3_ik", "fruitfly_v2_3_ik.xml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py, mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, REPO)
    from viz.core.colors import PALETTE

    def wing_cols(f):
        """qpos column of each wing DOF, from the file's OWN names_qpos.

        Never infer these from a separately-loaded XML: the IK model is
        fruitfly_v1 (nq=93, wings at 7..12) while fruitfly_v2_3_ik has nq=101
        and would silently map the same wing names onto leg columns.
        """
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["names_qpos"][()]]
        return {n: names.index(n) for n in DOFS if n in names}

    # Joint ranges come from the model named in the data's OWN config
    # (fruitfly_v1, nq=93) -- see wing_cols() for why this must not be guessed.
    import h5py as _h5
    with _h5.File(args.example, "r") as f:
        _cfg = f["config"][()].decode()
        _nq_data = f["qpos"].shape[1]
    _mjcf = re.search(r"MJCF_PATH:\s*(\S+)", _cfg).group(1)
    _mj = mujoco.MjModel.from_xml_path(_mjcf)
    assert _mj.nq == _nq_data, f"model nq {_mj.nq} != data nq {_nq_data} ({_mjcf})"
    rng = {}
    for n in DOFS:
        j = mujoco.mj_name2id(_mj, mujoco.mjtObj.mjOBJ_JOINT, n)
        rng[n] = np.degrees(_mj.jnt_range[j].copy())
    print("wing joint ranges (deg):", {k: [round(v[0], 1), round(v[1], 1)] for k, v in rng.items()})

    def sat_pct(x, lo, hi):
        """% of frames within 1% of the joint span of either stop."""
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        tol = 0.01 * (hi - lo)
        return float(np.mean((x <= lo + tol) | (x >= hi - tol)) * 100)

    rows = []
    for p in sorted(glob.glob(args.glob)):
        try:
            with h5py.File(p, "r") as f:
                if "qpos" not in f or "names_qpos" not in f:
                    continue
                q = f["qpos"][()]
                adr = wing_cols(f)
        except Exception:
            continue
        if len(q) < 64 or len(adr) != len(DOFS):
            continue
        deg = {n: np.degrees(q[:, adr[n]]) for n in DOFS}
        if not any(np.isfinite(v).any() for v in deg.values()):
            continue
        roll = ["wing_roll_left", "wing_roll_right"]
        yp = ["wing_yaw_left", "wing_yaw_right", "wing_pitch_left", "wing_pitch_right"]
        rows.append(dict(
            path=p,
            fly=("fly1" if "/fly1/" in p else "fly0"),
            T=int(len(q)),
            max_p2p=float(np.nanmax([np.ptp(deg[n][np.isfinite(deg[n])])
                                     if np.isfinite(deg[n]).any() else np.nan for n in DOFS])),
            roll_ac=float(np.nanmean([lag1_ac(deg[n]) for n in roll])),
            yp_ac=float(np.nanmean([lag1_ac(deg[n]) for n in yp])),
            d_roll=float(np.nanmean([np.nanmedian(np.abs(np.diff(deg[n]))) for n in roll])),
            d_yp=float(np.nanmean([np.nanmedian(np.abs(np.diff(deg[n]))) for n in yp])),
            sat={n: sat_pct(deg[n], *rng[n]) for n in DOFS},
        ))

    rows = [r for r in rows if np.isfinite(r["max_p2p"])]
    print(f"{len(rows)} bout-flies with fitted wing DOFs")

    ra = np.array([r["roll_ac"] for r in rows])
    ya = np.array([r["yp_ac"] for r in rows])
    dr = np.array([r["d_roll"] for r in rows])
    dy = np.array([r["d_yp"] for r in rows])
    mp = np.array([r["max_p2p"] for r in rows])
    isf1 = np.array([r["fly"] == "fly1" for r in rows])

    below = float(np.mean(ra[np.isfinite(ra) & np.isfinite(ya)] <
                          ya[np.isfinite(ra) & np.isfinite(ya)]) * 100)
    summary = dict(
        n_bout_flies=len(rows),
        roll_ac_mean=float(np.nanmean(ra)), yawpitch_ac_mean=float(np.nanmean(ya)),
        pct_bouts_roll_less_autocorrelated_than_yawpitch=below,
        d_roll_median_deg=float(np.nanmedian(dr)), d_yawpitch_median_deg=float(np.nanmedian(dy)),
        wing_excursion_median_deg=float(np.nanmedian(mp)),
        wing_excursion_p90_deg=float(np.nanpercentile(mp, 90)),
        wing_excursion_max_deg=float(np.nanmax(mp)),
        male_extension_reference_deg=MALE_EXTENSION_DEG,
    )
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "roll_noise_survey.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=2)

    # ---------------- figure ----------------
    sat_med = {n: float(np.nanmedian([r["sat"][n] for r in rows])) for n in DOFS}
    summary["saturation_median_pct"] = sat_med
    with open(os.path.join(args.out, "roll_noise_survey.json"), "w") as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=2)
    print("median joint-limit saturation (%):",
          {k: round(v, 2) for k, v in sat_med.items()})

    import matplotlib.colors as mcolors

    def mpl(key, fallback):
        """viz.core.colors.PALETTE is BGR 0-255 (cv2); matplotlib wants RGB 0-1."""
        v = PALETTE.get(key)
        return mcolors.to_hex([ch / 255.0 for ch in reversed(v)]) if v else fallback

    c0 = mpl("fly0", "#4C72B0")   # cyan  = fly0, per the shared palette
    c1 = mpl("fly1", "#E8871A")   # orange = fly1

    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    ROLL_C, YP_C = "#C44E52", "#4C72B0"

    a = ax[0, 0]
    a.scatter(ya[~isf1], ra[~isf1], s=14, alpha=.6, c=c0, label="fly0")
    a.scatter(ya[isf1], ra[isf1], s=14, alpha=.6, c=c1, label="fly1")
    lo = float(min(np.nanmin(ya), np.nanmin(ra))) - .03
    a.plot([lo, 1.005], [lo, 1.005], "k--", lw=1, label="y = x (equally noisy)")
    a.set_xlim(lo, 1.005); a.set_ylim(lo, 1.005)
    a.set_xlabel("wing yaw/pitch  lag-1 autocorrelation")
    a.set_ylabel("wing roll  lag-1 autocorrelation")
    a.set_title(f"A. Is roll noisier than its neighbours?\nroll {np.nanmean(ra):.3f} vs "
                f"yaw/pitch {np.nanmean(ya):.3f} — only {below:.0f}% below y=x",
                fontsize=11)
    a.legend(fontsize=8, loc="lower right")

    a = ax[0, 1]
    hi = float(np.nanpercentile(np.concatenate([dr, dy]), 98))
    bins = np.linspace(0, hi, 45)
    a.hist(dy, bins=bins, alpha=.6, color=YP_C, label=f"yaw/pitch (median {np.nanmedian(dy):.3f}°)")
    a.hist(dr, bins=bins, alpha=.6, color=ROLL_C, label=f"roll (median {np.nanmedian(dr):.3f}°)")
    a.set_xlabel("median frame-to-frame |Δangle| per bout-fly (deg, at 800 Hz)")
    a.set_ylabel("bout-flies")
    a.set_title("B. Frame-to-frame step size\n(prior assumes roll steps are larger)", fontsize=11)
    a.legend(fontsize=8)

    a = ax[0, 2]
    order = list(DOFS)
    vals = [sat_med[n] for n in order]
    cols = [ROLL_C if "roll" in n else YP_C for n in order]
    a.barh(range(len(order)), vals, color=cols, alpha=.85)
    a.set_yticks(range(len(order)))
    a.set_yticklabels(order, fontsize=9)
    a.invert_yaxis()
    a.set_xlabel("% of frames within 1% of a joint stop (median over bout-flies)")
    a.set_title("C. Which wing DOF parks at its limit?\n(prior assumes it is roll)", fontsize=11)

    a = ax[1, 0]
    a.hist(mp, bins=40, color="#55A868", alpha=.85)
    a.axvline(MALE_EXTENSION_DEG, color="k", ls="--", lw=1.5)
    a.text(MALE_EXTENSION_DEG - 3, a.get_ylim()[1] * .82,
           "singing male\nwing extension ≈90°", ha="right", fontsize=9)
    a.set_xlim(0, max(105.0, float(np.nanmax(mp))) * 1.03)
    a.set_xlabel("largest fitted wing-DOF excursion in the bout (deg, peak-to-peak)")
    a.set_ylabel("bout-flies")
    a.set_title(f"D. Do the fitted wings reach biological range?\nmedian {np.nanmedian(mp):.1f}°, "
                f"max {np.nanmax(mp):.1f}° over {len(rows)} bout-flies", fontsize=11)

    with h5py.File(args.example, "r") as f:
        qex = f["qpos"][()]
        adr = wing_cols(f)
    tt = np.arange(len(qex)) / 800.0
    styles = {"wing_yaw_left": ("-", YP_C), "wing_pitch_left": ("-", "#55A868"),
              "wing_roll_left": ("-", ROLL_C), "wing_yaw_right": ("--", YP_C),
              "wing_pitch_right": ("--", "#55A868"), "wing_roll_right": ("--", ROLL_C)}
    ex = args.example.split("courtship/")[1].replace("/pose/bouts", "").replace("/stac_ik.h5", "")

    for a, (t_lo, t_hi), ttl in ((ax[1, 1], (tt[0], tt[-1]), "E. Strongest-wing clip, full bout"),
                                 (ax[1, 2], (1.05, tt[-1]), "F. Zoom: the late-bout breakdown")):
        m = (tt >= t_lo) & (tt <= t_hi)
        for n in DOFS:
            ls, c = styles[n]
            a.plot(tt[m], np.degrees(qex[m, adr[n]]), ls, color=c, lw=1, alpha=.85, label=n)
        for n in DOFS:
            for b in rng[n]:
                a.axhline(b, color="0.75", lw=.6, zorder=0)
        a.set_xlabel("time (s, 800 Hz)")
        a.set_ylabel("joint angle (deg)")
        a.set_title(f"{ttl}\n{ex}   (grey lines = joint stops)", fontsize=11)
    ax[1, 1].legend(fontsize=7, ncol=2, loc="lower left")

    fig.suptitle("Wing blade-roll prior: does the evidence support turning it on?", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = os.path.join(args.out, "roll_noise_survey.png")
    fig.savefig(png, dpi=125)
    print("wrote", png)


if __name__ == "__main__":
    main()
