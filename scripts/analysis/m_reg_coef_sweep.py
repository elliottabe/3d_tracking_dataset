"""Sweep M_REG_COEF: does regularizing the wing-vein marker offsets restore blade observability?

THE DEFECT. STAC's marker-offset fit collapses the wing landmark triangle.
Measured on bout_00001 fly0 (frame-independent, distances/angles only):

                    b->V12   b->V13   V12-V13   angle@base   tri area
  MODEL XML sites   0.2151   0.2615    0.0572      8.09 deg    0.00396
  FITTED offsets    0.2139   0.2537    0.0423      3.49 deg    0.00165
  MEASURED 3D kp    0.2128   0.2504    0.0468      6.56 deg    0.00303

The measured keypoints agree with the model's anatomy; the FITTED offsets are
the outlier, squashing the three landmarks onto the wing's long axis to 40% of
the true triangle area. That collapse makes the marker cost insensitive to
rotation about that axis -- which is a way to LOWER residual when roll is badly
fit -- so blade roll/pitch then drift freely and the wing ends up passing
through the abdomen in 100% of frames.

Root cause is config, not code: `configs/anatomy/v1.yaml` sets M_REG_COEF: 0.0
and comments the four wing veins out of SITES_TO_REGULARIZE, even though the
comment above the list says the wing veins should be regularized. The
regularization reference is correct -- KEYPOINT_INITIAL_OFFSETS for the veins
equals the XML site_pos exactly -- so switching it on pulls the offsets back to
anatomy rather than to some stale value.

EXPECTATION, read the sweep against this:
  * angle@base and triangle AREA of the fitted offsets climb back toward the
    model's 8.09 deg / 0.00396 as the coefficient rises. If they do not move,
    the regularization is not reaching these sites (check SITES_TO_REGULARIZE
    actually contains them and that _is_regularized is built from it).
  * wing-abdomen PENETRATION falls WITHOUT any pose prior -- that is the whole
    claim, that the penetration was a downstream symptom of the offset collapse.
  * marker residual RISES somewhat: a collapsed marker set fits the data better
    by construction, so a modest rise is the honest price of a non-degenerate
    parameterisation, not a regression. A steep rise means the coefficient is
    over-constraining offsets that legitimately differ from the model.
  * too large a coefficient pins offsets at the model and stops them absorbing
    real per-fly anatomy -- expect residual to blow up at the top of the sweep.

Offsets are fit ONCE per recording and shared across bouts, exactly as
run_bout.py does it, including applying segment_scales BEFORE the fit so the
offsets are fit on the morphed model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "third_party/jarvis_jax"),
           os.path.join(REPO, "stac-mjx"), os.path.join(REPO, "scripts")):
    sys.path.insert(0, _p)

WING_VEINS = ["WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]
WING_JOINTS = ("wing_yaw_left", "wing_roll_left", "wing_pitch_left",
               "wing_yaw_right", "wing_roll_right", "wing_pitch_right")


def triangle(P3):
    """(b->V12, b->V13, V12-V13, angle at base [deg], area) -- frame independent."""
    b, v12, v13 = P3
    a1, a2 = v12 - b, v13 - b
    ang = np.degrees(np.arccos(np.clip(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)), -1, 1)))
    return (np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(v13 - v12),
            ang, 0.5 * np.linalg.norm(np.cross(a1, a2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="Session0")
    ap.add_argument("--recording", default="2025_10_20_13_20_04")
    ap.add_argument("--bout", type=int, default=1)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--coefs", default="0,0.1,1,10,100")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py
    import mujoco
    from omegaconf import OmegaConf, open_dict
    OmegaConf.register_new_resolver(
        "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)
    from hydra import initialize_config_dir, compose
    from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout
    import run_bout as rb

    PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
    VIDEO = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
    run_root = os.path.join(PROC, args.session, args.recording, "pose")
    fly_dir = os.path.join(run_root, "bouts", f"bout_{args.bout:05d}", f"fly{args.fly}")

    kp_path = (os.path.join(fly_dir, "kp3d_filt.npz") if
               os.path.exists(os.path.join(fly_dir, "kp3d_filt.npz"))
               else os.path.join(fly_dir, "kp3d.npz"))
    z = np.load(kp_path, allow_pickle=True)
    kp3d = np.asarray(z["kp3d"], float)
    # the kp3d npz files carry only kp3d/conf3d -- names live in the bout's h5
    with h5py.File(os.path.join(fly_dir, "stac_ik.h5"), "r") as f:
        kp_names = [x.decode() for x in f["kp_names"][()]]
    if len(kp_names) != kp3d.shape[1]:
        raise SystemExit(f"kp_names ({len(kp_names)}) != kp3d width ({kp3d.shape[1]})")
    print(f"kp3d from {os.path.basename(kp_path)}: {kp3d.shape}, {len(kp_names)} names")

    with open(os.path.join(run_root, "scale.json")) as f:
        scale = float(json.load(f)["scale"])
    print(f"body scale {scale:.6f}")

    coefs = [float(c) for c in args.coefs.split(",")]
    rows = []
    for coef in coefs:
        wdir = os.path.join(args.work, f"coef_{coef:g}")
        shutil.rmtree(wdir, ignore_errors=True)
        os.makedirs(os.path.join(wdir, "bout"), exist_ok=True)

        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.join(REPO, "configs")):
            cfg = compose(config_name="pipeline", overrides=[
                "paths=hyak", f"recording={'session0' if args.session=='Session0' else 'session1'}",
                f"recording.session_dir={os.path.join(VIDEO, args.session, args.recording)}"])
        with open_dict(cfg):
            cfg.model.M_REG_COEF = float(coef)
            base = [str(s) for s in cfg.model.SITES_TO_REGULARIZE]
            cfg.model.SITES_TO_REGULARIZE = base + [w for w in WING_VEINS if w not in base]
        print(f"\n=== M_REG_COEF={coef:g}  regularized: {list(cfg.model.SITES_TO_REGULARIZE)}")

        # segment_scales BEFORE the offset fit, as run_bout does (offsets must be
        # fit on the MORPHED model or they absorb the limb-proportion mismatch)
        seg_path = os.path.join(run_root, "segment_scales.json")
        if bool(cfg.model.get("segment_calibration", True)) and os.path.exists(seg_path):
            with open(seg_path) as f:
                rb.apply_segment_scales(cfg, json.load(f))
            print("  applied segment_scales.json")

        sample = rb.high_confidence_sample(kp3d)
        fit_offsets_once(cfg, kp3d[sample], kp_names,
                         offsets_path="offsets.h5", save_path=wdir, scale=scale)
        kp_solve, _ = rb.fill_short_gaps(kp3d[:args.nt])
        if not rb.finite_frame_mask(kp_solve).all():
            raise SystemExit("test clip has non-finite frames after gap fill; pick another")
        ik_only_bout(cfg, kp_solve, kp_names, offsets_path=os.path.join(wdir, "offsets.h5"),
                     out_h5="stac_ik.h5", save_path=os.path.join(wdir, "bout"), scale=scale)

        # ---------------- measure ----------------
        h5p = os.path.join(wdir, "bout", "stac_ik.h5")
        with h5py.File(h5p, "r") as f:
            q = f["qpos"][()]
            offs = f["offsets"][()]
            kpd = f["kp_data"][()].reshape(len(q), -1, 3)
            names = [x.decode() for x in f["kp_names"][()]]
            import re as _re
            xml = _re.search(r"MJCF_PATH:\s*(\S+)", f["config"][()].decode()).group(1)

        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
        S = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        xml_pos = {n: m.site_pos[S.index(f"tracking[{n}]")].copy()
                   for n in names if f"tracking[{n}]" in S}
        for i, n in enumerate(names):
            if f"tracking[{n}]" in S:
                m.site_pos[S.index(f"tracking[{n}]")] = offs[i]

        # landmark triangle of the FITTED offsets, at spring rest
        mujoco.mj_resetData(m, d)
        for j in WING_JOINTS:
            a = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
            d.qpos[a] = m.qpos_spring[a]
        mujoco.mj_forward(m, d)
        tri = {}
        for side in ("L", "R"):
            nm = [f"Wing{side}_base", f"Wing{side}_V12", f"Wing{side}_V13"]
            tri[side] = triangle(np.stack([d.site_xpos[S.index(f"tracking[{n}]")] for n in nm]))

        # penetration + residual over the solved clip
        bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        ag = [g for g in range(m.ngeom)
              if bn[m.geom_bodyid[g]] and "abd" in bn[m.geom_bodyid[g]].lower()]
        wg = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"wing_{'left' if s == 'L' else 'right'}")]
            for s in "LR"}
        used = [n for n in names if f"tracking[{n}]" in S]
        sidx = [S.index(f"tracking[{n}]") for n in used]
        kidx = [names.index(n) for n in used]
        wk = [i for i, n in enumerate(used) if "Wing" in n]
        gaps, R = [], np.empty((len(q), len(sidx)))
        for t in range(len(q)):
            d.qpos[:] = q[t]
            mujoco.mj_forward(m, d)
            gaps.append(min(mujoco.mj_geomDistance(m, d, gw, ga, 1.0, None)
                            for s in "LR" for gw in wg[s] for ga in ag))
            R[t] = np.linalg.norm(d.site_xpos[sidx] - kpd[t][kidx], axis=1)
        gaps = np.asarray(gaps)

        off_dev = float(np.median([np.linalg.norm(offs[names.index(w)] - xml_pos[w])
                                   for w in WING_VEINS if w in xml_pos]))
        rec = dict(coef=coef,
                   angle_L=tri["L"][3], area_L=tri["L"][4],
                   angle_R=tri["R"][3], area_R=tri["R"][4],
                   pen_median=float(-np.median(gaps)), pen_max=float(-gaps.min()),
                   pct_pen=float(100 * np.mean(gaps < 0)),
                   resid_all=float(np.nanmedian(R)),
                   resid_wing=float(np.nanmedian(R[:, wk])),
                   wing_offset_dev_from_xml=off_dev)
        rows.append(rec)
        print(f"  angle L/R {rec['angle_L']:.2f}/{rec['angle_R']:.2f} deg   "
              f"area {rec['area_L']:.5f}/{rec['area_R']:.5f}   "
              f"pen {rec['pen_median']:.4f}   resid {rec['resid_all']:.5f}")

    print(f"\n{'M_REG_COEF':>11}{'angleL':>8}{'angleR':>8}{'areaL':>9}{'areaR':>9}"
          f"{'pen med':>9}{'%pen':>7}{'resid all':>11}{'resid wing':>12}{'|off-xml|':>11}")
    print(f"{'MODEL':>11}{8.09:>8.2f}{8.09:>8.2f}{0.00396:>9.5f}{0.00396:>9.5f}"
          f"{'-':>9}{'-':>7}{'-':>11}{'-':>12}{0.0:>11.4f}")
    for r in rows:
        print(f"{r['coef']:>11g}{r['angle_L']:>8.2f}{r['angle_R']:>8.2f}"
              f"{r['area_L']:>9.5f}{r['area_R']:>9.5f}{r['pen_median']:>9.4f}"
              f"{r['pct_pen']:>7.1f}{r['resid_all']:>11.5f}{r['resid_wing']:>12.5f}"
              f"{r['wing_offset_dev_from_xml']:>11.4f}")

    with open(os.path.join(args.out, "m_reg_sweep.json"), "w") as f:
        json.dump(dict(session=args.session, recording=args.recording, bout=args.bout,
                       fly=args.fly, nt=args.nt,
                       model_reference=dict(angle=8.09, area=0.00396), rows=rows), f, indent=2)
    print(f"\nwrote {args.out}/m_reg_sweep.json")


if __name__ == "__main__":
    main()
