"""Re-fit offsets + IK from robustly-triangulated keypoints, and re-measure the wing margin.

The 2D check localized the wing error to the LIFTING step: the 2D detections sit
on the wing outline (WingL_V12 0.020 of fly size from the silhouette edge, conf
0.96, 7/7 views) but the triangulated 3D cannot reproject onto them (10.93 px
median, 46.78 px p90, against 4.64 px for V13), with one camera (Cam2012861)
contributing 46.5 px while its own V13 is 4.2 px.

`triangulate_keypoints(..., reproj_resid_px=)` already implements consensus
outlier-view rejection but is never configured, so it is off. Enabling it at
12 px drops WingL_V12's reprojection error 15.63 -> 5.03 px and moves the point
on 98.3% of frames (mean 2.64 mm on a 24.6 mm fly) -- this camera is
persistently wrong, not occasionally spiking.

THE TEST. That is a 3D-consistency improvement. It only matters if it survives
into the fit, so this re-runs the SAME pipeline stages run_bout uses
(segment_scales -> fit_offsets_once -> ik_only_bout) from each set of keypoints
and measures where the fitted wing markers land.

EXPECTATION:
  * `dist_to_margin` for V12 falls from ~0.034 (30% of the 0.1151 blade chord)
    toward the XML site's 0.0033. That is the claim; V13, already on the margin
    at 0.0008-0.0071, must not be pushed off it.
  * marker residual should not worsen much -- unlike the offset regularization,
    this changes the DATA rather than constraining the fit away from it, so
    there is no anatomy-vs-data tug of war to pay for.
  * wing-abdomen penetration is reported but NOT claimed: the M_REG sweep showed
    the landmark geometry and the penetration are separable defects, so a
    restored margin need not move it.

M_REG_COEF is left at its config default so this isolates triangulation. The
baseline arm re-triangulates with the gate OFF rather than reading kp3d.npz, so
both arms pass through identical code and the only difference is the gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ORDER MATTERS: scripts/ must come AFTER the repo root on sys.path, because
# scripts/viz/ would otherwise shadow the top-level `viz` package and
# `from viz.core import ...` fails.
for _p in (os.path.join(REPO, "scripts"), os.path.join(REPO, "stac-mjx"),
           os.path.join(REPO, "third_party/jarvis_jax"), REPO):
    sys.path.insert(0, _p)

VEINS = ["WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]


def wing_outline_dist(m, mujoco, site_local, side):
    """Distance from a wing-local point to the blade MARGIN (where veins end)."""
    from scipy.spatial import ConvexHull
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                            "wing_left" if side == "L" else "wing_right")
    Vs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        Vs.append(m.mesh_vert[a:a + n].reshape(-1, 3) @ R.reshape(3, 3).T + m.geom_pos[g])
    V = np.concatenate(Vs, 0)
    c0 = V.mean(0)
    X = V - c0
    _, _, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    e1, e2 = vt[0], vt[1]
    P2 = np.column_stack([X @ e1, X @ e2])
    poly = P2[ConvexHull(P2).vertices]
    q = np.array([(site_local - c0) @ e1, (site_local - c0) @ e2])

    def seg(p, a, b):
        ab = b - a
        t = np.clip((p - a) @ ab / (ab @ ab + 1e-18), 0, 1)
        return np.linalg.norm(p - (a + t * ab))
    return float(min(seg(q, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="Session0")
    ap.add_argument("--recording", default="2025_10_20_13_20_04")
    ap.add_argument("--bout", type=int, default=1)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--nt", type=int, default=300)
    ap.add_argument("--resid-px", type=float, default=12.0)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import h5py
    import mujoco
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver(
        "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)
    from hydra import initialize_config_dir, compose
    from viz.core import reproject
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout
    import run_bout as rb

    PROC = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
    VIDEO = "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship"
    run_root = os.path.join(PROC, args.session, args.recording, "pose")
    fly_dir = os.path.join(run_root, "bouts", f"bout_{args.bout:05d}", f"fly{args.fly}")
    cam_mats, _ = reproject.camera_matrices(
        os.path.join(VIDEO, args.session, args.recording, "calibration"))

    z = np.load(os.path.join(fly_dir, "kp2d.npz"), allow_pickle=True)
    kp2d = np.nan_to_num(np.asarray(z["kp2d"], np.float32))
    conf = np.asarray(z["conf"], np.float32)
    with h5py.File(os.path.join(fly_dir, "stac_ik.h5"), "r") as f:
        kp_names = [x.decode() for x in f["kp_names"][()]]
    with open(os.path.join(run_root, "scale.json")) as f:
        scale = float(json.load(f)["scale"])

    stored = np.asarray(np.load(os.path.join(fly_dir, "kp3d.npz"))["kp3d"], float)
    arms = {}
    for tag, thr in (("baseline (gate OFF)", None), (f"robust ({args.resid_px:g}px)", args.resid_px)):
        arms[tag] = np.asarray(triangulate_keypoints(
            kp2d, conf, cam_mats, conf_thresh=0.3, view_conf_thresh=0.6,
            reproj_resid_px=thr)[0])
    d0 = np.linalg.norm(arms["baseline (gate OFF)"] - stored, axis=-1)
    print(f"sanity: my gate-OFF triangulation vs the stored kp3d.npz -> "
          f"median {np.nanmedian(d0):.4f} mm (should be ~0)")

    rows = {}
    for tag, kp3d in arms.items():
        slug = "base" if "baseline" in tag else "robust"
        wdir = os.path.join(args.work, slug)
        shutil.rmtree(wdir, ignore_errors=True)
        os.makedirs(os.path.join(wdir, "bout"), exist_ok=True)
        with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "configs")):
            cfg = compose(config_name="pipeline", overrides=[
                "paths=hyak",
                f"recording={'session0' if args.session == 'Session0' else 'session1'}",
                f"recording.session_dir={os.path.join(VIDEO, args.session, args.recording)}"])
        seg = os.path.join(run_root, "segment_scales.json")
        if bool(cfg.model.get("segment_calibration", True)) and os.path.exists(seg):
            with open(seg) as f:
                rb.apply_segment_scales(cfg, json.load(f))
        print(f"\n=== {tag} ===")
        sample = rb.high_confidence_sample(kp3d)
        fit_offsets_once(cfg, kp3d[sample], kp_names, offsets_path="offsets.h5",
                         save_path=wdir, scale=scale)
        kps, _ = rb.fill_short_gaps(kp3d[:args.nt])
        if not rb.finite_frame_mask(kps).all():
            raise SystemExit("non-finite frames after gap fill")
        ik_only_bout(cfg, kps, kp_names, offsets_path=os.path.join(wdir, "offsets.h5"),
                     out_h5="stac_ik.h5", save_path=os.path.join(wdir, "bout"), scale=scale)

        with h5py.File(os.path.join(wdir, "bout", "stac_ik.h5"), "r") as f:
            q = f["qpos"][()]
            offs = f["offsets"][()]
            kpd = f["kp_data"][()].reshape(len(q), -1, 3)
            names = [x.decode() for x in f["kp_names"][()]]
            xml = re.search(r"MJCF_PATH:\s*(\S+)", f["config"][()].decode()).group(1)
        m = mujoco.MjModel.from_xml_path(xml)
        d = mujoco.MjData(m)
        S = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
        xmlpos = {n: m.site_pos[S.index(f"tracking[{n}]")].copy() for n in VEINS}
        margin = {n: wing_outline_dist(m, mujoco, offs[names.index(n)], n[4]) for n in VEINS}
        margin_xml = {n: wing_outline_dist(m, mujoco, xmlpos[n], n[4]) for n in VEINS}

        for i, n in enumerate(names):
            if f"tracking[{n}]" in S:
                m.site_pos[S.index(f"tracking[{n}]")] = offs[i]
        bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        ag = [g for g in range(m.ngeom)
              if bn[m.geom_bodyid[g]] and "abd" in bn[m.geom_bodyid[g]].lower()]
        wg = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"wing_{'left' if s == 'L' else 'right'}")] for s in "LR"}
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
        rows[tag] = dict(margin=margin, margin_xml=margin_xml,
                         pen=float(-np.median(gaps)),
                         resid=float(np.nanmedian(R)), resid_wing=float(np.nanmedian(R[:, wk])))
        print(f"  margin {[f'{margin[n]:.4f}' for n in VEINS]}  pen {rows[tag]['pen']:.4f}")

    b, r = rows["baseline (gate OFF)"], rows[f"robust ({args.resid_px:g}px)"]
    print(f"\n{'landmark':12}{'XML site':>11}{'baseline':>11}{'robust':>11}{'change':>10}"
          f"   (distance from the wing MARGIN; blade chord 0.1151)")
    for n in VEINS:
        print(f"{n:12}{b['margin_xml'][n]:11.4f}{b['margin'][n]:11.4f}{r['margin'][n]:11.4f}"
              f"{100*(r['margin'][n]-b['margin'][n])/max(b['margin'][n],1e-9):9.0f}%")
    print(f"\n{'':12}{'baseline':>11}{'robust':>11}{'change':>10}")
    for k, lab in (("resid", "marker residual"), ("resid_wing", "wing residual"),
                   ("pen", "wing-abd penetration")):
        print(f"{lab:12}{b[k]:11.5f}{r[k]:11.5f}{100*(r[k]-b[k])/max(abs(b[k]),1e-9):9.1f}%")

    with open(os.path.join(args.out, "robust_refit.json"), "w") as f:
        json.dump({k: {kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in v.items()}
                   for k, v in rows.items()}, f, indent=2)
    print(f"\nwrote {args.out}/robust_refit.json")


if __name__ == "__main__":
    main()
