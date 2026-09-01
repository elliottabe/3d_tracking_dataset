#!/usr/bin/env python3
"""All 7 rig cameras x [real frame | control pose | wing-mask-fit pose], one PNG.

Acceptance figure for the post-STAC wing-pitch refinement (Stage D2),
`docs/specs/2026-09-01-wing-orientation-from-masks-design.md` §5.6. The 3-view
`viz sidebyside` shows three cameras; this shows all seven, because the defect
is an orientation about an axis that is nearly invisible from some views and
obvious from others, and a still of one flattering camera would hide it.

WHAT THE FIGURE SHOULD LOOK LIKE IF THE CHANGE IS CORRECT -- written down before
generating it, so it can be wrong (CLAUDE.md: "a figure you can't be wrong about
proves nothing"):

  * FLY0, THE FEMALE (the hard fly: walls, occlusion, OOD poses). Both wings are
    folded and STAC puts their pitch at about -8 deg, so in the CONTROL column
    the blades stand up out of the abdomen and cut through it. In the TREATMENT
    column both blades should lie FLATTER along the abdomen -- rotated visibly
    toward the body -- and should fill the thin sliver of SAM mask that the body
    alone does not cover, WITHOUT spilling outside the grey mask outline. A
    blade that ends up outside its own row's outline is the coverage term
    overshooting, not a success.
  * FLY1'S NON-SINGING (folded, RIGHT) WING must move the same way.
  * FLY1'S SINGING (extended, LEFT) WING MUST LOOK ESSENTIALLY UNCHANGED between
    the two columns. The mask already agrees with the fit there (spec §4.1: mask
    optimum -10 deg vs fitted -1..-8 deg), so the term is supposed to be
    SELF-TARGETING -- it corrects the wing that is wrong and leaves the singer
    alone. A visible swing of the extended wing means the stage is stealing
    song, and acceptance criterion 1 should have failed.
  * IN EVERY ROW the rendered fly must sit INSIDE THAT ROW'S OWN mask outline.
    Each row's camera is built from that row's own calibration; a render that
    drifts outside its own outline means the row was built from the wrong
    camera's matrices -- the camera-order trap, which produces confident,
    self-consistent, completely wrong pictures.
  * EACH PANEL TITLE NAMES THE POSE FILE IT DRAWS. If the two right-hand columns
    look identical AND report the same filename, the figure is drawing the
    control twice and proves nothing about the stage. If they report different
    files and still look identical, the stage ran and was INERT -- a real
    finding, but a different one.

HOW THE VIEW IS BUILT (the `rigcam` recipe from viz/views/sidebyside.py):
`similarity_from_points(FK sites, outputs.h5 kp3d_mm)` recovers the model->world
similarity, then `mujoco_camera_from_affine(cam_mat, (W,H), s, R, t, anchor,
back_off=2.0)` turns that camera's own affine (orthographic) DLT into a MuJoCo
camera; `cam_pos/quat/fovy` are set and `mj_forward` is called AGAIN before
rendering.

THE CAMERA IS COMPUTED ONCE, FROM THE CONTROL ARM, AND REUSED FOR BOTH COLUMNS.
Deliberate: recovering it separately per arm would let the viewpoint move
between the columns (the wing markers enter the Umeyama fit), and then a
difference between the panels could be a difference in where the reader is
standing rather than in the pose. Both columns are therefore the same camera on
two poses.

Run (GPU node; `unset LD_LIBRARY_PATH`, MUJOCO_GL=egl is set below):
  python scripts/viz/wing_fit_7cam.py \
      --run  <run_root> --bout 28 --fly 0 --frames auto \
      --treat figures/2026-09-01-wing-mask-fit/arms_fly0.npz --treat-key q_cov0 \
      --treat-label "wing-mask fit, coverage_weight=0" \
      --out figures/2026-09-01-wing-mask-fit
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import cv2  # noqa: E402
import h5py  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from viz.config import resolve_body_model_xml  # noqa: E402
from viz.core import io as vio, overlays, reproject  # noqa: E402

_MASK = (180, 120, 60)        # BGR, the repo's SAM-mask colour
_MEAS_WING = (255, 0, 255)    # magenta: MEASURED wing keypoint (kp3d.npz)
_FIT_WING = (0, 255, 255)     # yellow:  FITTED wing marker
_TITLE_H = 44


def _band(width, lines, color=(255, 255, 255)):
    im = np.zeros((_TITLE_H, width, 3), np.uint8)
    for i, txt in enumerate(lines[:2]):
        cv2.putText(im, txt, (6, 17 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    return im


def build_rig(xml, W, H, offsets=None, kp_names=None):
    """One MuJoCo model + offscreen renderer carrying an orthographic `rigcam`.

    Shared across the 7 rows: the model is the same, only `cam_mat` differs per
    camera, and that is passed to `mujoco_camera_from_affine` per row rather
    than baked into a per-row model.

    `offsets` is the run's OWN fitted marker offsets (`stac_ik.h5:offsets`),
    applied to the `tracking[...]` sites. Without them the FK sites sit at the
    nominal landmarks the solve never targeted, which makes two things subtly
    wrong at once: the yellow "fitted marker" dots are not where the fit put
    them, and `similarity_from_points(FK sites, outputs.h5 kp3d_mm)` is fitting
    across an offset mismatch, since kp3d_mm WAS built at the fitted offsets.
    """
    import mujoco
    spec = mujoco.MjSpec.from_file(xml)
    spec.visual.global_.offwidth = int(W)
    spec.visual.global_.offheight = int(H)
    c = spec.worldbody.add_camera()
    c.name = "rigcam"
    c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    c.fovy = 1.0
    c.pos = [0.0, 0.0, 1.0]
    mj = spec.compile()
    sn = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
          for i in range(mj.nsite)]
    if offsets is not None:
        for i, n in enumerate(kp_names):                    # BY NAME
            if f"tracking[{n}]" in sn:
                mj.site_pos[sn.index(f"tracking[{n}]")] = offsets[i]
    return dict(mj=mj, dat=mujoco.MjData(mj), mujoco=mujoco,
                cid=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_CAMERA, "rigcam"),
                renderer=mujoco.Renderer(mj, height=int(H), width=int(W)),
                site_names=sn)


def render_pose(rig, q, cam_mat, cam_pose, W, H):
    """Render `q` through the PREBUILT camera pose (pos, quat, fovy)."""
    mj, dat, mjmod = rig["mj"], rig["dat"], rig["mujoco"]
    dat.qpos[:] = q
    mjmod.mj_forward(mj, dat)
    pos, quat, fovy = cam_pose
    mj.cam_pos[rig["cid"]] = pos
    mj.cam_quat[rig["cid"]] = quat
    mj.cam_fovy[rig["cid"]] = fovy
    mjmod.mj_forward(mj, dat)                      # AGAIN, after moving the cam
    rig["renderer"].update_scene(dat, camera="rigcam")
    return cv2.cvtColor(rig["renderer"].render(), cv2.COLOR_RGB2BGR)


def camera_pose_for_frame(rig, q_ctrl, world, sidx, kidx, cam_mat, W, H):
    """(pos, quat, fovy) from the CONTROL arm's model->world similarity, or None
    when too few fitted sites are finite to fit one."""
    from viz.core.mjcam import mujoco_camera_from_affine, similarity_from_points
    mj, dat, mjmod = rig["mj"], rig["dat"], rig["mujoco"]
    if not np.isfinite(q_ctrl).all():
        return None
    dat.qpos[:] = q_ctrl
    mjmod.mj_forward(mj, dat)
    Xm = dat.site_xpos[sidx]
    Xw = world[kidx]
    ok = np.isfinite(Xw).all(axis=1)
    if ok.sum() < 4:
        return None
    sc, R, tr = similarity_from_points(Xm[ok], Xw[ok])
    return mujoco_camera_from_affine(cam_mat, (W, H), sc, R, tr,
                                     Xm[ok].mean(0), back_off=2.0)


def fk_sites_mm(mj, dat, mjmod, sidx, q, bs, bR, bt):
    """Fitted marker sites at `q`, in the same mm frame as outputs.h5's kp3d_mm."""
    dat.qpos[:] = q
    mjmod.mj_forward(mj, dat)
    p = dat.site_xpos[sidx]
    return bs * (bR @ p.T).T + bt


def pick_frames(q_ctrl, q_treat, valid, adr, t0, t1, n=3):
    """Frames worth looking at, not flattering ones: the biggest pitch change,
    the WORST mask coverage in the window (occlusion / wall), and a mid-bout
    frame for the ordinary case."""
    fin = np.isfinite(q_ctrl).all(1) & np.isfinite(q_treat).all(1)
    idx = np.arange(t0, t1)
    idx = idx[fin[t0:t1] & (valid[t0:t1].sum(1) >= 4)]
    if idx.size == 0:
        raise SystemExit("no frame in the window has a finite pose and >=4 mask views")
    d = np.abs(np.degrees(q_treat[idx][:, adr] - q_ctrl[idx][:, adr])).max(1)
    out = [int(idx[np.argmax(d)]),
           int(idx[np.argmin(valid[idx].sum(1))]),
           int(idx[len(idx) // 2])]
    seen, keep = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            keep.append(t)
    return keep[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run root holding bouts/ (READ ONLY)")
    ap.add_argument("--bout", type=int, required=True)
    ap.add_argument("--fly", type=int, required=True)
    ap.add_argument("--treat", required=True,
                    help="npz holding the TREATMENT qpos (qpos_wingfit.npz, or "
                         "wing_mask_fit_ab.py's arms_flyN.npz)")
    ap.add_argument("--treat-key", default=None,
                    help="key inside --treat (default: 'qpos', else the single q_* key)")
    ap.add_argument("--treat-label", default=None)
    ap.add_argument("--frames", default="auto",
                    help="'auto' or a comma-separated list of frame indices")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--nt", type=int, default=1500)
    ap.add_argument("--session-dir", default=None)
    ap.add_argument("--predictions-dir", default=None)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--panel-h", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    from viz.config import courtship_recording
    rec = courtship_recording()
    cameras = [str(c) for c in rec["cameras"]]
    session_dir = args.session_dir or rec["session_dir"]
    predictions_dir = args.predictions_dir or rec["predictions_dir"]
    calib_dir = args.calib_dir or rec["calib_dir"]

    fdir = vio.fly_dir(args.run, args.bout, args.fly)
    with h5py.File(os.path.join(fdir, "stac_ik.h5"), "r") as f:
        icfg = OmegaConf.create(f["config"][()].decode())
        kp_names = [s.decode() for s in f["kp_names"][:]]
        offsets = np.asarray(f["offsets"][()], float)
    xml = resolve_body_model_xml(icfg.model.MJCF_PATH)

    # CONTROL: what the run itself committed to, read-only.
    q_ctrl, ctrl_src = vio.load_qpos(args.run, args.bout, args.fly, source="refined")
    outs = vio.load_outputs(args.run, args.bout, args.fly)
    world_ctrl = np.asarray(outs["kp3d_mm"], float)
    ref = np.load(os.path.join(fdir, "qpos_refined.npz"))
    bs, bR, bt = ref["bridge_s"], ref["bridge_R"], ref["bridge_t"]

    # TREATMENT
    with np.load(args.treat) as z:
        key = args.treat_key or ("qpos" if "qpos" in z.files else None)
        if key is None:
            cand = [k for k in z.files if k.startswith("q_") and k != "q_control"]
            if len(cand) != 1:
                raise SystemExit(f"--treat-key needed; {args.treat} holds {z.files}")
            key = cand[0]
        q_treat = np.asarray(z[key], float)
    treat_src = f"{os.path.basename(args.treat)}[{key}]"
    treat_label = args.treat_label or treat_src
    print(f"[7cam] control pose source  {ctrl_src}")
    print(f"[7cam] treatment pose source {treat_src}")
    if q_treat.shape != q_ctrl.shape:
        raise SystemExit(f"shape mismatch: control {q_ctrl.shape} treat {q_treat.shape}")
    ident = np.array_equal(np.nan_to_num(q_ctrl, nan=0.0),
                           np.nan_to_num(q_treat, nan=0.0))
    print(f"[7cam] control and treatment qpos identical: {ident}"
          f"{'   <-- THE FIGURE WOULD PROVE NOTHING' if ident else ''}")

    mj0 = mujoco.MjModel.from_xml_path(xml)
    adr = [int(mj0.jnt_qposadr[mujoco.mj_name2id(mj0, mujoco.mjtObj.mjOBJ_JOINT, n)])
           for n in ("wing_pitch_left", "wing_pitch_right")]     # BY NAME

    masks = vio.load_masks(predictions_dir, args.bout, args.fly, cameras)
    mk, mv, H, W = masks["masks"], masks["valid"], masks["H"], masks["W"]
    if list(masks["cameras"]) != cameras:
        raise SystemExit(f"mask camera axis {list(masks['cameras'])} != canonical {cameras}")

    kp3d_path = os.path.join(fdir, "kp3d.npz")
    meas = (np.asarray(np.load(kp3d_path)["kp3d"], float)
            if os.path.exists(kp3d_path) else None)
    wing_kp = [i for i, n in enumerate(kp_names) if "Wing" in n]

    cam_mats, cam_names = reproject.camera_matrices(calib_dir)
    cam_names = list(cam_names)
    if cam_names != cameras:
        # BY NAME, always: the calibration glob order is canonical here, but a
        # positional assumption is exactly the trap this repo has been bitten by.
        cam_mats = np.stack([cam_mats[cam_names.index(c)] for c in cameras])
    print(f"[7cam] cameras (canonical): {cameras}")

    if args.start_frame is not None:
        start_abs = int(args.start_frame)
    else:
        from scripts.run_bout import bout_start_frame
        from hydra import compose, initialize_config_dir
        OmegaConf.register_new_resolver(
            "basename", lambda p: os.path.basename(os.path.normpath(str(p))),
            replace=True)
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.join(_REPO, "configs")):
            start_abs = bout_start_frame(compose(config_name="pipeline"), args.bout)
    print(f"[7cam] bout {args.bout} starts at absolute video frame {start_abs}")

    frames = (pick_frames(q_ctrl, q_treat, np.asarray(mv, bool), adr,
                          args.t0, min(args.t0 + args.nt, len(q_ctrl)))
              if args.frames == "auto"
              else [int(x) for x in args.frames.split(",")])
    print(f"[7cam] frames: {frames}")

    rig = build_rig(xml, W, H, offsets=offsets, kp_names=kp_names)
    sn = rig["site_names"]
    pairs = [(sn.index(f"tracking[{n}]"), i) for i, n in enumerate(kp_names)
             if f"tracking[{n}]" in sn]
    sidx = [a for a, _ in pairs]
    kidx = [b for _, b in pairs]

    # SELF-CHECK, same idea as scripts/viz/render_prior_sidebyside.py: FK at the
    # CONTROL qpos, pushed through the stored bridge, must reproduce the
    # outputs.h5 kp3d_mm that the rigcam similarity is fitted against. If it
    # does not, the offsets or the bridge are being applied wrongly and both
    # render columns would be quietly misaligned -- which reads as an IK
    # failure, not as a bug in this script.
    chk, ref = [], []
    for t in frames:
        if not np.isfinite(q_ctrl[t]).all():
            continue
        p = fk_sites_mm(rig["mj"], rig["dat"], rig["mujoco"], sidx, q_ctrl[t],
                        bs[t], bR[t], bt[t])
        chk.append(p)
        ref.append(world_ctrl[t][kidx])
    if chk:
        chk, ref = np.asarray(chk), np.asarray(ref)
        err = float(np.nanmedian(np.linalg.norm(chk - ref, axis=-1)))
        span = float(np.nanmax(ref) - np.nanmin(ref))
        print(f"[7cam] self-check: FK+bridge reproduces outputs.h5 kp3d_mm to "
              f"{err:.4f} mm ({100 * err / span:.2f}% of the {span:.2f} mm span)")
        if err > 0.05 * span:
            raise SystemExit(
                "[7cam] FK+bridge does NOT reproduce the stored kp3d_mm; the "
                "rigcam similarity would be fitted across a mismatch and both "
                "render columns would be misaligned. Refusing to draw.")

    for t in frames:
        rows = []
        for ci, cam in enumerate(cameras):
            m = np.asarray(mk[t][ci], bool) if mv[t, ci] else None
            bgr = np.zeros((H, W, 3), np.uint8)
            vp = os.path.join(session_dir, f"{cam}.mp4")
            try:
                bgr = vio.read_frame(vp, start_abs + t)
            except Exception as e:                       # noqa: BLE001
                print(f"[7cam]   {cam} frame {start_abs+t} unreadable: {e}")
            real = bgr.copy()
            if m is not None:
                overlays.draw_mask(real, m, _MASK, alpha=0.25)

            cam_pose = camera_pose_for_frame(rig, q_ctrl[t], world_ctrl[t],
                                             sidx, kidx, cam_mats[ci], W, H)
            panels = [(f"{cam}  video + SAM mask", "real frame", real)]
            for lab, src, q in (("CONTROL (STAC/bridge pose)", ctrl_src, q_ctrl[t]),
                                (f"TREATMENT: {treat_label}", treat_src, q_treat[t])):
                if cam_pose is None or not np.isfinite(q).all():
                    im = np.zeros((H, W, 3), np.uint8)
                else:
                    im = render_pose(rig, q, cam_mats[ci], cam_pose, W, H)
                    if m is not None:
                        overlays.draw_mask(im, m, _MASK, alpha=0.0)   # OUTLINE only
                    world_q = np.full_like(world_ctrl[t], np.nan)
                    for si, ki in pairs:
                        world_q[ki] = fk_sites_mm(
                            rig["mj"], rig["dat"], rig["mujoco"], [si], q,
                            bs[t], bR[t], bt[t])[0]
                    for arr, col, rad in ((meas[t] if meas is not None else None,
                                           _MEAS_WING, 4),
                                          (world_q, _FIT_WING, 3)):
                        if arr is None:
                            continue
                        uv = reproject.project(cam_mats[ci], np.nan_to_num(arr))
                        for i in wing_kp:
                            if np.isfinite(arr[i]).all():
                                overlays.draw_points(im, uv[i][None], col, radius=rad)
                panels.append((lab, src, im))

            # one crop window per row, from THIS camera's own mask + fly extent
            if m is not None and m.any():
                ys, xs = np.where(m)
                pad = 55
                x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, W)
                y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, H)
            else:
                x0, y0, x1, y1 = 0, 0, W, H
            ph = int(args.panel_h)
            pw = max(1, int(round((x1 - x0) / max(1, (y1 - y0)) * ph)))
            cells = []
            for lab, src, im in panels:
                crop = cv2.resize(im[y0:y1, x0:x1], (pw, ph))
                cells.append(np.vstack([_band(pw, [lab, f"pose: {src}"]), crop]))
            sep = np.full((cells[0].shape[0], 3, 3), 60, np.uint8)
            rows.append(np.hstack([cells[0], sep, cells[1], sep, cells[2]]))

        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)])
                if r.shape[1] < wmax else r for r in rows]
        hdr = _band(wmax, [f"bout {args.bout} fly{args.fly}  frame t={t} "
                           f"(abs video frame {start_abs + t})   "
                           f"7 rig cameras x [real | control | treatment]",
                           "magenta = MEASURED wing kp (kp3d.npz), yellow = FITTED "
                           "wing marker, grey outline = SAM mask"],
                    (0, 255, 255))
        img = np.vstack([hdr] + rows)
        p = os.path.join(args.out, f"wingfit_7cam_bout{args.bout}_fly{args.fly}_t{t}.png")
        cv2.imwrite(p, img)
        print(f"[7cam] wrote {p}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
