"""rigcam view: render the fitted fly through MuJoCo cameras built from the
REAL rig calibration, beside the real video from the same camera.

The existing `sidebyside` mesh panel renders `track1`, a model-space camera, so
its viewpoint is unrelated to any real camera and orientation cannot be judged
from it. This view instead constructs, per real camera, a MuJoCo camera that
observes the model from the SAME viewpoint, so the render and the video are
directly comparable pixel for pixel -- the SAM mask outline is drawn on BOTH
panes as a shared fiducial.

Why orthographic: the rig's calibration is AFFINE (every camera matrix's third
row is [0,0,0,1]); apparent size is exactly depth-independent, so there is no
finite camera centre and an RQ decomposition is singular. See viz.core.mjcam.

Accuracy: the built cameras reproduce the real cameras to ~2.8-3.4 px median on
the exemplar, which is the model->world similarity residual (0.42 world units
x ~8.1 px/unit = 3.4 px), i.e. the STAC fit residual rather than camera error.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import numpy as np
import cv2
import h5py
import mujoco

from viz.core import io as vio
from viz.core import reproject
from viz.core.mjcam import mujoco_camera_from_affine, similarity_from_points
from viz.config import courtship_recording


def _bout_start(video_dir, bout):
    p = os.path.join(video_dir, "courtship_bouts_unified_summary.csv")
    with open(p) as f:
        for r in csv.DictReader(f):
            if int(r["bout_idx"]) == int(bout):
                return int(r["start_frame"])
    raise KeyError(f"bout {bout} not in {p}")


def _build_model(xml, w, h):
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.visual.global_.offwidth = int(w)
    spec.visual.global_.offheight = int(h)
    cam = spec.worldbody.add_camera()
    cam.name = "rigcam"
    cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    cam.fovy = 1.0
    cam.pos = [0.0, 0.0, 1.0]
    return spec.compile()


def run(args):
    rec = courtship_recording()
    session_dir = getattr(args, "session_dir", None) or rec["session_dir"]
    predictions_dir = getattr(args, "predictions_dir", None) or rec["predictions_dir"]
    calib_dir = getattr(args, "calib_dir", None) or rec["calib_dir"]
    bout, fly, t = int(args.bout), int(args.fly), int(args.frame)

    cam_mats, cam_names = reproject.camera_matrices(calib_dir)
    cam_names = list(cam_names)
    want = list(args.cams) if getattr(args, "cams", None) else cam_names

    fdir = vio.fly_dir(args.run, bout, fly)
    qpos = np.asarray(np.load(os.path.join(fdir, "qpos_refined.npz"))["qpos"], float)
    with h5py.File(os.path.join(fdir, "outputs.h5"), "r") as f:
        world = f["kp3d_mm"][()].astype(float)
        kp_names = [x.decode() if isinstance(x, bytes) else str(x)
                    for x in f["kp_names"][()]]
    if t >= len(qpos):
        raise ValueError(f"frame {t} out of range (T={len(qpos)})")

    cap = cv2.VideoCapture(os.path.join(session_dir, f"{cam_names[0]}.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = _build_model(args.model_xml, W, H)
    data = mujoco.MjData(model)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rigcam")
    msites = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
              for i in range(model.nsite)]
    pairs = [(msites.index(f"tracking[{n}]"), i) for i, n in enumerate(kp_names)
             if f"tracking[{n}]" in msites]
    if not pairs:
        raise RuntimeError("no tracking[...] sites matched the run's kp_names")
    sidx = [a for a, _ in pairs]; kidx = [b for _, b in pairs]

    data.qpos[:] = qpos[t]
    mujoco.mj_forward(model, data)
    Xm = data.site_xpos[sidx]
    Xw = world[t][kidx]
    ok = np.isfinite(Xw).all(axis=1)
    if ok.sum() < 4:
        raise RuntimeError(f"frame {t} has only {ok.sum()} finite fitted sites")
    s, R, tr = similarity_from_points(Xm[ok], Xw[ok])
    anchor = Xm[ok].mean(0)

    masks = None
    try:
        masks = vio.load_masks(predictions_dir, bout, fly, cam_names)
    except Exception as e:                       # noqa: BLE001
        print(f"[rigcam] masks unavailable ({e}); no outline overlay")

    start_abs = _bout_start(session_dir, bout)
    renderer = mujoco.Renderer(model, height=H, width=W)
    pad = int(getattr(args, "pad", 170) or 170)
    rows = []
    for cn in want:
        if cn not in cam_names:
            print(f"[rigcam] skipping unknown camera {cn}"); continue
        ci = cam_names.index(cn)
        pos, quat, fovy = mujoco_camera_from_affine(
            cam_mats[ci], (W, H), s, R, tr, anchor, back_off=2.0)
        model.cam_pos[cid] = pos
        model.cam_quat[cid] = quat
        model.cam_fovy[cid] = fovy
        # cam_xpos/cam_xmat are derived in mj_forward: re-run kinematics AFTER
        # editing the model camera or the scene keeps the previous pose.
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="rigcam")
        rend = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        cap = cv2.VideoCapture(os.path.join(session_dir, f"{cn}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_abs + t)
        okf, real = cap.read(); cap.release()
        if not okf:
            print(f"[rigcam] could not read {cn} frame {start_abs + t}"); continue

        if masks is not None and masks["valid"][t, ci]:
            mk = masks["masks"][t, ci].astype(np.uint8)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(real, cnts, -1, (255, 150, 0), 2)
            cv2.drawContours(rend, cnts, -1, (255, 150, 0), 2)

        uv = reproject.project(cam_mats[ci], Xw[ok])
        cx, cy = int(np.median(uv[:, 0])), int(np.median(uv[:, 1]))
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(W, cx + pad), min(H, cy + pad)
        L, Rr = real[y0:y1, x0:x1].copy(), rend[y0:y1, x0:x1].copy()
        for im, lab in ((L, f"{cn} REAL + SAM mask"), (Rr, f"{cn} MuJoCo @ rig camera")):
            cv2.putText(im, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        rows.append(np.hstack([L, np.full((L.shape[0], 3, 3), 60, np.uint8), Rr]))

    if not rows:
        raise RuntimeError("no cameras rendered")
    h = min(r.shape[0] for r in rows); w = min(r.shape[1] for r in rows)
    out = np.vstack([r[:h, :w] for r in rows])
    dst = args.out or f"rigcam_bout{bout}_fly{fly}_f{t}.png"
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    cv2.imwrite(dst, out)
    print(f"[rigcam] wrote {dst} ({out.shape[1]}x{out.shape[0]}, {len(rows)} cameras)")
    return 0
