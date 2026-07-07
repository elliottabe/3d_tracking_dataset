"""Legs vs mask vs detector, 2 cams x 2 conditions. Each panel: the MASK used
(translucent fill) + DETECTOR 2D leg skeleton (cyan) + that condition's FITTED leg
skeleton. Left col = current (body-only SAM mask, red fit); right col = legmask
(SAM mask + leg skeleton, green fit). Shows whether each fit follows its mask and
how both compare to the detector legs."""
import os, sys
import numpy as np, cv2
os.environ.setdefault("USER", "eabe")
from hydra import initialize_config_dir, compose
with initialize_config_dir(version_base=None, config_dir="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"):
    cfg = compose(config_name="courtship_pipeline")
sys.path.insert(0, "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset")
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks
from scripts.run_courtship_bout import project_points, bout_start_frame, all_cams_frames, open_video_captures

bout, fly, fr = 1, 1, 250
THICK = 18
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
CUR = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
LEG = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_legmask"
kn = list(cfg.model.KP_NAMES)
segs = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]
legchains = [[kn.index(f"{p}_{s}") for s in segs if f"{p}_{s}" in kn]
             for p in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")]
rt = ReprojectionTool(cfg.recording.calib_dir); cm = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
det = np.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/kp2d.npz")["kp2d"][fr]           # (C,K,2)
cur = np.asarray(ioh5.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]
leg = np.asarray(ioh5.load(f"{LEG}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]
md = load_bout_masks(f"{cfg.recording.predictions_dir}/bout_{bout:05d}/sam3_masks.npz", fly, expected_cameras=cams)

def leg_aug(mask_bool, kp2d_cam):
    canvas = mask_bool.astype(np.uint8)
    for chain in legchains:
        pts = [(int(kp2d_cam[k, 0]), int(kp2d_cam[k, 1])) for k in chain if np.isfinite(kp2d_cam[k]).all()]
        for j in range(len(pts) - 1): cv2.line(canvas, pts[j], pts[j + 1], 1, THICK)
    return canvas.astype(bool)

def fill(bgr, m, col, a=0.30):
    ov = bgr.copy(); ov[m] = col
    return cv2.addWeighted(ov, a, bgr, 1 - a, 0)

def chains3d(bgr, ci, P, col):
    for chain in legchains:
        pts = []
        for k in chain:
            if np.isfinite(P[k]).all():
                u = project_points(cm[ci], P[k][None])[0]
                if np.isfinite(u).all(): pts.append((int(u[0]), int(u[1])))
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, 1)
        for p in pts: cv2.circle(bgr, p, 2, col, -1)

def chains2d(bgr, kp, col):
    for chain in legchains:
        pts = [(int(kp[k, 0]), int(kp[k, 1])) for k in chain if np.isfinite(kp[k]).all()]
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, 2)
        for p in pts: cv2.circle(bgr, p, 3, col, -1)

showcams = ["Cam2012630", "Cam2012853"]
caps = open_video_captures(S0, cams)
frames = None
for t, fimgs in enumerate(all_cams_frames(caps, start, md["T"])):
    if t == fr: frames = np.asarray(fimgs); break
for cap in caps: cap.release()

rows = []
for cam in showcams:
    ci = cams.index(cam)
    base = cv2.cvtColor(frames[ci], cv2.COLOR_RGB2BGR)
    bodym = md["masks"][fr, ci]; augm = leg_aug(bodym, det[ci])
    panels = []
    for tag, m, fitP, fitcol in [("current: body-only mask", bodym, cur, (0, 0, 255)),
                                  ("legmask: mask + leg skeleton", augm, leg, (0, 255, 0))]:
        bgr = fill(base.copy(), m, (200, 200, 200))     # mask = translucent grey fill
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, (255, 0, 255), 1)   # mask edge = magenta
        chains2d(bgr, det[ci], (255, 255, 0))           # detector = cyan
        chains3d(bgr, ci, fitP, fitcol)                 # fit = red/green
        # crop to legs
        allk = np.array([project_points(cm[ci], cur[k][None])[0]
                         for chain in legchains for k in chain if np.isfinite(cur[k]).all()])
        x0, y0 = allk.min(0) - 55; x1, y1 = allk.max(0) + 55
        x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(bgr.shape[1], int(x1)); y1 = min(bgr.shape[0], int(y1))
        crop = bgr[y0:y1, x0:x1].copy()
        cv2.putText(crop, f"{cam} {tag}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        panels.append(crop)
    h = max(p.shape[0] for p in panels); w = max(p.shape[1] for p in panels)
    padp = [np.zeros((h, w, 3), np.uint8) for _ in panels]
    for i, p in enumerate(panels): padp[i][:p.shape[0], :p.shape[1]] = p
    rows.append(np.hstack(padp))
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows]
mont = np.vstack(rows)
cv2.putText(mont, "grey=mask magenta=mask-edge  cyan=DETECTOR legs  red/green=FIT", (5, mont.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
out = f"{SP}/legmask_overlay_f{fr}.png"; cv2.imwrite(out, mont); print("wrote", out, mont.shape)
