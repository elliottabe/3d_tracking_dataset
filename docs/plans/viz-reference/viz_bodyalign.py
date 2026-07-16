"""Visualize the body-alignment transform. Per frame, fit a single similarity
(scale+R+t) on the BODY sites mapping the fitted model (kp3d_mm) -> detector
triangulation (kp3d), apply it to ALL sites, and overlay:
  RED   = raw fit sites (kp3d_mm) reprojected        (as-is, ~30px off)
  GREEN = body-aligned fit sites reprojected          (after the global transform)
  CYAN  = detector 2D keypoints                        (truth)
If GREEN snaps onto CYAN for body+coxa/tro (and only the distal tips stay off),
the base error is a GLOBAL shift (bridge), not a per-site leg error."""
import os, sys
import numpy as np, cv2
os.environ.setdefault("USER", "eabe")
from hydra import initialize_config_dir, compose
with initialize_config_dir(version_base=None, config_dir="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"):
    cfg = compose(config_name="pipeline")
sys.path.insert(0, "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset")
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from scripts.run_bout import project_points, bout_start_frame

bout, fly, fr = 1, 1, 250
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
RUN = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
kn = list(cfg.model.KP_NAMES)
body = [i for i, n in enumerate(kn) if any(n.startswith(x) for x in ("Scutellum", "Wing", "Antenna", "Eye", "Abd"))]
segs = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]
legchains = [[kn.index(f"{p}_{s}") for s in segs if f"{p}_{s}" in kn]
             for p in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")]
rt = ReprojectionTool(cfg.recording.calib_dir); cm = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
kp3d = np.load(f"{RUN}/bouts/bout_{bout:05d}/fly{fly}/kp3d.npz")["kp3d"][fr]           # detector 3D
mm = np.asarray(ioh5.load(f"{RUN}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]
det = np.load(f"{RUN}/bouts/bout_{bout:05d}/fly{fly}/kp2d.npz")["kp2d"][fr]            # detector 2D

def umeyama(X, Y):
    muX, muY = X.mean(0), Y.mean(0); Xc, Yc = X - muX, Y - muY
    U, S, Vt = np.linalg.svd((Yc.T @ Xc) / len(X))
    d = np.sign(np.linalg.det(U @ Vt)); Dg = np.diag([1, 1, d])
    R = U @ Dg @ Vt; s = (S * np.array([1, 1, d])).sum() / (Xc ** 2).sum() * len(X)
    return s, R, muY - s * R @ muX

ok = np.isfinite(mm).all(-1) & np.isfinite(kp3d).all(-1)
b = [i for i in body if ok[i]]
s, R, t = umeyama(mm[b], kp3d[b])
mm_al = (s * (R @ mm.T).T + t)          # body-aligned fit sites

def chain2d(bgr, kp2d_cam, col, th=1):
    for chain in legchains:
        pts = [(int(kp2d_cam[k, 0]), int(kp2d_cam[k, 1])) for k in chain if np.isfinite(kp2d_cam[k]).all()]
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, th)

def chain3d(bgr, ci, P, col, th=1):
    for chain in legchains:
        pts = []
        for k in chain:
            u = project_points(cm[ci], P[k][None])[0]
            if np.isfinite(u).all(): pts.append((int(u[0]), int(u[1])))
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, th)

show = ["Cam2012630", "Cam2012853"]
tiles = []
for cam in show:
    ci = cams.index(cam)
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    okr, bgr = cap.read(); cap.release()
    if not okr: continue
    # all-site dots + leg chains
    for k in range(len(kn)):
        u = project_points(cm[ci], mm[k][None])[0]
        if np.isfinite(u).all(): cv2.circle(bgr, (int(u[0]), int(u[1])), 2, (0, 0, 255), -1)   # raw fit red
        ua = project_points(cm[ci], mm_al[k][None])[0]
        if np.isfinite(ua).all(): cv2.circle(bgr, (int(ua[0]), int(ua[1])), 2, (0, 255, 0), -1)  # aligned green
        if np.isfinite(det[ci, k]).all(): cv2.circle(bgr, (int(det[ci, k, 0]), int(det[ci, k, 1])), 2, (255, 255, 0), -1)  # detector cyan
    chain3d(bgr, ci, mm, (0, 0, 255)); chain3d(bgr, ci, mm_al, (0, 255, 0)); chain2d(bgr, det[ci], (255, 255, 0))
    allu = np.array([project_points(cm[ci], mm[k][None])[0] for k in range(len(kn)) if np.isfinite(mm[k]).all()])
    x0, y0 = allu.min(0) - 55; x1, y1 = allu.max(0) + 55
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(bgr.shape[1], int(x1)); y1 = min(bgr.shape[0], int(y1))
    crop = bgr[y0:y1, x0:x1].copy(); cv2.putText(crop, cam, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(crop)
h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
pad = [np.zeros((h, w, 3), np.uint8) for _ in range(2)]
for i, tt in enumerate(tiles): pad[i][:tt.shape[0], :tt.shape[1]] = tt
banner = np.zeros((90, 2 * w, 3), np.uint8)
for i, (txt, col) in enumerate([("RED = raw fit (as-is)", (0, 0, 255)),
                                ("GREEN = body-aligned fit (global transform)", (0, 255, 0)),
                                ("CYAN = detector keypoints (truth)", (255, 255, 0)),
                                (f"s={s:.3f}  |t|={np.linalg.norm(t):.3f}  bout{bout} fly{fly} f{fr}", (255, 255, 255))]):
    cv2.putText(banner, txt, (5, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
mont = np.vstack([np.hstack(pad), banner])
out = f"{SP}/bodyalign_f{fr}.png"; cv2.imwrite(out, mont); print("wrote", out, mont.shape)
