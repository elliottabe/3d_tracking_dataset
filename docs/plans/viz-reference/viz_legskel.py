"""Compare FITTED LEG SKELETONS (kp3d_mm leg-joint chains) between CURRENT (legs in
silhouette DOF) and LEGTEST (legs keypoint-driven), vs the DETECTOR leg keypoints.
Draws each leg as a connected chain proximal->distal so curled vs straight is legible."""
import os, sys
import numpy as np, cv2
os.environ.setdefault("USER", "eabe")
from hydra import initialize_config_dir, compose
with initialize_config_dir(version_base=None, config_dir="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"):
    cfg = compose(config_name="courtship_pipeline")
sys.path.insert(0, "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset")
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from scripts.run_courtship_bout import project_points, bout_start_frame

bout, fly, fr = 1, int(sys.argv[1]) if len(sys.argv) > 1 else 1, int(sys.argv[2]) if len(sys.argv) > 2 else 250
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
CUR = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
LEG = sys.argv[3] if len(sys.argv) > 3 else "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_legmask"
kn = list(cfg.model.KP_NAMES)
# 6 leg chains proximal->distal
segs = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]
legs = {}
for pref in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R"):
    chain = [kn.index(f"{pref}_{s}") for s in segs if f"{pref}_{s}" in kn]
    legs[pref] = chain
LEGTIP_IDX = [kn.index(f"{p}_TaTip") for p in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")]
rt = ReprojectionTool(cfg.recording.calib_dir); cam_mats = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
cur = np.asarray(ioh5.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]
leg = np.asarray(ioh5.load(f"{LEG}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]
z = np.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/kp2d.npz"); det = z["kp2d"][fr]

def draw_chains(bgr, ci, kp3d, col):
    for chain in legs.values():
        pts = []
        for k in chain:
            if np.isfinite(kp3d[k]).all():
                u = project_points(cam_mats[ci], kp3d[k][None])[0]
                if np.isfinite(u).all(): pts.append((int(u[0]), int(u[1])))
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, 1)
        for p in pts: cv2.circle(bgr, p, 2, col, -1)

def draw_chains_2d(bgr, kp2d_cam, col, thick=2):
    """Connect leg-chain joints directly from 2D detector keypoints (no projection)."""
    for chain in legs.values():
        pts = [(int(kp2d_cam[k, 0]), int(kp2d_cam[k, 1])) for k in chain
               if np.isfinite(kp2d_cam[k]).all()]
        for j in range(len(pts) - 1): cv2.line(bgr, pts[j], pts[j + 1], col, thick)
        for p in pts: cv2.circle(bgr, p, 3, col, -1)

show = ["Cam2012630", "Cam2012853", "Cam2012855"]
tiles = []
for cam in show:
    ci = cams.index(cam)
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    ok, bgr = cap.read(); cap.release()
    if not ok: continue
    draw_chains_2d(bgr, det[ci], (255, 255, 0), thick=2)  # DETECTOR 2D leg skeleton = yellow (on the legs)
    draw_chains(bgr, ci, cur, (0, 0, 255))     # current fit = red
    draw_chains(bgr, ci, leg, (0, 255, 0))     # legmask fit = green
    # crop around the union of current leg joints
    allk = np.array([project_points(cam_mats[ci], cur[k][None])[0]
                     for chain in legs.values() for k in chain if np.isfinite(cur[k]).all()])
    x0, y0 = allk.min(0) - 60; x1, y1 = allk.max(0) + 60
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(bgr.shape[1], int(x1)); y1 = min(bgr.shape[0], int(y1))
    crop = bgr[y0:y1, x0:x1].copy(); cv2.putText(crop, cam, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(crop)

h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
pad = [np.zeros((h, w, 3), np.uint8) for _ in range(4)]
for i, t in enumerate(tiles): pad[i][:t.shape[0], :t.shape[1]] = t
for i, (txt, col) in enumerate([("YELLOW = DETECTOR 2D leg skeleton (on legs)", (255, 255, 0)),
                                ("RED = current fit (body-only mask)", (0, 0, 255)),
                                ("GREEN = legmask fit (mask+leg skel)", (0, 255, 0)),
                                (f"bout{bout} fly{fly} f{fr}", (255, 255, 255))]):
    cv2.putText(pad[3], txt, (5, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
mont = np.vstack([np.hstack(pad[0:2]), np.hstack(pad[2:4])])
out = f"{SP}/legskel_fly{fly}_f{fr}.png"; cv2.imwrite(out, mont); print("wrote", out, mont.shape)
