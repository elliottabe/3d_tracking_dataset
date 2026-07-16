"""Compare mesh legs: CURRENT run (legs in silhouette DOF, curl?) vs LEGTEST
(legs dropped from silhouette DOF -> keypoint-driven). Overlay both meshes'
projected verts (current=red, legtest=green) + detector leg-tip keypoints (cyan)
on a few cams for one fly/frame, so we can see which matches the straight legs."""
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

bout, fly, fr = 1, int(sys.argv[1]) if len(sys.argv) > 1 else 1, 250
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
CUR = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
LEG = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_legtest"
kn = list(cfg.model.KP_NAMES)
LEGTIPS = [i for i, n in enumerate(kn) if n.endswith(("TaTip", "TaT3", "TiTa"))]
rt = ReprojectionTool(cfg.recording.calib_dir); cam_mats = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)

cur = np.asarray(ioh5.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["mesh_mm"])[fr]
leg = np.asarray(ioh5.load(f"{LEG}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["mesh_mm"])[fr]
z = np.load(f"{CUR}/bouts/bout_{bout:05d}/fly{fly}/kp2d.npz"); det = z["kp2d"][fr]  # (C,K,2)

# show the 3 cams where legs are most visible (side views)
show = ["Cam2012630", "Cam2012853", "Cam2012855"]
tiles = []
for cam in show:
    ci = cams.index(cam)
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    ok, bgr = cap.read(); cap.release()
    if not ok: continue
    for m, col in ((cur, (0, 0, 255)), (leg, (0, 255, 0))):    # current=red, legtest=green
        mm = m[np.isfinite(m).all(-1)]
        for x, y in project_points(cam_mats[ci], mm):
            if np.isfinite(x) and np.isfinite(y): cv2.circle(bgr, (int(x), int(y)), 1, col, -1)
    for k in LEGTIPS:                                           # detector leg tips = cyan
        p = det[ci, k]
        if np.isfinite(p).all(): cv2.circle(bgr, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)
    allp = project_points(cam_mats[ci], cur[np.isfinite(cur).all(-1)])
    x0, y0 = allp.min(0) - 70; x1, y1 = allp.max(0) + 70
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(bgr.shape[1], int(x1)); y1 = min(bgr.shape[0], int(y1))
    crop = bgr[y0:y1, x0:x1].copy(); cv2.putText(crop, cam, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(crop)

h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
pad = [np.zeros((h, w, 3), np.uint8) for _ in range(4)]
for i, t in enumerate(tiles): pad[i][:t.shape[0], :t.shape[1]] = t
for i, (txt, col) in enumerate([("RED = current (legs in silhouette DOF)", (0, 0, 255)),
                                ("GREEN = legtest (legs keypoint-driven)", (0, 255, 0)),
                                ("CYAN = detector leg-tip kp", (255, 255, 0)),
                                (f"bout{bout} fly{fly} f{fr}", (255, 255, 255))]):
    cv2.putText(pad[3], txt, (5, 24 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
mont = np.vstack([np.hstack(pad[0:2]), np.hstack(pad[2:4])])
out = f"{SP}/legcompare_fly{fly}.png"; cv2.imwrite(out, mont); print("wrote", out, mont.shape)
