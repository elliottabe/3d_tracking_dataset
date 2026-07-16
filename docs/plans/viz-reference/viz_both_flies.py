"""Overlay both flies' articulated mesh (from outputs.h5 mesh_mm) on each camera for a
frame, fly0=cyan, fly1=orange. Montage of 7 cams. Shows the corrected-mask fit quality."""
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

bout = int(sys.argv[1]) if len(sys.argv) > 1 else 1
fr = int(sys.argv[2]) if len(sys.argv) > 2 else 250
run = sys.argv[3] if len(sys.argv) > 3 else "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
rt = ReprojectionTool(cfg.recording.calib_dir); cam_mats = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
mesh = {f: np.asarray(ioh5.load(f"{run}/bouts/bout_{bout:05d}/fly{f}/outputs.h5")["mesh_mm"]) for f in (0, 1)}
COL = {0: (255, 255, 0), 1: (0, 165, 255)}   # fly0 cyan, fly1 orange (BGR)

tiles = []
for ci, cam in enumerate(cams):
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    ok, bgr = cap.read(); cap.release()
    if not ok: continue
    allpts = []
    for f in (0, 1):
        m = mesh[f][fr]; m = m[np.isfinite(m).all(-1)]
        if len(m):
            uv = project_points(cam_mats[ci], m)
            for x, y in uv:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(bgr, (int(round(x)), int(round(y))), 1, COL[f], -1)
                    allpts.append((x, y))
    allpts = np.array(allpts) if allpts else np.array([[bgr.shape[1]/2, bgr.shape[0]/2]])
    x0, y0 = allpts.min(0) - 55; x1, y1 = allpts.max(0) + 55
    x0 = max(0, int(x0)); y0 = max(0, int(y0)); x1 = min(bgr.shape[1], int(x1)); y1 = min(bgr.shape[0], int(y1))
    crop = bgr[y0:y1, x0:x1].copy()
    cv2.putText(crop, cam, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(crop)

h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
pad = [np.zeros((h, w, 3), np.uint8) for _ in range(8)]
for i, t in enumerate(tiles): pad[i][:t.shape[0], :t.shape[1]] = t
cv2.putText(pad[7], "fly0=CYAN", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.putText(pad[7], "fly1=ORANGE", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
cv2.putText(pad[7], f"bout{bout} frame{fr}", (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(pad[7], "corrected masks", (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
mont = np.vstack([np.hstack(pad[0:2]), np.hstack(pad[2:4]), np.hstack(pad[4:6]), np.hstack(pad[6:8])])
out = f"{SP}/bothflies_bout{bout}_f{fr}.png"
cv2.imwrite(out, mont); print("wrote", out, mont.shape)
