"""Localize the orientation error: overlay the DETECTOR's head (eyes/antenna, RED) vs
tail (abd_tip, BLUE) keypoints [kp2d] on the raw frame, AND the FITTED head/tail
[kp3d_mm reproj] as hollow circles. If detector head is at the fly's real head but
fitted head isn't -> STAC/mesh orientation bug; if detector head is wrong -> detector."""
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
fly = int(sys.argv[2]) if len(sys.argv) > 2 else 0
fr = int(sys.argv[3]) if len(sys.argv) > 3 else 250
run = sys.argv[4] if len(sys.argv) > 4 else "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
kn = list(cfg.model.KP_NAMES); EYES = [kn.index("EyeL"), kn.index("EyeR")]; ANT = kn.index("Antenna_Base"); TAIL = kn.index("Abd_tip")
rt = ReprojectionTool(cfg.recording.calib_dir); cam_mats = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
z = np.load(f"{run}/bouts/bout_{bout:05d}/fly{fly}/kp2d.npz"); det = z["kp2d"][fr]  # (C,K,2)
kp3d_mm = np.asarray(ioh5.load(f"{run}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")["kp3d_mm"])[fr]

tiles = []
for ci, cam in enumerate(cams):
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    ok, bgr = cap.read(); cap.release()
    if not ok: continue
    # DETECTOR head (eyes red filled, antenna magenta) + tail (abd_tip blue filled)
    for k in EYES:
        p = det[ci, k]
        if np.isfinite(p).all(): cv2.circle(bgr, (int(p[0]), int(p[1])), 4, (0,0,255), -1)
    p = det[ci, ANT]
    if np.isfinite(p).all(): cv2.circle(bgr, (int(p[0]), int(p[1])), 4, (255,0,255), -1)
    p = det[ci, TAIL]
    if np.isfinite(p).all(): cv2.circle(bgr, (int(p[0]), int(p[1])), 5, (255,0,0), -1)
    # FITTED head (eyes, hollow red) + tail (hollow blue) from kp3d_mm reproj
    fh = project_points(cam_mats[ci], kp3d_mm[EYES][np.isfinite(kp3d_mm[EYES]).all(-1)]) if np.isfinite(kp3d_mm[EYES]).all() else []
    for x,y in fh: cv2.circle(bgr, (int(x),int(y)), 7, (0,0,255), 2)
    ft = kp3d_mm[TAIL]
    if np.isfinite(ft).all():
        u = project_points(cam_mats[ci], ft[None])[0]; cv2.circle(bgr, (int(u[0]),int(u[1])), 8, (255,0,0), 2)
    # crop around detector kps
    dv = det[ci][np.isfinite(det[ci]).all(-1)]
    c = dv if len(dv) else np.array([[bgr.shape[1]/2,bgr.shape[0]/2]])
    x0,y0=c.min(0)-55; x1,y1=c.max(0)+55
    x0=max(0,int(x0)); y0=max(0,int(y0)); x1=min(bgr.shape[1],int(x1)); y1=min(bgr.shape[0],int(y1))
    crop=bgr[y0:y1,x0:x1].copy(); cv2.putText(crop,cam,(3,14),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)
    tiles.append(crop)

h=max(t.shape[0] for t in tiles); w=max(t.shape[1] for t in tiles)
pad=[np.zeros((h,w,3),np.uint8) for _ in range(8)]
for i,t in enumerate(tiles): pad[i][:t.shape[0],:t.shape[1]]=t
L=pad[7]
for i,(txt,col) in enumerate([("det EYES=red dot",(0,0,255)),("det antenna=magenta",(255,0,255)),("det TAIL=blue dot",(255,0,0)),("FITTED eyes=red ring",(0,0,255)),("FITTED tail=blue ring",(255,0,0)),(f"bout{bout} fly{fly} f{fr}",(255,255,255))]):
    cv2.putText(L,txt,(5,20+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
mont=np.vstack([np.hstack(pad[0:2]),np.hstack(pad[2:4]),np.hstack(pad[4:6]),np.hstack(pad[6:8])])
out=f"{SP}/headtail_bout{bout}_fly{fly}_f{fr}.png"; cv2.imwrite(out,mont); print("wrote",out,mont.shape)
