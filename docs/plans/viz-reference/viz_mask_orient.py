"""Per-fly viz: corrected SAM mask (green outline) + fitted mesh cloud (faint) + fitted
HEAD sites (eyes/antenna, red) + TAIL (abd_tip, blue) + head->tail axis (yellow), to check
mask alignment AND mesh orientation vs the fly. Montage of 7 cams."""
import os, sys
import numpy as np, cv2
os.environ.setdefault("USER", "eabe")
from hydra import initialize_config_dir, compose
with initialize_config_dir(version_base=None, config_dir="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"):
    cfg = compose(config_name="pipeline")
sys.path.insert(0, "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset")
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.bout_masks import load_bout_masks
from scripts.run_bout import project_points, bout_start_frame

bout = int(sys.argv[1]) if len(sys.argv) > 1 else 1
fly = int(sys.argv[2]) if len(sys.argv) > 2 else 0
fr = int(sys.argv[3]) if len(sys.argv) > 3 else 250
SP = "/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad"
S0 = str(cfg.recording.session_dir); cams = list(cfg.recording.cameras)
run = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07042026"
kn = list(cfg.model.KP_NAMES); HEAD = [kn.index(x) for x in ("EyeL","EyeR","Antenna_Base")]; TAIL = kn.index("Abd_tip")
rt = ReprojectionTool(cfg.recording.calib_dir); cam_mats = np.asarray(rt.camera_matrices, np.float32)
start = bout_start_frame(cfg, bout)
d = ioh5.load(f"{run}/bouts/bout_{bout:05d}/fly{fly}/outputs.h5")
mesh_mm = np.asarray(d["mesh_mm"]); kp3d_mm = np.asarray(d["kp3d_mm"])
mk = load_bout_masks(f"{S0}/Predictions_3D_36233268_fixed/bout_{bout:05d}/sam3_masks.npz", fly,
                     expected_cameras=list(cfg.recording.cameras))

tiles = []
for ci, cam in enumerate(cams):
    cap = cv2.VideoCapture(f"{S0}/{cam}.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, start + fr)
    ok, bgr = cap.read(); cap.release()
    if not ok: continue
    # corrected SAM mask outline (green)
    if mk["valid"][fr, ci] and mk["masks"][fr, ci].any():
        cnts, _ = cv2.findContours(mk["masks"][fr, ci].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, (0, 255, 0), 1)
    # faint mesh cloud (gray)
    m = mesh_mm[fr]; m = m[np.isfinite(m).all(-1)]
    if len(m):
        for x, y in project_points(cam_mats[ci], m):
            if np.isfinite(x) and np.isfinite(y): cv2.circle(bgr, (int(x), int(y)), 1, (200, 200, 200), -1)
    # fitted head sites (red) + tail (blue) + axis
    kp = kp3d_mm[fr]
    hh = kp[HEAD]; hh = hh[np.isfinite(hh).all(-1)]
    tt = kp[TAIL]
    if len(hh):
        huv = project_points(cam_mats[ci], hh)
        for x, y in huv: cv2.circle(bgr, (int(x), int(y)), 3, (0, 0, 255), -1)   # head red
        hc = huv.mean(0)
        if np.isfinite(tt).all():
            tuv = project_points(cam_mats[ci], tt[None])[0]
            cv2.circle(bgr, (int(tuv[0]), int(tuv[1])), 4, (255, 0, 0), -1)      # tail blue
            cv2.arrowedLine(bgr, (int(tuv[0]), int(tuv[1])), (int(hc[0]), int(hc[1])), (0, 255, 255), 2, tipLength=0.3)
    # crop around mask+mesh
    allp = np.array(list(project_points(cam_mats[ci], m)) if len(m) else [[bgr.shape[1]/2, bgr.shape[0]/2]])
    x0,y0 = allp.min(0)-50; x1,y1 = allp.max(0)+50
    x0=max(0,int(x0)); y0=max(0,int(y0)); x1=min(bgr.shape[1],int(x1)); y1=min(bgr.shape[0],int(y1))
    crop = bgr[y0:y1, x0:x1].copy()
    cv2.putText(crop, cam, (3,14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    tiles.append(crop)

h=max(t.shape[0] for t in tiles); w=max(t.shape[1] for t in tiles)
pad=[np.zeros((h,w,3),np.uint8) for _ in range(8)]
for i,t in enumerate(tiles): pad[i][:t.shape[0],:t.shape[1]]=t
L=pad[7]
for i,(txt,col) in enumerate([("green=SAM mask",(0,255,0)),("gray=mesh",(200,200,200)),("red=fitted HEAD",(0,0,255)),("blue=fitted TAIL",(255,0,0)),("yellow=tail->head axis",(0,255,255)),(f"bout{bout} fly{fly} f{fr}",(255,255,255))]):
    cv2.putText(L,txt,(5,22+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
mont=np.vstack([np.hstack(pad[0:2]),np.hstack(pad[2:4]),np.hstack(pad[4:6]),np.hstack(pad[6:8])])
out=f"{SP}/orient_bout{bout}_fly{fly}_f{fr}.png"; cv2.imwrite(out,mont); print("wrote",out,mont.shape)
