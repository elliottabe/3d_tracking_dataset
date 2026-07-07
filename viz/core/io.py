"""Artifact loaders + run-path resolution + frame reading for viz views."""
import os
import numpy as np
import cv2
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks

def fly_dir(run_root, bout, fly):
    return os.path.join(run_root, "bouts", f"bout_{int(bout):05d}", f"fly{int(fly)}")

def load_outputs(run_root, bout, fly):
    d = ioh5.load(os.path.join(fly_dir(run_root, bout, fly), "outputs.h5"))
    return {"kp3d_mm": np.asarray(d["kp3d_mm"]), "mesh_mm": np.asarray(d["mesh_mm"]),
            "kp_names": [str(n) for n in np.asarray(d["kp_names"]).tolist()]}

def load_kp2d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp2d.npz")) as z:
        return np.asarray(z["kp2d"]), np.asarray(z["conf"])

def load_kp3d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp3d.npz")) as z:
        return np.asarray(z["kp3d"]), np.asarray(z["conf3d"])

def load_masks(predictions_dir, bout, fly, cameras):
    npz = os.path.join(predictions_dir, f"bout_{int(bout):05d}", "sam3_masks.npz")
    return load_bout_masks(npz, fly, expected_cameras=list(cameras))

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read(); cap.release()
    if not ok:
        raise IndexError(f"frame {frame_idx} not readable in {video_path}")
    return bgr

def read_frames(session_dir, cameras, start, count):
    caps = [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4")) for c in cameras]
    try:
        for cap in caps: cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for _ in range(int(count)):
            imgs = []
            for cap in caps:
                ok, bgr = cap.read()
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
            yield imgs
    finally:
        for cap in caps: cap.release()
