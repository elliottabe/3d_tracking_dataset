"""3D(mm)->2D(px) reprojection, ReprojectionTool `ph @ M` convention.
M is a (4,3) camera matrix; ph=[pts,1]. Verbatim match to
scripts/run_bout.py::project_points."""
import numpy as np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

def camera_matrices(calib_dir):
    rt = ReprojectionTool(calib_dir)
    names = [c.name for c in rt._camera_list]
    return np.asarray(rt.camera_matrices, np.float32), names

def project(cam_mat_4x3, pts3d):
    pts = np.asarray(pts3d, np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), np.float32)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)   # (N,4)
    proj = ph @ np.asarray(cam_mat_4x3, np.float64)                  # (N,3)
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

def reproject_all(cam_mats, pts3d):
    return np.stack([project(m, pts3d) for m in np.asarray(cam_mats)])
