"""One-time quadric decimation of a canonical mesh for fast overlay rasterization.

The 875k-vertex visual mesh is too heavy to fillPoly per frame x camera x bout.
A ~5k-face decimated copy renders ~100x faster and is visually equivalent for
overlay silhouettes. Kinematics/outputs never use this — only the raster.
"""
from __future__ import annotations
import numpy as np
import trimesh


def decimate_mesh_npz(mesh_npz: str, out_npz: str, *, target_faces: int = 5000) -> str:
    z = np.load(mesh_npz, allow_pickle=True)
    m = trimesh.Trimesh(np.asarray(z["vertices"], np.float64),
                        np.asarray(z["faces"]), process=False)
    d = m.simplify_quadric_decimation(face_count=target_faces)
    np.savez(out_npz,
             vertices=np.asarray(d.vertices, np.float32),
             faces=np.asarray(d.faces, np.int32),
             source=str(mesh_npz), target_faces=np.int32(target_faces))
    return out_npz
