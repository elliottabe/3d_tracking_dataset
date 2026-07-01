"""Silhouette wing-tip landmark extraction (productionized from demo_wingtip_triangulation).

For each wing side, march along the predicted wing axis (proximal->tip) to the far
edge of the fly's SAM mask in each camera, then affine-DLT-triangulate those 2-D
tips into one 3-D point. Cameras are telecentric (affine). Confidence = #cameras used.
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.cse.affine_camera import project_affine


def wing_side_vertices(mesh_npz):
    z = np.load(mesh_npz, allow_pickle=True)
    fps = z["fps_300"] if "fps_300" in z.files else z[f"fps_{len(z['vertex_segment'])}"]
    seg = z["vertex_segment"][fps]; C = z["vertices"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(z["seg_ids"], z["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    thorax_c = C[[("thorax" in n) for n in names]].mean(0)
    out = {}
    for side in ("left", "right"):
        si = np.where([("wing" in n and side in n) for n in names])[0]
        d = np.linalg.norm(C[si] - thorax_c, axis=1)
        out[side] = {"tip": int(si[d.argmax()]), "prox": int(si[d.argmin()])}
    return out


def mask_wing_tip_2d(mask, prox2d, tip2d, corridor=12.0):
    ax = np.asarray(tip2d, float) - np.asarray(prox2d, float)
    L = np.linalg.norm(ax)
    if L < 5:
        return None
    ax = ax / L
    ys, xs = np.where(mask)
    rel = np.stack([xs - prox2d[0], ys - prox2d[1]], 1)
    along = rel @ ax
    perp = np.abs(rel[:, 0] * (-ax[1]) + rel[:, 1] * ax[0])
    corr = (perp < corridor) & (along > 0.3 * L)
    if corr.sum() < 3:
        return None
    k = int(np.argmax(along[corr]))
    return np.array([xs[corr][k], ys[corr][k]], float)


def _triangulate(cam_mats, cam_ids, pts2d):
    A = np.zeros((2 * len(cam_ids), 4))
    for i, c in enumerate(cam_ids):
        P = np.asarray(cam_mats[c]); uv = pts2d[c]
        A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
    _, _, Vh = np.linalg.svd(A)
    Xh = Vh[-1]
    return (Xh / Xh[3])[:3]


def triangulate_wing_tips(masks, cam_mats, prox3d, predtip3d, *, corridor=12.0):
    C = len(cam_mats)
    out = {}
    for side in ("left", "right"):
        pts = np.zeros((C, 2)); used = []
        for c in range(C):
            if masks[c] is None:
                continue
            prox2d = project_affine(cam_mats[c], prox3d[side])
            tip2d = project_affine(cam_mats[c], predtip3d[side])
            t = mask_wing_tip_2d(masks[c], prox2d, tip2d, corridor=corridor)
            if t is not None:
                pts[c] = t; used.append(c)
        out[side] = (_triangulate(cam_mats, used, pts), len(used)) if len(used) >= 2 else None
    return out
