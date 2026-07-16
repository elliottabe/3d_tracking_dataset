"""Design-1 augmented datasets: keypoints (50) + canonical vertices (M) = 50+M joints.

A canonical vertex is treated as an extra "joint", so the existing ViTPose /
reproject / V2VNet / soft-argmax / STAC-IK path runs unchanged at J = 50 + M.

* CSEImageDataset      (2D, wraps V3Dataset): per image, keypoints become
                       (50+M, 3) = [annotated kp ; projected vertex 2D].
* CSEFramesetDataset   (3D, wraps V3FramesetDataset): kp3d/vis become (50+M, *)
                       = [triangulated kp ; STAC vertex 3D].

Vertex 2D/3D labels come from the auto-labeler (cse_labels.py) merged per split
(cse_labels_<split>_M200.npz).  Augmented keypoint names + a kNN vertex edge
graph (for the graph-Laplacian shape prior) are provided for trainer wiring.
"""
from __future__ import annotations

import json
import os

import numpy as np

from jarvis_jax.data.v3 import V3Dataset
from jarvis_jax.data.v3_3d import V3FramesetDataset


def load_aux(npz_path):
    """Load merged per-split labels -> (img2lab2d, img2verts3d, fps_indices, M)."""
    z = np.load(npz_path, allow_pickle=True)
    M = int(z["M"])
    img_ids = z["image_ids"]; lab2d = z["labels2d"]
    img2lab2d = {int(i): lab2d[k] for k, i in enumerate(img_ids)}
    fs_img = z["fs_image_ids"]; v3d = z["verts3d"]          # (F,nc),(F,M,3)
    img2verts3d = {}
    for f in range(len(fs_img)):
        for iid in fs_img[f]:
            img2verts3d[int(iid)] = v3d[f]
    return img2lab2d, img2verts3d, z["fps_indices"], M


def augmented_keypoint_names(base_names, M):
    return list(base_names) + [f"vtx_{i}" for i in range(M)]


def vertex_edges(mesh_npz, fps_indices, k=4):
    """kNN graph among the M FPS vertices (canonical mesh) as (vtx_i, vtx_j) name pairs.

    Provides a smoothness/shape prior over the dense vertices for graph_laplacian.
    """
    from scipy.spatial import cKDTree
    z = np.load(mesh_npz, allow_pickle=True)
    pts = z["vertices"][fps_indices]                       # (M,3)
    tree = cKDTree(pts)
    _, nn = tree.query(pts, k=k + 1)                       # incl self
    edges = set()
    for i in range(len(pts)):
        for j in nn[i, 1:]:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))
    return [{"keypointA": f"vtx_{a}", "keypointB": f"vtx_{b}"} for a, b in sorted(edges)]


def _file2imgid(root, split):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    return {im["file_name"]: int(im["id"]) for im in coco["images"]}


class CSEImageDataset(V3Dataset):
    """2D dataset with keypoints augmented to 50 + M (vertex 2D labels appended)."""

    def __init__(self, root, split, aux_npz, *, recordings=None, **kw):
        super().__init__(root, split, recordings=recordings, **kw)
        img2lab2d, _, _, M = load_aux(aux_npz)
        self.M = M
        f2id = _file2imgid(root, split)
        n_missing = 0
        for i, fn in enumerate(self.file_names):
            iid = f2id.get(fn, -1)
            vlab = img2lab2d.get(iid)
            if vlab is None:
                vlab = np.zeros((M, 3), np.float32); n_missing += 1
            # keypoints[i]: (50,3) full-image px [x,y,vis]; append (M,3) -> (50+M,3)
            self.keypoints[i] = np.concatenate(
                [self.keypoints[i], vlab.astype(np.float32)], axis=0)
        self.num_joints = self.keypoints[0].shape[0]
        if n_missing:
            print(f"[CSEImageDataset] {n_missing}/{len(self.file_names)} imgs without "
                  f"vertex labels (zero-filled, vis=0)")


class CSEFramesetDataset(V3FramesetDataset):
    """3D frameset dataset with kp3d/vis augmented to 50 + M (STAC vertex 3D appended)."""

    def __init__(self, root, split, aux_npz, *, recordings=None, **kw):
        super().__init__(root, split, recordings=recordings, **kw)
        _, self._img2verts3d, _, self.M = load_aux(aux_npz)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)                     # kp3d (50,3), vis (50,)
        fs = self.framesets[idx]
        v3d = None
        for iid in fs["frames"]:
            if int(iid) in self._img2verts3d:
                v3d = self._img2verts3d[int(iid)]; break
        if v3d is None:
            v3d = np.zeros((self.M, 3), np.float32)
            vvis = np.zeros(self.M, bool)
        else:
            vvis = np.ones(self.M, bool)
        out["kp3d"] = np.concatenate([out["kp3d"], v3d.astype(np.float32)], 0)
        out["vis"] = np.concatenate([out["vis"], vvis], 0)
        return out
