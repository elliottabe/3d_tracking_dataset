"""Window dataset over the v12 root for the multi-view query model (mvq).

A sample is (recording, host fly, start frame, T): T consecutive labelled
frames of the host, all 7 cameras, cropped at 448 around the projection of
ONE window-level center3D -- the inference convention of
predict/session_frameset.build_frameset, not the per-camera bbox crop of
data/v5_3d.py. Every other labelled fly inside the crops is an extra
instance (fly_valid), so the model can be trained as a set predictor.

Camera order is rt.cameras' sorted-glob order and slots are placed BY NAME
(see data/v5_3d.py docstring for why). Keypoint order is asserted against
annotations/keypoint_names.json.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from jarvis_jax.data.build_v5 import iter_resolved_slots
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.data.v5_3d import _load_mask, _resolve_sex, _frameset_own_sex
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

CROP = 448
WINDOW_KEYS = ("crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
               "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "prompt_mask")


def _parse_key(key):
    rec, frame, fly = key.split("/")
    return rec, int(frame.split("_")[1]), int(fly[3:])


def _affine_np(cam_mats):
    P = np.swapaxes(np.asarray(cam_mats, np.float64), 1, 2)          # (C,3,4)
    if not np.allclose(P[:, 2, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("non-affine calibration")
    return P[:, :2, :3], P[:, :2, 3]


class V12WindowDataset:
    def __init__(self, root, split, T=1, *, max_flies=2, jitter_units=3.0, seed=0,
                 train=True, recordings=None):
        self.root, self.split, self.T = root, split, int(T)
        self.max_flies, self.jitter, self.train = int(max_flies), float(jitter_units), bool(train)
        self.rng = np.random.default_rng(seed)
        coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
        self.manifest = json.load(open(os.path.join(root, "manifest.json")))["recordings"]
        self.keypoint_names = list(coco["keypoint_names"])
        canon = json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))
        if self.keypoint_names != canon:
            raise ValueError("instances keypoint_names != annotations/keypoint_names.json")
        self.K = len(self.keypoint_names)
        self._img = {i["id"]: i for i in coco["images"]}
        self._ann = {a["id"]: a for a in coco["annotations"]}
        self._fs = {}                                   # (rec, frame, fly) -> frameset
        self._tools = {}
        for key, fsv in coco["framesets"].items():
            rec, frame, fly = _parse_key(key)
            if recordings is not None and rec not in recordings:
                continue
            grp = self.manifest[rec]["calib_group"]
            if grp not in self._tools:
                self._tools[grp] = ReprojectionTool(os.path.join(root, "calibrations", str(grp)))
            self._fs[(rec, frame, fly)] = fsv
        self.windows = []
        for (rec, frame, fly) in sorted(self._fs):
            if all((rec, frame + k, fly) in self._fs for k in range(self.T)):
                self.windows.append((rec, fly, frame))
        self._sex = {}
        for (rec, frame, fly), fsv in self._fs.items():
            self._sex[(rec, fly)] = _resolve_sex(_frameset_own_sex(fsv, self._ann), fly,
                                                 self.manifest.get(rec, {}))

    def __len__(self):
        return len(self.windows)

    def calib_group(self, i):
        return self.manifest[self.windows[i][0]]["calib_group"]

    def is_female(self, i):
        rec, fly, _ = self.windows[i]
        return self._sex[(rec, fly)] == "female"

    def n_flies(self, i):
        rec, fly, f0 = self.windows[i]
        others = {k[2] for k in self._fs if k[0] == rec and f0 <= k[1] < f0 + self.T and k[2] != fly}
        return 1 + min(len(others), self.max_flies - 1)

    # ------------------------------------------------------------------ helpers
    def _rt(self, rec):
        return self._tools[self.manifest[rec]["calib_group"]]

    def _labels_full(self, fsv, rt):
        """Per camera (BY NAME) full-frame (K,3) labels, image infos, ann infos."""
        C = rt.num_cameras
        cam_to_row = {n: i for i, n in enumerate(rt.cameras.keys())}
        kp = np.zeros((C, self.K, 3), np.float32)
        infos = [None] * C
        for img_id, ann_id in iter_resolved_slots(fsv):
            info, ann = self._img[img_id], self._ann[ann_id]
            c = cam_to_row.get(info["file_name"].split("/")[1])
            if c is None:
                continue
            k = np.asarray(ann["keypoints"], np.float32)
            if k.size == self.K * 3:
                kp[c] = k.reshape(-1, 3)
            infos[c] = (info, ann)
        return kp, infos

    def _dlt(self, kp, rt):
        C = rt.num_cameras
        X = np.zeros((self.K, 3), np.float32); has = np.zeros(self.K, bool)
        for j in range(self.K):
            use = [c for c in range(C) if kp[c, j, 2] > 0]
            if len(use) >= 2:
                pts = np.zeros((C, 2)); pts[use] = kp[use, j, :2]
                X[j] = rt.reconstruct_point(pts, cams_to_use=use); has[j] = True
        return X, has

    def _decode(self, info):
        with Image.open(os.path.join(self.root, "images", info["file_name"])) as im:
            return np.asarray(im.convert("RGB"), np.uint8)

    # ------------------------------------------------------------------ sample
    def __getitem__(self, i):
        rec, host, f0 = self.windows[i]
        rt = self._rt(rec); C = rt.num_cameras; T, K, F = self.T, self.K, self.max_flies
        cams = list(rt.cameras.keys())
        M, t = _affine_np(rt.camera_matrices)                         # (C,2,3),(C,2) float64

        # --- labels per frame per fly (full-frame), 3D via DLT, host first
        frames = [f0 + k for k in range(T)]
        others = sorted({k[2] for k in self._fs if k[0] == rec and k[1] in frames and k[2] != host})
        flies = [host] + others[: F - 1]
        kp_full = np.zeros((F, T, C, K, 3), np.float32)
        X3 = np.zeros((F, T, K, 3), np.float32); has3d = np.zeros((F, T, K), bool)
        infos = {}
        for fi, fly in enumerate(flies):
            for ti, f in enumerate(frames):
                fsv = self._fs.get((rec, f, fly))
                if fsv is None:
                    continue
                kp, inf = self._labels_full(fsv, rt)
                kp_full[fi, ti] = kp
                X3[fi, ti], has3d[fi, ti] = self._dlt(kp, rt)
                for c in range(C):
                    if inf[c] is not None:
                        infos.setdefault((ti, c), inf[c])
        cam_valid = np.zeros((T, C), bool)
        for (ti, c) in infos:
            cam_valid[ti, c] = True
        # host frameset None-slots: camera absent for this window frame
        for ti, f in enumerate(frames):
            fsv = self._fs[(rec, f, host)]
            present = {self._img[img]["file_name"].split("/")[1] for img, _ in iter_resolved_slots(fsv)}
            for c, name in enumerate(cams):
                if name not in present:
                    cam_valid[ti, c] = False

        # --- window centre from the host's 3D (frame 0), jittered in train mode
        vis0 = has3d[0, 0]
        pts = X3[0, 0][vis0] if vis0.any() else np.zeros((1, 3), np.float32)
        center = 0.5 * (pts.max(0) + pts.min(0))
        if self.train and self.jitter > 0:
            center = center + self.rng.uniform(-self.jitter, self.jitter, size=3)
        center = center.astype(np.float32)

        # --- crops around the projection of center (same origin for all frames of the window)
        origin = np.zeros((C, 2), np.int32)
        for c in range(C):
            info = next((infos[(ti, c)][0] for ti in range(T) if (ti, c) in infos), None)
            w, h = (info["width"], info["height"]) if info else (1936, 448)
            u, v = M[c] @ center + t[c]
            origin[c] = crop_origin([u, v, 0, 0], w, h, CROP)
        crops = np.zeros((T, C, CROP, CROP, 3), np.uint8)
        prompt = np.zeros((T, C, CROP, CROP), bool)
        for (ti, c), (info, ann) in infos.items():
            if not cam_valid[ti, c]:
                continue
            img = self._decode(info)
            x0, y0 = origin[c]
            crops[ti, c] = img[y0:y0 + CROP, x0:x0 + CROP]
            # host mask (may be absent -> zeros)
            fsv = self._fs[(rec, frames[ti], host)]
            for img_id, ann_id in iter_resolved_slots(fsv):
                if self._img[img_id]["file_name"] == info["file_name"]:
                    a = self._ann[ann_id]
                    m = _load_mask(self.root, info["file_name"], a.get("src_ann_id", ann_id),
                                   ann_id, info["width"], info["height"])
                    prompt[ti, c] = m[y0:y0 + CROP, x0:x0 + CROP].astype(bool)

        # --- to crop/local coordinates
        t_local = np.zeros((T, C, 2), np.float32)
        for ti in range(T):
            t_local[ti] = (M @ center + t - origin).astype(np.float32)
        kp2d = kp_full[..., :2] - origin[None, None, :, None, :]
        inside = ((kp2d >= 0) & (kp2d <= CROP - 1)).all(-1)
        vis2d = (kp_full[..., 2] > 0) & inside & cam_valid[None, :, :, None]
        fly_valid = np.array([fi < len(flies) and vis2d[fi].any() for fi in range(F)])
        fly_valid[0] = True
        px_scale = float(np.mean(np.sqrt((M ** 2).sum((1, 2)) / 2.0)))
        return {
            "crops": crops, "cam_valid": cam_valid,
            "M": M.astype(np.float32), "t_local": t_local, "center3D": center,
            "kp3d_local": (X3 - center).astype(np.float32) * has3d[..., None],
            "has3d": has3d, "kp2d": kp2d.astype(np.float32), "vis2d": vis2d,
            "fly_valid": fly_valid, "px_scale": np.float32(px_scale),
            "is_female": np.bool_(self.is_female(i)), "prompt_mask": prompt,
        }


def window_batches(ds, batch_size, *, shuffle=True, seed=0, weights=None, num_workers=8,
                   drop_last=True):
    rng = np.random.default_rng(seed)
    n = len(ds)
    if weights is not None:
        w = np.asarray(weights, np.float64); w = w / w.sum()
        idx = rng.choice(n, size=n, replace=True, p=w)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
    stop = (n // batch_size) * batch_size if drop_last else n
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for s in range(0, stop, batch_size):
            samples = list(pool.map(ds.__getitem__, [int(i) for i in idx[s:s + batch_size]]))
            yield {k: np.stack([smp[k] for smp in samples]) for k in WINDOW_KEYS}
