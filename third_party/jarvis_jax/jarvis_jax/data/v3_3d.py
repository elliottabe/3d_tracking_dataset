"""3D frameset dataset: multi-camera crops + camera matrices + triangulated GT.

Port of third_party/JARVIS-HybridNet/jarvis/dataset/dataset3D.py, adapted for
the JAX pipeline.  Pure NumPy/Pillow; no torch/jax required.

Public API
----------
V3FramesetDataset(root, split, *, recordings=None)
    .framesets          : list of frameset dicts (datasetName, frames)
    .__len__()          : number of framesets
    .__getitem__(i)     : dict with keys listed below
    .get_joint_2d(i, j) : (num_cam, 3) float32 [x,y,v] for frameset i, joint j

frameset_batches(ds, batch_size, *, shuffle, seed, drop_last) -> iterator

__getitem__ returns
-------------------
crops4         (num_cam, 448, 448, 4) uint8      RGB+mask crop per camera
centerHM       (num_cam, 2)           float32    crop centre in full-image pixels
                                                  (same space as projection matrices)
center3D       (3,)                   float32    ROI centre in world coords
                                                  (quantised midrange of vis kp3d)
cameraMatrices (num_cam, 4, 3)        float32    DLT projection matrices
kp3d           (50, 3)                float32    triangulated world keypoints
                                                  (0,0,0) sentinel for non-visible
vis            (50,)                  bool       True if triangulated from >=2 cams

centerHM definition
-------------------
Matches dataset3D.py: [center_x, center_y] in full-image pixels where the crop
is centred.  Passed directly to ReprojectionLayer which clamps reprojected pixels
around centerHM in heatmap space (heatmap_size = crop/2 + 2 = 226).

center3D definition
-------------------
Matches dataset3D.py: axis-wise truncated midrange of the visible (non-zero)
triangulated keypoints, quantised to grid_spacing (1 mm here).

  center3D[d] = int((max_d + min_d) / grid_spacing / 2) * grid_spacing
"""

import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

_CROP = 448
_GRID_SPACING = 1  # mm; matches unified_V3 config GRID_SPACING: 1


def _load_mask(root: str, split: str, file_name: str, ann_id: int,
               img_w: int, img_h: int) -> np.ndarray:
    """Load SAM3 mask for one annotation; returns float32 (H,W) 0/1."""
    npz_path = os.path.join(
        root, "sam3_masks", split,
        os.path.splitext(file_name)[0] + ".npz")
    if not os.path.exists(npz_path):
        return np.zeros((img_h, img_w), dtype=np.float32)
    try:
        z = np.load(npz_path, allow_pickle=True)
        masks, ids, matched = z["masks"], z["ann_ids"], z["matched"]
        if masks.shape[0] == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        sel = np.where((ids == ann_id) & matched)[0]
        if sel.size == 0:
            return np.zeros((img_h, img_w), dtype=np.float32)
        return masks[sel[0]].astype(np.float32)
    except Exception:
        return np.zeros((img_h, img_w), dtype=np.float32)


def _build_crop(root: str, split: str, file_name: str, bbox: np.ndarray,
                ann_id: int, img_w: int, img_h: int):
    """Build (448,448,4) uint8 crop and its (x0,y0) origin.

    Mirrors v3.py V3Dataset.__getitem__ exactly so crops match the B-loader.
    """
    with Image.open(os.path.join(root, split, file_name)) as pil:
        img = np.asarray(pil.convert("RGB"), dtype=np.uint8)

    mask = _load_mask(root, split, file_name, ann_id, img_w, img_h)

    x0, y0 = crop_origin(bbox, img_w, img_h, _CROP)
    rgb_crop = img[y0:y0 + _CROP, x0:x0 + _CROP]
    mask_crop = mask[y0:y0 + _CROP, x0:x0 + _CROP][..., None]
    img4 = np.concatenate([rgb_crop, mask_crop.astype(np.uint8)], axis=-1)
    return img4, x0, y0  # (448,448,4) uint8, crop offsets


class V3FramesetDataset:
    """Multi-camera frameset dataset for HybridNet 3D training.

    Parameters
    ----------
    root : str
        Root of the V3 dataset (contains annotations/, train/, val/, …).
    split : str
        'train' or 'val'.
    recordings : list[str] | None
        If given, only include framesets whose datasetName is in this list.
    """

    def __init__(self, root: str, split: str, *, recordings=None):
        self.root = root
        self.split = split

        ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        with open(ann_path) as f:
            coco = json.load(f)

        # Expose keypoint names and skeleton edges for graph-Laplacian prior wiring
        self.keypoint_names: list[str] = coco.get("keypoint_names", [])
        self.skeleton: list = coco.get("skeleton", [])

        # Build lookup tables
        self._id2img = {im["id"]: im for im in coco["images"]}
        # Map image_id -> single annotation (one fly per image in this dataset)
        self._id2ann: dict[int, dict] = {}
        for a in coco["annotations"]:
            img_id = a["image_id"]
            # Keep first annotation if multiple (shouldn't happen for single-fly)
            if img_id not in self._id2ann:
                self._id2ann[img_id] = a

        # Build per-datasetName ReprojectionTool
        calib_base = os.path.join(root, "calib_params")
        self._repro_tools: dict[str, ReprojectionTool] = {}
        for dataset_name in coco["calibrations"]:
            calib_dir = os.path.join(calib_base, dataset_name)
            if os.path.isdir(calib_dir):
                self._repro_tools[dataset_name] = ReprojectionTool(calib_dir)

        # Index framesets
        raw_framesets = coco["framesets"]  # key -> {datasetName, frames:[img_ids]}
        self.framesets: list[dict] = []
        # Store per-frameset data (pre-computed during indexing for speed and
        # to support get_joint_2d)
        self._fs_kp2d: list[np.ndarray] = []   # (num_cam, 50, 3) per frameset
        self._fs_anns: list[list[dict]] = []    # list of per-camera ann dicts

        for fs_key, fs_val in raw_framesets.items():
            dataset_name = fs_val["datasetName"]
            frame_ids = fs_val["frames"]

            if recordings is not None and dataset_name not in recordings:
                continue

            if dataset_name not in self._repro_tools:
                continue  # skip if no calibration found

            rt = self._repro_tools[dataset_name]
            num_cameras = rt.num_cameras

            # Collect per-camera annotations
            anns = []
            ok = True
            for img_id in frame_ids[:num_cameras]:
                ann = self._id2ann.get(img_id)
                if ann is None:
                    ok = False
                    break
                anns.append(ann)
            if not ok or len(anns) != num_cameras:
                continue

            # Collect per-camera 2D keypoints (num_cam, 50, 3)
            kp2d = np.zeros((num_cameras, 50, 3), dtype=np.float32)
            for c, ann in enumerate(anns):
                kp2d[c] = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)

            self.framesets.append(fs_val)
            self._fs_kp2d.append(kp2d)
            self._fs_anns.append(anns)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.framesets)

    def get_joint_2d(self, idx: int, joint: int) -> np.ndarray:
        """Return (num_cam, 3) float32 [x, y, v] for frameset idx, joint joint."""
        return self._fs_kp2d[idx][:, joint, :]  # (num_cam, 3)

    def __getitem__(self, idx: int) -> dict:
        fs = self.framesets[idx]
        dataset_name = fs["datasetName"]
        frame_ids = fs["frames"]
        rt = self._repro_tools[dataset_name]
        num_cameras = rt.num_cameras
        anns = self._fs_anns[idx]
        kp2d = self._fs_kp2d[idx]  # (num_cam, 50, 3) float32

        crops4 = np.zeros((num_cameras, _CROP, _CROP, 4), dtype=np.uint8)
        centerHM = np.zeros((num_cameras, 2), dtype=np.float32)

        for c, (img_id, ann) in enumerate(zip(frame_ids[:num_cameras], anns)):
            img_info = self._id2img[img_id]
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]
            bbox = np.asarray(ann["bbox"], dtype=np.float32)  # [x, y, w, h]
            ann_id = ann["id"]

            crop, x0, y0 = _build_crop(
                self.root, self.split, file_name, bbox, ann_id, img_w, img_h)
            crops4[c] = crop

            # centerHM = crop centre in full-image pixel coords
            # Matches dataset3D: center_x = x0 + crop/2, center_y = y0 + crop/2
            cx = float(x0 + _CROP / 2)
            cy = float(y0 + _CROP / 2)
            centerHM[c] = [cx, cy]

        # ------------------------------------------------------------------
        # Triangulate each joint using full-image 2D coords
        # ------------------------------------------------------------------
        num_joints = kp2d.shape[1]  # 50
        kp3d = np.zeros((num_joints, 3), dtype=np.float32)
        vis = np.zeros(num_joints, dtype=bool)

        for j in range(num_joints):
            pts2d = np.zeros((num_cameras, 2), dtype=np.float64)
            cams_to_use = []
            for c in range(num_cameras):
                x, y, v = kp2d[c, j]
                if v > 0:
                    pts2d[c] = [x, y]
                    cams_to_use.append(c)
            if len(cams_to_use) >= 2:
                p3d = rt.reconstruct_point(pts2d, cams_to_use=cams_to_use)
                kp3d[j] = p3d.astype(np.float32)
                vis[j] = True

        # ------------------------------------------------------------------
        # center3D: truncated midrange of visible triangulated keypoints
        # Matches dataset3D.py __getitem__ (grid_spacing=1):
        #   x = [kp3d[j][0] for j in vis if kp3d[j][0] != 0]  (non-zero x)
        #   center3D[d] = int((max_d + min_d) / grid_spacing / 2) * grid_spacing
        # ------------------------------------------------------------------
        vis_kp = kp3d[vis]  # (K, 3) where K = number of visible joints
        if vis_kp.shape[0] == 0:
            center3D = np.zeros(3, dtype=np.float32)
        else:
            # Filter out zero coordinates per axis (matching dataset3D sentinel check)
            center3D_parts = []
            for d in range(3):
                coords = vis_kp[:, d]
                non_zero = coords[coords != 0]
                if non_zero.size == 0:
                    non_zero = coords  # fallback
                mid = (float(np.max(non_zero)) + float(np.min(non_zero))) / float(_GRID_SPACING) / 2.0
                center3D_parts.append(int(mid) * _GRID_SPACING)
            center3D = np.array(center3D_parts, dtype=np.float32)

        return {
            "crops4": crops4,            # (num_cam, 448, 448, 4) uint8
            "centerHM": centerHM,        # (num_cam, 2) float32
            "center3D": center3D,        # (3,) float32
            "cameraMatrices": rt.camera_matrices,  # (num_cam, 4, 3) float32
            "kp3d": kp3d,                # (50, 3) float32
            "vis": vis,                  # (50,) bool
        }


def frameset_batches(ds: V3FramesetDataset, batch_size: int, *,
                     shuffle: bool = True, seed: int = 0,
                     drop_last: bool = True):
    """Iterator that yields batched dicts with a leading B dimension.

    All dict fields are stacked along axis 0 to produce shape (B, ...).
    """
    n = len(ds)
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    stop = (n // batch_size) * batch_size if drop_last else n
    for s in range(0, stop, batch_size):
        sel = idx[s:s + batch_size]
        samples = [ds[int(i)] for i in sel]
        batch = {}
        for key in samples[0]:
            batch[key] = np.stack([s[key] for s in samples], axis=0)
        yield batch
