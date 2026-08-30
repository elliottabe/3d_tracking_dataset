"""Per-fly frameset dataset for red_data_3d_v5.

Differs from data/v3_3d.py in exactly three ways, all deliberate:
  1. A frameset is keyed by (recording, frame, FLY) rather than (recording,
     frame), so the 264 two-fly framesets contribute both flies instead of the
     first one only.
  2. Media paths are flat -- images/<rec>/<cam>/Frame_*.jpg with no train/ or
     val/ component -- because the split is metadata here.
  3. Calibration is looked up by GROUP, not per-recording, so the three
     distinct rig calibrations are explicit and filterable.

The __getitem__ dict schema is IDENTICAL to V3FramesetDataset so that
scripts/precompute_repro_cache.py and train/train_3d_cached.py consume it
unchanged.

Camera-order note (repo has a history of confident-wrong-3D index bugs):
`frameset["frames"]`/`frameset["ann_ids"]` run in whatever order the SOURCE
per-recording frameset used -- verified on real red_data_3d_v5 data this does
NOT match `ReprojectionTool`'s glob-sorted Cam*.yaml order (e.g. one real
frameset carries frames in Cam853,631,857,630,862,861,855 order while the
calibration's cameras are 630,631,853,855,857,861,862). So every (image_id,
ann_id) pair is placed into its camera's row by NAME (via `rt.cameras`, which
preserves the sorted-glob insertion order that `camera_matrices` rows follow)
-- never by position in `frames`/`ann_ids`.

`ann_ids` may contain `None` (ruling R15: a camera whose per-frame fly count
disagreed with the frameset max cannot be assigned an identity by position,
so it is recorded ABSENT for that fly). `iter_resolved_slots` from
data/build_v5.py is the shared accessor for this -- three consumers already
hand-rolled `is not None` checks and it lives in one place now. A camera with
an unresolved slot contributes a black crop and invisible (v=0) keypoints;
the frameset itself still loads.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.data.build_v5 import iter_resolved_slots
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.geometry.center3d import quantize_center3d
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

_CROP = 448
_GRID_SPACING = 1


def _resolve_sex(ann_sex, fly_id, rec_meta):
    """Resolve one (fly, recording)'s sex via a fallback chain -- duplicated
    verbatim from `data/v5_2d.py::_resolve_sex` (see that module's docstring
    for the measured counts motivating it) rather than imported, mirroring
    this repo's existing precedent of keeping `data/v5_2d.py` and
    `data/v5_3d.py` deliberately un-coupled (see both modules' `_load_mask`):

      1. the annotation's own `sex`, if not "unknown"
      2. else `rec_meta["fly_sex"]["fly<fly_id>"]` -- the per-fly label
         carried by the 4 two-fly recordings, keyed by THIS fly, not the
         recording as a whole (a mixed-sex two-fly recording has no single
         recording-level sex to fall back to)
      3. else `rec_meta["sex"]` -- the recording-level label
      4. else "unknown"

    Never raises: a recording missing from the manifest, or missing both
    `fly_sex` and `sex`, degrades to "unknown" rather than erroring.
    """
    if ann_sex and ann_sex != "unknown":
        return ann_sex
    fly_sex = rec_meta.get("fly_sex")
    if fly_sex:
        per_fly = fly_sex.get(f"fly{fly_id}")
        if per_fly and per_fly != "unknown":
            return per_fly
    rec_sex = rec_meta.get("sex")
    if rec_sex and rec_sex != "unknown":
        return rec_sex
    return "unknown"


def _frameset_own_sex(fsv, id2ann):
    """The frameset's own annotation-level `sex`, from the first camera
    whose slot resolved to a real annotation id (ruling R15 may leave some
    cameras unresolved as None -- see module docstring); "unknown" if every
    camera is unresolved or the resolved annotation lacks the field."""
    for ann_id in fsv.get("ann_ids", []):
        if ann_id is not None:
            return id2ann[ann_id].get("sex", "unknown")
    return "unknown"


def _load_mask(v5_root, file_name, src_ann_id, merged_ann_id, w, h):
    """Mask for ONE annotation.

    Two DIFFERENT keying schemes coexist in the same v5 masks/ tree, both
    verified against real data:
      - The 17 recordings whose masks were borrowed from
        red_data_unified_V3's own sam3_masks/ store `ann_ids` in V3's
        annotation id space -- `src_ann_id` (the merged annotation's
        `src_ann_id` field) matches these directly, PROVIDED the merge's
        annotation source for that recording is also V3 (see
        `discover_sources`'s V3-annotation preference for the 21
        recordings it overlaps).
      - The 9 recordings whose masks were generated fresh against the v5
        tree key `ann_ids` by the merged v5 `id` instead (there is no V3
        copy to borrow an id space from).
    Trying `src_ann_id` FIRST costs nothing (the array is small) and is
    correct for the large majority; falling back to `merged_ann_id` makes a
    future source-preference change unable to silently zero out this
    channel again the way the id-space mismatch previously did (verified:
    ~0.1% hit rate for the 12 recordings general_model sourced but V3
    masked, before that mismatch was fixed upstream).
    """
    rec, cam, fn = file_name.split("/")
    path = os.path.join(v5_root, "masks", rec, cam, fn.replace(".jpg", ".npz"))
    if not os.path.exists(path):
        return np.zeros((h, w), np.uint8)
    with np.load(path) as z:
        ids = z["ann_ids"]
        hit = np.nonzero(ids == src_ann_id)[0]
        if hit.size == 0:
            hit = np.nonzero(ids == merged_ann_id)[0]
        if hit.size == 0 or not z["matched"][hit[0]]:
            return np.zeros((h, w), np.uint8)
        return z["masks"][hit[0]].astype(np.uint8)


class V5FramesetDataset:
    """Per-fly frameset dataset over red_data_3d_v5.

    Parameters
    ----------
    root : str
        Root of the v5 dataset (contains annotations/, images/, masks/,
        calibrations/, manifest.json).
    split : str
        'train' or 'val'.
    recordings : list[str] | None
        If given, only include framesets whose recording is in this list.
    calib_groups : list[str] | None
        If given, only include framesets whose recording's calib_group (per
        manifest.json) is in this list.
    sex : str | None
        If given, only include framesets whose resolved sex (see
        `_resolve_sex`: the frameset's own annotation `sex`, else the
        manifest recording's per-fly `fly_sex["fly<fly_id>"]`, else the
        manifest recording's `sex`, else "unknown") equals this.
    """

    def __init__(self, root: str, split: str, *, recordings=None,
                 calib_groups=None, sex=None):
        self.root = root
        self.split = split
        with open(os.path.join(root, "annotations", f"instances_{split}.json")) as f:
            coco = json.load(f)
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)["recordings"]

        self.keypoint_names = coco.get("keypoint_names", [])
        self.skeleton = coco.get("skeleton", [])
        self._id2img = {i["id"]: i for i in coco["images"]}
        self._id2ann = {a["id"]: a for a in coco["annotations"]}

        self._tools: dict[str, ReprojectionTool] = {}
        self.keys: list[str] = []
        self._fs: list[dict] = []
        self._sex: list[str] = []   # resolved sex, parallel to self._fs/keys
        for key, fsv in sorted(coco["framesets"].items()):
            rec = fsv["recording"]
            meta = self.manifest.get(rec, {})
            grp = meta.get("calib_group")
            if recordings is not None and rec not in recordings:
                continue
            if calib_groups is not None and grp not in calib_groups:
                continue
            resolved_sex = _resolve_sex(
                _frameset_own_sex(fsv, self._id2ann), fsv["fly_id"], meta)
            if sex is not None and resolved_sex != sex:
                continue
            if grp not in self._tools:
                d = os.path.join(root, "calibrations", str(grp))
                if not os.path.isdir(d):
                    continue
                self._tools[grp] = ReprojectionTool(d)
            self.keys.append(key)
            self._fs.append(fsv)
            self._sex.append(resolved_sex)

    def __len__(self) -> int:
        return len(self._fs)

    def is_female(self, idx: int) -> bool:
        # Shares `_resolve_sex` with the `sex=` filter above (not the raw
        # recording-level manifest field) -- a two-fly recording's female
        # fly must be seen as female even when the RECORDING-level sex is
        # "unknown" (see module note / `_resolve_sex`).
        return self._sex[idx] == "female"

    def __getitem__(self, idx: int) -> dict:
        fsv = self._fs[idx]
        rec = fsv["recording"]
        grp = self.manifest[rec]["calib_group"]
        rt = self._tools[grp]
        n_cam = rt.num_cameras
        # rt.cameras preserves the sorted-glob insertion order that
        # camera_matrices rows follow -- the ONLY authority on camera index.
        cam_to_row = {name: i for i, name in enumerate(rt.cameras.keys())}

        n_joints = len(self.keypoint_names) or 50
        crops4 = np.zeros((n_cam, _CROP, _CROP, 4), np.uint8)
        centerHM = np.zeros((n_cam, 2), np.float32)
        kp2d = np.zeros((n_cam, n_joints, 3), np.float32)

        for img_id, ann_id in iter_resolved_slots(fsv):
            # Ruling R15: a camera that could not resolve which fly is which
            # is skipped by iter_resolved_slots already, leaving its row
            # black and its keypoints invisible (v=0) -- triangulation
            # already ignores non-visible views.
            info, ann = self._id2img[img_id], self._id2ann[ann_id]
            cam_name = info["file_name"].split("/")[1]
            c = cam_to_row.get(cam_name)
            if c is None:
                continue  # camera absent from this recording's calib group
            w, h = info["width"], info["height"]
            kps = np.asarray(ann["keypoints"], np.float32)
            if kps.size == n_joints * 3:
                kp2d[c] = kps.reshape(-1, 3)
            # else: this annotation uses a REDUCED keypoint schema (real v5
            # data: "headless" recordings carry 47 -- no head -- and leg-
            # amputation recordings carry 44 -- no T1L -- both verified as
            # ordered NAME subsets of the canonical schema against the
            # source general_model annotation files, but that name mapping
            # lives outside the v5 root and isn't recoverable from this
            # annotation alone). Guessing a positional pad/truncate would
            # silently assign the wrong joint name to the wrong index --
            # exactly the keypoint-order-scrambles-anatomy failure mode
            # this repo has been bitten by before. So this camera's
            # keypoints are left all-invisible (v=0, the existing sentinel)
            # rather than guessed at; its crop/mask/bbox are still loaded
            # normally below since those are valid regardless of schema.
            path = os.path.join(self.root, "images", info["file_name"])
            with Image.open(path) as pil:
                img = np.asarray(pil.convert("RGB"), np.uint8)
            mask = _load_mask(self.root, info["file_name"],
                              ann.get("src_ann_id", ann_id), ann_id, w, h)
            bbox = np.asarray(ann["bbox"], np.float32)
            x0, y0 = crop_origin(bbox, w, h, _CROP)
            crops4[c, ..., :3] = img[y0:y0 + _CROP, x0:x0 + _CROP]
            crops4[c, ..., 3] = mask[y0:y0 + _CROP, x0:x0 + _CROP]
            centerHM[c] = [x0 + _CROP / 2, y0 + _CROP / 2]

        kp3d = np.zeros((n_joints, 3), np.float32)
        vis = np.zeros(n_joints, bool)
        for j in range(n_joints):
            pts = np.zeros((n_cam, 2), np.float64)
            use = []
            for c in range(n_cam):
                x, y, v = kp2d[c, j]
                if v > 0:
                    pts[c] = [x, y]
                    use.append(c)
            if len(use) >= 2:
                kp3d[j] = rt.reconstruct_point(pts, cams_to_use=use).astype(np.float32)
                vis[j] = True

        return {
            "crops4": crops4,
            "centerHM": centerHM,
            "center3D": quantize_center3d(kp3d[vis], _GRID_SPACING),
            "cameraMatrices": rt.camera_matrices,
            "kp3d": kp3d,
            "vis": vis,
            "fly_id": np.int32(fsv["fly_id"]),
            "calib_group": grp,
        }


def frameset_batches(ds: V5FramesetDataset, batch_size: int, *,
                     shuffle: bool = True, seed: int = 0,
                     drop_last: bool = True, female_weight: float = 1.0):
    """Batch iterator. `female_weight` > 1 repeats female framesets, because
    female data is 7% of the set and is the binding constraint on the goal."""
    idx = []
    for i in range(len(ds)):
        reps = int(round(female_weight)) if ds.is_female(i) else 1
        idx.extend([i] * max(reps, 1))
    idx = np.asarray(idx)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    stop = (len(idx) // batch_size) * batch_size if drop_last else len(idx)
    for s in range(0, stop, batch_size):
        samples = [ds[int(i)] for i in idx[s:s + batch_size]]
        yield {k: np.stack([smp[k] for smp in samples], 0)
               for k in samples[0] if k != "calib_group"}
