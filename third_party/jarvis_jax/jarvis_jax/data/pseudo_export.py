"""Write gated P3b pseudo-labels as a v12-format export (spec 2026-09-05 §3.2).

The whole point is that NOTHING downstream needs to learn a new format:
`data/v12_windows.V12WindowDataset` opens this root unchanged, so the sampler,
the cohort code and Plan B's T=2 loader all apply as they stand. What the
export adds is per-frameset provenance -- `source: "pseudo"`, `checkpoint`,
the `gates` quantities that admitted the frame, `weight`, `role`, `stratum`
and the `partners` map -- which the loader ignores and Plan B reads.

Three things are load-bearing and each has its own test:

* **Keypoints are written BY NAME, in EXPORT order.** The campaign npz
  keypoint axis is `mvq_meta.json`'s `keypoint_names_written` (which starts
  `Scutellum, WingL_base, WingR_base, ...`); the export's axis is
  `annotations/keypoint_names.json` (which starts `Antenna_Base, EyeL, ...`).
  They are DIFFERENT permutations of the same 50 landmarks. `to_export_order`
  is the only way arrays cross that boundary here, and it refuses an
  ambiguous keypoint axis rather than guessing one (CLAUDE.md's
  keypoint-order trap: a by-index write is self-consistent and completely
  wrong, and every jitter/confidence/residual metric rates it GOOD).

* **The 2D labels are the REPROJECTION of the gated 3D, not the model's own
  2D head.** `V12WindowDataset` never reads a 3D field: it triangulates the
  written 2D (`_dlt`). Writing the head's pixels would hand the loader a 3D
  point up to `reproj_px` (3 px) away from the pose that actually passed the
  gates, and the pseudo-label the model trains on would not be the one that
  was admitted. Reprojecting makes the loader's DLT reproduce the gated 3D
  exactly.

* **Cameras are placed BY NAME.** `file_name` is `<rec>/<cam>/Frame_<n>.jpg`
  and the loader resolves a camera row by splitting that name, so the camera
  axis of `kp2d`/`vis` (canonical `cfg.recording.cameras` == the calibration
  sorted-glob order, what `load_bout_arrays` verifies) is written out under
  the matching camera NAME rather than a row index.

One record is one (recording, frame, host fly): the arrays carry every fly of
the frame (axis F) so a companion fly's pixels come from the same decode, but
a FRAMESET -- and therefore a training window -- is written only for a fly
that is some record's `host_fly`. That keeps the stratified sample the writer
is handed and the window count the loader builds exactly equal; a fly that
was admitted but not sampled stays a present-but-unlabelled animal, which
`V12WindowDataset.unlabelled_sex` already handles from `n_flies`.
"""
from __future__ import annotations

import collections
import dataclasses
import glob
import json
import os

import numpy as np
from PIL import Image

SEX_NAME = {0: "female", 1: "male", -1: "unknown"}
JPEG_QUALITY = 95


def _write_jpeg(path, rgb):
    """Write an (H,W,3) RGB array as JPEG at `JPEG_QUALITY`.

    cv2 is ~4x faster than PIL here (2.2 ms vs 8.0 ms on a 1936x448 frame,
    measured 2026-09-05) and this runs ~200k times, but it writes BGR -- hence
    the reversed channel view. `test_written_jpeg_round_trips_the_rgb_channels`
    guards the flip: a swapped export would train the model on blue flies and
    nothing else in the pipeline would notice.
    """
    try:
        import cv2
        if not cv2.imwrite(path, np.ascontiguousarray(rgb[:, :, ::-1]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
            raise RuntimeError(f"cv2.imwrite failed for {path}")
    except ImportError:
        Image.fromarray(rgb).save(path, quality=JPEG_QUALITY)


@dataclasses.dataclass
class PseudoRecord:
    """One admitted (recording, frame, host fly).

    kp3d (F,K,3), kp2d (F,C,K,2), vis (F,C,K) and sex (F,) cover EVERY fly of
    the frame in campaign (fly0, fly1) order and in the campaign's own
    keypoint order; `host_fly` says which of them this record is the frameset
    for. A fly is "labelled" here iff it has at least one visible keypoint and
    a finite 3D pose -- a fly the gates rejected is passed in with `vis` all
    False rather than as a separate absent-fly convention.

    `partners` maps delta -> the ABSOLUTE video frame of the T=2 partner, and
    is empty when no partner of that delta passed the gates (spec §3.2: the
    pair is dropped, the single frame kept). `role` is "anchor", "partner" or
    "negative"; a negative has `host_fly=None` and a `center3D`.
    """
    recording: str
    frame: int
    host_fly: int | None
    kp3d: np.ndarray
    kp2d: np.ndarray
    vis: np.ndarray
    sex: np.ndarray
    stratum: dict
    gates: dict
    partners: dict
    role: str
    center3D: np.ndarray | None = None
    bout: int | None = None


def _keypoint_axis(shape, K):
    """The axis of `shape` that is the keypoint axis, or a ValueError naming
    the shape. `(C,K,2)`/`(K,3)` pixel and world arrays are resolved by their
    trailing coordinate axis; anything else must have exactly one axis of
    length K. Never falls back to a positional guess -- picking the wrong axis
    here is the failure mode this module exists to prevent."""
    if len(shape) >= 2 and shape[-1] in (2, 3) and shape[-2] == K:
        return len(shape) - 2
    cand = [i for i, n in enumerate(shape) if n == K]
    if len(cand) != 1:
        raise ValueError(
            f"cannot identify the keypoint axis of an array with shape {tuple(shape)} "
            f"for K={K}: {len(cand)} candidate axes {cand}. Pass axis= explicitly -- "
            f"guessing one silently permutes cameras or flies instead of keypoints.")
    return cand[0]


def to_export_order(arr, written_names, export_names, *, axis=None):
    """`arr[..., k, ...]` in `written_names` order -> `export_names` order, BY NAME."""
    from jarvis_jax.tracking.predict_2d import detector_to_model_perm
    perm = detector_to_model_perm(list(written_names), list(export_names))
    arr = np.asarray(arr)
    ax = _keypoint_axis(arr.shape, len(perm)) if axis is None else int(axis)
    return np.take(arr, perm, axis=ax)


def reprojected_labels(kp3d, cam_mats, vis, hw):
    """(C,K,2) label pixels = the PROJECTION of the pseudo 3D (see the module
    docstring) and (C,K) visibility = vis & finite & inside the frame."""
    from jarvis_jax.tracking.lift_mvq import project_points
    uv = project_points(cam_mats, kp3d)
    H, W = hw
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) \
            & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
    return np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), \
        (np.asarray(vis, bool) & inside)


def bbox_from_visible(uv, vis):
    """COCO [x, y, w, h] over the visible pixels; [0,0,0,0] when none are."""
    if not np.any(vis):
        return [0.0, 0.0, 0.0, 0.0]
    p = np.asarray(uv, np.float64)[np.asarray(vis, bool)]
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def _empty_coco(export_names, skeleton=None):
    return {"keypoint_names": list(export_names), "skeleton": list(skeleton or []),
            "categories": [{"id": 1, "name": "fly", "num_keypoints": len(export_names)}],
            "images": [], "annotations": [], "framesets": {}}


def _link_calibration(out_root, group, calib_dir):
    """`calibrations/<group>/Cam*.yaml` as symlinks to the recording's own
    calibration. One group per recording (each has its own calibration), so
    `ReprojectionTool` sees exactly the cameras `cam_mats` was built from."""
    dst = os.path.join(out_root, "calibrations", str(group))
    os.makedirs(dst, exist_ok=True)
    src = sorted(glob.glob(os.path.join(str(calib_dir), "Cam*.yaml")))
    if not src:
        raise FileNotFoundError(f"no Cam*.yaml in {calib_dir} for calibration group {group!r}")
    for p in src:
        link = os.path.join(dst, os.path.basename(p))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(p), link)
    return [os.path.splitext(os.path.basename(p))[0] for p in src]


def write_pseudo_export(out_root, records, *, export_names, cameras, recordings,
                        checkpoint, gates, weight=0.3, frame_reader, mask_reader=None,
                        split="train", subset="pseudo_p3b", version=None,
                        skeleton=None, write_images=True, extra_manifest=None,
                        progress=True, min_cams=2):
    """Write `records` as a v12 root at `out_root`; returns a summary dict.

    Args:
        records: `PseudoRecord`s, in any order (grouped and sorted here so each
            video frame is decoded exactly once).
        export_names: the export's keypoint axis (`annotations/keypoint_names.json`).
            The records' own axis is `recordings[rec]["kp_names"]`, defaulting
            to `export_names` when the caller has already reordered.
        cameras: canonical camera NAMES, matching the camera axis of `kp2d`/`vis`.
        recordings: `{rec: {"calib_dir": ..., "calib_group": rec, "fly_sex": {...},
            "cam_mats": (C,4,3), "kp_names": [...], "behavior": ..., "sex": ...}}`.
            `cam_mats` defaults to the linked calibration's own matrices.
        frame_reader: `(rec, frame) -> (C,H,W,3) uint8` (or `(frames, present)`),
            camera axis in `cameras` order.
        mask_reader: optional `(rec, frame) -> (F,C,H,W) bool`, the SAM3 masks
            of every fly, written as `masks/<rec>/<cam>/Frame_<n>.npz` keyed by
            `ann_ids` exactly as `v5_3d._load_mask` reads them.
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

    out_root = str(out_root)
    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)
    export_names = list(export_names)
    cameras = [str(c) for c in cameras]
    recs = {str(k): dict(v) for k, v in recordings.items()}

    # ---- calibration: one group per recording, linked and read back by name
    cam_mats, groups = {}, {}
    for rec, meta in recs.items():
        grp = str(meta.get("calib_group") or rec)
        names = _link_calibration(out_root, grp, meta["calib_dir"])
        if names != cameras:
            raise ValueError(
                f"{rec}: calibration group {grp!r} has cameras {names} but the record "
                f"arrays' camera axis is {cameras} -- they must agree BY NAME.")
        groups[rec] = grp
        if meta.get("cam_mats") is None:
            # float64 from the cameras' own (3,4) matrices, NOT the float32
            # `camera_matrices`: the loader triangulates these labels back with
            # `ReprojectionTool`'s float64 stack, and matching it keeps the
            # 2D -> 3D round trip exact instead of ~1e-4 px off.
            rt = ReprojectionTool(os.path.join(out_root, "calibrations", grp))
            meta["cam_mats"] = np.stack([c.cameraMatrix.T for c in rt.cameras.values()])
        cam_mats[rec] = np.asarray(meta["cam_mats"], np.float64)

    coco = _empty_coco(export_names, skeleton)
    by_frame = collections.defaultdict(list)
    for r in records:
        by_frame[(r.recording, int(r.frame))].append(r)

    n_img = n_ann = 0
    per_rec = collections.Counter()
    per_role = collections.Counter()
    per_stratum = collections.Counter()
    per_bout = collections.Counter()
    partner_avail = collections.Counter()
    n_masks = n_dropped_cams = 0
    last_rec = None
    for (rec, frame) in sorted(by_frame):
        if progress and rec != last_rec:
            print(f"[pseudo_export] writing {rec} ...", flush=True)
            last_rec = rec
        group = by_frame[(rec, frame)]
        kp_names = list(recs[rec].get("kp_names") or export_names)
        fly_sex = recs[rec].get("fly_sex") or {}

        got = frame_reader(rec, frame)
        frames, present = got if isinstance(got, tuple) else (got, None)
        frames = np.asarray(frames)
        if frames.shape[0] != len(cameras):
            raise ValueError(f"{rec} Frame_{frame}: frame_reader returned {frames.shape[0]} "
                             f"camera rows for {len(cameras)} cameras {cameras}")
        present = np.ones(len(cameras), bool) if present is None else np.asarray(present, bool)
        H, W = int(frames.shape[1]), int(frames.shape[2])
        masks = mask_reader(rec, frame) if mask_reader is not None else None
        if masks is not None and tuple(np.shape(masks)[-2:]) != (H, W):
            # The loader slices the mask with the IMAGE's crop origin
            # (`v12_windows._build`), so a mask of a different size is not a
            # smaller mask -- it is a mask of somewhere else.
            raise ValueError(
                f"{rec} Frame_{frame}: mask_reader returned {np.shape(masks)[-2:]} masks for "
                f"a {(H, W)} frame -- they must be the same size, by camera and by pixel.")

        # a frameset (== a training window) only for flies that are a record's host
        hosts = {}
        for r in group:
            hosts.setdefault(r.host_fly, r)

        # Labels FIRST, pixels second: a camera no frameset resolves is never
        # opened by the loader (`_build` only decodes resolved slots), so its
        # JPEG would be dead weight in a ~150k-file tree.
        labels, used = {}, np.zeros(len(cameras), bool)
        for fly in sorted(hosts, key=lambda f: (f is None, f)):
            r = hosts[fly]
            if fly is None or r.role == "negative":
                used[:] = True
                continue
            X = to_export_order(np.asarray(r.kp3d, np.float64)[fly], kp_names, export_names)
            V = to_export_order(np.asarray(r.vis, bool)[fly], kp_names, export_names)
            uv, vis = reprojected_labels(X, cam_mats[rec], V, (H, W))
            vis = vis & present[:, None]
            if int((vis.any(-1)).sum()) < min_cams:
                # The loader triangulates the written 2D; a fly seen in fewer
                # than two cameras has no 3D at all and its window would be
                # centred on the origin. Dropped and counted, never written as
                # a silently-degenerate frameset.
                n_dropped_cams += 1
                continue
            labels[fly] = (uv, vis)
            used |= vis.any(-1)

        img_ids = []
        for c, cam in enumerate(cameras):
            n_img += 1
            fn = f"{rec}/{cam}/Frame_{frame}.jpg"
            coco["images"].append({"id": n_img, "width": W, "height": H,
                                   "recording": rec, "file_name": fn})
            img_ids.append(n_img)
            if write_images and present[c] and used[c]:
                p = os.path.join(out_root, "images", rec, cam)
                os.makedirs(p, exist_ok=True)
                _write_jpeg(os.path.join(p, f"Frame_{frame}.jpg"), frames[c])

        mask_rows = collections.defaultdict(list)
        for fly in sorted(hosts, key=lambda f: (f is None, f)):
            r = hosts[fly]
            negative = fly is None or r.role == "negative"
            if not negative and fly not in labels:
                continue
            if negative:
                ann_ids, num_kp = [], 0
                for c in range(len(cameras)):
                    n_ann += 1
                    coco["annotations"].append({
                        "id": n_ann, "image_id": img_ids[c],
                        "keypoints": [0.0] * (3 * len(export_names)), "num_keypoints": 0,
                        "bbox": [0.0, 0.0, 0.0, 0.0], "sex": "unknown", "fly_id": -1,
                        "subset": subset, "src_ann_id": n_ann, "negative": True})
                    ann_ids.append(n_ann)
                key = f"{rec}/Frame_{frame}/neg0"
                fsv = {"recording": rec, "fly_id": -1, "subset": subset,
                       "frames": img_ids, "ann_ids": ann_ids, "negative": True,
                       "center3D": [float(v) for v in np.asarray(r.center3D, np.float64)]
                       if r.center3D is not None else None}
            else:
                uv, vis = labels[fly]
                sex = SEX_NAME.get(int(np.asarray(r.sex).reshape(-1)[fly]), "unknown")
                if sex == "unknown":
                    sex = fly_sex.get(f"fly{fly}", "unknown")
                ann_ids = []
                for c in range(len(cameras)):
                    if not present[c] or not vis[c].any():
                        ann_ids.append(None)
                        continue
                    n_ann += 1
                    kp = np.concatenate([uv[c], vis[c][:, None].astype(np.float32)], -1)
                    coco["annotations"].append({
                        "id": n_ann, "image_id": img_ids[c],
                        "keypoints": [round(float(v), 3) for v in kp.reshape(-1)],
                        "num_keypoints": int(vis[c].sum()),
                        "bbox": [round(v, 3) for v in bbox_from_visible(uv[c], vis[c])],
                        "sex": sex, "sex_source": recs[rec].get("sex_source", "mvq_p3b_mask_identity"),
                        "behavior": recs[rec].get("behavior", "courtship"),
                        "fly_id": int(fly), "subset": subset, "src_ann_id": n_ann})
                    ann_ids.append(n_ann)
                    if masks is not None:
                        mask_rows[c].append((n_ann, np.asarray(masks[fly, c], bool)))
                key = f"{rec}/Frame_{frame}/fly{int(fly)}"
                fsv = {"recording": rec, "fly_id": int(fly), "subset": subset,
                       "frames": img_ids, "ann_ids": ann_ids}
            fsv.update({"source": "pseudo", "weight": float(weight), "role": str(r.role),
                        "bout": None if r.bout is None else int(r.bout),
                        "stratum": dict(r.stratum or {}),
                        "partners": {str(int(d)): int(f) for d, f in (r.partners or {}).items()},
                        "gates": {k: (None if v is None else round(float(v), 4))
                                  for k, v in (r.gates or {}).items()}})
            coco["framesets"][key] = fsv
            per_rec[rec] += 1
            per_role[str(r.role)] += 1
            per_bout[(rec, r.bout)] += 1
            st = fsv["stratum"]
            per_stratum[(st.get("host_sex", "unknown"),
                         "contact" if st.get("contact") else
                         ("apart" if st.get("apart") else "mid"))] += 1
            for d in fsv["partners"]:
                partner_avail[d] += 1

        if masks is not None:
            for c, rows in mask_rows.items():
                p = os.path.join(out_root, "masks", rec, cameras[c])
                os.makedirs(p, exist_ok=True)
                np.savez_compressed(
                    os.path.join(p, f"Frame_{frame}.npz"),
                    masks=np.stack([m for _, m in rows]),
                    ann_ids=np.asarray([a for a, _ in rows], np.int64),
                    matched=np.ones(len(rows), bool))
                n_masks += len(rows)

    # ---- annotations, manifest
    with open(os.path.join(out_root, "annotations", f"instances_{split}.json"), "w") as f:
        json.dump(coco, f)
    other = "val" if split == "train" else "train"
    with open(os.path.join(out_root, "annotations", f"instances_{other}.json"), "w") as f:
        json.dump(_empty_coco(export_names, skeleton), f)
    with open(os.path.join(out_root, "annotations", "keypoint_names.json"), "w") as f:
        json.dump(export_names, f)

    gate_block = dataclasses.asdict(gates) if dataclasses.is_dataclass(gates) else dict(gates)
    gate_block = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gate_block.items()}
    manifest = {
        "version": version or os.path.basename(out_root.rstrip("/")),
        "source": "pseudo", "checkpoint": str(checkpoint), "gates": gate_block,
        "weight": float(weight),
        "calib_groups": sorted(set(groups.values())),
        "recordings": {rec: {
            "calib_group": groups[rec], "n_framesets": int(per_rec[rec]),
            "n_flies": int(recs[rec].get("n_flies", 2)),
            "behavior": recs[rec].get("behavior", "courtship"),
            "sex": recs[rec].get("sex", "mixed"),
            "sex_source": recs[rec].get("sex_source", "mvq_p3b_mask_identity"),
            "fly_sex": recs[rec].get("fly_sex") or {},
            "has_masks": bool(mask_reader is not None),
            "split": split, "source": "pseudo", "weight": float(weight),
            "subset": subset} for rec in sorted(recs)},
    }
    manifest.update(extra_manifest or {})
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    return {"n_framesets": len(coco["framesets"]), "n_images": n_img,
            "n_annotations": n_ann, "n_mask_rows": n_masks,
            "n_dropped_too_few_cameras": n_dropped_cams,
            "per_recording": dict(per_rec), "per_role": dict(per_role),
            "per_stratum": {f"{a}/{b}": n for (a, b), n in sorted(per_stratum.items())},
            "per_bout": {f"{r}/bout_{b}": n for (r, b), n in sorted(
                per_bout.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1))},
            "partner_availability": dict(partner_avail)}
