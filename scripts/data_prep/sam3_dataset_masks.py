#!/usr/bin/env python3
"""Generate SAM3 silhouette masks for a v5-shaped LABEL dataset root.

WHY THIS EXISTS. The 4th input channel of the ViTPose detector is a SAM
silhouette. On red_data_3d_v5 those masks were BORROWED from
red_data_unified_V3's `sam3_masks/`; that root was deleted on 2026-09-02 and
general_model subsets mostly carry no masks, so on a general_model-derived root
only 2 of 17 recordings have any. Measured with the real loader on
red_data_3d_v10: 5.8% of train annotations and 11.8% of val return a non-empty
channel 3. A mask-ON arm trained there would be mask-on in name only, and a
mask-on/mask-off A/B would be comparing mask-off against mask-off-plus-noise.

`scripts/sam3_masks.py` cannot do this: it is a bout/session front-end that
wants videos and a bout summary, and it TRACKS through time. A label root is
loose, non-contiguous frames.

IDENTITY IS NOT MATCHED, IT IS PROMPTED. Segment-everything-then-assign is the
step that produces chimeras on two-fly frames, and there is no temporal context
here to disambiguate. Instead each ANNOTATION is prompted individually with its
own COCO bbox, so the returned mask belongs to that annotation by construction
and a two-fly image simply gets two prompts. Nothing is ever assigned.

OUTPUT contract, matching `jarvis_jax/data/v5_2d._load_mask` exactly:
    <root>/masks/<rec>/<cam>/Frame_<N>.npz
      masks    (K, H, W) bool   -- one per annotation on that image
      ann_ids  (K,)      int64  -- the MERGED annotation id
      matched  (K,)      bool   -- False where SAM3 returned nothing
`_load_mask` tries `src_ann_id` first and falls back to `id`; we write `id`,
and assert no annotation's `src_ann_id` collides with a DIFFERENT annotation's
`id` on the same image (which would silently hand over the wrong fly's mask).

Shard for SLURM with --shard/--num-shards; sharding is by (recording, camera)
so one task owns whole camera directories and never races another on a file.

    python scripts/data_prep/sam3_dataset_masks.py \
        --root <v12 root> --shard $SLURM_ARRAY_TASK_ID --num-shards 16
"""
import argparse
import collections
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

# Pre-load huggingface_hub.file_download BEFORE anything drags in the SAM3
# import chain (jarvis -> timm -> torch). That chain leaves `tqdm` without
# `set_lock`, after which huggingface_hub's LAZY file_download import fails and
# `from huggingface_hub import hf_hub_download` -- which sam3.model_builder
# does at module scope -- raises ImportError, killing every array task at
# import. Loading it here, while tqdm is still intact, caches the module.
# Same guard as scripts/sam3_masks.py and predict/sam3_driver.py; this script
# was written without it and job 39501387 died exactly that way.
import huggingface_hub.file_download  # noqa: F401

import numpy as np


def load_annotations(root):
    """{(rec, cam, frame): [(ann_id, src_ann_id, bbox, w, h, file_name)]}"""
    per_image = collections.defaultdict(list)
    for split in ("train", "val"):
        p = Path(root) / "annotations" / f"instances_{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        id2im = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = id2im[a["image_id"]]
            rec, cam, fn = im["file_name"].split("/")
            frame = os.path.splitext(fn)[0]
            per_image[(rec, cam, frame)].append(
                (int(a["id"]), int(a.get("src_ann_id", a["id"])),
                 list(a["bbox"]), im["width"], im["height"], im["file_name"]))
    return per_image


def assert_no_id_collisions(per_image):
    """`_load_mask` tries src_ann_id BEFORE the merged id. If one annotation's
    src_ann_id equals a DIFFERENT annotation's id on the same image, that lookup
    hits the wrong row and the model trains on the other fly's silhouette --
    silently, and only on two-fly images, which is where it does most damage."""
    bad = []
    for key, anns in per_image.items():
        ids = {a[0] for a in anns}
        for ann_id, src_id, *_ in anns:
            if src_id != ann_id and src_id in ids:
                bad.append((key, ann_id, src_id))
    if bad:
        raise SystemExit(
            f"FATAL: {len(bad)} annotation(s) whose src_ann_id collides with a "
            f"different annotation's id on the same image, e.g. {bad[:3]}. "
            f"Writing masks keyed by merged id would be resolved by the "
            f"src_ann_id branch to the WRONG fly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--text", default="insect")
    ap.add_argument("--sam3-version", default="sam3.1")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--box-pad", type=float, default=0.15,
                    help="fraction of the bbox side to pad the prompt box by")
    ap.add_argument("--limit-groups", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    per_image = load_annotations(root)
    assert_no_id_collisions(per_image)

    groups = collections.defaultdict(list)
    for (rec, cam, frame), anns in per_image.items():
        groups[(rec, cam)].append((frame, anns))
    keys = sorted(groups)
    mine = [k for i, k in enumerate(keys) if i % a.num_shards == a.shard]
    if a.limit_groups:
        mine = mine[:a.limit_groups]
    n_img = sum(len(groups[k]) for k in mine)
    n_ann = sum(len(x[1]) for k in mine for x in groups[k])
    print(f"[shard {a.shard}/{a.num_shards}] {len(mine)} (recording,camera) groups, "
          f"{n_img} images, {n_ann} annotations", flush=True)
    if a.dry_run:
        for k in mine[:10]:
            print("   ", k, len(groups[k]), "images")
        return

    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker
    t0 = time.time()
    tracker = SAM3VideoTracker(gpu_id=a.gpu_id, text_prompt=a.text,
                               sam3_version=a.sam3_version, compile=False)
    print(f"[shard {a.shard}] SAM3VideoTracker loaded in {time.time()-t0:.1f}s", flush=True)
    P = tracker.predictor

    done = empty = skipped = 0
    for gi, (rec, cam) in enumerate(mine):
        items = sorted(groups[(rec, cam)], key=lambda x: x[0])
        out_dir = root / "masks" / rec / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [(f, anns) for f, anns in items
                if a.overwrite or not (out_dir / f"{f}.npz").exists()]
        skipped += len(items) - len(todo)
        if not todo:
            continue

        # SAM3 sessions read a DIRECTORY of frames; give it this camera's
        # frames, in a fixed order, as symlinks (no pixel copies).
        tmp = tempfile.mkdtemp(prefix=f"sam3_ds_{rec}_{cam}_")
        try:
            cam_dir = os.path.join(tmp, "cam")
            os.makedirs(cam_dir)
            for i, (frame, anns) in enumerate(todo):
                src = root / "images" / anns[0][5]
                os.symlink(os.path.realpath(src), os.path.join(cam_dir, f"{i:06d}.jpg"))
            sid = P.handle_request({"type": "start_session", "resource_path": cam_dir,
                                    "offload_video_to_cpu": True})["session_id"]
            for i, (frame, anns) in enumerate(todo):
                W, H = anns[0][3], anns[0][4]
                masks, ids, matched = [], [], []
                for ann_id, _src, bbox, _w, _h, _fn in anns:
                    # One prompt per annotation: the mask is that fly's by
                    # construction, so nothing has to be assigned afterwards.
                    x, y, bw, bh = bbox
                    px, py = a.box_pad * bw, a.box_pad * bh
                    box = [max(0.0, (x - px)) / W, max(0.0, (y - py)) / H,
                           min(1.0, (bw + 2 * px) / W), min(1.0, (bh + 2 * py) / H)]
                    P.handle_request({"type": "reset_session", "session_id": sid})
                    resp = P.handle_request({
                        "type": "add_prompt", "session_id": sid, "frame_index": i,
                        "text": a.text, "bounding_boxes": [box],
                        "bounding_box_labels": [1]})
                    m = None
                    out = resp.get("outputs") if isinstance(resp, dict) else None
                    if out and out.get("out_obj_ids") is not None and len(out["out_obj_ids"]):
                        m = np.asarray(out["out_binary_masks"][0]).astype(bool)
                    if m is None or not m.any():
                        m = np.zeros((H, W), dtype=bool); empty += 1
                        matched.append(False)
                    else:
                        matched.append(True)
                    masks.append(m); ids.append(ann_id)
                np.savez_compressed(out_dir / f"{frame}.npz",
                                    masks=np.stack(masks),
                                    ann_ids=np.asarray(ids, dtype=np.int64),
                                    matched=np.asarray(matched, dtype=bool))
                done += 1
            P.handle_request({"type": "close_session", "session_id": sid})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"[shard {a.shard}] {gi+1}/{len(mine)} {rec}/{cam}: "
              f"{len(todo)} images written ({done} total, {empty} empty masks)",
              flush=True)

    print(f"[shard {a.shard}] DONE  wrote {done} npz, skipped {skipped} existing, "
          f"{empty} annotations got an EMPTY mask", flush=True)


if __name__ == "__main__":
    main()
