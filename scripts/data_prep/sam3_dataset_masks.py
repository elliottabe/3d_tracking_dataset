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

USES THE IMAGE PREDICTOR, NOT THE VIDEO TRACKER. The first version of this
script drove SAM3VideoTracker: a video session per camera directory, then
reset_session + add_prompt per annotation. That paid full video-tracker setup
for what is single-image segmentation and measured **4 images/min** -- 9 hours
for one 1,556-image shard, past even the array's 6 h limit. `build_sam3_image_model
(enable_inst_interactivity=True)` + `Sam3Processor` is the right primitive:
`set_image_batch` embeds a batch of images in one pass and `predict_inst_batch`
takes a LIST of Nx4 box arrays, so every annotation on an image is decoded from
one embedding and a two-fly image costs no more embedding than a one-fly image.

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
                 list(a["bbox"]), im["width"], im["height"], im["file_name"],
                 list(a.get("keypoints", []))))
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
    ap.add_argument("--batch", type=int, default=8,
                    help="images embedded per set_image_batch call")
    # DEFAULT 0 = OFF. Points were tried (job 39501907) to fix the 91%
    # over-inclusive rate on 2026_01_13_18_47_45 and REGRESSED HARD: 1523 of
    # 1538 annotations came back EMPTY. The 15 that survived are exactly the
    # 15 two-fly images, so single-object calls are the ones failing -- a
    # collapsed dimension when n_obj == 1, in either the multimask argmax
    # selection or the per-object point block, NOT a modelling failure. The
    # collision guard also caught 1 mis-associated pair among those 15.
    # Do not re-enable without first printing the real shapes that
    # predict_inst_batch returns for n_obj == 1 vs n_obj == 2.
    ap.add_argument("--max-points", type=int, default=0,
                    help="visible GT keypoints as positive point prompts. "
                         "0 = OFF (box only), the only configuration verified "
                         "to produce non-empty masks -- see the comment above.")
    ap.add_argument("--bpe-path", default=None)
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

    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    t0 = time.time()
    model = build_sam3_image_model(bpe_path=a.bpe_path, enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    print(f"[shard {a.shard}] SAM3 image model loaded in {time.time()-t0:.1f}s", flush=True)

    def xyxy(bbox, W, H):
        """COCO xywh -> padded XYXY, clipped to the image."""
        x, y, bw, bh = bbox
        px, py = a.box_pad * bw, a.box_pad * bh
        return [max(0.0, x - px), max(0.0, y - py),
                min(float(W), x + bw + px), min(float(H), y + bh + py)]

    done = empty = skipped = 0
    # Two-fly images are the only place points could go wrong SILENTLY: if a
    # per-object point block were mis-associated, both flies' prompts would
    # land on one animal and the two masks would coincide. Box-only prompting
    # was verified visually to keep them apart; this makes the property a
    # counted assertion rather than a spot check.
    twofly = collided = 0
    t_start = time.time()
    for gi, (rec, cam) in enumerate(mine):
        items = sorted(groups[(rec, cam)], key=lambda x: x[0])
        out_dir = root / "masks" / rec / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        # NEVER write through a symlink: masks inherited from general_model are
        # symlinks into that PROTECTED tree, and np.savez would follow them.
        todo = [(f, anns) for f, anns in items
                if not (out_dir / f"{f}.npz").exists()
                or (a.overwrite and not (out_dir / f"{f}.npz").is_symlink())]
        skipped += len(items) - len(todo)
        if not todo:
            continue

        for s0 in range(0, len(todo), a.batch):
            chunk = todo[s0:s0 + a.batch]
            imgs, boxes_batch, pts_batch, lbl_batch = [], [], [], []
            for frame, anns in chunk:
                W, H = anns[0][3], anns[0][4]
                imgs.append(Image.open(root / "images" / anns[0][5]).convert("RGB"))
                boxes_batch.append(np.asarray(
                    [xyxy(bb, W, H) for _i, _s, bb, _w, _h, _f, _k in anns],
                    dtype=np.float32))
                if a.max_points > 0:
                    pts, lbls = [], []
                    for *_rest, kps in anns:
                        k = np.asarray(kps, dtype=np.float32).reshape(-1, 3)
                        vis = k[k[:, 2] > 0][:, :2]
                        if len(vis) == 0:                 # keep row counts aligned
                            vis = np.zeros((1, 2), np.float32)
                            lab = np.zeros((1,), np.int32)      # 0 = ignore/background
                        else:
                            if len(vis) > a.max_points:   # even subsample, keeps
                                idx = np.linspace(0, len(vis) - 1,  # body+wing+leg spread
                                                  a.max_points).astype(int)
                                vis = vis[idx]
                            lab = np.ones((len(vis),), np.int32)
                        pts.append(vis); lbls.append(lab)
                    # pad ragged per-object point counts to a common P
                    P = max(len(x) for x in pts)
                    pp = np.zeros((len(pts), P, 2), np.float32)
                    ll = np.zeros((len(pts), P), np.int32)
                    for j, (v, l) in enumerate(zip(pts, lbls)):
                        pp[j, :len(v)] = v; ll[j, :len(l)] = l
                    pts_batch.append(pp); lbl_batch.append(ll)
                else:
                    pts_batch.append(None); lbl_batch.append(None)
            if a.max_points <= 0:
                pts_batch = lbl_batch = None
            state = processor.set_image_batch(imgs)
            # POINTS + BOX, not box alone. A bare box left the mask free to
            # settle on the box itself wherever the animal is small and
            # low-contrast against a bright floor: measured 91% over-inclusive
            # on 2026_01_13_18_47_45 (441/487), with the worst masks holding
            # only 10% of their own keypoints. The GT keypoints sit ON the
            # animal and are the strongest disambiguator available; they cost
            # nothing extra because the embedding is already computed.
            # multimask_output=True + argmax(score) is the standard remedy for
            # an ambiguous prompt -- False takes SAM's single guess.
            if a.max_points > 0:
                masks_batch, scores_batch, _ = model.predict_inst_batch(
                    state, pts_batch, lbl_batch, box_batch=boxes_batch,
                    multimask_output=True)
                # (n_obj, n_mask, H, W) + (n_obj, n_mask) -> best per object.
                # UNVERIFIED for n_obj == 1; see --max-points.
                masks_batch = [np.asarray(m)[range(len(m)), np.argmax(np.asarray(sc), axis=-1)]
                               for m, sc in zip(masks_batch, scores_batch)]
            else:
                # The verified path: box only, single mask. 1553 masks, 0 empty.
                masks_batch, _scores, _ = model.predict_inst_batch(
                    state, None, None, box_batch=boxes_batch,
                    multimask_output=False)

            for (frame, anns), mset in zip(chunk, masks_batch):
                W, H = anns[0][3], anns[0][4]
                masks, ids, matched = [], [], []
                for k, (ann_id, _src, _bb, _w, _h, _fn, _kp) in enumerate(anns):
                    m = np.asarray(mset[k])
                    m = m.squeeze()               # (1,H,W) or (H,W) -> (H,W)
                    m = m > 0.0 if m.dtype != bool else m
                    if m.shape != (H, W) or not m.any():
                        m = np.zeros((H, W), dtype=bool)
                        matched.append(False); empty += 1
                    else:
                        matched.append(True)
                    masks.append(m.astype(bool)); ids.append(ann_id)
                if len(masks) > 1:
                    twofly += 1
                    for i0 in range(len(masks)):
                        for i1 in range(i0 + 1, len(masks)):
                            inter = np.logical_and(masks[i0], masks[i1]).sum()
                            union = np.logical_or(masks[i0], masks[i1]).sum()
                            if union and inter / union > 0.5:
                                collided += 1
                np.savez_compressed(out_dir / f"{frame}.npz",
                                    masks=np.stack(masks),
                                    ann_ids=np.asarray(ids, dtype=np.int64),
                                    matched=np.asarray(matched, dtype=bool))
                done += 1
        rate = done / max(1e-6, (time.time() - t_start)) * 60.0
        print(f"[shard {a.shard}] {gi+1}/{len(mine)} {rec}/{cam}: "
              f"{len(todo)} written ({done} total, {empty} empty, {rate:.0f} img/min)",
              flush=True)

    print(f"[shard {a.shard}] DONE  wrote {done} npz, skipped {skipped} existing, "
          f"{empty} annotations got an EMPTY mask", flush=True)
    print(f"[shard {a.shard}] two-fly images {twofly}; mask pairs with IoU>0.5 "
          f"(BOTH ON ONE FLY -- should be 0): {collided}", flush=True)


if __name__ == "__main__":
    main()
