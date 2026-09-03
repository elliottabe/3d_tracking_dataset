#!/usr/bin/env python3
"""Probe: SAM3 TEXT prompt ("insect") vs the box prompt, on the same images.

WHY TRY IT. The box prompt works (0 empty, 0 over-inclusive, worst kp-in-mask
75%), so this is not a rescue -- it is a cross-check and a capability probe.
Text grounding is what the BOUT pipeline already uses, it needs no GT box, and
if it matches or beats the box prompt it would also work on unlabelled frames.

HOW IDENTITY IS RESOLVED. A text prompt returns EVERY insect it finds, with no
notion of which annotation each belongs to. Rather than guess, each detection
is assigned to the annotation whose visible GT keypoints most fall inside it,
and an annotation whose best detection is below --min-inside is recorded
matched=False. That is the R15 rule: record ABSENT rather than attribute
wrongly. A detection matched by nothing (shadow, debris, the other fly) is
dropped.

WRITES TO ITS OWN DIRECTORY (--masks-dir, default masks_text/). The box masks
under masks/ are the working set and are never touched -- an unproven change
overwrote them once already.

    python scripts/data_prep/sam3_text_prompt_probe.py --root <v12 root> \
        --recording 2026_01_13_18_47_45 --limit 20
"""
import argparse, collections, json, os
from pathlib import Path

import huggingface_hub.file_download  # noqa: F401  (see sam3_dataset_masks.py)
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--recording", default=None, help="limit to one recording")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--text", default="insect")
    ap.add_argument("--masks-dir", default="masks_text")
    ap.add_argument("--min-inside", type=float, default=0.5,
                    help="min share of an annotation's visible keypoints inside a "
                         "detection for it to be attributed to that annotation")
    ap.add_argument("--conf", type=float, default=None,
                    help="SAM3 confidence threshold (default: model's own)")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    root = Path(a.root)
    per = collections.defaultdict(list)
    for split in ("train", "val"):
        p = root / "annotations" / f"instances_{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        id2im = {im["id"]: im for im in d["images"]}
        for an in d["annotations"]:
            im = id2im[an["image_id"]]
            rec, cam, fn = im["file_name"].split("/")
            if a.recording and rec != a.recording:
                continue
            if a.camera and cam != a.camera:
                continue
            per[(rec, cam, os.path.splitext(fn)[0])].append((an, im))
    keys = sorted(per)[:a.limit]
    if not keys:
        raise SystemExit("no images matched")
    print(f"probing {len(keys)} images from "
          f"{sorted({k[0] for k in keys})}", flush=True)

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    # SAM3's own examples run the whole notebook under bf16 autocast, and the
    # video tracker builds its own bf16_context. Sam3Processor does NOT, so the
    # text-grounding forward hits "mat1 and mat2 must have the same dtype, but
    # got BFloat16 and Float" in the ViTDet MLP. (The box path via
    # predict_inst_batch does not need this -- the interactive predictor
    # handles its own casting.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    model = build_sam3_image_model(enable_inst_interactivity=False)
    processor = Sam3Processor(model)
    if a.conf is not None:
        processor.set_confidence_threshold(a.conf)

    rows = []
    for (rec, cam, frame) in keys:
        anns = per[(rec, cam, frame)]
        img = Image.open(root / "images" / anns[0][1]["file_name"]).convert("RGB")
        W, H = img.size
        with amp:
            state = processor.set_image(img)
            state = processor.set_text_prompt(a.text, state)
        masks = state.get("masks")
        # `state["masks"]` is a CUDA tensor (and bf16 under autocast); np.asarray
        # on it raises. Detach -> float32 -> cpu -> numpy, in that order.
        def _np(m):
            if hasattr(m, "detach"):
                m = m.detach().float().cpu()
            return np.asarray(m).squeeze()
        masks = [] if masks is None or len(masks) == 0 else [
            (_np(m) > 0.0) if _np(m).dtype != bool else _np(m) for m in masks]
        out = []
        for an, im in anns:
            kp = np.asarray(an["keypoints"], float).reshape(-1, 3)
            v = kp[:, 2] > 0
            best, best_i = 0.0, -1
            for mi, m in enumerate(masks):
                if m.shape != (H, W):
                    continue
                xi = np.clip(kp[v, 0].astype(int), 0, W - 1)
                yi = np.clip(kp[v, 1].astype(int), 0, H - 1)
                frac = float(m[yi, xi].mean()) if v.any() else 0.0
                if frac > best:
                    best, best_i = frac, mi
            ok = best >= a.min_inside
            out.append(dict(ann_id=int(an["id"]), inside=best, matched=bool(ok),
                            fg=float(masks[best_i].mean()) if ok else 0.0,
                            mask_idx=best_i if ok else -1))
        # save, in its OWN directory
        md = root / a.masks_dir / rec / cam
        md.mkdir(parents=True, exist_ok=True)
        arr, ids, matched = [], [], []
        for o in out:
            arr.append(masks[o["mask_idx"]] if o["matched"]
                       else np.zeros((H, W), bool))
            ids.append(o["ann_id"]); matched.append(o["matched"])
        np.savez_compressed(md / f"{frame}.npz", masks=np.stack(arr),
                            ann_ids=np.asarray(ids, np.int64),
                            matched=np.asarray(matched, bool))
        rows.append(dict(rec=rec, cam=cam, frame=frame, n_det=len(masks),
                         n_ann=len(anns), res=out))
        print(f"  {rec}/{cam}/{frame}: {len(masks)} detections, {len(anns)} annotations, "
              f"matched {sum(o['matched'] for o in out)}/{len(out)}, "
              f"inside {[round(o['inside'],2) for o in out]}", flush=True)

    n_ann = sum(r["n_ann"] for r in rows)
    n_ok = sum(o["matched"] for r in rows for o in r["res"])
    dets = [r["n_det"] for r in rows]
    ins = [o["inside"] for r in rows for o in r["res"]]
    fgs = [o["fg"] for r in rows for o in r["res"] if o["matched"]]
    print(f"\n=== TEXT PROMPT '{a.text}' ===")
    print(f"images {len(rows)}  annotations {n_ann}  attributed {n_ok} "
          f"({100*n_ok/max(1,n_ann):.0f}%)")
    print(f"detections per image: min {min(dets)} median {int(np.median(dets))} max {max(dets)}")
    print(f"kp-in-mask: mean {np.mean(ins):.3f} min {np.min(ins):.3f}")
    if fgs:
        print(f"fg fraction (attributed): p50 {np.median(fgs)*100:.2f}%  max {np.max(fgs)*100:.2f}%")
    if a.out_json:
        json.dump(rows, open(a.out_json, "w"), indent=2, default=float)
        print("wrote", a.out_json)


if __name__ == "__main__":
    main()
