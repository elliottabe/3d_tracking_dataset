"""Per-recording contact sheets for the manual sexing pass.

WHY THIS EXISTS: 54% of red_data framesets carry sex: "unknown", including the
three big Group-A courtship recordings (677 framesets in bout 28's own
calibration). Sex is a per-RECORDING property, so ~30 judgements settle it --
far cheaper than a GUI.

WHAT TO LOOK FOR: the male is smaller with a darker, blunter abdomen tip and
carries sex combs on the T1 tarsi; the female is larger with a pointed
ovipositor. NOTE the sexing gotcha recorded in ab-hybridnet-vs-dlt-ik: mask
AREA is backwards as a size cue during courtship, because the male extends a
wing during song and so has the LARGER silhouette. Judge by body shape, not
extent.

EXPECTATION for a correct sheet: every crop shows one fly filling most of the
frame with keypoints landing on eyes, thorax, abdomen and legs. Keypoints
scattered into empty space mean the annotation-to-image mapping is wrong and
the sheet must not be used for sexing.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from viz.core.colors import PALETTE, keypoint_groups


def _crop_box(bbox, w, h, pad=1.4, size=256):
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    half = max(bw, bh) * pad / 2
    x0 = int(np.clip(cx - half, 0, max(w - 1, 0)))
    y0 = int(np.clip(cy - half, 0, max(h - 1, 0)))
    x1 = int(np.clip(cx + half, 1, w))
    y1 = int(np.clip(cy + half, 1, h))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def build_contact_sheet(v5_root: str, recording: str, *, n_frames: int = 6,
                        cams=("Cam2012630", "Cam2012855"),
                        out_dir: str, cell: int = 256, fly_id: int = 0) -> str:
    with open(os.path.join(v5_root, "annotations", "instances.json")) as f:
        blob = json.load(f)
    img_by_id = {i["id"]: i for i in blob["images"]}
    ann_by_id = {a["id"]: a for a in blob["annotations"]}
    keys = sorted(k for k, v in blob["framesets"].items()
                  if v["recording"] == recording and v["fly_id"] == fly_id)
    if not keys:
        raise ValueError(f"no framesets for {recording} fly{fly_id}")
    pick = keys[:: max(1, len(keys) // max(n_frames, 1))][:n_frames]

    groups = keypoint_groups(blob["keypoint_names"])
    # keypoint_groups returns {group_name: [indices]}; invert it so a keypoint
    # index can look up its own group name below.
    idx2group = {i: g for g, idxs in groups.items() for i in idxs}
    sheet = Image.new("RGB", (cell * len(pick), cell * len(cams)), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)

    for col, key in enumerate(pick):
        fs = blob["framesets"][key]
        for row, cam in enumerate(cams):
            hit = next(((i, a) for i, a in zip(fs["frames"], fs["ann_ids"])
                        if img_by_id[i]["file_name"].split("/")[-2] == cam), None)
            if hit is None:
                continue
            img_id, ann_id = hit
            info, ann = img_by_id[img_id], ann_by_id[ann_id]
            path = os.path.join(v5_root, "images", info["file_name"])
            if not os.path.exists(path):
                continue
            with Image.open(path) as pil:
                im = pil.convert("RGB")
            x0, y0, x1, y1 = _crop_box(ann["bbox"], info["width"], info["height"])
            crop = im.crop((x0, y0, x1, y1)).resize((cell, cell))
            kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
            cd = ImageDraw.Draw(crop)
            sx, sy = cell / (x1 - x0), cell / (y1 - y0)
            for j, (px, py, v) in enumerate(kp):
                if v <= 0:
                    continue
                cx, cy = (px - x0) * sx, (py - y0) * sy
                if not (0 <= cx < cell and 0 <= cy < cell):
                    continue
                colour = PALETTE.get(idx2group.get(j), (255, 255, 255))
                cd.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=tuple(colour))
            sheet.paste(crop, (col * cell, row * cell))
            draw.text((col * cell + 4, row * cell + 4), f"{cam}", fill=(230, 230, 230))
        draw.text((col * cell + 4, cell * len(cams) - 14),
                  key.split("/")[1], fill=(180, 180, 180))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{recording}_fly{fly_id}.png")
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recordings", nargs="*", default=None)
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.v5_root, "manifest.json")))
    recs = args.recordings or sorted(man["recordings"])
    with open(os.path.join(args.v5_root, "annotations", "instances.json")) as f:
        blob = json.load(f)
    for rec in recs:
        flies = sorted({v["fly_id"] for v in blob["framesets"].values()
                        if v["recording"] == rec})
        for k in flies:
            print(build_contact_sheet(args.v5_root, rec, n_frames=args.n_frames,
                                      out_dir=args.out_dir, fly_id=k))


if __name__ == "__main__":
    main()
