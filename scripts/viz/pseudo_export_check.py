"""Read a pseudo-label export back the way the trainer will and LOOK at it.

EXPECTATION (state it before generating, CLAUDE.md): on every panel the
written keypoints sit ON the fly in that camera's JPEG -- head (red) at the
head, abdomen (magenta) at the abdomen, the six leg chains (orange) running
outward from the thorax -- and the grey SAM3 mask outline of that same
annotation wraps the same animal. Anything else names the bug directly:

  * keypoints on the OTHER fly of the pair       -> identity/`host_fly` wrong;
  * an anatomically scrambled skeleton (head
    points on a leg, chains crossing the body)   -> keypoint order written by
                                                    INDEX, not by name;
  * a plausible-looking skeleton on the wrong
    camera's image                               -> camera axis written by
                                                    index, not by name;
  * blue flies on a red background               -> the JPEG's RGB/BGR flip.

The panels are chosen to be the HARD cases, not the flattering ones: the
female host, a contact frame and a wall frame are picked explicitly when the
export's strata contain them, and each frameset is rendered in two cameras.

Usage:
    python scripts/viz/pseudo_export_check.py --root <export> --out figures/<topic>/check.png
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "third_party", "jarvis_jax"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from viz.core.colors import PALETTE, keypoint_groups, leg_chains   # noqa: E402

GROUP_COLOR = {"head": "head", "thorax": "thorax", "abdomen": "abdomen", "legs": "legs"}


def _panel(root, coco, img, ann, fsv, key, cam_row, kp_names, pad=40):
    img_id, ann_id = fsv["frames"][cam_row], fsv["ann_ids"][cam_row]
    if ann_id is None:
        return None
    info, a = img[img_id], ann[ann_id]
    cam = info["file_name"].split("/")[1]
    path = os.path.join(root, "images", info["file_name"])
    if not os.path.exists(path):
        return None
    im = cv2.imread(path)                      # BGR, as cv2 draws
    kp = np.asarray(a["keypoints"], np.float32).reshape(-1, 3)
    vis = kp[:, 2] > 0
    if not vis.any():
        return None
    groups = keypoint_groups(kp_names)
    for g, idx in groups.items():
        c = PALETTE[GROUP_COLOR[g]]
        for i in idx:
            if vis[i]:
                cv2.circle(im, (int(kp[i, 0]), int(kp[i, 1])), 3, c, -1)
    for leg, chain in leg_chains(kp_names).items():
        pts = [(int(kp[i, 0]), int(kp[i, 1])) for i in chain if vis[i]]
        for p, q in zip(pts, pts[1:]):
            cv2.line(im, p, q, PALETTE["legs"], 1)
    # the annotation's OWN SAM3 mask outline, resolved exactly as the loader does
    from jarvis_jax.data.v5_3d import _load_mask
    m = _load_mask(root, info["file_name"], a.get("src_ann_id", ann_id), ann_id,
                   info["width"], info["height"])
    if m.any():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, PALETTE["mask"], 1)
    x0 = max(int(kp[vis, 0].min()) - pad, 0)
    x1 = min(int(kp[vis, 0].max()) + pad, im.shape[1])
    y0 = max(int(kp[vis, 1].min()) - pad, 0)
    y1 = min(int(kp[vis, 1].max()) + pad, im.shape[0])
    crop = im[y0:y1, x0:x1]
    st = fsv.get("stratum", {})
    label = (f"{key}  {cam}  {a.get('sex', '?')}  {fsv.get('role')}  "
             f"{'contact' if st.get('contact') else ('apart' if st.get('apart') else 'mid')}"
             f"{' WALL' if st.get('wall') else ''}  kpdist={st.get('kp_dist_units')}u "
             f"cent={st.get('sep_units')}u  nkp={int(vis.sum())}  "
             f"mask={'yes' if m.any() else 'NO'}")
    return crop, label


def pick(framesets, n):
    """The hard cases first: female host, contact, wall -- then random fill."""
    want = [("female host", lambda v: v["stratum"].get("host_sex") == "female"),
            ("female+contact", lambda v: v["stratum"].get("host_sex") == "female"
             and v["stratum"].get("contact")),
            ("male+contact", lambda v: v["stratum"].get("host_sex") == "male"
             and v["stratum"].get("contact")),
            ("wall", lambda v: v["stratum"].get("wall")),
            ("partner", lambda v: v.get("role") == "partner")]
    out, seen = [], set()
    for why, f in want:
        hits = [k for k, v in framesets.items() if f(v)]
        if hits:
            k = hits[len(hits) // 2]
            if k not in seen:
                out.append((why, k)); seen.add(k)
    rest = [k for k in framesets if k not in seen]
    random.Random(0).shuffle(rest)
    for k in rest[: max(0, n - len(out))]:
        out.append(("random", k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--cams", type=int, default=2, help="camera rows per frameset")
    a = ap.parse_args()
    coco = json.load(open(os.path.join(a.root, "annotations", f"instances_{a.split}.json")))
    kp_names = coco["keypoint_names"]
    img = {i["id"]: i for i in coco["images"]}
    ann = {x["id"]: x for x in coco["annotations"]}
    rows = []
    for why, key in pick(coco["framesets"], a.n):
        fsv = coco["framesets"][key]
        got = []
        for c in range(len(fsv["frames"])):
            p = _panel(a.root, coco, img, ann, fsv, key, c, kp_names)
            if p is not None:
                got.append(p)
            if len(got) >= a.cams:
                break
        for crop, label in got:
            rows.append((crop, f"[{why}] {label}"))
    if not rows:
        raise SystemExit("nothing to render")
    W = max(max(c.shape[1] for c, _ in rows), 1000)      # room for the whole label line
    H = sum(c.shape[0] + 22 for c, _ in rows)
    canvas = np.zeros((H, W, 3), np.uint8)
    y = 0
    for crop, label in rows:
        cv2.putText(canvas, label[:170], (4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
        canvas[y:y + crop.shape[0], :crop.shape[1]] = crop
        y += crop.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, canvas)
    print(f"wrote {a.out}  ({len(rows)} panels)")


if __name__ == "__main__":
    main()
