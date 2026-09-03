#!/usr/bin/env python3
"""Show the framesets where the two flies' labels land on the SAME animal.

WHAT R15 DID AND WHY YOU ARE LOOKING AT THIS. `build_generalmodel_split`
attributes identity from the SUBSET a label came from (`courtship_28_34_female`
vs `_male`). When both subsets' annotations for one camera sit on the same
animal -- median visible-keypoint displacement under SAME_FLY_PX = 50 px --
neither can be attributed, so BOTH slots are recorded ABSENT rather than
guessed. That is a labelling defect being caught, not a builder bug, and it is
worth a human ruling because the fix is upstream in the labels.

EXPECTATION IF R15 IS RIGHT. In the flagged panels the two skeletons (orange =
the subset filed as MALE, cyan = the subset filed as FEMALE) lie on top of each
other on ONE fly, while a second fly sits nearby unlabelled. In the control row
-- a normal frameset from the same recording -- the two skeletons sit on two
clearly separated animals. If a flagged panel instead shows two separated
skeletons, the 50 px threshold is wrong and R15 is discarding good data.

Run:
    python scripts/viz/ambiguous_slot_check.py \
        --stage /gscratch/.../red_data/courtship_labels_2026_09_02 \
        --out figures/2026-09-02-ambiguous-slots
"""
import argparse, json, os, collections
from pathlib import Path

# (female subset, male subset, recording, [flagged frames], control frame)
CASES = [
    ("courtship_28_34_female", "courtship_28_34_male", "2026_04_02_17_28_34",
     [621799, 621803], None),
    ("headless_56_42_female", "headless_56_42_1_female", "2026_06_09_15_38_35",
     [13912], None),
    ("courtship_25_51_female", "courtship_25_51_male", "2026_04_02_15_25_51",
     [431686], None),
]


def load(stage, subset):
    p = Path(stage) / subset / "annotations" / "instances_train.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text())
    id2im = {im["id"]: im for im in d["images"]}
    by = collections.defaultdict(list)
    for a in d["annotations"]:
        im = id2im[a["image_id"]]
        parts = im["file_name"].split("/")
        frame = int(os.path.splitext(parts[-1])[0].split("_")[-1])
        by[(parts[1], frame)].append((a, str(Path(stage) / subset / "train" / im["file_name"])))
    return by, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--gm", default=None, help="general_model root, for subsets not staged")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", type=int, default=3, help="cameras per frameset to draw")
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for fsub, msub, rec, flagged, _ in CASES:
        fby, _ = load(a.stage, fsub)
        mby, _ = load(a.stage, msub)
        if not fby and a.gm:
            fby, _ = load(a.gm, fsub); mby, _ = load(a.gm, msub)
        if not fby or not mby:
            print(f"skip {rec}: subsets not found under --stage/--gm"); continue
        all_frames = sorted({fr for _, fr in fby})
        control = next((f for f in all_frames if f not in flagged), None)
        for frame, tag in [(f, "FLAGGED") for f in flagged] + \
                          ([(control, "control")] if control is not None else []):
            cams = sorted({c for c, fr in fby if fr == frame})[:a.cams]
            for cam in cams:
                rows.append((rec, frame, cam, tag, fby.get((cam, frame), []),
                             mby.get((cam, frame), [])))

    if not rows:
        raise SystemExit("nothing to draw")
    ncol = a.cams
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes: ax.axis("off")

    for ax, (rec, frame, cam, tag, fa, ma) in zip(axes, rows):
        path = (fa or ma)[0][1]
        try:
            img = np.asarray(Image.open(path).convert("L"))
        except Exception as e:
            ax.set_title(f"{cam} {frame}: {e}", fontsize=6); continue
        ax.imshow(img, cmap="gray"); ax.axis("off")
        med = None
        for anns, color, lbl in ((fa, "#00d5ff", "filed FEMALE"),
                                 (ma, "#ff8c1a", "filed MALE")):
            for ann, _ in anns:
                k = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
                v = k[:, 2] > 0
                ax.scatter(k[v, 0], k[v, 1], s=3, c=color, linewidths=0, label=lbl)
        if fa and ma:
            kf = np.asarray(fa[0][0]["keypoints"], float).reshape(-1, 3)
            km = np.asarray(ma[0][0]["keypoints"], float).reshape(-1, 3)
            both = (kf[:, 2] > 0) & (km[:, 2] > 0)
            if both.any():
                med = float(np.median(np.linalg.norm(kf[both, :2] - km[both, :2], axis=1)))
        col = "crimson" if tag == "FLAGGED" else "green"
        ax.set_title(f"{tag}  {rec}  Frame_{frame}  {cam}"
                     + (f"   median |F-M| = {med:.1f} px" if med is not None else ""),
                     fontsize=7, color=col)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        seen = {}
        for hi, li in zip(h, l): seen.setdefault(li, hi)
        fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("R15 same-animal slots: both subsets' labels on ONE fly (crimson) "
                 "vs a normal two-fly frameset (green).\n"
                 "SAME_FLY_PX = 50; below that the builder records BOTH slots absent "
                 "rather than guess identity.", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    png = os.path.join(a.out, "ambiguous_slots.png")
    fig.savefig(png, dpi=125)
    print("wrote", png, f"({len(rows)} panels)")


if __name__ == "__main__":
    main()
