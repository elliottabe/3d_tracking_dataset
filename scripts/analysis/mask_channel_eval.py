#!/usr/bin/env python3
"""Mask-channel ablation: per-sample, per-keypoint 2D eval on red_data_3d_v5 val.

GPU-bound step of the mask-channel-ablation task (see
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/mask-channel-ablation.md``).
Restores a ViTPose checkpoint and evaluates it on the FULL val split twice --
mask channel populated as normal, and mask channel ZEROED at inference (the
cheap Step-1 probe) -- saving raw per-(annotation, keypoint) predictions/GT/
visibility plus per-annotation metadata (sex, behavior, recording, ann_id) to
an .npz. All the aggregation (overall/female MPJPE, per-keypoint,
per-body-group via viz/core/colors.py, male-vs-female, behavior-tag proxy for
close-interaction) happens LOCALLY afterwards from this .npz -- no GPU needed
for that -- see scripts/analysis/mask_channel_report.py.

CAVEAT (state this every time these numbers are quoted): a checkpoint TRAINED
with the mask populated has never seen a zeroed mask at train time, so its
"zeroed" condition here is out-of-distribution. A large error increase under
zeroing is therefore AMBIGUOUS (real mask information, or mere sensitivity to
an unseen-blank input) -- but a SMALL increase is still strong evidence the
mask carries little, since an OOD penalty would only ever inflate the gap.
This caveat does NOT apply to a checkpoint trained with train.mask_ablation=true
(jarvis_jax.train.train.TrainConfig) -- that model's own val split (zeroed) is
in-distribution for it by construction.

Usage (run via the SLURM queue -- scripts/slurm/submit_task.sh -- never on a
shared interactive GPU node):
    python scripts/analysis/mask_channel_eval.py \\
        --ckpt-dir /gscratch/.../jax_vitpose_runs/v5_s70_bal_augdef_full/final \\
        --data-root /gscratch/.../red_data/red_data_3d_v5 \\
        --conditions populated,zeroed \\
        --out figures/2026-08-31-mask-ablation/v5_s70_bal_augdef_full.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_VAL_RECORDING = "2026_05_27_11_56_05"   # the female/courtship val recording


def build_arrays(model, ds, batch_size=16, in_size=448):
    """Run `model` over every sample of `ds` (in index order, no shuffling)
    and return (pred_xy (N,K,2) f32, gt_xy (N,K,2) f32, vis (N,K) bool)."""
    import jax.numpy as jnp
    import numpy as np
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.data.v3 import batches
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
    from jarvis_jax.train.train import _eval_forward

    model.eval()
    scale = in_size / float(ds.heatmap_size)
    preds, gts, viss = [], [], []
    for img4_u8, kp_xy, vis in batches(ds, batch_size, shuffle=False, drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pred = _eval_forward(model, img)
        pk = heatmaps_to_keypoints(pred, in_size=in_size)
        preds.append(np.asarray(pk))
        gts.append(np.asarray(kp_xy) * scale)
        viss.append(np.asarray(vis))
    return (np.concatenate(preds, axis=0), np.concatenate(gts, axis=0),
            np.concatenate(viss, axis=0))


def collect_metadata(ds):
    """Pull per-annotation metadata arrays (sex/behavior/recording/ann_id/
    bbox/img_wh) straight off a V5Dataset (or ZeroMaskDataset wrapping one)
    in index order. bbox/img_wh let a LOCAL (no-GPU) follow-up script
    reconstruct the exact same crop from the on-disk image for a hard-frame
    overlay figure, without a second GPU pass."""
    import numpy as np
    file_names = list(ds.file_names)
    recordings = np.asarray([fn.split("/")[0] for fn in file_names])
    return {
        "file_name": np.asarray(file_names),
        "recording": recordings,
        "sex": np.asarray(list(ds.sex)),
        "behavior": np.asarray(list(ds.behavior)),
        "ann_id": np.asarray(list(ds.ann_ids)),
        "bbox": np.asarray(list(ds.bboxes)),
        "img_wh": np.asarray(list(ds.img_wh)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--conditions", default="populated,zeroed",
                    help="comma-separated subset of {populated,zeroed}")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-recording", default=DEFAULT_VAL_RECORDING)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import numpy as np
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.data.v5_2d import V5Dataset
    from jarvis_jax.data.mask_zero import ZeroMaskDataset
    from jarvis_jax.scripts.eval_keypoints_2d import restore_model

    cfg = ViTPoseConfig()
    print(f"restoring checkpoint: {a.ckpt_dir}")
    model = restore_model(a.ckpt_dir, "vitpose", cfg)

    ds = V5Dataset(a.data_root, "val")
    meta = collect_metadata(ds)
    print(f"val set: {len(ds)} annotations, "
          f"{len(set(meta['recording']))} recordings")

    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    out = dict(meta)
    out["kp_names"] = np.asarray(
        __import__("json").load(
            open(os.path.join(a.data_root, "annotations", "keypoint_names.json"))))
    out["val_recording"] = np.asarray(a.val_recording)
    fem_mask = meta["recording"] == a.val_recording

    for cond in conditions:
        this_ds = ZeroMaskDataset(ds) if cond == "zeroed" else ds
        pred, gt, vis = build_arrays(model, this_ds, batch_size=a.batch_size)
        err = np.linalg.norm(pred - gt, axis=-1)   # (N,K) px
        out[f"pred_{cond}"] = pred.astype(np.float32)
        out[f"gt_{cond}"] = gt.astype(np.float32)
        out[f"vis_{cond}"] = vis.astype(bool)
        overall = float((err * vis).sum() / max(vis.sum(), 1))
        fem_vis = vis[fem_mask]
        fem_err = err[fem_mask]
        female = (float((fem_err * fem_vis).sum() / max(fem_vis.sum(), 1))
                  if fem_vis.sum() > 0 else float("nan"))
        print(f"[{cond}] overall val MPJPE: {overall:.3f}px  "
              f"female ({a.val_recording}): {female:.3f}px")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
