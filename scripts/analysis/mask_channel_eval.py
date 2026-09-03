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

FEMALE AXIS: reported from the per-annotation resolved `sex`, not from a
recording name. See DEFAULT_VAL_RECORDING below for the one-recording trap
that motivated this (a 14-annotation sliver quoted as the female headline).

KEYPOINT ORDER: verified, not assumed. This script calls
`jarvis_jax.tracking.predict_2d.verify_detector_kp_order` against the
checkpoint's own training order (recovered from its .hydra/overrides.yaml ->
paths.data_root -> annotations/keypoint_names.json) and records whether the
guard actually VERIFIED or fell back to warn-only into the .npz
(`kp_order_verified`). A warn-only run is exactly how a two-fly eval was
scrambled on 2026-08-31.

Usage (run via the SLURM queue -- scripts/slurm/submit_task.sh -- unless the
GPU node is idle and the user has authorised running directly on it):
    python scripts/analysis/mask_channel_eval.py \\
        --ckpt-dir /gscratch/.../jax_vitpose_runs/v5vf_maskoff/final \\
        --data-root /gscratch/.../red_data/red_data_3d_v5_valfix \\
        --conditions zeroed \\
        --out figures/2026-09-02-vitpose-maskoff-ab/v5vf_maskoff.npz
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

# NO default recording. The previous default was "2026_05_27_11_56_05",
# commented "the female/courtship val recording", and it is a trap that was
# quoted as a headline female number:
#   * it IS present in both red_data_3d_v5 and red_data_3d_v5_valfix -- but
#     with only 14 val annotations out of 1871 (0.7%), and its manifest split
#     is "mixed" (10 train / 2 val framesets), so it is a razor-thin, partly
#     train-adjacent sliver, NOT the female cohort;
#   * its sibling 2026_05_27_11_57_05 (105 val annotations) is the MALE half of
#     the same courtship pair -- "fixing" the one-digit difference points the
#     "female" metric at a male recording.
# The real female axis on this val set is the per-annotation resolved sex
# (V5Dataset._resolve_sex): 304 female vs 1567 male annotations in
# red_data_3d_v5_valfix val, of which 181 female come from the two-fly
# recording 2026_04_07_11_33_33 that valfix moved from TRAIN to val. So the
# female MPJPE below is computed from `sex`, and --val-recording is an
# OPTIONAL extra per-recording slice that must select something or die.
DEFAULT_VAL_RECORDING = None


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


import numpy as np   # module scope: DistractorGrayFill runs inside the loader
                    # thread pool, where main()'s local import is not visible


from jarvis_jax.data.distractor import DistractorGrayFillDataset


class DistractorGrayFill(DistractorGrayFillDataset):
    """Always-on (p=1) distractor gray-fill, as inference applies it. The
    implementation moved to ``jarvis_jax.data.distractor`` on 2026-09-03 so
    training can apply the same fill (train.distractor_fill_p); this thin
    subclass keeps the eval CLI's name and always-fill behaviour."""

    def __init__(self, ds, root, *, dilate=15, protect=60, crop=448):
        super().__init__(ds, root, p=1.0, dilate=dilate, protect=protect, crop=crop)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--distractor-grayfill", action="store_true",
                    help="gray-fill the OTHER annotations' flies out of the RGB "
                         "crop, reproducing what predict_2d does at inference "
                         "but training never did. See DistractorGrayFill below.")
    ap.add_argument("--distractor-dilate", type=int, default=15)
    ap.add_argument("--target-protect", type=int, default=60)
    ap.add_argument("--conditions", default="populated,zeroed",
                    help="comma-separated subset of {populated,zeroed}")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--val-recording", default=DEFAULT_VAL_RECORDING,
                    help="OPTIONAL extra per-recording MPJPE slice. Must match "
                         "at least one val annotation or the run aborts -- a "
                         "typo'd/absent name used to select nothing (or a "
                         "0.7%% sliver) and still print a confident number.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-kp-order", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="abort if the keypoint-order guard can only warn "
                         "(default on).")
    a = ap.parse_args(argv)

    import json
    import warnings

    import numpy as np
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.data.v5_2d import V5Dataset
    from jarvis_jax.data.mask_zero import ZeroMaskDataset
    from jarvis_jax.scripts.eval_keypoints_2d import restore_model
    from jarvis_jax.tracking.predict_2d import verify_detector_kp_order

    kp_names = json.load(
        open(os.path.join(a.data_root, "annotations", "keypoint_names.json")))

    # --- keypoint-order guard, BEFORE any inference. Compares the checkpoint's
    # own training order (recovered from its .hydra/overrides.yaml) against the
    # order of the root we are evaluating on. `None` means the guard could not
    # resolve the training order and only WARNED -- record that, because a
    # warn-only run is how a two-fly eval got scrambled on 2026-08-31.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trained = verify_detector_kp_order(a.ckpt_dir, kp_names, strict=True)
    kp_order_verified = trained is not None
    print(f"keypoint-order guard: "
          f"{'VERIFIED against ckpt training order' if kp_order_verified else 'WARN-ONLY (unverified)'}"
          f" ({len(kp_names)} names)")
    for w in caught:
        print(f"  warning: {w.message}")
    if a.require_kp_order and not kp_order_verified:
        raise SystemExit(
            "aborting: keypoint-order guard is WARN-ONLY for "
            f"{a.ckpt_dir} -- the eval would index a keypoint axis it cannot "
            "verify. Pass --no-require-kp-order only if you accept that.")

    cfg = ViTPoseConfig()
    print(f"restoring checkpoint: {a.ckpt_dir}")
    model = restore_model(a.ckpt_dir, "vitpose", cfg)

    ds = V5Dataset(a.data_root, "val")
    meta = collect_metadata(ds)
    print(f"val set: {len(ds)} annotations, "
          f"{len(set(meta['recording']))} recordings, "
          f"{int((meta['sex'] == 'female').sum())} female / "
          f"{int((meta['sex'] == 'male').sum())} male annotations")

    conditions = [c.strip() for c in a.conditions.split(",") if c.strip()]
    out = dict(meta)
    out["kp_names"] = np.asarray(kp_names)
    out["kp_order_verified"] = np.asarray(kp_order_verified)
    out["ckpt_dir"] = np.asarray(str(a.ckpt_dir))
    out["data_root"] = np.asarray(str(a.data_root))
    out["val_recording"] = np.asarray("" if a.val_recording is None
                                      else a.val_recording)

    # --- optional per-recording slice, with the guard the old default lacked
    rec_mask = None
    if a.val_recording:
        rec_mask = meta["recording"] == a.val_recording
        if not rec_mask.any():
            raise SystemExit(
                f"--val-recording {a.val_recording!r} matches 0 of {len(ds)} "
                f"val annotations. Present recordings: "
                f"{sorted(set(meta['recording'].tolist()))}")
        print(f"--val-recording {a.val_recording}: {int(rec_mask.sum())} "
              f"annotations ({100 * rec_mask.mean():.1f}% of val)")

    female_mask = meta["sex"] == "female"
    male_mask = meta["sex"] == "male"

    def _mpjpe(err, vis, sel=None):
        e, v = (err, vis) if sel is None else (err[sel], vis[sel])
        d = v.sum()
        return (float((e * v).sum() / d) if d > 0 else float("nan")), int(d)

    for cond in conditions:
        this_ds = ZeroMaskDataset(ds) if cond == "zeroed" else ds
        if a.distractor_grayfill:
            this_ds = DistractorGrayFill(this_ds, a.data_root,
                                         dilate=a.distractor_dilate,
                                         protect=a.target_protect)
        pred, gt, vis = build_arrays(model, this_ds, batch_size=a.batch_size)
        err = np.linalg.norm(pred - gt, axis=-1)   # (N,K) px
        out[f"pred_{cond}"] = pred.astype(np.float32)
        out[f"gt_{cond}"] = gt.astype(np.float32)
        out[f"vis_{cond}"] = vis.astype(bool)
        overall, n_all = _mpjpe(err, vis)
        female, n_f = _mpjpe(err, vis, female_mask)
        male, n_m = _mpjpe(err, vis, male_mask)
        line = (f"[{cond}] overall {overall:.3f}px (n={n_all}) | "
                f"FEMALE {female:.3f}px (n={n_f}) | male {male:.3f}px (n={n_m})")
        if rec_mask is not None:
            rec_v, rec_n = _mpjpe(err, vis, rec_mask)
            line += f" | {a.val_recording} {rec_v:.3f}px (n={rec_n})"
        print(line)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
