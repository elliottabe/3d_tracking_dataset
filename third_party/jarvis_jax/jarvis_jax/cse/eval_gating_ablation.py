"""Phase-5 ablation: GATED vs UNGATED cse_vit350 on val.

Metrics per (checkpoint arm x decode mode):
  * mpjpe_px         : kp channels 0..49
  * pck@10 / pck@5   : kp PCK
  * dense_mpjpe_px   : dense vertex channels 50..349 (visible verts)
  * off_fly_mass_frac: HEADLINE -- mean fraction of predicted positive heatmap mass
                       OUTSIDE the dilated SAM mask (lower is better; the containment
                       loss target). Computed with mask_containment(dilate=k).
Also writes qualitative overlays. Honest reporting: if gating does not reduce
off-fly mass / improve accuracy, the numbers say so.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def off_fly_mass_frac(pred, mask224, dilate=0):
    """Mean fraction of positive predicted heatmap mass outside the dilated mask."""
    from jarvis_jax.train.losses import mask_containment
    return float(mask_containment(pred, mask224, dilate=dilate))


def _pck(pred_kp, gt_kp, vis, thresh):
    import numpy as np
    d = np.linalg.norm(np.asarray(pred_kp) - np.asarray(gt_kp), axis=-1)  # (B,K)
    v = np.asarray(vis).astype(bool)
    return float((d[v] <= thresh).mean()) if v.any() else float("nan")


def _eval_arm(ckpt, ds, batch, num_joints, dilate, gate_decode, in_size=448):
    import numpy as np, jax.numpy as jnp, orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.data.v3 import batches
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe
    from jarvis_jax.cse.gating import gate_heatmaps
    import jax

    m = ViTPose(ViTPoseConfig(num_keypoints=num_joints), rngs=nnx.Rngs(0))
    g, st = nnx.split(m); m = nnx.merge(g, ocp.StandardCheckpointer().restore(ckpt, st)); m.eval()
    scale = in_size / float(ds.heatmap_size)

    off_num = off_den = 0.0
    kp_err_sum = kp_n = 0.0
    dv_err_sum = dv_n = 0.0
    pck10_hits = pck5_hits = pck_n = 0.0
    for img4_u8, kp_xy, vis in batches(ds, batch, shuffle=False, drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pred = m(img, use_running_average=True)                       # (B,224,224,J)
        mask = img[..., 3]
        mask224 = jax.image.resize(mask, (mask.shape[0], pred.shape[1], pred.shape[2]),
                                   method="nearest")
        # headline off-fly mass (weighted by batch size)
        off_num += off_fly_mass_frac(pred, mask224, dilate=dilate) * img4_u8.shape[0]
        off_den += img4_u8.shape[0]
        dec = gate_heatmaps(pred, mask, dilate=dilate) if gate_decode else pred
        pk = np.asarray(heatmaps_to_keypoints(dec, in_size=in_size))  # (B,J,2)
        gk = np.asarray(kp_xy) * scale
        vb = np.asarray(vis).astype(bool)
        # keypoints 0..49
        kp_err_sum += float(mpjpe(jnp.asarray(pk[:, :50]), jnp.asarray(gk[:, :50]),
                                  jnp.asarray(vb[:, :50]))) * int(vb[:, :50].sum())
        kp_n += int(vb[:, :50].sum())
        # dense verts 50..
        dv_err_sum += float(mpjpe(jnp.asarray(pk[:, 50:]), jnp.asarray(gk[:, 50:]),
                                  jnp.asarray(vb[:, 50:]))) * int(vb[:, 50:].sum())
        dv_n += int(vb[:, 50:].sum())
        pck10_hits += _pck(pk[:, :50], gk[:, :50], vb[:, :50], 10.0) * int(vb[:, :50].sum())
        pck5_hits += _pck(pk[:, :50], gk[:, :50], vb[:, :50], 5.0) * int(vb[:, :50].sum())
        pck_n += int(vb[:, :50].sum())
    return {
        "mpjpe_px": kp_err_sum / max(kp_n, 1),
        "dense_mpjpe_px": dv_err_sum / max(dv_n, 1),
        "off_fly_mass_frac": off_num / max(off_den, 1),
        "pck@10px": pck10_hits / max(pck_n, 1),
        "pck@5px": pck5_hits / max(pck_n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3")
    ap.add_argument("--aux-val", default="/gscratch/portia/eabe/data/Johnson_lab/cse_work/cse_labels_val_M300.npz")
    ap.add_argument("--ungated-ckpt", required=True)
    ap.add_argument("--gated-ckpt", required=True)
    ap.add_argument("--num-joints", type=int, default=350)
    ap.add_argument("--dilate", type=int, default=11)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from jarvis_jax.cse.cse_dataset import CSEImageDataset
    ds = CSEImageDataset(a.root, "val", a.aux_val)
    summary = {}
    for arm, ckpt in (("ungated_ckpt", a.ungated_ckpt), ("gated_ckpt", a.gated_ckpt)):
        summary[arm] = {
            "raw": _eval_arm(ckpt, ds, a.batch, a.num_joints, a.dilate, gate_decode=False),
            "gated_decode": _eval_arm(ckpt, ds, a.batch, a.num_joints, a.dilate, gate_decode=True),
        }
        print(arm, json.dumps(summary[arm], indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    print(f"[ablation] wrote {a.out}")


if __name__ == "__main__":
    main()
