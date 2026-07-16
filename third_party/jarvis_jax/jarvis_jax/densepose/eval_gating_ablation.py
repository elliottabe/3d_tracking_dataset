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
    from jarvis_jax.densepose.gating import gate_heatmaps
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


def _restore_model(ckpt, num_joints):
    """Restore a ViTPose checkpoint (same pattern as _eval_arm) for viz reuse."""
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    m = ViTPose(ViTPoseConfig(num_keypoints=num_joints), rngs=nnx.Rngs(0))
    g, st = nnx.split(m); m = nnx.merge(g, ocp.StandardCheckpointer().restore(ckpt, st)); m.eval()
    return m


def _viz_overlays(ungated_ckpt, gated_ckpt, ds, num_joints, dilate, viz_dir, n=6, in_size=448):
    """Qualitative ungated-vs-gated overlays (raw decode). For the first ``n`` val crops:
    the RGB crop, then the summed positive predicted heatmap mass over the crop with the
    dilated SAM-mask boundary drawn for each arm -- mass landing outside the boundary is
    exactly the failure the containment loss targets, so the per-panel off-mask fraction
    makes the gating effect visible. Saves PNGs to ``viz_dir``. Numbers-only if matplotlib
    is unavailable."""
    import numpy as np, jax.numpy as jnp, jax
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                            # pragma: no cover
        print(f"[ablation] viz skipped (matplotlib unavailable: {e})")
        return
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.data.v3 import batches
    from jarvis_jax.models.dilation import dilate_mask_jax
    os.makedirs(viz_dir, exist_ok=True)

    ung = _restore_model(ungated_ckpt, num_joints)
    gat = _restore_model(gated_ckpt, num_joints)

    saved = 0
    for img4_u8, kp_xy, vis in batches(ds, 1, shuffle=False, drop_last=False):
        if saved >= n:
            break
        img = normalize_image(jnp.asarray(img4_u8))
        rgb = np.asarray(img4_u8)[0, ..., :3].astype("uint8")
        raw_m = np.asarray(img4_u8)[0, ..., 3].astype("float32")
        mask448 = (raw_m > (raw_m.max() * 0.5)).astype("float32") if raw_m.max() > 0 else raw_m
        dmask = np.asarray(dilate_mask_jax(jnp.asarray(mask448), int(dilate)))   # (448,448)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        axes[0].imshow(rgb)
        axes[0].contour(dmask, levels=[0.5], colors="cyan", linewidths=1.0)
        axes[0].set_title("crop + dilated mask"); axes[0].axis("off")
        for ax, (name, model) in zip(axes[1:], (("ungated", ung), ("gated", gat))):
            pred = model(img, use_running_average=True)                          # (1,224,224,J)
            mass = np.asarray(jax.nn.relu(pred)[0].sum(-1))                      # (224,224)
            mass = np.asarray(jax.image.resize(jnp.asarray(mass), (in_size, in_size),
                                               method="linear"))
            off = float((mass * (1.0 - dmask)).sum() / (mass.sum() + 1e-6))
            ax.imshow(rgb)
            ax.imshow(mass, cmap="inferno", alpha=0.55)
            ax.contour(dmask, levels=[0.5], colors="cyan", linewidths=1.0)
            ax.set_title(f"{name}: off-mask mass={off:.3f}"); ax.axis("off")
        fig.tight_layout()
        p = os.path.join(viz_dir, f"overlay_{saved:02d}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        saved += 1
    print(f"[ablation] wrote {saved} overlays to {viz_dir}")


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
    ap.add_argument("--viz-n", type=int, default=0, help="write this many qualitative overlay PNGs (0=off)")
    ap.add_argument("--viz-dir", default=None, help="overlay dir (default: <out-dir>/viz)")
    a = ap.parse_args()

    from jarvis_jax.densepose.cse_dataset import CSEImageDataset
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

    if a.viz_n > 0:
        viz_dir = a.viz_dir or os.path.join(os.path.dirname(a.out), "viz")
        _viz_overlays(a.ungated_ckpt, a.gated_ckpt, ds, a.num_joints, a.dilate,
                      viz_dir, n=a.viz_n)


if __name__ == "__main__":
    main()
