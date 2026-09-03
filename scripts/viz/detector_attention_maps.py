#!/usr/bin/env python3
"""Where does the mask-off ViTPose (v12_bal_maskoff) look when it misses a tarsal tip?

Diagnostic promoted from the 2026-09-03 spike so it can be rerun on any checkpoint pair
(e.g. v12_bal_maskon once it finishes). Run on a GPU node: CUDA_VISIBLE_DEVICES=<free gpu>.

EXPECTATION, written before plotting. v12's val error is concentrated in tarsal
tips (32-37 px) and, on two-fly frames, extremity predictions scatter by more
than a body length while the body core stays put. Two mechanisms would look
different here:
  (A) ATTENTION HOPS: the token under the GT tip attends to the OTHER fly (or
      the wall/arena band) in the mid/late layers, and the predicted tip lands
      on the other fly's leg. Then the query-from-GT-tip attention maps on bad
      frames put mass on the other fly's mask, the per-layer "mass on other
      fly" curve is higher for bad frames than good ones, and the tip heatmap
      is bimodal with the second mode on the other fly.
  (B) ATTENTION IS FINE, THE READOUT ISN'T: attention stays on the target's
      own leg chain but the heatmap for that tip is diffuse/multi-modal on the
      target itself (e.g. the wrong leg's tip, or T1 vs T2 confusion). Then the
      mass curves for bad and good frames overlap, and the failure shows in the
      heatmap panel not the attention panel.
v5vf_maskoff (trained on most of these val recordings, so it "knows" them) is
the comparison arm on the SAME crops: if its attention on the same bad frames
stays on the target while v12's leaves, the failure is a generalisation gap in
where the network looks, not in the decoder.

Both checkpoints are mask-ablation arms (channel 3 zeroed, v12 provably
invariant), so channel 3 is zeroed here too.

Outputs (figures/2026-09-03-maskoff-attention/):
  attn_query_<tag>.png   per selected frame: crop with GT/pred/masks, tip heatmap,
                         query-from-GT-tip attention at 4 layers + rollout, for
                         v12 and v5vf
  attn_mass_by_layer.png bad-vs-good tip frames: attention mass from GT-tip tokens
                         onto target mask / other-fly mask / background per layer
  attn_summary.npz       the numbers behind attn_mass_by_layer.png
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "third_party" / "jarvis_jax")):
    sys.path.insert(0, p)

import jax, jax.numpy as jnp
from flax import nnx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.v5_2d import V5Dataset
from jarvis_jax.data.mask_zero import ZeroMaskDataset
from jarvis_jax.data.device import normalize_image
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.scripts.eval_keypoints_2d import restore_model
from jarvis_jax.tracking.predict_2d import verify_detector_kp_order
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from viz.core.colors import leg_chains

PATCH, GRID, CROP = 16, 28, 448


# ----------------------------------------------------------------------------
# forward pass that also returns every layer's softmax attention
# ----------------------------------------------------------------------------
def forward_with_attn(model, x):
    """x (B,448,448,4) float -> heatmaps (B,224,224,50), attn (L,B,heads,N+1,N+1)."""
    vit = model.backbone
    t = vit.patch_embed(x)
    b = t.shape[0]
    cls = jnp.broadcast_to(vit.cls_token.value, (b, 1, t.shape[-1]))
    t = jnp.concatenate([cls, t], axis=1) + vit.pos_embed.value
    attns = []
    for blk in vit.blocks:
        h = blk.norm1(t)
        bb, n, d = h.shape
        at = blk.attn
        qkv = at.qkv(h).reshape(bb, n, 3, at.num_heads, at.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * at.scale, axis=-1)
        attns.append(a)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(bb, n, d)
        t = t + at.proj(out)
        t = t + blk.mlp(blk.norm2(t))
    t = vit.norm(t)[:, 1:]
    hm = model.decoder(t, use_running_average=True)
    return hm, jnp.stack(attns, 0)


def rollout(attn_layers):
    """Abnar & Zuidema attention rollout. attn_layers (L,heads,N,N) for ONE
    sample -> (N,N): row i = where token i's final representation drew from."""
    L, H, N, _ = attn_layers.shape
    eye = jnp.eye(N)
    r = eye
    for l in range(L):
        a = attn_layers[l].mean(0)
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(-1, keepdims=True)
        r = a @ r
    return r


def token_of_xy(x, y):
    """crop px -> token index (1 + row*28 + col); cls is token 0."""
    c = int(np.clip(x // PATCH, 0, GRID - 1)); r = int(np.clip(y // PATCH, 0, GRID - 1))
    return 1 + r * GRID + c


def patch_fraction(mask_crop):
    """(448,448) bool -> (784,) fraction of each 16x16 patch covered."""
    m = mask_crop.astype(np.float32).reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
    return m.reshape(-1)


def masks_for(root, ds, i):
    """target mask crop, other-fly mask crop (bool, 448x448) for ds index i."""
    fn = ds.file_names[i]; rec, cam, base = fn.split("/")
    npz = os.path.join(root, "masks", rec, cam, os.path.splitext(base)[0] + ".npz")
    w, h = ds.img_wh[i]
    x0, y0 = crop_origin(ds.bboxes[i], int(w), int(h), crop=CROP)
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    tgt = np.zeros((CROP, CROP), bool); oth = np.zeros((CROP, CROP), bool)
    if not os.path.exists(npz):
        return tgt, oth
    with np.load(npz) as z:
        masks, ids, matched = z["masks"], z["ann_ids"], z["matched"]
    me = int(ds.ann_ids[i])
    for j in range(len(ids)):
        if not matched[j]:
            continue
        m = masks[j].astype(bool)[sl]
        pad = np.zeros((CROP, CROP), bool); pad[:m.shape[0], :m.shape[1]] = m
        if int(ids[j]) == me:
            tgt |= pad
        else:
            oth |= pad
    return tgt, oth


def draw_chains(ax, kp, vis, names, color, lw=1.0, ls="-"):
    for leg, chain in leg_chains(names).items():
        pts = [kp[k] for k in chain if vis[k]]
        if len(pts) >= 2:
            pts = np.asarray(pts); ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, alpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--ckpt-v12", default="/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v12_bal_maskoff/final")
    ap.add_argument("--ckpt-ref", default="/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5vf_maskoff/final")
    ap.add_argument("--eval-npz", default="figures/2026-09-03-v12-detector-ab/v12_bal_maskoff.npz")
    ap.add_argument("--out", default="figures/2026-09-03-maskoff-attention")
    ap.add_argument("--n-mass", type=int, default=160, help="frames per group for the mass curves")
    ap.add_argument("--layers", default="1,4,8,12")
    ap.add_argument("--skip-mass", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    names = json.load(open(os.path.join(a.root, "annotations", "keypoint_names.json")))
    assert verify_detector_kp_order(a.ckpt_v12, names, strict=True) is not None, a.ckpt_v12
    # v5vf's training root (red_data_3d_v5_valfix) was deleted, so the guard
    # cannot resolve its order and is warn-only. Evidence it matches: on the
    # control recording both arms trained on, the 2026-09-03 A/B measured
    # 5.42 vs 5.43 px -- a scrambled order would be tens of px.
    ref_ok = verify_detector_kp_order(a.ckpt_ref, names, strict=True) is not None
    print(f"keypoint order: v12 VERIFIED; ref {'VERIFIED' if ref_ok else 'WARN-ONLY (training root deleted)'}")
    name_idx = {n: i for i, n in enumerate(names)}
    tips = [name_idx[f"{leg}_TaTip"] for leg in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")]

    ds_raw = V5Dataset(a.root, "val")
    ds = ZeroMaskDataset(ds_raw)
    z = np.load(a.eval_npz)
    assert list(z["file_name"]) == list(ds_raw.file_names), "eval npz is not in val index order"
    pred, gt, vis = z["pred_zeroed"], z["gt_zeroed"], z["vis_zeroed"]   # crop px (448)
    err = np.linalg.norm(pred - gt, axis=-1); err[~vis] = np.nan
    tip_err = np.nanmax(err[:, tips], axis=1)            # worst tip per annotation
    two_fly = np.array([masks_for(a.root, ds_raw, i)[1].any() for i in range(len(ds_raw))])
    rec = z["recording"]
    print(f"val {len(ds_raw)} anns; two-fly {two_fly.sum()}; "
          f"worst-tip median {np.nanmedian(tip_err):.1f}px, >30px: {(tip_err > 30).mean():.1%}")

    cfg = ViTPoseConfig()
    models = {"v12_bal_maskoff": restore_model(a.ckpt_v12, "vitpose", cfg),
              "v5vf_maskoff": restore_model(a.ckpt_ref, "vitpose", cfg)}
    for m in models.values():
        m.eval()
    fwd = {k: nnx.jit(forward_with_attn) for k in models}

    def run(mkey, i):
        img4, kp, v = ds[i]
        x = normalize_image(jnp.asarray(img4[None]))
        hm, at = fwd[mkey](models[mkey], x)
        return np.asarray(img4), np.asarray(hm[0]), at[:, 0]       # at: (L,heads,N,N) device

    # ------------------------------------------------------------------ frames
    order = np.argsort(-np.nan_to_num(tip_err, nan=-1))
    sel = []
    sel += [(int(i), "worst_twofly") for i in order if two_fly[i]][:3]
    sel += [(int(i), "worst_single") for i in order if not two_fly[i]][:2]
    ctl = np.where(rec == "2026_01_29_14_09_33")[0]
    sel += [(int(ctl[np.argsort(tip_err[ctl])[len(ctl) // 2]]), "control_median")]
    fem = np.where((rec == "2026_06_09_15_46_55"))[0]
    sel += [(int(fem[np.argsort(tip_err[fem])[len(fem) // 2]]), "headless_female_median")]
    layers = [int(s) for s in a.layers.split(",")]

    for i, tag in sel:
        tgt, oth = masks_for(a.root, ds_raw, i)
        worst_tip = tips[int(np.nanargmax(err[i, tips]))]
        gx, gy = gt[i, worst_tip]; px_, py_ = pred[i, worst_tip]
        qtok = token_of_xy(gx, gy)
        ptok = token_of_xy(px_, py_)
        ncol = 2 + len(layers) + 1
        fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.6))
        for r, mkey in enumerate(models):
            img4, hm, at = run(mkey, i)
            pk = np.asarray(heatmaps_to_keypoints(jnp.asarray(hm[None]), in_size=CROP))[0]
            rgb = img4[..., :3]
            ax = axes[r, 0]; ax.imshow(rgb)
            ax.contour(tgt, levels=[0.5], colors="lightgrey", linewidths=0.8)
            if oth.any():
                ax.contour(oth, levels=[0.5], colors="orange", linewidths=0.8)
            draw_chains(ax, gt[i], vis[i], names, "white", lw=0.8)
            draw_chains(ax, pk, vis[i], names, "cyan", lw=0.8, ls="--")
            ax.plot(gx, gy, "o", mfc="none", mec="white", ms=9, mew=1.5)
            ax.plot(pk[worst_tip, 0], pk[worst_tip, 1], "x", color="cyan", ms=9, mew=1.5)
            e = np.linalg.norm(pk[worst_tip] - gt[i, worst_tip])
            ax.set_title(f"{mkey}\n{names[worst_tip]} err {e:.0f}px", fontsize=8)
            # heatmap channel of the worst tip
            ax = axes[r, 1]
            h = np.maximum(hm[..., worst_tip], 0)
            ax.imshow(rgb, alpha=0.5); ax.imshow(h, extent=(0, CROP, CROP, 0), cmap="magma", alpha=0.7)
            ax.plot(gx, gy, "o", mfc="none", mec="white", ms=9, mew=1.5)
            ax.set_title(f"heatmap {names[worst_tip]}\nmax {h.max():.2f}", fontsize=8)
            # attention rows from the GT-tip token
            at_np = {l: np.asarray(at[l - 1].mean(0)) for l in layers}
            for c, l in enumerate(layers):
                ax = axes[r, 2 + c]
                row = at_np[l][qtok, 1:].reshape(GRID, GRID)
                ax.imshow(rgb, alpha=0.45)
                ax.imshow(row, extent=(0, CROP, CROP, 0), cmap="viridis", alpha=0.7,
                          vmin=0, vmax=max(row.max(), 1e-6))
                ax.plot(gx, gy, "o", mfc="none", mec="white", ms=9, mew=1.5)
                on_o = float((row.reshape(-1) * patch_fraction(oth)).sum()) if oth.any() else 0.0
                on_t = float((row.reshape(-1) * patch_fraction(tgt)).sum())
                ax.set_title(f"attn from GT tip, layer {l}\ntarget {on_t:.2f}  other {on_o:.2f}", fontsize=8)
            ro = np.asarray(rollout(at))
            ax = axes[r, -1]
            row = ro[qtok, 1:].reshape(GRID, GRID)
            ax.imshow(rgb, alpha=0.45)
            ax.imshow(row, extent=(0, CROP, CROP, 0), cmap="viridis", alpha=0.7, vmin=0, vmax=row.max())
            ax.plot(gx, gy, "o", mfc="none", mec="white", ms=9, mew=1.5)
            ax.plot(px_, py_, "x", color="cyan", ms=8, mew=1.5)
            ax.set_title("rollout from GT tip\n(x = v12 eval pred)", fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{tag}: {ds_raw.file_names[i]}  ann {int(ds_raw.ann_ids[i])}  "
                     f"{'two-fly' if oth.any() else 'single'}  v12 worst-tip err {tip_err[i]:.0f}px\n"
                     "white = GT, cyan dashed = prediction, grey = target mask, orange = other fly mask; "
                     "channel 3 zeroed for both arms", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"attn_query_{tag}_{i}.png", dpi=130); plt.close(fig)
        print("wrote", tag, i)

    # ------------------------------------------------- heatmap modes on bad tips
    # If attention is the same for both arms (see mass curves), the failure must
    # be in the READOUT: is the GT tip still a local mode of the heatmap that a
    # competing mode simply out-votes, or is there no response at GT at all?
    # EXPECTATION for (B): on bad tips the heatmap value at GT is a sizeable
    # fraction of the max for v12 (a "competing mode" miss), and the SAME frames
    # under v5vf have the competing mode suppressed (ratio GT/max ~1).
    def group(mask_sel, n):
        idx = np.where(mask_sel)[0]
        return idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    bad = np.concatenate([group(two_fly & (tip_err > 30), a.n_mass), group(~two_fly & (tip_err > 30), a.n_mass)])
    ratios = {m: [] for m in models}; where = {m: [] for m in models}
    hm_fwd = {k: nnx.jit(lambda mdl, x: mdl(x, use_running_average=True)) for k in models}
    for i in bad:
        img4, _, _ = ds[int(i)]
        tgt, oth = masks_for(a.root, ds_raw, int(i))
        x = normalize_image(jnp.asarray(img4[None]))
        for mkey in models:
            hm = np.asarray(hm_fwd[mkey](models[mkey], x)[0])
            pk = np.asarray(heatmaps_to_keypoints(jnp.asarray(hm[None]), in_size=CROP))[0]
            for k in tips:
                if not vis[i, k] or err[i, k] <= 30:
                    continue
                h = np.maximum(hm[..., k], 0)
                gx, gy = np.clip(gt[i, k] / 2, 0, 223).astype(int)
                r = 3
                at_gt = h[max(gy - r, 0):gy + r + 1, max(gx - r, 0):gx + r + 1].max()
                ratios[mkey].append(at_gt / max(h.max(), 1e-6))
                px, py = np.clip(pk[k], 0, CROP - 1).astype(int)
                where[mkey].append("other" if oth[py, px] else ("target" if tgt[py, px] else "bg"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1.2, 25)
    for mkey in models:
        rr = np.asarray(ratios[mkey])
        axes[0].hist(rr, bins=bins, histtype="step", lw=2, label=f"{mkey} n={len(rr)} (>0.5: {(rr > 0.5).mean():.0%})")
    axes[0].set_xlabel("heatmap at GT tip / heatmap max  (v12-bad tips, err>30px)")
    axes[0].set_ylabel("count"); axes[0].legend(fontsize=8)
    axes[0].set_title("Is GT still a mode? 1.0 = GT is the peak (v5vf gets these right)", fontsize=9)
    cats = ["target", "other", "bg"]
    w = 0.38
    for j, mkey in enumerate(models):
        cnt = [where[mkey].count(c) / max(len(where[mkey]), 1) for c in cats]
        axes[1].bar(np.arange(3) + (j - 0.5) * w, cnt, w, label=mkey)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["on target fly mask", "on OTHER fly mask", "background"])
    axes[1].set_ylabel("fraction of predictions"); axes[1].legend(fontsize=8)
    axes[1].set_title("Where does the argmax land on those tips?", fontsize=9)
    fig.suptitle(f"v12_bal_maskoff bad tarsal tips (err>30px, {len(bad)} frames sampled): heatmap readout, both arms on identical crops", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "heatmap_modes_bad_tips.png", dpi=130); plt.close(fig)
    for mkey in models:
        rr = np.asarray(ratios[mkey])
        print(f"{mkey}: n={len(rr)} GT/max median {np.median(rr):.2f}, >0.5: {(rr > 0.5).mean():.1%}, >0.9: {(rr > 0.9).mean():.1%}; "
              f"argmax on other {where[mkey].count('other')/len(rr):.1%}, target {where[mkey].count('target')/len(rr):.1%}, bg {where[mkey].count('bg')/len(rr):.1%}")
    np.savez(out / "heatmap_modes_bad_tips.npz", **{f"ratio__{m}": np.asarray(ratios[m]) for m in models},
             **{f"where__{m}": np.asarray(where[m]) for m in models}, bad_idx=bad)
    if a.skip_mass:
        print("skip-mass: done"); return

    # ------------------------------------------------------------- mass curves
    # attention mass from the GT-tip tokens onto target / other / background,
    # per layer (head-mean), bad vs good tip frames, both arms. Two-fly only for
    # the "other" region to mean anything; single-fly frames report target/bg.
    def group(mask_sel, n):
        idx = np.where(mask_sel)[0]
        return idx[np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)]
    groups = {
        "bad_twofly (tip>30px)": group(two_fly & (tip_err > 30), a.n_mass),
        "good_twofly (tip<10px)": group(two_fly & (tip_err < 10), a.n_mass),
        "bad_single (tip>30px)": group(~two_fly & (tip_err > 30), a.n_mass),
        "good_single (tip<10px)": group(~two_fly & (tip_err < 10), a.n_mass),
    }
    for g, idx in groups.items():
        print(g, len(idx))

    @nnx.jit
    def mass_fn(model, x, qmask, regions):
        """qmask (N+1,) float: 1 at query (GT tip) tokens; regions (3,N) patch fractions.
        -> (L, 3) attention mass from the mean query row onto each region."""
        _, at = forward_with_attn(model, x)             # (L,1,H,N+1,N+1)
        rows = at[:, 0].mean(1)                           # (L,N+1,N+1)
        q = (rows * qmask[None, :, None]).sum(1) / jnp.maximum(qmask.sum(), 1)  # (L,N+1)
        return q[:, 1:] @ regions.T                       # (L,3)

    L = 12
    summary = {}
    for mkey in models:
        for g, idx in groups.items():
            acc = []
            for i in idx:
                img4, _, _ = ds[int(i)]
                tgt, oth = masks_for(a.root, ds_raw, int(i))
                ft, fo = patch_fraction(tgt), patch_fraction(oth)
                fb = np.clip(1 - ft - fo, 0, 1)
                qm = np.zeros(GRID * GRID + 1, np.float32)
                for k in tips:
                    if vis[i, k]:
                        qm[token_of_xy(*gt[i, k])] = 1
                if qm.sum() == 0:
                    continue
                m = mass_fn(models[mkey], normalize_image(jnp.asarray(img4[None])),
                            jnp.asarray(qm), jnp.asarray(np.stack([ft, fo, fb])))
                acc.append(np.asarray(m))
            summary[(mkey, g)] = np.stack(acc) if acc else np.zeros((0, L, 3))
            print(mkey, g, summary[(mkey, g)].shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    styles = {"v12_bal_maskoff": "-", "v5vf_maskoff": "--"}
    colors = {"bad_twofly (tip>30px)": "C3", "good_twofly (tip<10px)": "C2",
              "bad_single (tip>30px)": "C1", "good_single (tip<10px)": "C0"}
    for (mkey, g), arr in summary.items():
        if len(arr) == 0:
            continue
        mu = arr.mean(0); se = arr.std(0) / np.sqrt(len(arr))
        for r, ax in enumerate(axes):
            ax.plot(range(1, L + 1), mu[:, r], styles[mkey], color=colors[g],
                    label=f"{mkey} {g} n={len(arr)}")
            ax.fill_between(range(1, L + 1), mu[:, r] - se[:, r], mu[:, r] + se[:, r],
                            color=colors[g], alpha=0.12)
    for ax, t in zip(axes, ["onto TARGET fly mask", "onto OTHER fly mask", "onto background"]):
        ax.set_title(f"attention mass from GT tarsal-tip tokens {t}", fontsize=9)
        ax.set_xlabel("layer"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("head-mean attention mass")
    axes[1].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Expectation if attention hops (A): red/orange (bad) above green/blue (good) on the OTHER-fly panel "
                 "in mid/late layers, solid (v12) above dashed (v5vf). Overlapping curves = mechanism (B), look at heatmaps.",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "attn_mass_by_layer.png", dpi=130); plt.close(fig)
    np.savez(out / "attn_summary.npz",
             **{f"{m}__{g.split(' ')[0]}": v for (m, g), v in summary.items()},
             tip_err=tip_err, two_fly=two_fly, selected=np.asarray([i for i, _ in sel]),
             selected_tags=np.asarray([t for _, t in sel]), layers=np.asarray(layers))
    print("wrote", out / "attn_mass_by_layer.png")


if __name__ == "__main__":
    main()
