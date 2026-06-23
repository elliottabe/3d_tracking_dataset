"""Diagnose 3D skeleton shrinkage: is it soft-argmax center-bias + ROI clamping?

For a trained cached-3D run, per visible joint over the val set, measure:
  - shrink      = ||gt-center|| - ||pred-center||   (>0 = predicted inward)
  - spread      = spatial std of the normalized 3D heatmap (world units;
                  high = diffuse = expectation pulled toward grid center)
  - conf        = peak voxel height (soft_argmax_3d's confidence)
  - margin      = 24 - max(|gt_local axis|)         (<0 = GT outside the ROI cube,
                  structurally unrepresentable by soft-argmax)
  - pos_err     = ||pred - gt||

Hypothesis (center-bias): shrink rises as spread rises, as conf falls, and as
margin shrinks/goes negative. Reports correlations + a global shrink factor +
the worst frameset, and writes scatter plots.
"""
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.repro_cache import load_cache
from jarvis_jax.hybridnet.v2vnet import V2VNet

NUM_J = 50
G = 24            # soft-argmax grid side
ROI = 48          # cube side (world units); local range = idx*2 - 24


def load_v2v(final_dir):
    v2v = V2VNet(NUM_J, NUM_J, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    ckptr = ocp.StandardCheckpointer()
    try:
        restored = ckptr.restore(final_dir, state)
    except TypeError:
        restored = ckptr.restore(final_dir, args=ocp.args.StandardRestore(state))
    return nnx.merge(gdef, restored)


def forward_volume(v2v, volumes_np, batch=16):
    """Return the softplus'd v2vNet output volume (N,J,G,G,G) float32 on host."""
    v2v.eval()

    @nnx.jit
    def step(v2v, vol):
        vol = vol.astype(jnp.float32)
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))      # (B,48,48,48,J)
        vol = v2v(vol, use_running_average=True)        # (B,24,24,24,J)
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))      # (B,J,24,24,24)
        return jax.nn.softplus(vol)

    out = []
    for i in range(0, len(volumes_np), batch):
        v = jnp.asarray(np.asarray(volumes_np[i:i + batch]))
        out.append(np.asarray(step(v2v, v)))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v3")
    ap.add_argument("--run", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/here_run3/final")
    ap.add_argument("--out", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/viz_compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[diag] loading val cache + run3 model")
    cache = load_cache(args.cache_dir, "val")
    vols = cache["volumes"]                  # (N,J,48,48,48) fp16
    gt = np.asarray(cache["kp3d"])           # (N,J,3) world
    c3d = np.asarray(cache["center3D"])      # (N,3)
    vis = np.asarray(cache["vis"])           # (N,J) bool
    N = vols.shape[0]
    v2v = load_v2v(args.run)

    print("[diag] forward (softplus volumes)")
    vol = forward_volume(v2v, vols)          # (N,J,G,G,G)

    # --- soft-argmax + spread + conf from the volume ---
    idx = np.arange(G, dtype=np.float32)
    flat = vol.reshape(N, NUM_J, -1)         # (N,J,G^3)
    prob = flat / np.clip(flat.sum(-1, keepdims=True), 1e-8, None)
    P = prob.reshape(N, NUM_J, G, G, G)
    # mean grid index per axis (xx=dim2, yy=dim3, zz=dim4)
    mx = (P.sum((3, 4)) * idx).sum(-1)       # (N,J)
    my = (P.sum((2, 4)) * idx).sum(-1)
    mz = (P.sum((2, 3)) * idx).sum(-1)
    # variance per axis -> spread (grid units -> world *2)
    vx = (P.sum((3, 4)) * (idx[None, None] - mx[..., None]) ** 2).sum(-1)
    vy = (P.sum((2, 4)) * (idx[None, None] - my[..., None]) ** 2).sum(-1)
    vz = (P.sum((2, 3)) * (idx[None, None] - mz[..., None]) ** 2).sum(-1)
    spread = np.sqrt(vx + vy + vz) * 2.0     # (N,J) world units
    conf = np.clip(flat.max(-1), 0, 255) / 255.0

    # pred local (world, center-relative) = idx*2 - ROI/2
    pred_local = np.stack([mx, my, mz], -1) * 2.0 - ROI / 2.0   # (N,J,3)
    gt_local = gt - c3d[:, None, :]                              # (N,J,3)

    pred_r = np.linalg.norm(pred_local, axis=-1)   # (N,J)
    gt_r = np.linalg.norm(gt_local, axis=-1)
    shrink = gt_r - pred_r                          # >0 = inward
    pos_err = np.linalg.norm(pred_local - gt_local, axis=-1)
    margin = (ROI / 2.0) - np.max(np.abs(gt_local), axis=-1)   # <0 = GT outside cube

    m = vis  # mask
    def msel(a): return a[m]

    # --- global shrink factor + outside-cube fraction ---
    glob_pred = pred_r[m].mean(); glob_gt = gt_r[m].mean()
    frac_out = float((margin[m] < 0).mean())
    print("\n================ SHRINKAGE DIAGNOSTIC (run3, val) ================")
    print(f"visible joints: {int(m.sum())} over {N} framesets")
    print(f"GLOBAL mean radius-from-center:  pred={glob_pred:.2f}  gt={glob_gt:.2f}  "
          f"-> pred/gt = {glob_pred/glob_gt:.3f}  ({100*(1-glob_pred/glob_gt):.1f}% shrink)")
    print(f"fraction of visible joints with GT OUTSIDE the ROI cube: {100*frac_out:.1f}%")
    print(f"mean pos error: {pos_err[m].mean():.2f}   mean inward shrink: {shrink[m].mean():.2f}")

    # --- correlations (hypothesis tests) ---
    def corr(a, b):
        a, b = msel(a), msel(b)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    print("\ncorrelations over visible joints (Pearson r):")
    print(f"  shrink vs spread   : r={corr(shrink, spread):+.3f}  (expect +, diffuse->inward)")
    print(f"  shrink vs conf     : r={corr(shrink, conf):+.3f}  (expect -, low conf->inward)")
    print(f"  shrink vs margin   : r={corr(shrink, margin):+.3f}  (expect -, near/outside edge->inward)")
    print(f"  pos_err vs spread  : r={corr(pos_err, spread):+.3f}")
    print(f"  pos_err vs conf    : r={corr(pos_err, conf):+.3f}")

    # --- worst frameset ---
    fs_err = np.array([pos_err[i][vis[i]].mean() if vis[i].any() else np.nan
                       for i in range(N)])
    worst = int(np.nanargmax(fs_err))
    print(f"\nWORST frameset = #{worst}  (mean pos err {fs_err[worst]:.2f})")
    # top shrunk joints in worst frameset
    try:
        from jarvis_jax.data.v3_3d import V3FramesetDataset
        names = list(V3FramesetDataset(
            "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3", "val").keypoint_names)
    except Exception:
        names = [f"j{j}" for j in range(NUM_J)]
    js = np.where(vis[worst])[0]
    order = js[np.argsort(-shrink[worst][js])][:10]
    print(f"  {'joint':<12} {'shrink':>7} {'pred_r':>7} {'gt_r':>6} {'spread':>7} {'conf':>5} {'margin':>7}")
    for j in order:
        print(f"  {names[j][:12]:<12} {shrink[worst,j]:7.2f} {pred_r[worst,j]:7.2f} "
              f"{gt_r[worst,j]:6.2f} {spread[worst,j]:7.2f} {conf[worst,j]:5.2f} {margin[worst,j]:7.2f}")

    # --- figure: scatters confirming the mechanism ---
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    sh, sp, cf, mg, pr, gr = (msel(shrink), msel(spread), msel(conf),
                              msel(margin), msel(pred_r), msel(gt_r))
    ax[0, 0].scatter(gr, pr, s=6, alpha=0.3)
    lim = max(gr.max(), pr.max())
    ax[0, 0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0, 0].set_xlabel("GT radius from center"); ax[0, 0].set_ylabel("pred radius from center")
    ax[0, 0].set_title(f"Radius: pred vs GT (below diagonal = shrunk)\n"
                       f"global pred/gt = {glob_pred/glob_gt:.3f}")
    ax[0, 1].scatter(sp, sh, s=6, alpha=0.3)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set_xlabel("heatmap spread (world units)"); ax[0, 1].set_ylabel("inward shrink")
    ax[0, 1].set_title(f"shrink vs spread  (r={corr(shrink,spread):+.3f})")
    ax[1, 0].scatter(cf, sh, s=6, alpha=0.3)
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_xlabel("confidence (peak height)"); ax[1, 0].set_ylabel("inward shrink")
    ax[1, 0].set_title(f"shrink vs confidence  (r={corr(shrink,conf):+.3f})")
    ax[1, 1].scatter(mg, sh, s=6, alpha=0.3)
    ax[1, 1].axhline(0, color="k", lw=0.5); ax[1, 1].axvline(0, color="r", lw=1)
    ax[1, 1].set_xlabel("ROI margin (24 - max|gt_local|);  <0 = outside cube")
    ax[1, 1].set_ylabel("inward shrink")
    ax[1, 1].set_title(f"shrink vs ROI margin  (r={corr(shrink,margin):+.3f})  "
                       f"| {100*frac_out:.1f}% outside")
    fig.suptitle("Shrinkage diagnostic — run3 val (soft-argmax center-bias test)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(args.out, "shrinkage_diag.png")
    fig.savefig(out_png, dpi=110)
    print(f"\n[diag] wrote {out_png}")


if __name__ == "__main__":
    main()
