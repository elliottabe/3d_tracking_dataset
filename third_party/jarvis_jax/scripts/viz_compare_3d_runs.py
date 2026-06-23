"""Compare two cached-3D runs: 3D keypoint accuracy + 2D reprojection overlays.

run1 = silent no-Laplacian (graph term had zero edges)
run2 = bone-prior fix (44 skeleton edges wired)

For each of 4 representative val framesets (best / ~33% / ~66% / worst by run2
per-frameset MPJPE) produces a PNG with:
  - 7 camera panels: the RGB crop with GT (green), run1 (red), run2 (blue)
    reprojected keypoints + skeleton edges overlaid.
  - a 3D scatter of the three skeletons.
  - a per-joint MPJPE bar chart (run1 vs run2).
Plus an overall summary PNG (mean MPJPE + per-joint mean error across val).

Reprojection: full-image px = (x,y,z,1) @ cameraMatrices[c] (4,3), then /w.
Crop-local px = full_px - centerHM[c] + CROP/2  (centerHM = crop_origin + CROP/2).
"""
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.v3_3d import V3FramesetDataset
from jarvis_jax.data.repro_cache import load_cache
from jarvis_jax.hybridnet.v2vnet import V2VNet
from jarvis_jax.hybridnet.model import soft_argmax_3d
from jarvis_jax.train.losses_3d import build_skeleton_edges

CROP = 448
NUM_J = 50


# --------------------------------------------------------------------------
# Model loading + inference
# --------------------------------------------------------------------------
def load_v2v(final_dir):
    v2v = V2VNet(NUM_J, NUM_J, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    ckptr = ocp.StandardCheckpointer()
    try:
        restored = ckptr.restore(final_dir, state)
    except TypeError:
        restored = ckptr.restore(final_dir, args=ocp.args.StandardRestore(state))
    return nnx.merge(gdef, restored)


def predict_3d(v2v, volumes_np, center3D_np, batch=32, sharpen=1.0):
    """volumes_np (N,J,48,48,48) fp16, center3D_np (N,3) -> (N,J,3) world."""
    v2v.eval()

    @nnx.jit
    def step(v2v, vol, c3d):
        vol = vol.astype(jnp.float32)
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))      # (B,48,48,48,J)
        vol = v2v(vol, use_running_average=True)        # (B,24,24,24,J)
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))      # (B,J,24,24,24)
        vol = jax.nn.softplus(vol)
        pl, _ = soft_argmax_3d(vol, grid_spacing=1, roi_cube=48, sharpen=sharpen)
        return pl + c3d[:, None, :]

    out = []
    for i in range(0, len(volumes_np), batch):
        v = jnp.asarray(np.asarray(volumes_np[i:i + batch]))
        c = jnp.asarray(center3D_np[i:i + batch])
        out.append(np.asarray(step(v2v, v, c)))
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def project_full(p3d, M):
    """p3d (3,), M (4,3) -> (2,) full-image px."""
    ph = np.concatenate([p3d, [1.0]])
    proj = ph @ M                      # (3,)
    return proj[:2] / proj[2]


def to_crop(p3d, M, center_hm):
    """3D world -> crop-local px (x,y)."""
    return project_full(p3d, M) - center_hm + CROP / 2.0


def per_frameset_mpjpe(pred, gt, vis):
    """pred/gt (J,3), vis (J,) -> mean joint error over visible joints."""
    if vis.sum() == 0:
        return np.nan
    d = np.linalg.norm(pred[vis] - gt[vis], axis=-1)
    return float(d.mean())


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
COL = {"GT": "#00d000", "run1": "#e02020", "run2": "#2050ff"}


def draw_overlay(ax, crop_rgb, kps_by, vis, edges, center_hm, M, title):
    ax.imshow(crop_rgb)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ei, ej = edges
    for name, p3d in kps_by.items():
        xy = np.stack([to_crop(p3d[j], M, center_hm) for j in range(NUM_J)])
        # skeleton edges (both endpoints visible)
        for a, b in zip(ei, ej):
            if vis[a] and vis[b]:
                ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]],
                        "-", color=COL[name], lw=0.6, alpha=0.7)
        m = vis
        ax.scatter(xy[m, 0], xy[m, 1], s=6, c=COL[name],
                   edgecolors="k", linewidths=0.2, label=name, zorder=3)
    ax.set_xlim(0, CROP); ax.set_ylim(CROP, 0)


def draw_3d(ax, kps_by, vis, edges):
    ei, ej = edges
    for name, p3d in kps_by.items():
        m = vis
        ax.scatter(p3d[m, 0], p3d[m, 1], p3d[m, 2], s=8, c=COL[name], label=name)
        for a, b in zip(ei, ej):
            if vis[a] and vis[b]:
                ax.plot([p3d[a, 0], p3d[b, 0]], [p3d[a, 1], p3d[b, 1]],
                        [p3d[a, 2], p3d[b, 2]], "-", color=COL[name], lw=0.6, alpha=0.7)
    ax.set_title("3D skeletons (world)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")


def draw_bars(ax, pred1, pred2, gt, vis, kp_names):
    js = np.where(vis)[0]
    e1 = np.linalg.norm(pred1[js] - gt[js], axis=-1)
    e2 = np.linalg.norm(pred2[js] - gt[js], axis=-1)
    x = np.arange(len(js))
    ax.bar(x - 0.2, e1, width=0.4, color=COL["run1"], label="run1 (no Lap)")
    ax.bar(x + 0.2, e2, width=0.4, color=COL["run2"], label="run2 (bone prior)")
    ax.set_xticks(x)
    ax.set_xticklabels([kp_names[j][:10] for j in js], rotation=90, fontsize=5)
    ax.set_ylabel("joint err (world units)", fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Per-joint error", fontsize=8)


def make_frameset_fig(idx, tag, crops, vis, gt, p1, p2, cams, center_hm,
                      edges, kp_names, out_png):
    nc = crops.shape[0]
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8])
    # camera panels: first 2 rows (8 cells); use nc cells for cams, rest for 3D
    cells = [(r, c) for r in range(2) for c in range(4)]
    for ci in range(nc):
        r, c = cells[ci]
        ax = fig.add_subplot(gs[r, c])
        draw_overlay(ax, crops[ci, :, :, :3], {"GT": gt, "run1": p1, "run2": p2},
                     vis, edges, center_hm[ci], cams[ci], f"cam {ci}")
        if ci == 0:
            ax.legend(fontsize=6, loc="upper right")
    # 3D scatter in the cell after the last camera
    r, c = cells[nc]
    ax3d = fig.add_subplot(gs[r, c], projection="3d")
    draw_3d(ax3d, {"GT": gt, "run1": p1, "run2": p2}, vis, edges)
    # per-joint bars across bottom row
    axb = fig.add_subplot(gs[2, :])
    draw_bars(axb, p1, p2, gt, vis, kp_names)
    m1 = per_frameset_mpjpe(p1, gt, vis)
    m2 = per_frameset_mpjpe(p2, gt, vis)
    fig.suptitle(f"val frameset #{idx}  [{tag}]   "
                 f"MPJPE  run1={m1:.2f}  run2={m2:.2f}  "
                 f"(visible joints={int(vis.sum())})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")


def make_summary_fig(m1_all, m2_all, pe1, pe2, kp_names, out_png):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(20, 7))
    # overall mean MPJPE
    axa.bar(["run1\n(no Lap)", "run2\n(bone prior)"],
            [np.nanmean(m1_all), np.nanmean(m2_all)],
            color=[COL["run1"], COL["run2"]])
    for i, v in enumerate([np.nanmean(m1_all), np.nanmean(m2_all)]):
        axa.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    axa.set_title(f"Overall val MPJPE (N={len(m1_all)} framesets)", fontsize=12)
    axa.set_ylabel("MPJPE (world units)")
    # per-joint mean error across val
    x = np.arange(NUM_J)
    axb.bar(x - 0.2, pe1, width=0.4, color=COL["run1"], label="run1 (no Lap)")
    axb.bar(x + 0.2, pe2, width=0.4, color=COL["run2"], label="run2 (bone prior)")
    axb.set_xticks(x)
    axb.set_xticklabels([n[:10] for n in kp_names], rotation=90, fontsize=5)
    axb.set_ylabel("mean joint err (world units)", fontsize=8)
    axb.legend(); axb.set_title("Per-joint mean error across val", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3")
    ap.add_argument("--cache-dir", default="/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v3")
    ap.add_argument("--run1", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/here_run2/final")
    ap.add_argument("--run2", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/here_run3/final")
    ap.add_argument("--out", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/viz_compare")
    ap.add_argument("--sharpen1", type=float, default=1.0,
                    help="soft-argmax sharpen exponent for run1 (center-bias fix)")
    ap.add_argument("--sharpen2", type=float, default=1.0,
                    help="soft-argmax sharpen exponent for run2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[viz] loading val cache + dataset")
    cache = load_cache(args.cache_dir, "val")
    vols = cache["volumes"]                 # (N,J,48,48,48) fp16 memmap
    gt_all = np.asarray(cache["kp3d"])      # (N,J,3)
    c3d_all = np.asarray(cache["center3D"]) # (N,3)
    vis_all = np.asarray(cache["vis"])      # (N,J) bool
    N = vols.shape[0]
    ds = V3FramesetDataset(args.root, "val")
    kp_names = list(ds.keypoint_names)
    ei, ej = build_skeleton_edges(kp_names, ds.skeleton)
    edges = (np.asarray(ei), np.asarray(ej))
    print(f"[viz] N={N} framesets, {len(ei)} skeleton edges")

    # Alignment guard: cache index must match dataset index
    for chk in (0, N // 2, N - 1):
        s = ds[chk]
        assert np.allclose(s["center3D"], c3d_all[chk], atol=1e-3), \
            f"cache/dataset center3D misaligned at {chk}"
        assert np.allclose(s["kp3d"], gt_all[chk], atol=1e-2), \
            f"cache/dataset kp3d misaligned at {chk}"
    print("[viz] alignment OK (cache idx == dataset idx)")

    print("[viz] running inference for run1 + run2")
    v1 = load_v2v(args.run1)
    v2 = load_v2v(args.run2)
    pred1 = predict_3d(v1, vols, c3d_all, sharpen=args.sharpen1)
    pred2 = predict_3d(v2, vols, c3d_all, sharpen=args.sharpen2)

    # per-frameset MPJPE
    m1 = np.array([per_frameset_mpjpe(pred1[i], gt_all[i], vis_all[i]) for i in range(N)])
    m2 = np.array([per_frameset_mpjpe(pred2[i], gt_all[i], vis_all[i]) for i in range(N)])
    print(f"[viz] overall MPJPE  run1={np.nanmean(m1):.3f}  run2={np.nanmean(m2):.3f}")

    # per-joint mean error across val (visible only)
    def per_joint_mean(pred):
        errs = np.full(NUM_J, np.nan)
        for j in range(NUM_J):
            m = vis_all[:, j]
            if m.sum() > 0:
                errs[j] = np.linalg.norm(pred[m, j] - gt_all[m, j], axis=-1).mean()
        return errs
    pe1 = np.nan_to_num(per_joint_mean(pred1))
    pe2 = np.nan_to_num(per_joint_mean(pred2))

    # choose 4 framesets by run2 error percentile
    order = np.argsort(np.nan_to_num(m2, nan=1e9))
    picks = [(order[0], "best"),
             (order[len(order) // 3], "p33"),
             (order[2 * len(order) // 3], "p66"),
             (order[-1], "worst")]

    print("[viz] rendering frameset figures")
    for idx, tag in picks:
        s = ds[int(idx)]
        make_frameset_fig(
            int(idx), tag,
            s["crops4"], vis_all[idx], gt_all[idx], pred1[idx], pred2[idx],
            np.asarray(s["cameraMatrices"]), np.asarray(s["centerHM"]),
            edges, kp_names,
            os.path.join(args.out, f"frameset_{tag}_{idx}.png"))

    make_summary_fig(m1, m2, pe1, pe2, kp_names,
                     os.path.join(args.out, "summary.png"))
    print(f"[viz] done -> {args.out}")


if __name__ == "__main__":
    main()
