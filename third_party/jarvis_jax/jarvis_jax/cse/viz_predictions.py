"""Overlay TRAINED ViTPose dense-pose predictions on val images (GPU).

Works for any output count via --num-joints (250 = 50 kp + 200 body/leg,
350 = + 100 wing).  Body/leg vertices are colored by MuJoCo segment, wing
vertices drawn red so the learned wing surface is easy to spot.  Reports the
2-D median error split body/leg vs wing.
"""
import argparse, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--n-kp", type=int, default=50)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--show-gt", action="store_true",
                    help="overlay GT wing labels (green) vs pred wing (red) + spread ratio")
    a = ap.parse_args()
    import numpy as np, jax.numpy as jnp, orbax.checkpoint as ocp
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
    from jarvis_jax.cse.cse_dataset import CSEImageDataset

    cfg = ViTPoseConfig(num_keypoints=a.num_joints); m = ViTPose(cfg, rngs=nnx.Rngs(0))
    g, st = nnx.split(m); m = nnx.merge(g, ocp.StandardCheckpointer().restore(a.ckpt, st)); m.eval()
    mz = np.load(a.mesh, allow_pickle=True); fps = np.load(a.aux, allow_pickle=True)["fps_indices"]
    seg = mz["vertex_segment"][fps]; uniq = np.unique(seg)
    cseg = np.array([{int(s): i for i, s in enumerate(uniq)}[int(s)] for s in seg])
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    iswing = np.array(["wing" in id2n[int(s)] for s in seg])
    nkp = a.n_kp
    ds = CSEImageDataset(a.root, "val", a.aux)
    idxs = [int(x) for x in np.linspace(0, len(ds) - 1, a.n)]
    imgs = np.stack([ds[i][0] for i in idxs]); kps = np.stack([ds[i][1] for i in idxs])
    viss = np.stack([ds[i][2] for i in idxs]).astype(bool)  # (n, 350) visibility
    pred = m(normalize_image(jnp.asarray(imgs)), use_running_average=True)
    pk = np.asarray(heatmaps_to_keypoints(pred, in_size=448)); gk = kps * (448 / 224.)
    err = np.linalg.norm(pk - gk, axis=2)
    vtx_err = err[:, nkp:]; vtx_vis = viss[:, nkp:]          # only score vis=1 targets
    bl_m = (~iswing)[None] & vtx_vis; wing_m = iswing[None] & vtx_vis
    bl_err = float(np.median(vtx_err[bl_m])) if bl_m.any() else float("nan")
    wing_err = float(np.median(vtx_err[wing_m])) if wing_m.any() else float("nan")
    wing_cov = float(vtx_vis[:, iswing].mean()) if iswing.any() else float("nan")

    def _spread(P):                                          # median radius from centroid
        if len(P) < 2: return np.nan
        return float(np.nanmedian(np.linalg.norm(P - np.nanmean(P, 0), axis=1)))
    pred_wing_spread = gt_wing_spread = float("nan")
    if iswing.any():                                         # spread over vis=1 wing verts only
        sp_p, sp_g = [], []
        for b in range(a.n):
            mb = iswing & vtx_vis[b]
            sp_p.append(_spread(pk[b, nkp:][mb])); sp_g.append(_spread(gk[b, nkp:][mb]))
        pred_wing_spread = float(np.nanmedian(sp_p)); gt_wing_spread = float(np.nanmedian(sp_g))
    ratio = pred_wing_spread / gt_wing_spread if gt_wing_spread else float("nan")

    c = (a.n + 1) // 2
    fig, ax = plt.subplots(2, c, figsize=(4.2 * c, 9))
    for k, b in enumerate(range(a.n)):
        aa = ax.flat[k]; aa.imshow(imgs[b, :, :, :3].astype(np.uint8))
        vp = pk[b, nkp:]; vg = gk[b, nkp:]; vv = vtx_vis[b]
        bm = (~iswing) & vv; wm = iswing & vv               # only draw visible targets/preds
        aa.scatter(vp[bm, 0], vp[bm, 1], s=9, c=cseg[bm], cmap="tab20", linewidths=0)
        if iswing.any():
            if a.show_gt:                                    # GT wing target (vis=1) = green ring
                aa.scatter(vg[wm, 0], vg[wm, 1], s=22, facecolors="none",
                           edgecolors="lime", linewidths=0.7)
            aa.scatter(vp[wm, 0], vp[wm, 1], s=11, c="red", linewidths=0)
        aa.scatter(pk[b, :nkp, 0], pk[b, :nkp, 1], s=14, c="cyan", marker="^", linewidths=0)
        aa.set_title(f"val {idxs[b]}"); aa.axis("off")
    for k in range(a.n, 2 * c): ax.flat[k].axis("off")
    gt_tag = "  GT-wing=lime◦" if a.show_gt else ""
    fig.suptitle(f"TRAINED ViTPose-{a.num_joints} (cyan▲ kp, seg verts, red wing{gt_tag}; vis=1 only) | "
                 f"2D err body/leg {bl_err:.1f}px wing {wing_err:.1f}px | "
                 f"wing vis-cov {wing_cov*100:.0f}% | "
                 f"wing spread pred/GT {pred_wing_spread:.0f}/{gt_wing_spread:.0f}px ({ratio:.2f})")
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  body/leg {bl_err:.1f}px  wing {wing_err:.1f}px  "
          f"wing_vis_cov {wing_cov*100:.1f}%  "
          f"wing_spread pred {pred_wing_spread:.1f} / GT {gt_wing_spread:.1f} px  ratio {ratio:.2f}")


if __name__ == "__main__":
    main()
