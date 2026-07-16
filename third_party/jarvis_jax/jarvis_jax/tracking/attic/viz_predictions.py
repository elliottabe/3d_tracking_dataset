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
    ap.add_argument("--ckpt", default=None, help="StandardCheckpointer 'final' dir")
    ap.add_argument("--ckpt-dir", default=None,
                    help="CheckpointManager dir (model+opt) — restores the LATEST committed step "
                         "(use to viz a mid-training checkpoint)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--n-kp", type=int, default=50)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--show-gt", action="store_true",
                    help="overlay GT wing labels (green) vs pred wing (red) + spread ratio")
    a = ap.parse_args()
    if not (a.ckpt or a.ckpt_dir):
        ap.error("one of --ckpt or --ckpt-dir is required")
    import numpy as np, jax.numpy as jnp, orbax.checkpoint as ocp
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.data.device import normalize_image
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
    from jarvis_jax.densepose.cse_dataset import CSEImageDataset

    cfg = ViTPoseConfig(num_keypoints=a.num_joints); m = ViTPose(cfg, rngs=nnx.Rngs(0))
    ckpt_step = None
    if a.ckpt_dir:                                          # restore latest mid-training step
        from jarvis_jax.train.checkpoint import make_manager, restore_latest
        from jarvis_jax.train.train import make_optimizer, TrainConfig
        opt = make_optimizer(m, TrainConfig())
        m, opt, ckpt_step = restore_latest(make_manager(a.ckpt_dir), m, opt)
        print(f"restored {a.ckpt_dir} @ step {ckpt_step}")
    else:
        g, st = nnx.split(m); m = nnx.merge(g, ocp.StandardCheckpointer().restore(a.ckpt, st))
    m.eval()
    mz = np.load(a.mesh, allow_pickle=True); fps = np.load(a.aux, allow_pickle=True)["fps_indices"]
    seg = mz["vertex_segment"][fps]; uniq = np.unique(seg)
    cseg = np.array([{int(s): i for i, s in enumerate(uniq)}[int(s)] for s in seg])
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    leg_sub = ("coxa", "trochanter", "femur", "tibia", "tarsus", "claw")
    iswing = np.array(["wing" in nm for nm in names])
    isleg = np.array([any(s in nm for s in leg_sub) for nm in names])
    isbody = ~iswing & ~isleg
    nkp = a.n_kp
    ds = CSEImageDataset(a.root, "val", a.aux)
    idxs = [int(x) for x in np.linspace(0, len(ds) - 1, a.n)]
    imgs = np.stack([ds[i][0] for i in idxs]); kps = np.stack([ds[i][1] for i in idxs])
    viss = np.stack([ds[i][2] for i in idxs]).astype(bool)  # (n, 350) visibility
    pred = m(normalize_image(jnp.asarray(imgs)), use_running_average=True)
    pk = np.asarray(heatmaps_to_keypoints(pred, in_size=448)); gk = kps * (448 / 224.)
    err = np.linalg.norm(pk - gk, axis=2)
    vtx_err = err[:, nkp:]; vtx_vis = viss[:, nkp:]          # only score vis=1 targets

    def _err(mask):
        m2 = mask[None] & vtx_vis
        return float(np.median(vtx_err[m2])) if m2.any() else float("nan")
    wing_err, leg_err, body_err = _err(iswing), _err(isleg), _err(isbody)
    wing_cov = float(vtx_vis[:, iswing].mean()) if iswing.any() else float("nan")

    def _spread(P):                                          # median radius from centroid
        if len(P) < 2: return np.nan
        return float(np.nanmedian(np.linalg.norm(P - np.nanmean(P, 0), axis=1)))

    def _spread_ratio(mask):                                 # pred/GT spread over vis=1 verts
        if not mask.any(): return (np.nan, np.nan, np.nan)
        sp, sg = [], []
        for b in range(a.n):
            mb = mask & vtx_vis[b]
            sp.append(_spread(pk[b, nkp:][mb])); sg.append(_spread(gk[b, nkp:][mb]))
        p, g = float(np.nanmedian(sp)), float(np.nanmedian(sg))
        return p, g, (p / g if g else np.nan)
    pred_wing_spread, gt_wing_spread, ratio = _spread_ratio(iswing)
    pred_leg_spread, gt_leg_spread, leg_ratio = _spread_ratio(isleg)

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
    step_tag = f" @step{ckpt_step}" if ckpt_step is not None else ""
    fig.suptitle(f"ViTPose-{a.num_joints}{step_tag} (cyan▲ kp, seg verts, red wing{gt_tag}; vis=1 only) | "
                 f"2D err body {body_err:.1f} / leg {leg_err:.1f} / wing {wing_err:.1f} px | "
                 f"spread pred/GT  wing {ratio:.2f}  leg {leg_ratio:.2f}  (wing vis-cov {wing_cov*100:.0f}%)")
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  err body {body_err:.1f} / leg {leg_err:.1f} / wing {wing_err:.1f} px  "
          f"wing_vis_cov {wing_cov*100:.1f}%  "
          f"wing_spread {pred_wing_spread:.1f}/{gt_wing_spread:.1f} ({ratio:.2f})  "
          f"leg_spread {pred_leg_spread:.1f}/{gt_leg_spread:.1f} ({leg_ratio:.2f})")


if __name__ == "__main__":
    main()
