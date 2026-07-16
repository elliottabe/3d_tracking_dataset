"""Visualize current 3-D triangulation (V2VNet) from the prebuilt reproject cache.

Loads the LATEST V2VNet checkpoint (training may still be running), runs the
cached val volumes through V2VNet -> soft-argmax -> world 3-D, and plots predicted
vs GT points for a few framesets.  Body/leg verts gray, wing verts red, GT wing
lime.  Reports the 3-D wing spread ratio (pred/GT) so we can compare against the
2-D under-spread (0.79) and see whether multi-view triangulation widens the wing.
"""
import argparse, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=350)
    ap.add_argument("--n-kp", type=int, default=50)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--sharpen", type=float, default=3.0)
    a = ap.parse_args()

    import numpy as np, jax.numpy as jnp
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from flax import nnx
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.data.repro_cache import load_cache
    from jarvis_jax.train.checkpoint import make_manager, restore_latest
    from jarvis_jax.train.train_3d_cached import make_eval_step_jitted, make_v2v_optimizer, CachedConfig

    J, nkp = a.num_joints, a.n_kp
    # wing mask over the vertex block
    mz = np.load(a.mesh, allow_pickle=True); nv = J - nkp
    fps = mz[f"fps_{nv}"]; seg = mz["vertex_segment"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    leg_sub = ("coxa", "trochanter", "femur", "tibia", "tarsus", "claw")
    iswing = np.array(["wing" in nm for nm in names])
    isleg = np.array([any(s in nm for s in leg_sub) for nm in names])

    # restore the LATEST committed V2VNet checkpoint (training may still be writing newer steps)
    v2v = V2VNet(J, J, rngs=nnx.Rngs(0))
    opt = make_v2v_optimizer(v2v, CachedConfig())
    mngr = make_manager(a.ckpt_dir)
    v2v, opt, step = restore_latest(mngr, v2v, opt)
    v2v.eval()
    print(f"restored V2VNet @ step {step}")

    cache = load_cache(a.cache_dir, "val")
    vol, kp3d, c3d, vis = cache["volumes"], cache["kp3d"], cache["center3D"], cache["vis"]
    # pick framesets with the most visible wing verts (so there's a wing to judge)
    wing_vis_ct = vis[:, nkp:][:, iswing].sum(1)
    order = np.argsort(-wing_vis_ct)
    idxs = sorted(order[:a.n].tolist())

    eval_step = make_eval_step_jitted(grid_spacing=1, roi_cube=48, sharpen=a.sharpen)
    vn = jnp.asarray(np.asarray(vol[idxs]))            # (n,J,48,48,48) f16
    cn = jnp.asarray(c3d[idxs])
    pred = np.asarray(eval_step(v2v, vn, cn))          # (n,J,3) world
    gt = kp3d[idxs]; visn = vis[idxs]

    def _spread(P):
        return float(np.nanmedian(np.linalg.norm(P - np.nanmean(P, 0), axis=1))) if len(P) >= 2 else np.nan

    def _stats(mask):                                   # (spread_ratio, pred_sp, gt_sp, mpjpe) over vis=1
        sp, sg, er = [], [], []
        for b in range(len(idxs)):
            mm = mask & visn[b, nkp:]
            sp.append(_spread(pred[b, nkp:][mm])); sg.append(_spread(gt[b, nkp:][mm]))
            if mm.any():
                er.append(np.linalg.norm(pred[b, nkp:][mm] - gt[b, nkp:][mm], axis=1))
        p, g = float(np.nanmedian(sp)), float(np.nanmedian(sg))
        return (p / g if g else np.nan), p, g, (float(np.median(np.concatenate(er))) if er else np.nan)
    ratio, rp, rg, wing_mpjpe = _stats(iswing)
    leg_ratio, lp, lg, leg_mpjpe = _stats(isleg)

    c = (a.n + 1) // 2
    fig = plt.figure(figsize=(4.4 * c, 9))
    for k, b in enumerate(range(len(idxs))):
        ax = fig.add_subplot(2, c, k + 1, projection="3d")
        bodym = (~iswing) & (~isleg) & visn[b, nkp:]
        legm = isleg & visn[b, nkp:]; wm = iswing & visn[b, nkp:]
        P = pred[b, nkp:]; G = gt[b, nkp:]
        ax.scatter(P[bodym, 0], P[bodym, 1], P[bodym, 2], s=6, c="0.6", linewidths=0)
        ax.scatter(P[legm, 0], P[legm, 1], P[legm, 2], s=7, c="dodgerblue", linewidths=0)
        ax.scatter(G[wm, 0], G[wm, 1], G[wm, 2], s=26, facecolors="none",
                   edgecolors="lime", linewidths=0.6, label="GT wing")
        ax.scatter(P[wm, 0], P[wm, 1], P[wm, 2], s=12, c="red", linewidths=0, label="pred wing")
        ax.set_title(f"fs {idxs[b]}", fontsize=9); ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
    fig.suptitle(f"V2VNet-{J} @step{step} 3-D (pred wing=red, GT wing=lime, leg=blue, body=gray) | "
                 f"3D MPJPE wing {wing_mpjpe:.2f} / leg {leg_mpjpe:.2f} | "
                 f"3D spread pred/GT  wing {ratio:.2f}  leg {leg_ratio:.2f}   [2D: wing 0.79 leg 0.96]")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  3D MPJPE wing {wing_mpjpe:.2f}/leg {leg_mpjpe:.2f}  "
          f"wing_spread {rp:.2f}/{rg:.2f} ({ratio:.2f})  leg_spread {lp:.2f}/{lg:.2f} ({leg_ratio:.2f})")


if __name__ == "__main__":
    main()
