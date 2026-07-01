"""Proof-of-concept: fitting the canonical mesh to the (compressed) 3-D predictions
restores correct wing SIZE, despite the raw per-vertex under-spread.

Pipeline mirrors what STAC-IK does, isolated to the wing for clarity:
  1. Estimate the fly scale `s` from the well-localized BODY verts (Umeyama,
     canonical->pred).  Body is high-contrast so this is reliable.
  2. For each wing (left/right) rigidly fit the canonical wing *at that fixed
     scale* to the predicted wing verts (Kabsch, R/t only).  Size therefore comes
     from the known geometry, orientation from the data.
Reports raw-pred wing spread/GT vs canonical-fit wing spread/GT, and renders
raw (red) vs fitted (orange) vs GT (lime).
"""
import argparse, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def umeyama(src, dst, with_scale=True):
    import numpy as np
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / ((Sc ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


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
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from flax import nnx
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.data.repro_cache import load_cache
    from jarvis_jax.train.checkpoint import make_manager, restore_latest
    from jarvis_jax.train.train_3d_cached import make_eval_step_jitted, make_v2v_optimizer, CachedConfig

    J, nkp = a.num_joints, a.n_kp
    mz = np.load(a.mesh, allow_pickle=True); nv = J - nkp
    fps = mz[f"fps_{nv}"]; seg = mz["vertex_segment"][fps]
    C = mz["vertices"][fps]                                  # (nv,3) canonical reference verts
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    leg_sub = ("coxa", "trochanter", "femur", "tibia", "tarsus", "claw")
    iswing = np.array(["wing" in nm for nm in names])
    isbody = np.array([("wing" not in nm) and not any(x in nm for x in leg_sub) for nm in names])
    wingL = np.array(["wing" in nm and "left" in nm for nm in names])
    wingR = np.array(["wing" in nm and "right" in nm for nm in names])

    v2v = V2VNet(J, J, rngs=nnx.Rngs(0)); opt = make_v2v_optimizer(v2v, CachedConfig())
    v2v, opt, step = restore_latest(make_manager(a.ckpt_dir), v2v, opt); v2v.eval()
    print(f"restored V2VNet @ step {step}")
    cache = load_cache(a.cache_dir, "val")
    vol, kp3d, c3d, vis = cache["volumes"], cache["kp3d"], cache["center3D"], cache["vis"]
    order = np.argsort(-vis[:, nkp:][:, iswing].sum(1)); idxs = sorted(order[:a.n].tolist())
    eval_step = make_eval_step_jitted(grid_spacing=1, roi_cube=48, sharpen=a.sharpen)
    pred = np.asarray(eval_step(v2v, jnp.asarray(np.asarray(vol[idxs])), jnp.asarray(c3d[idxs])))
    gt = kp3d[idxs]; visn = vis[idxs, nkp:]

    def spread(P):
        return float(np.nanmedian(np.linalg.norm(P - np.nanmean(P, 0), axis=1))) if len(P) >= 2 else np.nan

    raw_r, fit_r = [], []
    fitted = [None] * len(idxs)
    for b in range(len(idxs)):
        vv = visn[b]; P = pred[b, nkp:]; G = gt[b, nkp:]
        bodm = isbody & vv
        if bodm.sum() < 4:
            continue
        s, _, _ = umeyama(C[bodm], P[bodm], with_scale=True)   # fly scale from body
        Ffull = np.full_like(C, np.nan)
        for side in (wingL, wingR):
            sm = side & vv
            if sm.sum() < 4:
                continue
            _, R, t = umeyama(s * C[sm], P[sm], with_scale=False)  # Kabsch at fixed scale
            Ffull[side] = (R @ (s * C[side]).T).T + t              # place ALL side verts (incl tip)
        fitted[b] = Ffull
        for side in (wingL, wingR):
            sm = side & vv
            if sm.sum() < 4 or np.isnan(Ffull[side]).all():
                continue
            g = spread(G[sm])
            if g and not np.isnan(g):
                raw_r.append(spread(P[sm]) / g)
                fit_r.append(spread(Ffull[sm]) / g)
    raw_ratio = float(np.nanmedian(raw_r)) if raw_r else float("nan")
    fit_ratio = float(np.nanmedian(fit_r)) if fit_r else float("nan")

    c = (a.n + 1) // 2
    fig = plt.figure(figsize=(4.4 * c, 9))
    for k, b in enumerate(range(len(idxs))):
        ax = fig.add_subplot(2, c, k + 1, projection="3d")
        vv = visn[b]; P = pred[b, nkp:]; G = gt[b, nkp:]; wm = iswing & vv
        ax.scatter(P[(isbody | (np.array([any(x in n for x in leg_sub) for n in names]))) & vv, 0],
                   P[(isbody | (np.array([any(x in n for x in leg_sub) for n in names]))) & vv, 1],
                   P[(isbody | (np.array([any(x in n for x in leg_sub) for n in names]))) & vv, 2],
                   s=5, c="0.7", linewidths=0)
        ax.scatter(G[wm, 0], G[wm, 1], G[wm, 2], s=26, facecolors="none", edgecolors="lime", linewidths=0.6)
        ax.scatter(P[wm, 0], P[wm, 1], P[wm, 2], s=10, c="red", linewidths=0)
        if fitted[b] is not None:
            Fw = fitted[b][iswing]; m = ~np.isnan(Fw).any(1)
            ax.scatter(Fw[m, 0], Fw[m, 1], Fw[m, 2], s=12, c="orange", linewidths=0)
        ax.set_title(f"fs {idxs[b]}", fontsize=9); ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.set_ticklabels([])
    fig.suptitle(f"Canonical-mesh fit to compressed 3-D preds @V2VNet step{step} | "
                 f"wing spread/GT  RAW pred {raw_ratio:.2f}  ->  MESH-FIT {fit_ratio:.2f}  "
                 f"(red=raw pred, orange=mesh-fit, lime=GT)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  RAW wing spread/GT {raw_ratio:.3f}  ->  MESH-FIT {fit_ratio:.3f}")


if __name__ == "__main__":
    main()
