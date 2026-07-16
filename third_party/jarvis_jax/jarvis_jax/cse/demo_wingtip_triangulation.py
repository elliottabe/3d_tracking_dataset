"""POC: recover the true 3-D wing tip from SAM silhouettes by multi-view triangulation.

For a few extended-wing val framesets:
  * GT wing tip          = cache 3-D label for the distal wing vertex (per side)
  * predicted wing tip   = V2VNet 3-D prediction for that vertex (compressed inboard)
  * SAM wing tip         = per camera, march along the predicted wing axis to the
                           far edge of the fly's SAM mask -> 2-D tip; triangulate
                           those across cameras (DLT) -> 3-D tip.
Reports, per wing, the 3-D distance of the predicted vs SAM tip from the GT tip,
and the wing-length ratio pred/GT vs SAM/GT.  If SAM >> pred toward 1.0, the mask
silhouette recovers the extent the heatmaps miss.  cache/pred/GT and rt triangulation
all live in the same mm world frame, so distances are directly comparable.
"""
import argparse, json, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=350)
    ap.add_argument("--n-kp", type=int, default=50)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--corridor", type=float, default=12.0)   # px perpendicular half-width
    ap.add_argument("--sharpen", type=float, default=3.0)
    a = ap.parse_args()

    import numpy as np, jax.numpy as jnp
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from flax import nnx
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.data.repro_cache import load_cache
    from jarvis_jax.train.checkpoint import make_manager, restore_latest
    from jarvis_jax.train.train_3d_cached import make_eval_step_jitted, make_v2v_optimizer, CachedConfig
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.densepose.cse_dataset import CSEFramesetDataset

    J, nkp = a.num_joints, a.n_kp
    # --- canonical mesh: wing tip/proximal vertices per side ---
    mz = np.load(a.mesh, allow_pickle=True); nv = J - nkp
    fps = mz[f"fps_{nv}"]; seg = mz["vertex_segment"][fps]; C = mz["vertices"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    thorax_c = C[[("thorax" in n) for n in names]].mean(0)
    sides = {}
    for side in ("left", "right"):
        m = np.array([("wing" in n and side in n) for n in names])
        si = np.where(m)[0]
        if not len(si):
            continue
        d = np.linalg.norm(C[si] - thorax_c, axis=1)
        sides[side] = dict(tip=int(si[d.argmax()]), prox=int(si[d.argmin()]))
    print("wing tip/prox fps-idx:", sides)

    # --- coco: image_id -> (file, ann_id) ---
    coco = json.load(open(os.path.join(a.root, "annotations", "instances_val.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}

    # --- frameset dataset (aligned with cache order) + cache GT/pred ---
    ds = CSEFramesetDataset(a.root, "val", a.aux)
    cache = load_cache(a.cache_dir, "val")
    vol, kp3d, c3d, vis = cache["volumes"], cache["kp3d"], cache["center3D"], cache["vis"]
    assert len(ds) == vol.shape[0], f"frameset/cache mismatch {len(ds)} vs {vol.shape[0]}"
    v2v = V2VNet(J, J, rngs=nnx.Rngs(0)); opt = make_v2v_optimizer(v2v, CachedConfig())
    v2v, opt, step = restore_latest(make_manager(a.ckpt_dir), v2v, opt); v2v.eval()
    print(f"restored V2VNet @ step {step}")

    iswing = np.array([("wing" in n) for n in names])
    order = np.argsort(-vis[:, nkp:][:, iswing].sum(1))
    chosen = [int(i) for i in order[:a.n]]
    eval_step = make_eval_step_jitted(grid_spacing=1, roi_cube=48, sharpen=a.sharpen)
    pred = np.asarray(eval_step(v2v, jnp.asarray(np.asarray(vol[chosen])), jnp.asarray(c3d[chosen])))

    rt_cache = {}
    def get_rt(rec):
        if rec not in rt_cache:
            rt_cache[rec] = ReprojectionTool(os.path.join(a.root, "calib_params", rec))
        return rt_cache[rec]

    def load_mask(file_name, ann_id):
        p = os.path.join(a.root, "sam3_masks", "val", os.path.splitext(file_name)[0] + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        if z["masks"].shape[0] == 0:
            return None
        sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
        if not len(sel):
            sel = np.where(z["ann_ids"] == ann_id)[0]
        return z["masks"][sel[0]].astype(bool) if len(sel) else None

    rows = []
    panels = []
    for pi, fs in enumerate(chosen):
        rec = ds.framesets[fs]["datasetName"]; frames = ds.framesets[fs]["frames"]
        rt = get_rt(rec); cam_names = list(rt.cameras.keys())
        # map rt camera index -> this frameset's image_id (by cam name in file path)
        cam2img = {}
        for iid in frames:
            fn = id2file.get(int(iid), "")
            cam = fn.split("/")[1] if "/" in fn else ""
            if cam in cam_names:
                cam2img[cam_names.index(cam)] = int(iid)
        for side, idx in sides.items():
            tj, pj = nkp + idx["tip"], nkp + idx["prox"]
            if not (vis[fs, tj] and vis[fs, pj]):
                continue
            GT_tip, prox3d = kp3d[fs, tj], pred[pi, pj]
            pred_tip = pred[pi, tj]
            ptip2d = rt.reproject_point(pred_tip); prox2d = rt.reproject_point(prox3d)
            gtip2d = rt.reproject_point(GT_tip)
            sam_pts = np.zeros((rt.num_cameras, 2)); cams_used = []; per_cam = {}
            for c in range(rt.num_cameras):
                if c not in cam2img:
                    continue
                fn = id2file[cam2img[c]]; ann = id2ann.get(cam2img[c])
                if ann is None:
                    continue
                mask = load_mask(fn, ann["id"])
                if mask is None:
                    continue
                ax = ptip2d[c] - prox2d[c]; L = np.linalg.norm(ax)
                if L < 5:
                    continue
                ax = ax / L
                ys, xs = np.where(mask)
                rel = np.stack([xs - prox2d[c, 0], ys - prox2d[c, 1]], 1)
                along = rel @ ax; perp = np.abs(rel[:, 0] * (-ax[1]) + rel[:, 1] * ax[0])
                corr = (perp < a.corridor) & (along > 0.3 * L)
                if corr.sum() < 3:
                    continue
                k = np.argmax(along[corr])
                tip2d = np.array([xs[corr][k], ys[corr][k]], float)
                sam_pts[c] = tip2d; cams_used.append(c)
                per_cam[c] = dict(tip2d=tip2d, fn=fn, ann=ann, along=float(along[corr][k]))
            if len(cams_used) < 2:
                continue
            SAM_tip = rt.reconstruct_point(sam_pts, cams_to_use=cams_used)
            len_gt = np.linalg.norm(GT_tip - prox3d)
            row = dict(fs=fs, side=side,
                       pred_err=float(np.linalg.norm(pred_tip - GT_tip)),
                       sam_err=float(np.linalg.norm(SAM_tip - GT_tip)),
                       len_pred=float(np.linalg.norm(pred_tip - prox3d) / len_gt),
                       len_sam=float(np.linalg.norm(SAM_tip - prox3d) / len_gt),
                       ncam=len(cams_used))
            rows.append(row)
            # choose the camera with the largest wing extent for the panel
            cbest = max(per_cam, key=lambda c: per_cam[c]["along"])
            panels.append((rec, per_cam[cbest], prox2d[cbest], ptip2d[cbest], gtip2d[cbest], row))

    # --- report ---
    if rows:
        pe = np.median([r["pred_err"] for r in rows]); se = np.median([r["sam_err"] for r in rows])
        lp = np.median([r["len_pred"] for r in rows]); ls = np.median([r["len_sam"] for r in rows])
        print(f"\n=== WING-TIP TRIANGULATION ({len(rows)} wings) ===")
        for r in rows:
            print(f"  fs{r['fs']:3d} {r['side']:5s} ncam{r['ncam']}  tip-err pred {r['pred_err']:.2f} "
                  f"SAM {r['sam_err']:.2f}  | wing-len pred/GT {r['len_pred']:.2f}  SAM/GT {r['len_sam']:.2f}")
        print(f"  MEDIAN  tip-err pred {pe:.2f} -> SAM {se:.2f}   wing-len pred/GT {lp:.2f} -> SAM/GT {ls:.2f}")
    else:
        pe = se = lp = ls = float("nan"); print("no triangulable wings found")

    # --- render panels (mask + reprojected tips) ---
    n = max(1, len(panels)); cols = min(4, n); rowsN = (n + cols - 1) // cols
    fig, ax = plt.subplots(rowsN, cols, figsize=(5 * cols, 4.4 * rowsN), squeeze=False)
    for k, (rec, pc, prox2d, ptip2d, gtip2d, r) in enumerate(panels):
        aa = ax.flat[k]
        try:
            img = mpimg.imread(os.path.join(a.root, "val", pc["fn"]))
            aa.imshow(img)
        except Exception:
            pass
        m = load_mask(pc["fn"], pc["ann"]["id"])
        if m is not None:
            aa.imshow(np.ma.masked_where(~m, m), alpha=0.3, cmap="cool")
        aa.plot([prox2d[0], ptip2d[0]], [prox2d[1], ptip2d[1]], "-", c="red", lw=1)
        aa.scatter(*ptip2d, c="red", s=40, marker="x", label="pred tip")
        aa.scatter(*pc["tip2d"], c="orange", s=40, label="SAM tip")
        aa.scatter(*gtip2d, c="lime", s=40, marker="+", label="GT tip")
        allx = [prox2d[0], ptip2d[0], pc["tip2d"][0], gtip2d[0]]
        ally = [prox2d[1], ptip2d[1], pc["tip2d"][1], gtip2d[1]]
        aa.set_xlim(min(allx) - 80, max(allx) + 80); aa.set_ylim(max(ally) + 80, min(ally) - 80)
        aa.set_title(f"fs{r['fs']} {r['side']}: len pred/GT {r['len_pred']:.2f} SAM/GT {r['len_sam']:.2f}",
                     fontsize=9)
        if k == 0:
            aa.legend(fontsize=7, loc="lower right")
    for k in range(len(panels), rowsN * cols):
        ax.flat[k].axis("off")
    fig.suptitle(f"SAM wing-tip triangulation @V2VNet step{step} | wing-len pred/GT {lp:.2f} -> SAM/GT {ls:.2f} | "
                 f"tip-err pred {pe:.2f} -> SAM {se:.2f} mm  (red x=pred, orange=SAM-mask, lime +=GT)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
