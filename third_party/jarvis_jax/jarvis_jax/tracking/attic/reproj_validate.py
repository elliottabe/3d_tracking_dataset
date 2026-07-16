"""Multi-view reprojection validation of the 3-D dense pose + SAM wing-tip.

For an extended-wing frameset, project into ALL cameras and overlay on each image:
  * predicted 50 keypoints (cyan)        vs annotated 2-D keypoints (lime +)
  * raw V2VNet wing tip (red x)          -> reprojects short (the compression)
  * SAM-triangulated wing tip (orange)   -> should land on the wing edge in EVERY view
Reports per-camera keypoint reprojection error, and a LEAVE-ONE-OUT wing-tip test:
triangulate the wing tip from the other cameras, reproject into the held-out one,
measure the pixel error vs that camera's own SAM-mask tip (true generalization).
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
    ap.add_argument("--fs-rank", type=int, default=0, help="which extended-wing frameset (0=most wing-visible)")
    ap.add_argument("--corridor", type=float, default=12.0)
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
    mz = np.load(a.mesh, allow_pickle=True); nv = J - nkp
    fps = mz[f"fps_{nv}"]; seg = mz["vertex_segment"][fps]; C = mz["vertices"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n) for s, n in zip(mz["seg_ids"], mz["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    thorax_c = C[[("thorax" in n) for n in names]].mean(0)
    sides = {}
    for side in ("left", "right"):
        si = np.where([("wing" in n and side in n) for n in names])[0]
        if len(si):
            d = np.linalg.norm(C[si] - thorax_c, axis=1)
            sides[side] = dict(tip=int(si[d.argmax()]), prox=int(si[d.argmin()]))

    coco = json.load(open(os.path.join(a.root, "annotations", "instances_val.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    src_kpnames = coco["keypoint_names"]

    ds = CSEFramesetDataset(a.root, "val", a.aux)
    cache = load_cache(a.cache_dir, "val")
    vol, kp3d, c3d, vis = cache["volumes"], cache["kp3d"], cache["center3D"], cache["vis"]
    v2v = V2VNet(J, J, rngs=nnx.Rngs(0)); opt = make_v2v_optimizer(v2v, CachedConfig())
    v2v, opt, step = restore_latest(make_manager(a.ckpt_dir), v2v, opt); v2v.eval()

    iswing = np.array([("wing" in n) for n in names])
    fs = int(np.argsort(-vis[:, nkp:][:, iswing].sum(1))[a.fs_rank])
    eval_step = make_eval_step_jitted(grid_spacing=1, roi_cube=48, sharpen=a.sharpen)
    pred = np.asarray(eval_step(v2v, jnp.asarray(np.asarray(vol[fs:fs+1])), jnp.asarray(c3d[fs:fs+1])))[0]
    rec = ds.framesets[fs]["datasetName"]; frames = ds.framesets[fs]["frames"]
    rt = ReprojectionTool(os.path.join(a.root, "calib_params", rec))
    cam_names = list(rt.cameras.keys())
    cam2img = {}
    for iid in frames:
        fn = id2file.get(int(iid), ""); cam = fn.split("/")[1] if "/" in fn else ""
        if cam in cam_names:
            cam2img[cam_names.index(cam)] = int(iid)

    def load_mask(fn, ann_id):
        p = os.path.join(a.root, "sam3_masks", "val", os.path.splitext(fn)[0] + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        if z["masks"].shape[0] == 0:
            return None
        sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
        if not len(sel):
            sel = np.where(z["ann_ids"] == ann_id)[0]
        return z["masks"][sel[0]].astype(bool) if len(sel) else None

    def sam_tip2d(c, side):                                  # per-camera SAM mask tip for a wing side
        if c not in cam2img:
            return None
        fn = id2file[cam2img[c]]; ann = id2ann.get(cam2img[c])
        if ann is None:
            return None
        mask = load_mask(fn, ann["id"])
        if mask is None:
            return None
        prox2d = rt.reproject_point(pred[nkp + sides[side]["prox"]])[c]
        ptip2d = rt.reproject_point(pred[nkp + sides[side]["tip"]])[c]
        ax = ptip2d - prox2d; L = np.linalg.norm(ax)
        if L < 5:
            return None
        ax = ax / L
        ys, xs = np.where(mask)
        rel = np.stack([xs - prox2d[0], ys - prox2d[1]], 1)
        along = rel @ ax; perp = np.abs(rel[:, 0] * (-ax[1]) + rel[:, 1] * ax[0])
        corr = (perp < a.corridor) & (along > 0.3 * L)
        if corr.sum() < 3:
            return None
        k = np.argmax(along[corr])
        return np.array([xs[corr][k], ys[corr][k]], float)

    # SAM 3D wing tip per side (all cams) + leave-one-out reprojection error
    sam3d = {}; loo_err = []
    for side in sides:
        tips = {c: sam_tip2d(c, side) for c in range(rt.num_cameras)}
        tips = {c: t for c, t in tips.items() if t is not None}
        if len(tips) >= 2:
            pts = np.zeros((rt.num_cameras, 2))
            for c, t in tips.items():
                pts[c] = t
            sam3d[side] = rt.reconstruct_point(pts, cams_to_use=list(tips))
            for c in tips:                                   # leave-one-out
                others = [k for k in tips if k != c]
                if len(others) < 2:
                    continue
                X = rt.reconstruct_point(pts, cams_to_use=others)
                proj = rt.reproject_point(X)[c]
                loo_err.append(float(np.linalg.norm(proj - tips[c])))

    # reproject everything into every camera
    proj_kp = np.stack([rt.reproject_point(pred[j]) for j in range(nkp)], 1)   # (nc, nkp, 2)
    reorder = {n: i for i, n in enumerate(src_kpnames)}
    kp_err = {}
    ncam = rt.num_cameras
    cols = min(4, ncam); rowsN = (ncam + cols - 1) // cols
    fig, ax = plt.subplots(rowsN, cols, figsize=(5.2 * cols, 4.0 * rowsN), squeeze=False)
    for c in range(ncam):
        aa = ax.flat[c]
        if c in cam2img:
            fn = id2file[cam2img[c]]
            try:
                aa.imshow(mpimg.imread(os.path.join(a.root, "val", fn)))
            except Exception:
                pass
            ann = id2ann.get(cam2img[c])
            ann_kp = np.asarray(ann["keypoints"], float).reshape(-1, 3) if ann else None
        else:
            ann_kp = None
        aa.scatter(proj_kp[c, :, 0], proj_kp[c, :, 1], s=10, c="cyan", linewidths=0)   # pred kp
        if ann_kp is not None:
            v = ann_kp[:, 2] > 0
            aa.scatter(ann_kp[v, 0], ann_kp[v, 1], s=26, marker="+", c="lime", linewidths=0.7)
            # order-robust reproj error: NN distance from each projected pred kp to the
            # annotated-kp set (avoids any model/src ordering mismatch).
            if v.sum():
                from scipy.spatial import cKDTree
                d, _ = cKDTree(ann_kp[v, :2]).query(proj_kp[c])
                kp_err[cam_names[c]] = float(np.median(d))
        for side in sides:
            rawtip = rt.reproject_point(pred[nkp + sides[side]["tip"]])[c]
            aa.scatter(*rawtip, s=45, marker="x", c="red", linewidths=1.5)
            st = sam_tip2d(c, side)
            if st is not None:
                aa.scatter(*st, s=45, c="orange", linewidths=0)
            if side in sam3d:
                pt = rt.reproject_point(sam3d[side])[c]
                aa.scatter(*pt, s=70, facecolors="none", edgecolors="orange", linewidths=1.5)
        # crop tightly to the annotated fly (these are 2-fly courtship frames full-frame)
        allpts = [proj_kp[c]]
        for side in sides:
            allpts.append(rt.reproject_point(pred[nkp + sides[side]["tip"]])[c][None])
        P = np.concatenate(allpts, 0)
        x0, y0 = P.min(0) - 70; x1, y1 = P.max(0) + 70
        aa.set_xlim(x0, x1); aa.set_ylim(y1, y0)
        ke = kp_err.get(cam_names[c], float("nan"))
        aa.set_title(f"{cam_names[c]}  kp-reproj {ke:.1f}px", fontsize=9); aa.axis("off")
    for c in range(ncam, rowsN * cols):
        ax.flat[c].axis("off")
    mk = np.nanmedian(list(kp_err.values())) if kp_err else float("nan")
    ml = np.median(loo_err) if loo_err else float("nan")
    fig.suptitle(f"Multi-view reprojection (fs{fs}, {ncam} cams) | median kp-reproj {mk:.1f}px | "
                 f"SAM wing-tip leave-one-out reproj {ml:.1f}px | "
                 f"cyan=pred kp, lime+=annot kp, red x=raw wing tip, orange=SAM tip (filled=this cam, ring=triangulated)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=115)
    print(f"wrote {a.out}")
    print(f"median kp reproj err: {mk:.2f}px | per-cam:", {k: round(v, 1) for k, v in kp_err.items()})
    print(f"SAM wing-tip leave-one-out reproj err: {ml:.2f}px  (n={len(loo_err)})")


if __name__ == "__main__":
    main()
