"""Render the STAC-posed V1 mesh into every camera and compare to the SAM mask.

This validates the full geometry chain for silhouette IK:
  qpos --(mjx FK)--> model-frame verts --(Umeyama via 50 markers)--> calibrated mm
       --(P_c)--> pixels,  overlaid on each camera image + its SAM mask.
Body/leg verts cyan, wing verts red.  Shows whether the point-fit (STAC) wings fill
the SAM silhouette or fall short — the baseline the silhouette term will correct.
Anatomy-agnostic: pass any (xml, canonical-mesh) pair.
"""
import argparse, json, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def umeyama(src, dst):
    import numpy as np
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    cov = Dc.T @ Sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((Sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ik-h5", required=True)
    ap.add_argument("--bout-h5", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--frame", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np, jax.numpy as jnp, h5py
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose

    anat = load_anatomy(a.xml, a.mesh); fk = make_fk_repose(anat)
    names = [ (n.decode() if isinstance(n, bytes) else n).lower() for n in anat["seg_names"] ]
    id2nm = {int(s): names[i] for i, s in enumerate(anat["seg_ids"])}
    vseg = anat["vertex_segment"]
    iswing = np.array([("wing" in id2nm[int(s)]) for s in vseg])

    ik = ioh5.load(a.ik_h5)
    qpos = np.asarray(ik["qpos"]); marker_sites = np.asarray(ik["marker_sites"])
    ik_kpnames = [ (n.decode() if isinstance(n, bytes) else n) for n in ik["kp_names"] ]
    with h5py.File(a.bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()]
    coco = json.load(open(os.path.join(a.root, "annotations", f"instances_{a.split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    coco_kpnames = coco["keypoint_names"]
    rt = ReprojectionTool(os.path.join(a.root, "calib_params", a.recording))
    cam_names = list(rt.cameras.keys())

    # pick frame: 'auto' = max wing spread in FK (extended wing)
    if a.frame == "auto":
        wsd = []
        for i in range(len(qpos)):
            w = np.asarray(fk(jnp.asarray(qpos[i].astype(np.float32))))[iswing]
            wsd.append(np.linalg.norm(w - w.mean(0), axis=1).mean())
        fr = int(np.argmax(wsd))
    else:
        fr = int(a.frame)
    print(f"frame {fr}/{len(qpos)}")

    # model->mm via Umeyama(marker_sites -> triangulated coco kp), matched by name
    ids = fs_imgids[fr]
    cam2img = {}
    for iid in ids:
        fn = id2file.get(int(iid), ""); cam = fn.split("/")[1] if "/" in fn else ""
        if cam in cam_names:
            cam2img[cam_names.index(cam)] = int(iid)
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    kp_mm = np.zeros((len(ik_kpnames), 3)); kok = np.zeros(len(ik_kpnames), bool)
    for j, nm in enumerate(ik_kpnames):
        ci = name2coco.get(nm)
        if ci is None:
            continue
        obs = np.zeros((rt.num_cameras, 2)); cams = []
        for c, iid in cam2img.items():
            ann = id2ann.get(iid)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            if kp[ci, 2] > 0:
                obs[c] = kp[ci, :2]; cams.append(c)
        if len(cams) >= 2:
            kp_mm[j] = rt.reconstruct_point(obs, cams_to_use=cams); kok[j] = True
    s, R, t = umeyama(marker_sites[fr][kok], kp_mm[kok])
    print(f"model->mm: scale {s:.4f}  (n_kp {int(kok.sum())})")

    vmodel = np.asarray(fk(jnp.asarray(qpos[fr].astype(np.float32))))   # model frame
    vmm = (s * (R @ vmodel.T).T + t)                                    # calibrated mm

    ncam = rt.num_cameras
    cols = min(4, ncam); rowsN = (ncam + cols - 1) // cols
    fig, ax = plt.subplots(rowsN, cols, figsize=(5.2 * cols, 3.8 * rowsN), squeeze=False)

    def load_mask(fn, ann_id):
        p = os.path.join(a.root, "sam3_masks", a.split, os.path.splitext(fn)[0] + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=True)
        if z["masks"].shape[0] == 0:
            return None
        sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
        if not len(sel):
            sel = np.where(z["ann_ids"] == ann_id)[0]
        return z["masks"][sel[0]].astype(bool) if len(sel) else None

    uv_all = np.stack([rt.reproject_point(vmm[k]) for k in range(0, len(vmm), max(1, len(vmm)//4000))], 1)
    sub = np.arange(0, len(vmm), max(1, len(vmm)//4000)); isw_sub = iswing[sub]
    for c in range(ncam):
        aa = ax.flat[c]
        if c in cam2img:
            fn = id2file[cam2img[c]]
            try:
                aa.imshow(mpimg.imread(os.path.join(a.root, a.split, fn)))
            except Exception:
                pass
            m = load_mask(fn, id2ann[cam2img[c]]["id"])
            if m is not None:
                aa.imshow(np.ma.masked_where(~m, m), alpha=0.3, cmap="autumn")
        uv = uv_all[c]
        aa.scatter(uv[~isw_sub, 0], uv[~isw_sub, 1], s=2, c="cyan", linewidths=0)
        aa.scatter(uv[isw_sub, 0], uv[isw_sub, 1], s=3, c="red", linewidths=0)
        x0, y0 = uv.min(0) - 60; x1, y1 = uv.max(0) + 60
        aa.set_xlim(x0, x1); aa.set_ylim(y1, y0)
        aa.set_title(cam_names[c], fontsize=9); aa.axis("off")
    for c in range(ncam, rowsN * cols):
        ax.flat[c].axis("off")
    fig.suptitle(f"STAC-posed V1 mesh -> all cameras (frame {fr}, {a.recording}) | "
                 f"cyan=body/leg verts, red=wing verts, orange=SAM mask  [does mesh fill the mask?]")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=115)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
