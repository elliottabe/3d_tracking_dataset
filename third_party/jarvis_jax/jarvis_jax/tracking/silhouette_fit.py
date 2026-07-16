"""Differentiable multi-view soft-silhouette fit (V1, anatomy-agnostic).

Renders the posed mesh's soft silhouette per camera (windowed-Gaussian splat of the
reposed verts) and fits qpos so the silhouettes match the SAM masks across all cameras.

Controlled test (--mode recover): take STAC qpos, PERTURB the wing joints, then
optimize ONLY the wing joints back from the multi-view silhouette -> proves the
differentiable silhouette-IK loop (reports wing-joint recovery + soft-IoU before/after).
"""
import argparse, json, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def umeyama(src, dst):
    import numpy as np
    mu_s, mu_d = src.mean(0), dst.mean(0); Sc, Dc = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Dc.T @ Sc / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt; s = np.trace(np.diag(D) @ S) / ((Sc ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True); ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--recording", required=True)
    ap.add_argument("--ik-h5", required=True); ap.add_argument("--bout-h5", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--frame", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--G", type=int, default=110); ap.add_argument("--sigma", type=float, default=1.3)
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--perturb", type=float, default=0.6, help="rad added to wing joints for the recovery test")
    a = ap.parse_args()

    import numpy as np, jax, jax.numpy as jnp, h5py, optax, mujoco
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.silhouette_ik import load_anatomy, make_fk_repose

    anat = load_anatomy(a.xml, a.mesh); fk = make_fk_repose(anat); m = anat["m"]
    names = [(n.decode() if isinstance(n, bytes) else n).lower() for n in anat["seg_names"]]
    id2nm = {int(s): names[i] for i, s in enumerate(anat["seg_ids"])}
    iswing = np.array([("wing" in id2nm[int(s)]) for s in anat["vertex_segment"]])
    # wing qpos indices (joints whose body name contains 'wing')
    wing_qpos = []
    for j in range(m.njnt):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]) or ""
        if "wing" in bn.lower():
            adr = m.jnt_qposadr[j]
            n = {mujoco.mjtJoint.mjJNT_HINGE: 1, mujoco.mjtJoint.mjJNT_SLIDE: 1,
                 mujoco.mjtJoint.mjJNT_BALL: 4, mujoco.mjtJoint.mjJNT_FREE: 7}[m.jnt_type[j]]
            wing_qpos += list(range(adr, adr + n))
    wing_qpos = np.array(wing_qpos); print("wing qpos idx:", wing_qpos)

    ik = ioh5.load(a.ik_h5); qpos = np.asarray(ik["qpos"]); marker_sites = np.asarray(ik["marker_sites"])
    ik_kpnames = [(n.decode() if isinstance(n, bytes) else n) for n in ik["kp_names"]]
    with h5py.File(a.bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()]
    coco = json.load(open(os.path.join(a.root, "annotations", f"instances_{a.split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    name2coco = {n: i for i, n in enumerate(coco["keypoint_names"])}
    rt = ReprojectionTool(os.path.join(a.root, "calib_params", a.recording)); cams = list(rt.cameras.keys())

    if a.frame == "auto":
        wsd = [np.linalg.norm((w := np.asarray(fk(jnp.asarray(qpos[i].astype(np.float32))))[iswing]) - w.mean(0), axis=1).mean()
               for i in range(len(qpos))]
        fr = int(np.argmax(wsd))
    else:
        fr = int(a.frame)
    print("frame", fr)

    ids = fs_imgids[fr]; cam2img = {}
    for iid in ids:
        fn = id2file.get(int(iid), ""); cn = fn.split("/")[1] if "/" in fn else ""
        if cn in cams:
            cam2img[cams.index(cn)] = int(iid)
    # model->mm via markers
    kp_mm = np.zeros((len(ik_kpnames), 3)); kok = np.zeros(len(ik_kpnames), bool)
    for j, nm in enumerate(ik_kpnames):
        ci = name2coco.get(nm)
        if ci is None: continue
        obs = np.zeros((rt.num_cameras, 2)); cc = []
        for c, iid in cam2img.items():
            ann = id2ann.get(iid)
            if ann is None: continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            if kp[ci, 2] > 0: obs[c] = kp[ci, :2]; cc.append(c)
        if len(cc) >= 2: kp_mm[j] = rt.reconstruct_point(obs, cams_to_use=cc); kok[j] = True
    s, R, t = umeyama(marker_sites[fr][kok], kp_mm[kok])
    sR = jnp.asarray(s * R); tt = jnp.asarray(t)
    print(f"model->mm scale {s:.3f}")

    def load_mask(fn, ann_id):
        p = os.path.join(a.root, "sam3_masks", a.split, os.path.splitext(fn)[0] + ".npz")
        if not os.path.exists(p): return None
        z = np.load(p, allow_pickle=True)
        if z["masks"].shape[0] == 0: return None
        sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
        if not len(sel): sel = np.where(z["ann_ids"] == ann_id)[0]
        return z["masks"][sel[0]].astype(np.float32) if len(sel) else None

    # per-camera: projection matrix, bbox (from init verts), SAM target on the bbox grid
    G = a.G
    v0 = np.asarray(fk(jnp.asarray(qpos[fr].astype(np.float32)))) @ (s * R).T + t
    cam_data = []
    for c in cam2img:
        P = jnp.asarray(rt._camera_list[c].cameraMatrix)            # (3,4)
        uv0 = np.asarray(rt.reproject_point(v0[::20]).T if False else np.stack([rt.reproject_point(v0[k]) for k in range(0, len(v0), 50)], 0)[:, c])
        x0, y0 = uv0.min(0) - 55; x1, y1 = uv0.max(0) + 55
        mask = load_mask(id2file[cam2img[c]], id2ann[cam2img[c]]["id"])
        if mask is None: continue
        gy, gx = np.meshgrid(np.linspace(y0, y1, G), np.linspace(x0, x1, G), indexing="ij")
        H, W = mask.shape
        tgt = mask[np.clip(gy.astype(int), 0, H-1), np.clip(gx.astype(int), 0, W-1)]
        cam_data.append(dict(P=P, x0=float(x0), y0=float(y0),
                             sgx=float(G/(x1-x0)), sgy=float(G/(y1-y0)),
                             tgt=jnp.asarray(tgt.astype(np.float32)), c=c, fn=id2file[cam2img[c]]))
    print(f"cameras used: {len(cam_data)}")

    K = 2
    offs = [(di, dj) for di in range(-K, K+1) for dj in range(-K, K+1)]
    def soft_sil(vmm, cd):
        ph = vmm @ cd["P"][:, :3].T + cd["P"][:, 3]
        uv = ph[:, :2] / ph[:, 2:3]
        gx = (uv[:, 0] - cd["x0"]) * cd["sgx"]; gy = (uv[:, 1] - cd["y0"]) * cd["sgy"]
        bx = jnp.floor(gx).astype(jnp.int32); by = jnp.floor(gy).astype(jnp.int32)
        grid = jnp.zeros((G * G,))
        for di, dj in offs:
            cx = bx + dj; cy = by + di
            w = jnp.exp(-(((cx + 0.5 - gx) ** 2 + (cy + 0.5 - gy) ** 2)) / (2 * a.sigma ** 2))
            valid = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G)
            idx = jnp.clip(cy, 0, G-1) * G + jnp.clip(cx, 0, G-1)
            grid = grid.at[idx].add(jnp.where(valid, w, 0.0))
        return 1.0 - jnp.exp(-grid.reshape(G, G))

    q_full = jnp.asarray(qpos[fr].astype(np.float32))
    widx = jnp.asarray(wing_qpos)
    def render_all(qw):
        q = q_full.at[widx].set(qw)
        vmm = fk(q) @ sR.T + tt
        return [soft_sil(vmm, cd) for cd in cam_data]
    def loss_fn(qw):
        L = 0.0
        for r, cd in zip(render_all(qw), cam_data):
            I = (r * cd["tgt"]).sum(); U = (r + cd["tgt"] - r * cd["tgt"]).sum()
            L = L + (1.0 - I / (U + 1e-6))
        return L / len(cam_data)

    qw_true = q_full[widx]
    iou0_each = lambda qw: [float((r*cd["tgt"]).sum()/((r+cd["tgt"]-r*cd["tgt"]).sum()+1e-6)) for r,cd in zip(render_all(qw),cam_data)]
    rng = np.random.default_rng(0)
    qw_init = qw_true + jnp.asarray(rng.normal(0, a.perturb, size=widx.shape[0]).astype(np.float32))
    print(f"IoU at TRUE wing qpos:      {np.mean(iou0_each(qw_true)):.3f}")
    print(f"IoU at PERTURBED wing qpos: {np.mean(iou0_each(qw_init)):.3f}")

    opt = optax.adam(a.lr); qw = qw_init; st = opt.init(qw)
    lossfn = jax.jit(jax.value_and_grad(loss_fn))
    for it in range(a.steps):
        l, g = lossfn(qw); upd, st = opt.update(g, st); qw = optax.apply_updates(qw, upd)
        if (it+1) % 30 == 0: print(f"  step {it+1} loss {float(l):.4f}")
    print(f"IoU after FIT:              {np.mean(iou0_each(qw)):.3f}")
    print(f"wing-qpos recovery: ||init-true|| {float(np.linalg.norm(np.asarray(qw_init-qw_true))):.3f} -> "
          f"||fit-true|| {float(np.linalg.norm(np.asarray(qw-qw_true))):.3f} rad")

    # viz before/after on up to 3 cams
    show = cam_data[:3]
    fig, ax = plt.subplots(3, len(show), figsize=(4.2*len(show), 11))
    rper = render_all(qw_init); rfit = render_all(qw)
    for k, cd in enumerate(show):
        ext = [cd["x0"], cd["x0"]+G/cd["sgx"], cd["y0"]+G/cd["sgy"], cd["y0"]]
        for row, (title, R_) in enumerate([("SAM target", cd["tgt"]), ("perturbed", rper[k]), ("fit", rfit[k])]):
            aa = ax[row, k] if len(show) > 1 else ax[row]
            try: aa.imshow(mpimg.imread(os.path.join(a.root, a.split, cd["fn"])))
            except Exception: pass
            aa.imshow(np.asarray(R_), alpha=0.5, cmap="hot", extent=ext)
            aa.set_xlim(ext[0], ext[1]); aa.set_ylim(ext[2], ext[3])
            aa.set_title(f"{cams[cd['c']]} {title}", fontsize=8); aa.axis("off")
    fig.suptitle(f"Silhouette-IK wing recovery (frame {fr}) | IoU true {np.mean(iou0_each(qw_true)):.2f} "
                 f"perturbed {np.mean(iou0_each(qw_init)):.2f} -> fit {np.mean(iou0_each(qw)):.2f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=115); print("wrote", a.out)


if __name__ == "__main__":
    main()
