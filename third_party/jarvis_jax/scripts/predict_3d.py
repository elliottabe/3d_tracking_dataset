"""Hydra entrypoint: batched JAX 3D inference + V3-val validation + npz output."""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, run_dir_for
register_resolvers()


def run_predict(*, root, split, vitpose_ckpt, v2v_final, out, sharpen, batch,
                limit, validate):
    from jarvis_jax.data.v3_3d import V3FramesetDataset
    from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch
    from jarvis_jax.geometry.center3d import estimate_center3d_from_masks
    from jarvis_jax.eval.mpjpe_3d import mpjpe_3d
    import jax

    ds = V3FramesetDataset(root, split)
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    nd = jax.device_count()
    n -= n % nd if nd else 0                          # device-multiple

    model = load_inference_model(vitpose_ckpt, v2v_final, sharpen=sharpen)

    all_kp, all_conf, all_c3d, all_gt, all_vis = [], [], [], [], []
    for i0 in range(0, n, batch):
        idx = list(range(i0, min(i0 + batch, n)))
        if len(idx) % nd:                            # pad to device-multiple, trim after
            pad = nd - (len(idx) % nd); idx += idx[-1:] * pad
        else:
            pad = 0
        crops = np.stack([ds[i]["crops4"] for i in idx])
        chm = np.stack([ds[i]["centerHM"] for i in idx]).astype("float32")
        cams = np.stack([ds[i]["cameraMatrices"] for i in idx]).astype("float32")
        kp, conf, c3d = predict_batch(model, crops, chm, cams)
        keep = len(idx) - pad
        all_kp.append(kp[:keep]); all_conf.append(conf[:keep]); all_c3d.append(c3d[:keep])
        if validate:
            all_gt.append(np.stack([ds[i]["kp3d"] for i in idx[:keep]]))
            all_vis.append(np.stack([ds[i]["vis"] for i in idx[:keep]]))

    kp = np.concatenate(all_kp); conf = np.concatenate(all_conf)
    c3d = np.concatenate(all_c3d)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, kp3d=kp, conf=conf, center3D=c3d, index=np.arange(len(kp)))
    print(f"[predict] wrote {out}  ({len(kp)} framesets)")

    result = {"out": out, "n": len(kp)}
    if validate:
        gt = np.concatenate(all_gt); vis = np.concatenate(all_vis)
        import jax.numpy as jnp
        mpjpe = float(mpjpe_3d(jnp.asarray(kp), jnp.asarray(gt), jnp.asarray(vis)))
        # center3D error vs GT center (recompute GT center via dataset)
        gt_c = np.stack([ds[i]["center3D"] for i in range(len(kp))])
        cerr = float(np.linalg.norm(c3d - gt_c, axis=1).mean())
        print(f"[predict] VAL MPJPE {mpjpe:.3f}   mean center3D err {cerr:.2f}")

        # In-cube sanity: fraction of GT-visible joints whose GT kp3d lies
        # outside the ROI cube (roi_cube/2 = 24 units on any axis).
        roi_half = 24.0  # roi_cube / 2
        # gt: (N, J, 3), gt_c: (N, 3), vis: (N, J)
        gt_c_exp = gt_c[:, np.newaxis, :]                 # (N, 1, 3)
        gt_rel = np.abs(gt - gt_c_exp)                    # (N, J, 3)
        out_cube = (gt_rel.max(axis=-1) > roi_half)       # (N, J) bool
        vis_any = vis.any()
        if vis_any:
            n_vis = vis.sum()
            n_out = (out_cube & vis).sum()
            frac_out = float(n_out) / float(n_vis)
        else:
            frac_out = float("nan")
        print(f"[predict] in-cube diagnostic: {frac_out*100:.2f}% of GT-visible joints "
              f"exceed roi_cube/2={roi_half:.0f} on any axis (expect ~0%)")

        result.update(mpjpe=mpjpe, center3d_err=cerr, incube_out_frac=frac_out)
    return result


def main_from_cfg(cfg):
    run_dir = run_dir_for(cfg)
    return run_predict(
        root=cfg.paths.data_root,
        split=cfg.predict.split,
        vitpose_ckpt=cfg.paths.vitpose_ckpt,
        v2v_final=os.path.join(run_dir, "final"),
        out=cfg.predict.out,
        sharpen=cfg.model.sharpen,
        batch=cfg.predict.batch,
        limit=cfg.predict.limit,
        validate=cfg.predict.validate,
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
