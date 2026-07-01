"""Full-dataset dense-pose inference: ViTPose-250 -> reproject -> V2VNet-250.

For every frameset (all recordings, a split) produces:
  pred3d  (n, 250, 3)   3-D dense pose (50 kp + 200 vertices), world mm
  conf    (n, 250)      soft-argmax confidence
  pred2d  (n, nc, 250, 2) per-camera 2-D predictions (for multi-view consistency)
  gt3d    (n, 250, 3)   STAC/triangulated GT (for accuracy eval)
  vis     (n, 250)
Reuses a single ViTPose forward per batch for both the 2-D decode and the 3-D
volume (mirrors HybridNet3D.__call__).  GPU; chained after V2VNet-250.
"""
import argparse, os, json
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--aux", required=True)
    ap.add_argument("--vitpose-ckpt", required=True)
    ap.add_argument("--v2v-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sharpen", type=float, default=3.0)
    ap.add_argument("--gate-dilate", type=int, default=0,
                    help="inference-time dilated-mask gating kernel (0 = off; 21 = jarvis3D default)")
    ap.add_argument("--recordings", nargs="*", default=None,
                    help="optional recording filter (default: all in split)")
    a = ap.parse_args()

    import numpy as np, jax, jax.numpy as jnp, orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.hybridnet.model import HybridNet3D, soft_argmax_3d
    from jarvis_jax.hybridnet.reproject import reproject_heatmaps
    from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
    from jarvis_jax.cse.cse_dataset import CSEFramesetDataset
    from jarvis_jax.data.v3_3d import frameset_batches

    J = a.num_joints
    cfg = ViTPoseConfig(num_keypoints=J)
    vit = ViTPose(cfg, rngs=nnx.Rngs(0))
    g, st = nnx.split(vit); vit = nnx.merge(g, ocp.StandardCheckpointer().restore(a.vitpose_ckpt, st)); vit.eval()
    v2v = V2VNet(J, J, rngs=nnx.Rngs(0))
    g2, st2 = nnx.split(v2v); v2v = nnx.merge(g2, ocp.StandardCheckpointer().restore(a.v2v_ckpt, st2)); v2v.eval()
    hyb = HybridNet3D(vit, v2v, cfg)

    ds = CSEFramesetDataset(a.root, a.split, a.aux, recordings=a.recordings)
    P3, CF, P2, G3, VIS, CHM, CM = [], [], [], [], [], [], []
    for batch in frameset_batches(ds, a.batch, shuffle=False, drop_last=False):
        crops = jnp.asarray(batch["crops4"]); c3 = jnp.asarray(batch["center3D"])
        cHM = jnp.asarray(batch["centerHM"]); cM = jnp.asarray(batch["cameraMatrices"])
        b, nc = crops.shape[:2]
        # Materialise between stages (device->host->device) so XLA compiles each
        # stage separately and frees it — otherwise the fused ViTPose+reproject+
        # V2VNet graph needs ~33 GiB and OOMs even at batch 1.
        hm = np.asarray(hyb.predict_heatmaps(crops))              # (b,nc,224,224,J)
        if a.gate_dilate > 0:
            from jarvis_jax.cse.gating import gate_heatmaps
            mask448 = jnp.asarray(crops[..., 3]).reshape(b * nc, crops.shape[2], crops.shape[3])
            hm = np.asarray(gate_heatmaps(
                jnp.asarray(hm).reshape(b * nc, 224, 224, J), mask448, a.gate_dilate)
            ).reshape(b, nc, 224, 224, J)
        p2 = heatmaps_to_keypoints(jnp.asarray(hm.reshape(b * nc, 224, 224, J)), in_size=448)
        P2.append(np.asarray(p2).reshape(b, nc, J, 2))
        # 3-D: pad 224->226, reproject (stage), then v2vnet+soft-argmax (stage)
        hmp = jnp.transpose(jnp.pad(jnp.asarray(hm), [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)]),
                            (0, 1, 4, 2, 3))
        vol = np.asarray(reproject_heatmaps(hmp, c3, cHM, cM, grid_size=48,
                                            grid_spacing=1, heatmap_size=226) / 255.0)
        vol = jnp.transpose(jnp.asarray(vol), (0, 2, 3, 4, 1))
        vol = v2v(vol, use_running_average=True)
        vol = jax.nn.softplus(jnp.transpose(vol, (0, 4, 1, 2, 3)))
        pts, cf = soft_argmax_3d(vol, grid_spacing=1, roi_cube=48, sharpen=a.sharpen)
        pts = pts + c3[:, None, :]
        P3.append(np.asarray(pts)); CF.append(np.asarray(cf))
        G3.append(np.asarray(batch["kp3d"])); VIS.append(np.asarray(batch["vis"]))
        CHM.append(np.asarray(batch["centerHM"])); CM.append(np.asarray(batch["cameraMatrices"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out,
        pred3d=np.concatenate(P3).astype(np.float32),
        conf=np.concatenate(CF).astype(np.float32),
        pred2d=np.concatenate(P2).astype(np.float32),
        gt3d=np.concatenate(G3).astype(np.float32),
        vis=np.concatenate(VIS),
        centerHM=np.concatenate(CHM).astype(np.float32),
        cameraMatrices=np.concatenate(CM).astype(np.float32))
    print(f"[predict_full] {a.split}: {sum(len(x) for x in P3)} framesets x {J} joints -> {a.out}")


if __name__ == "__main__":
    main()
