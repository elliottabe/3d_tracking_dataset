"""C3: build the 250-channel reproject cache from the trained ViTPose-250.

For each frameset: run the (frozen) ViTPose-250 over its cameras, reproject the
250 heatmaps into the shared 3-D volume (HybridNet3D.reproject_volume), and store
(volume, kp3d[250], center3D, vis[250]) in the repro-cache layout that
train_3d_cached consumes (it reads J = volumes.shape[1], so V2VNet-250 is
automatic).  meta carries the augmented keypoint names + skeleton (kp edges +
vertex kNN graph) for the graph-Laplacian prior.

GPU; run via sbatch after the ViTPose-250 checkpoint exists.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

_GRID = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--aux", required=True)
    ap.add_argument("--vitpose-ckpt", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import numpy as np
    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.densepose.cse_dataset import (CSEFramesetDataset,
                                            augmented_keypoint_names, vertex_edges)

    print("jax devices:", jax.device_count())
    cfg = ViTPoseConfig(num_keypoints=a.num_joints)
    vit = ViTPose(cfg, rngs=nnx.Rngs(0))
    gdef, st = nnx.split(vit)
    vit = nnx.merge(gdef, ocp.StandardCheckpointer().restore(a.vitpose_ckpt, st))
    vit.eval()
    hyb = HybridNet3D(vit, V2VNet(a.num_joints, a.num_joints, rngs=nnx.Rngs(0)), cfg)

    ds = CSEFramesetDataset(a.root, a.split, a.aux)
    from jarvis_jax.data.v3_3d import frameset_batches
    n = len(ds)
    os.makedirs(a.cache_dir, exist_ok=True)
    vol_path = os.path.join(a.cache_dir, f"{a.split}_volumes.f16")
    mm = np.memmap(vol_path, dtype=np.float16, mode="w+",
                   shape=(n, a.num_joints, _GRID, _GRID, _GRID))

    kp3d, c3d, vis, w = [], [], [], 0
    for batch in frameset_batches(ds, a.batch, shuffle=False, drop_last=False):
        vols = hyb.reproject_volume(
            np.asarray(batch["crops4"]), np.asarray(batch["center3D"]),
            np.asarray(batch["centerHM"]), np.asarray(batch["cameraMatrices"]))
        vols = np.asarray(vols).astype(np.float16)         # (b, J, 48,48,48)
        for v in vols:
            mm[w] = v; w += 1
        kp3d.append(batch["kp3d"]); c3d.append(batch["center3D"]); vis.append(batch["vis"])
        if w % 200 < a.batch:
            print(f"  cached {w}/{n}", flush=True)
    mm.flush()
    assert w == n, (w, n)

    np.savez(os.path.join(a.cache_dir, f"{a.split}_labels.npz"),
             kp3d=np.concatenate(kp3d).astype(np.float32),
             center3D=np.concatenate(c3d).astype(np.float32),
             vis=np.concatenate(vis))

    # augmented names + skeleton (kp edges from coco + vertex kNN graph)
    coco = json.load(open(os.path.join(a.root, "annotations", f"instances_{a.split}.json")))
    fps = np.load(a.aux, allow_pickle=True)["fps_indices"]
    names = augmented_keypoint_names(coco["keypoint_names"], a.num_joints - 50)
    skeleton = list(coco.get("skeleton", [])) + vertex_edges(a.mesh, fps, k=4)
    json.dump({"n": n, "split": a.split, "n_joints": a.num_joints, "grid_size": _GRID,
               "vitpose_ckpt": a.vitpose_ckpt, "keypoint_names": names,
               "skeleton": skeleton},
              open(os.path.join(a.cache_dir, f"{a.split}_meta.json"), "w"))
    print(f"wrote cache: {n} framesets x {a.num_joints} joints -> {a.cache_dir}")


if __name__ == "__main__":
    main()
