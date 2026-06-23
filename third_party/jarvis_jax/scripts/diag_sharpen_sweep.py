"""Sweep the soft-argmax `sharpen` exponent on a trained run (inference-only).

Computes the v2vNet output volume ONCE, then for each sharpen value recomputes
the soft-argmax post-hoc and reports val MPJPE + the global pred/gt radius ratio
(1.0 = no shrink). Answers: does sharpening fix the inward-shrink WITHOUT
retraining, and which exponent is best?
"""
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.data.repro_cache import load_cache
from jarvis_jax.hybridnet.v2vnet import V2VNet

NUM_J = 50
G = 24
ROI = 48


def load_v2v(final_dir):
    v2v = V2VNet(NUM_J, NUM_J, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    ckptr = ocp.StandardCheckpointer()
    try:
        restored = ckptr.restore(final_dir, state)
    except TypeError:
        restored = ckptr.restore(final_dir, args=ocp.args.StandardRestore(state))
    return nnx.merge(gdef, restored)


def forward_volume(v2v, volumes_np, batch=16):
    v2v.eval()

    @nnx.jit
    def step(v2v, vol):
        vol = vol.astype(jnp.float32)
        vol = jnp.transpose(vol, (0, 2, 3, 4, 1))
        vol = v2v(vol, use_running_average=True)
        vol = jnp.transpose(vol, (0, 4, 1, 2, 3))
        return jax.nn.softplus(vol)

    out = []
    for i in range(0, len(volumes_np), batch):
        v = jnp.asarray(np.asarray(volumes_np[i:i + batch]))
        out.append(np.asarray(step(v2v, v)))
    return np.concatenate(out, axis=0)


def soft_argmax_np(vol, sharpen):
    """vol (N,J,G,G,G) >=0 -> pred_local (N,J,3) using the sharpen exponent."""
    hm = vol ** sharpen if sharpen != 1.0 else vol
    flat = hm.reshape(hm.shape[0], NUM_J, -1)
    s = np.clip(flat.sum(-1, keepdims=True), 1e-12, None)
    P = (flat / s).reshape(hm.shape[0], NUM_J, G, G, G)
    idx = np.arange(G, dtype=np.float32)
    mx = (P.sum((3, 4)) * idx).sum(-1)
    my = (P.sum((2, 4)) * idx).sum(-1)
    mz = (P.sum((2, 3)) * idx).sum(-1)
    return np.stack([mx, my, mz], -1) * 2.0 - ROI / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v3")
    ap.add_argument("--run", default="/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/here_run3/final")
    ap.add_argument("--sharpens", default="1,1.5,2,2.5,3,4,6,8")
    args = ap.parse_args()

    cache = load_cache(args.cache_dir, "val")
    gt = np.asarray(cache["kp3d"])
    c3d = np.asarray(cache["center3D"])
    vis = np.asarray(cache["vis"])
    gt_local = gt - c3d[:, None, :]
    gt_r = np.linalg.norm(gt_local, axis=-1)[vis]

    print(f"[sweep] forward volumes for {args.run}")
    v2v = load_v2v(args.run)
    vol = forward_volume(v2v, cache["volumes"])

    print(f"\n{'sharpen':>8} {'MPJPE':>8} {'pred/gt_radius':>15} {'%shrink':>8}")
    best = None
    for p in [float(x) for x in args.sharpens.split(",")]:
        pred_local = soft_argmax_np(vol, p)
        pos_err = np.linalg.norm(pred_local - gt_local, axis=-1)[vis]
        pred_r = np.linalg.norm(pred_local, axis=-1)[vis]
        ratio = pred_r.mean() / gt_r.mean()
        mpjpe = pos_err.mean()
        print(f"{p:8.2f} {mpjpe:8.3f} {ratio:15.3f} {100*(1-ratio):7.1f}%")
        if best is None or mpjpe < best[1]:
            best = (p, mpjpe, ratio)
    print(f"\n[sweep] best: sharpen={best[0]}  MPJPE={best[1]:.3f}  pred/gt={best[2]:.3f}")


if __name__ == "__main__":
    main()
