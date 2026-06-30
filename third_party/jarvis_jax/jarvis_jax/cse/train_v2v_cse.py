"""C3: train V2VNet-250 on the CSE reproject cache (thin wrapper, no hydra).

train_3d_cached.run_cached_training infers J = volumes.shape[1] (=250) and wires
the graph-Laplacian prior from the cache meta (kp edges + vertex kNN graph), so
no trainer changes are needed beyond a smaller batch (250-channel 3-D volumes are
~5x the 50-channel memory).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)   # 250ch volumes: smaller than 50ch=64
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--laplacian-weight", type=float, default=0.05)
    ap.add_argument("--sharpen", type=float, default=3.0)
    a = ap.parse_args()

    import jax
    from jarvis_jax.train.train_3d_cached import run_cached_training, CachedConfig
    print("jax devices:", jax.device_count())
    tcfg = CachedConfig(lr=a.lr, total_steps=a.steps, batch_size=a.batch,
                        laplacian_weight=a.laplacian_weight, sharpen=a.sharpen)
    run_cached_training(a.cache_dir, out_dir=a.out, ckpt_dir=a.ckpt_dir, tcfg=tcfg,
                        save_every=1000, log_every=50, eval_every=500)


if __name__ == "__main__":
    main()
