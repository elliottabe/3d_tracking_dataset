"""Design-1 pilot: train ViTPose with 50+M outputs (keypoints + canonical vertices).

Minimal single-host loop (no data-parallel mesh) to confirm the augmented head
trains and the dense vertex heatmaps localize.  Backbone is MAE-initialised;
augmentation is disabled (flip needs a 250-joint L/R map — a full-run refinement).
The full run will add v3-checkpoint warm-start of the first 50 channels + flip aug.

    python -m jarvis_jax.cse.train_keypoints_cse \
        --root <V3> --aux-train <..._train_M200.npz> --aux-val <..._val_M200.npz> \
        --out <run>/final --ckpt-dir <run>/ckpt --steps 2000 --batch 16
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aux-train", required=True)
    ap.add_argument("--aux-val", required=True)
    ap.add_argument("--mae", default="/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()

    import jax
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.convert.build_checkpoint import build
    from jarvis_jax.data.v3 import batches
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step, eval_mpjpe
    from jarvis_jax.cse.cse_dataset import CSEImageDataset

    print("jax devices:", jax.device_count())
    cfg = ViTPoseConfig(num_keypoints=a.num_joints)
    tcfg = TrainConfig(lr=a.lr, total_steps=a.steps, batch_size=a.batch)

    model = build(a.mae, cfg) if os.path.exists(a.mae) else ViTPose(cfg, rngs=nnx.Rngs(0))
    opt = make_optimizer(model, tcfg)
    step = make_train_step(tcfg.mask_weight, aug_params=None, lr_swap=None,
                           heatmap_size=cfg.heatmap_size)

    train_ds = CSEImageDataset(a.root, "train", a.aux_train)
    val_ds = CSEImageDataset(a.root, "val", a.aux_val)
    print(f"train {len(train_ds)} imgs, val {len(val_ds)} imgs, J={a.num_joints}")

    key = jax.random.PRNGKey(0)
    epoch = 0
    it = batches(train_ds, a.batch, shuffle=True, seed=epoch)
    for i in range(a.steps):
        try:
            img, kp, vis = next(it)
        except StopIteration:
            epoch += 1
            it = batches(train_ds, a.batch, shuffle=True, seed=epoch)
            img, kp, vis = next(it)
        loss = float(step(model, opt, jax.random.fold_in(key, i), img, kp, vis))
        if (i + 1) % a.log_every == 0:
            print(f"step {i+1}/{a.steps} loss {loss:.5f}", flush=True)
        if (i + 1) % a.eval_every == 0:
            # MPJPE split: keypoints (first 50) vs vertices (rest)
            mp = eval_mpjpe(model, val_ds, a.batch)
            print(f"  val MPJPE(all {a.num_joints}) {mp:.3f}px", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(a.out, nnx.split(model)[1], force=True)
    ckptr.wait_until_finished()
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
