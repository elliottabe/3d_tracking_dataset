"""Train ViTPose KeypointDetect on red_data_unified_V3 (JAX, data-parallel).

Full fine-tune from the MAE-pretrained backbone with a lower backbone LR
(see TrainConfig.backbone_lr_mult). Everything trains, so the 4th (SAM-mask)
patch-embed channel learns. End-to-end localization evidence comes from a real
run of this script (the synthetic overfit gate was dropped as flaky — see
tests/test_train_step.py).

CLI:
    python -m jarvis_jax.scripts.train_keypoints \\
        --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \\
        --out /tmp/vitpose_kp_ckpt --steps 2000 --batch 8 --lr 3e-4
"""
import argparse
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.convert.build_checkpoint import build
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.sharding import data_parallel_mesh, shard_batch
from jarvis_jax.train.train import (
    TrainConfig, make_optimizer, make_train_step, eval_mpjpe,
)

DEFAULT_MAE_NPZ = "/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz"


def _epochs(ds, batch_size, base_seed):
    """Infinite stream of batches, reshuffled each epoch."""
    epoch = 0
    while True:
        yield from batches(ds, batch_size, shuffle=True, seed=base_seed + epoch)
        epoch += 1


def run_training(root, *, out_dir, mae_npz=DEFAULT_MAE_NPZ, tcfg=None,
                 val_recording="2026_05_27_11_56_05",
                 log_every=50, eval_every=500, smoke=False):
    cfg = ViTPoseConfig()
    tcfg = tcfg or TrainConfig()
    if smoke:
        # batch must be divisible by the number of devices for data-parallel sharding
        n_devices = len(jax.devices())
        tcfg.total_steps, tcfg.batch_size, eval_every, log_every = 4, n_devices, 4, 1

    n_dev = len(jax.devices())
    if tcfg.batch_size % n_dev != 0:
        raise ValueError(
            f"batch_size ({tcfg.batch_size}) must be divisible by the JAX device "
            f"count ({n_dev}) for data-parallel sharding.")

    if mae_npz and os.path.exists(mae_npz):
        model = build(mae_npz, cfg)          # MAE-pretrained backbone
    else:
        print(f"WARNING: MAE npz not found ({mae_npz}); using RANDOM init "
              f"(real training should use MAE init).")
        model = ViTPose(cfg, rngs=nnx.Rngs(tcfg.seed))

    opt = make_optimizer(model, tcfg)
    step = make_train_step(tcfg.mask_weight)
    mesh = data_parallel_mesh()

    train_ds = V3Dataset(root, "train")
    val_ds = V3Dataset(root, "val", recordings=[val_recording])

    src = _epochs(train_ds, tcfg.batch_size, tcfg.seed)

    final_loss = 0.0
    for i in range(tcfg.total_steps):
        img4, hm, vis = next(src)
        b = [shard_batch(jnp.asarray(a), mesh) for a in (img4, hm, vis)]
        final_loss = float(step(model, opt, *b))
        if (i + 1) % log_every == 0:
            print(f"step {i+1}/{tcfg.total_steps} loss {final_loss:.5f}")
        if (i + 1) % eval_every == 0:
            print(f"  val MPJPE {eval_mpjpe(model, val_ds, tcfg.batch_size):.3f}px")

    val_mpjpe = eval_mpjpe(model, val_ds, tcfg.batch_size)
    print(f"final val MPJPE {val_mpjpe:.3f}px")

    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, nnx.split(model)[1], force=True)
    ckptr.wait_until_finished()
    return {"final_loss": final_loss, "val_mpjpe": val_mpjpe,
            "steps": tcfg.total_steps}


def main():
    ap = argparse.ArgumentParser(description="Train ViTPose KeypointDetect (JAX).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mae-npz", default=DEFAULT_MAE_NPZ)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tcfg = TrainConfig(total_steps=args.steps, batch_size=args.batch, lr=args.lr,
                       backbone_lr_mult=args.backbone_lr_mult)
    run_training(args.root, out_dir=args.out, mae_npz=args.mae_npz,
                 tcfg=tcfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
