"""Design-1 FULL run: ViTPose with 50+M outputs, v3 warm-start, 8-GPU data-parallel.

Mirrors scripts/train_keypoints.run_training (mesh / replicate / prefetch /
checkpoint-resume) with three changes for CSE:
  * CSEImageDataset (250 joints = 50 kp + M vertex 2D labels),
  * model warm-started from the trained v3 50-kp checkpoint (warm_start_from_v3),
  * augmentation: flip is OFF by default (identity L/R swap), since the
    stratified vertex subset isn't guaranteed closed under mirroring (affine +
    cutout + photometric still applied). Pass --flip-p>0 to enable flip using
    the dense (50+M) L/R involution from build_dense_lr_swap (Phase 5).

Phase 5 also adds optional mask gating: --mask-weight>0 turns on a dilated-mask
(--mask-dilate px) containment loss penalizing off-fly heatmap mass.

ckpt-g2 is preemptible -> fixed --ckpt-dir + restore_latest gives auto-resume.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _epochs(ds, batch_size, base_seed):
    from jarvis_jax.data.v3 import batches
    epoch = 0
    while True:
        yield from batches(ds, batch_size, shuffle=True, seed=base_seed + epoch)
        epoch += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aux-train", required=True)
    ap.add_argument("--aux-val", required=True)
    ap.add_argument("--mae", default="/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz")
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--warmstart-joints", type=int, default=50,
                    help="output count of the warm-start source ckpt (50 for v3, "
                         "250 for the body+leg CSE model when adding wings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--num-joints", type=int, default=250)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--mask-weight", type=float, default=0.0,
                    help="dilated-mask containment loss weight (>0 turns gating ON)")
    ap.add_argument("--mask-dilate", type=int, default=11,
                    help="containment-loss mask dilation kernel (px); used only when mask-weight>0")
    ap.add_argument("--flip-p", type=float, default=0.0,
                    help="horizontal-flip prob; >0 enables flip aug using the dense L/R swap")
    ap.add_argument("--base-names",
                    default="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/stac-mjx/configs/anatomy/v1.yaml",
                    help="anatomy yaml providing the 50 base keypoint names (for the flip swap)")
    ap.add_argument("--val-recording", default="2026_03_18_15_31_22")
    # Wing fine-tune: render densely-packed wing vertices with a tighter Gaussian
    # (so neighbouring peaks stay resolvable, fixing the inboard tip compression)
    # and up-weight their loss. Defaults (7.0 / 1.0) reproduce the original run.
    ap.add_argument("--mesh-npz", default=None,
                    help="canonical wings mesh — needed when wing/leg sigma or weight differ from default")
    ap.add_argument("--wing-sigma", type=float, default=7.0)
    ap.add_argument("--wing-weight", type=float, default=1.0)
    ap.add_argument("--leg-sigma", type=float, default=7.0)
    ap.add_argument("--leg-weight", type=float, default=1.0)
    a = ap.parse_args()

    import jax, jax.numpy as jnp, numpy as np
    import orbax.checkpoint as ocp
    from flax import nnx
    from jarvis_jax.config import ViTPoseConfig
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.convert.build_checkpoint import build
    from jarvis_jax.data.prefetch import prefetch
    from jarvis_jax.data.augment import AugParams
    from jarvis_jax.sharding import data_parallel_mesh, replicate
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step, eval_mpjpe
    from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest
    from jarvis_jax.densepose.cse_dataset import CSEImageDataset
    from jarvis_jax.densepose.warm_start import warm_start_from_v3

    n_dev = jax.device_count()
    print("jax devices:", n_dev)
    if a.batch % n_dev != 0:
        raise ValueError(f"batch {a.batch} not divisible by device count {n_dev}")

    cfg = ViTPoseConfig(num_keypoints=a.num_joints)
    tcfg = TrainConfig(lr=a.lr, total_steps=a.steps, batch_size=a.batch,
                       backbone_lr_mult=a.backbone_lr_mult,
                       mask_weight=a.mask_weight, mask_dilate=a.mask_dilate)

    model = build(a.mae, cfg) if os.path.exists(a.mae) else ViTPose(cfg, rngs=nnx.Rngs(0))
    model = warm_start_from_v3(model, a.v3_ckpt,
                               ViTPoseConfig(num_keypoints=a.warmstart_joints))
    opt = make_optimizer(model, tcfg)

    mngr = make_manager(a.ckpt_dir)
    start = 0
    model, opt, start = restore_latest(mngr, model, opt)
    if start:
        print(f"resuming from checkpoint at step {start}")

    # aug: flip ON iff --flip-p>0, using the dense (50+M) L/R involution.
    if a.flip_p > 0.0:
        from jarvis_jax.densepose.dense_lr_swap import build_dense_lr_swap
        from jarvis_jax.densepose.cse_labels import model_kp_order  # 50 canonical STAC names
        base_names = model_kp_order(a.base_names)
        M = a.num_joints - 50
        lr_swap = build_dense_lr_swap(a.mesh_npz, f"fps_{M}", base_names)
        aug = AugParams(enabled=True, flip_p=a.flip_p)
        print(f"[cse-train] flip ON (p={a.flip_p}) with dense L/R swap "
              f"({int((lr_swap != np.arange(len(lr_swap))).sum())}/{len(lr_swap)} channels swapped)")
    else:
        aug = AugParams(enabled=True, flip_p=0.0)
        lr_swap = np.arange(a.num_joints, dtype=np.int32)

    # Per-channel target sigma + loss weight: tighten/emphasise the thin densely-
    # sampled structures (wings, legs) whose verts merge into one blob at sigma 7.
    sigma = np.full(a.num_joints, 7.0, np.float32)
    jweight = np.ones(a.num_joints, np.float32)
    if any(v != d for v, d in ((a.wing_sigma, 7.0), (a.wing_weight, 1.0),
                               (a.leg_sigma, 7.0), (a.leg_weight, 1.0))):
        if not a.mesh_npz:
            raise ValueError("--mesh-npz required when wing/leg sigma or weight are set")
        mz = np.load(a.mesh_npz, allow_pickle=True)
        nv = a.num_joints - 50
        fps = mz[f"fps_{nv}"]; seg = mz["vertex_segment"][fps]
        id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
                for s, n in zip(mz["seg_ids"], mz["seg_names"])}
        names = [id2n[int(s)].lower() for s in seg]
        leg_sub = ("coxa", "trochanter", "femur", "tibia", "tarsus", "claw")
        iswing = np.array(["wing" in nm for nm in names])
        isleg = np.array([any(s in nm for s in leg_sub) for nm in names])
        wing_joint = np.zeros(a.num_joints, bool); wing_joint[50:][iswing] = True
        leg_joint = np.zeros(a.num_joints, bool);  leg_joint[50:][isleg] = True
        sigma[wing_joint] = a.wing_sigma; jweight[wing_joint] = a.wing_weight
        sigma[leg_joint] = a.leg_sigma;   jweight[leg_joint] = a.leg_weight
        print(f"fine-tune: {int(wing_joint.sum())} wing ch @ sigma {a.wing_sigma} w {a.wing_weight} | "
              f"{int(leg_joint.sum())} leg ch @ sigma {a.leg_sigma} w {a.leg_weight} | "
              f"{int(a.num_joints - wing_joint.sum() - leg_joint.sum())} body/kp ch @ sigma 7.0 w 1.0")
    step = make_train_step(tcfg.mask_weight, aug, lr_swap, heatmap_size=cfg.heatmap_size,
                           sigma=jnp.asarray(sigma), joint_weight=jnp.asarray(jweight),
                           mask_dilate=tcfg.mask_dilate)

    mesh = data_parallel_mesh()
    gdef_m, st_m = nnx.split(model); model = nnx.merge(gdef_m, replicate(st_m, mesh))
    gdef_o, st_o = nnx.split(opt);   opt = nnx.merge(gdef_o, replicate(st_o, mesh))

    train_ds = CSEImageDataset(a.root, "train", a.aux_train)
    val_ds = CSEImageDataset(a.root, "val", a.aux_val, recordings=[a.val_recording])
    val_all = CSEImageDataset(a.root, "val", a.aux_val)
    print(f"train {len(train_ds)} imgs, val {len(val_all)} imgs, J={a.num_joints}")

    base_key = jax.random.PRNGKey(0)
    dev_stream = prefetch(_epochs(train_ds, a.batch, 0), mesh, depth=2)
    for i in range(start, a.steps):
        img, kp, vis = next(dev_stream)
        loss = float(step(model, opt, jax.random.fold_in(base_key, i), img, kp, vis))
        if (i + 1) % a.log_every == 0:
            print(f"step {i+1}/{a.steps} loss {loss:.5f}", flush=True)
        if (i + 1) % a.eval_every == 0:
            mp = eval_mpjpe(model, val_all, a.batch)
            fem = eval_mpjpe(model, val_ds, a.batch)
            print(f"  val MPJPE(all) {mp:.3f}px (val-rec {fem:.3f}px)", flush=True)
        if (i + 1) % a.save_every == 0:
            save_step(mngr, i + 1, model, opt)
    save_step(mngr, a.steps, model, opt); mngr.wait_until_finished()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(a.out, nnx.split(model)[1], force=True); ckptr.wait_until_finished()
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
