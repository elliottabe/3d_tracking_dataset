"""Error-based hard-example mining for the 2D keypoint detector: per-annotation
2D MPJPE (px) on a split (default TRAIN), saved to an .npz for
`configs/sampling/hard_error.yaml` to drive error-weighted resampling in a
hard-frame fine-tune (see `jarvis_jax.scripts.train_keypoints.error_weights`).

Reuses (does not reinvent) existing machinery:
  * ``jarvis_jax.scripts.eval_keypoints_2d.restore_model`` for the Orbax
    checkpoint restore (arch in {vitpose, efficienttrack, efficienttrack_bn}).
  * ``jarvis_jax.data.v3.V3Dataset``/``batches`` -- iterated
    ``shuffle=False, drop_last=False`` so per-sample errors land in DATASET
    ORDER (index i of the returned ``error`` array == ``ds[i]``), matching
    what ``configs/sampling/hard_error.yaml``'s ``weights_file`` is later
    zipped against (``len(error) == len(train_ds)``, same split/order).
  * ``jarvis_jax.eval.mpjpe.heatmaps_to_keypoints`` + ``jarvis_jax.train.
    train._eval_forward`` -- the SAME jitted decode ``eval_mpjpe`` uses,
    just kept per-sample here instead of averaged over the dataset.

CLI (Hydra; mirrors eval_keypoints_2d.py). The checkpoint dir / output path
aren't registered config groups -- add them ad hoc with ``+mine.*``::

    python -m jarvis_jax.scripts.mine_hard_frames \\
        paths=hyak model=efficienttrack_bn \\
        +mine.ckpt_dir=/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/et2d_bn_imagenet/final \\
        +mine.out=/gscratch/portia/eabe/data/Johnson_lab/hard_mining/et2d_bn_train_errors.npz \\
        +mine.split=train

Optional overrides (each a NEW key under the ad-hoc ``mine.*`` node, so also
``+``-prefixed): ``mine.batch_size`` (default 16), ``mine.data_root``
(default ``paths.data_root``).
"""
import hydra
import jax.numpy as jnp
import numpy as np

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass

register_resolvers()

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.device import normalize_image
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from jarvis_jax.scripts.eval_keypoints_2d import restore_model
from jarvis_jax.train.train import _eval_forward


def mine_errors(model, ds, batch_size, *, in_size=448):
    """Per-annotation 2D MPJPE (px), in DATASET ORDER (index i == ds[i]).

    Same decode primitives as ``jarvis_jax.train.train.eval_mpjpe``
    (``_eval_forward`` + ``heatmaps_to_keypoints``), but kept per-sample
    instead of averaged. NaN for annotations with no visible joints (nothing
    to score). Iterates ``batches(ds, batch_size, shuffle=False,
    drop_last=False)`` so every annotation is scored exactly once, in order.
    """
    model.eval()
    scale = in_size / float(ds.heatmap_size)
    n = len(ds)
    error = np.full(n, np.nan, dtype=np.float32)
    pos = 0
    for img4_u8, kp_xy, vis in batches(ds, batch_size, shuffle=False,
                                       drop_last=False):
        b = img4_u8.shape[0]
        img = normalize_image(jnp.asarray(img4_u8))
        pred = _eval_forward(model, img)
        pk = heatmaps_to_keypoints(pred, in_size=in_size)          # (B,K,2)
        gk = jnp.asarray(kp_xy) * scale                             # (B,K,2)
        d = np.asarray(jnp.linalg.norm(pk - gk, axis=-1))           # (B,K)
        w = np.asarray(vis, dtype=np.float64)                       # (B,K)
        num = (d * w).sum(axis=-1)
        den = w.sum(axis=-1)
        per_sample = np.full(b, np.nan, dtype=np.float32)
        np.divide(num, den, out=per_sample, where=den > 0, casting="unsafe")
        error[pos:pos + b] = per_sample
        pos += b
    assert pos == n
    return error


def mine(ckpt_dir, arch, root, *, split="train", vitpose_cfg=None,
        batch_size=16, dataset_cls=V3Dataset, in_size=448, quiet=False):
    """Restore `arch`'s checkpoint at `ckpt_dir`, score every annotation in
    `root`'s `split` and return a dict of arrays (see module docstring for
    what's saved). `dataset_cls` defaults to ``V3Dataset``; tests inject a
    tiny stand-in the same way ``eval_keypoints_2d.evaluate`` does.
    """
    cfg = vitpose_cfg if vitpose_cfg is not None else ViTPoseConfig()
    model = restore_model(ckpt_dir, arch, cfg)

    ds = dataset_cls(root, split)
    error = mine_errors(model, ds, batch_size, in_size=in_size)

    result = {
        "error": error.astype(np.float32),
        "file_name": np.asarray(ds.file_names),
        "sex": np.asarray(ds.sex),
        "behavior": np.asarray(ds.behavior),
        "ann_id": np.asarray(ds.ann_ids),
        "split": np.asarray(split),
    }

    if not quiet:
        finite = error[np.isfinite(error)]
        print(f"checkpoint: {ckpt_dir}")
        print(f"arch:       {arch}")
        print(f"split:      {split}  (n={len(ds)}, {finite.size} with visible joints)")
        if finite.size:
            print(f"error(px):  mean={finite.mean():.3f} median={np.median(finite):.3f} "
                 f"p90={np.percentile(finite, 90):.3f} p99={np.percentile(finite, 99):.3f}")
        # Worst-first; NaN (no visible joints) sorts last, never "worst".
        sort_key = np.where(np.isfinite(error), error, -np.inf)
        worst = np.argsort(-sort_key)[:10]
        print("worst 10 file_names:")
        for i in worst:
            print(f"  {error[i]:.3f}px  {ds.file_names[i]}")

    return result


def main_from_cfg(cfg):
    model_node = cfg.model.get("vitpose", cfg.model)
    arch = model_node.get("arch", "vitpose")
    vitpose_cfg = build_dataclass(ViTPoseConfig, model_node)
    root = cfg.mine.get("data_root", cfg.paths.data_root)
    split = cfg.mine.get("split", "train")
    result = mine(
        cfg.mine.ckpt_dir, arch, root, split=split, vitpose_cfg=vitpose_cfg,
        batch_size=cfg.mine.get("batch_size", 16))
    out = cfg.mine.out
    np.savez(out, **result)
    print(f"saved: {out if str(out).endswith('.npz') else str(out) + '.npz'}")
    return result


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
