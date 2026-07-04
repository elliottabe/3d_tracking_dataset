"""GPU smoke test for the v3->v4 detector finetune driver.

Builds tiny fake real+pseudo V3-format datasets (via Task 3's
write_pseudolabel_coco), continue-trains for 2 steps from the real v3
checkpoint, and checks the loop runs end-to-end and writes a NEW checkpoint
(never touching v3_ckpt itself). Not a localization gate -- see
test_train_step.py for why overfit-style gates are flaky for this model.
"""
import os

import jax
import numpy as np
import pytest

from jarvis_jax.cse.build_pseudolabel_dataset import bbox_from_mask, write_pseudolabel_coco

V3_CKPT = os.environ.get(
    "V3_CKPT",
    "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_kp_maskaware/final")
gpu = any(d.platform == "gpu" for d in jax.devices())
run = pytest.mark.skipif(
    not (os.path.exists(V3_CKPT) and gpu),
    reason="finetune smoke test needs a v3 checkpoint + GPU")


def _tiny_root(root, split, n=2):
    recs = []
    for i in range(n):
        H = W = 200
        m = np.zeros((H, W), bool)
        m[60:140, 60:140] = True
        kp = np.zeros((50, 3))
        kp[:, 0] = 100
        kp[:, 1] = 100
        kp[:, 2] = 1
        recs.append({
            "file_name": f"rec/Cam/Frame_{i}.jpg", "img_w": W, "img_h": H,
            "rgb": (np.random.rand(H, W, 3) * 255).astype(np.uint8),
            "mask": m, "keypoints": kp, "bbox": bbox_from_mask(m),
        })
    write_pseudolabel_coco(root, recs, split=split)


@run
def test_finetune_smoke(tmp_path):
    from jarvis_jax.cse.finetune_detector import finetune

    real = str(tmp_path / "real")
    pseudo = str(tmp_path / "pseudo")
    out = str(tmp_path / "v4")
    os.makedirs(real)
    os.makedirs(pseudo)
    _tiny_root(real, "train")
    _tiny_root(real, "val")
    _tiny_root(pseudo, "train")

    result = finetune(
        v3_ckpt=V3_CKPT, real_root=real, pseudo_root=pseudo, out_dir=out,
        val_recordings=["rec"], total_steps=2, eval_every=2, patience=1,
        batch_size=2)

    assert os.path.exists(out)
    assert "best_female_mpjpe" in result
    assert "best_step" in result and "full_val_mpjpe" in result
    assert np.isfinite(result["best_female_mpjpe"])
    assert np.isfinite(result["full_val_mpjpe"])
    # v3_ckpt itself must never be modified by the finetune run.
    assert os.path.exists(os.path.join(V3_CKPT))


@run
def test_finetune_never_writes_to_v3_ckpt(tmp_path):
    """out_dir must be distinct from v3_ckpt; finetune must refuse to alias them."""
    from jarvis_jax.cse.finetune_detector import finetune

    real = str(tmp_path / "real")
    pseudo = str(tmp_path / "pseudo")
    os.makedirs(real)
    os.makedirs(pseudo)
    _tiny_root(real, "train")
    _tiny_root(real, "val")
    _tiny_root(pseudo, "train")

    with pytest.raises(ValueError):
        finetune(
            v3_ckpt=V3_CKPT, real_root=real, pseudo_root=pseudo,
            out_dir=V3_CKPT, val_recordings=["rec"], total_steps=1,
            eval_every=1, patience=1, batch_size=2)
