import os
import jax
import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
gpu = any(d.platform == "gpu" for d in jax.devices())
run = pytest.mark.skipif(
    not (os.path.isdir(ROOT) and gpu),
    reason="smoke run needs V3 data + GPU")


@run
def test_smoke_training_runs_and_checkpoints(tmp_path):
    from jarvis_jax.scripts.train_keypoints import run_training
    out = str(tmp_path / "ckpt")
    res = run_training(ROOT, out_dir=out, smoke=True)
    assert res["steps"] == 4
    assert res["final_loss"] >= 0.0
    assert res["val_mpjpe"] >= 0.0
    assert os.path.isdir(out)  # checkpoint written
