"""Task 9 fix-round-1 regression test.

The 2D training entrypoint (jarvis_jax/scripts/train_keypoints.py::
run_training) must forward `train.target_sigma` (TrainConfig.target_sigma)
to make_train_step's `sigma` kwarg. Before this fix, run_training called
make_train_step WITHOUT a `sigma=` kwarg at all, so it silently fell back to
make_train_step's own hardcoded default (7.0) no matter what
configs/train/vit2d.yaml's `target_sigma` said -- a run launched with
`train.target_sigma=2.0` would render heatmap targets at sigma=7.0, unchanged
from before Task 9. See task-9-report.md, "Fix round 1".

This test invokes the REAL run_training() call site (not a reimplementation
of the wiring, which would not catch a regression there) and intercepts
make_train_step just long enough to capture the `sigma` value it was
actually called with, aborting immediately after (before any GPU
data-parallel setup or dataset loading runs). It then renders an actual
heatmap target with that captured sigma and checks its concentration matches
sigma=2.0 (tight) rather than sigma=7.0 (diffuse) -- proving the value
reaches rendering, not just that a config key exists.
"""
import json

import numpy as np
import pytest

import jarvis_jax.scripts.train_keypoints as train_keypoints_mod
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.data.device import render_heatmaps
from jarvis_jax.train.train import TrainConfig

NUM_JOINTS = 4
HEATMAP_SIZE = 224


class _StopAtMakeTrainStep(Exception):
    """Raised by the make_train_step spy to abort run_training right after
    the call site under test."""


def _concentration(hm, radius=7):
    """Fraction of heatmap mass within `radius` px of the peak (same
    statistic as tests/test_transforms_sigma.py)."""
    j, i = np.unravel_index(np.argmax(hm), hm.shape)
    yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
    near = ((yy - j) ** 2 + (xx - i) ** 2) <= radius ** 2
    return float(hm[near].sum() / max(hm.sum(), 1e-9))


def test_run_training_forwards_target_sigma_to_make_train_step(tmp_path, monkeypatch):
    root = tmp_path / "toy_root"
    (root / "annotations").mkdir(parents=True)
    names = [f"kp{i}" for i in range(NUM_JOINTS)]
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps({"keypoint_names": names}))

    captured = {}

    def _spy_make_train_step(*args, **kwargs):
        # Reproduce make_train_step's OWN default (7.0) exactly, so an
        # un-fixed call site (no `sigma=` kwarg) is captured as 7.0 -- what
        # training would actually render today.
        captured["sigma"] = kwargs.get("sigma", 7.0)
        raise _StopAtMakeTrainStep

    monkeypatch.setattr(train_keypoints_mod, "make_train_step", _spy_make_train_step)

    tcfg = TrainConfig(target_sigma=2.0, mask_weight=0.0)
    vitpose_cfg = ViTPoseConfig(num_keypoints=NUM_JOINTS, in_ch=4,
                                heatmap_size=HEATMAP_SIZE)

    with pytest.raises(_StopAtMakeTrainStep):
        train_keypoints_mod.run_training(
            str(root), out_dir=str(tmp_path / "out"), tcfg=tcfg,
            vitpose_cfg=vitpose_cfg, arch="efficienttrack", smoke=True)

    assert "sigma" in captured, "make_train_step was never reached"
    assert captured["sigma"] == tcfg.target_sigma, (
        f"run_training must forward tcfg.target_sigma ({tcfg.target_sigma}) to "
        f"make_train_step's sigma kwarg; got {captured['sigma']} instead "
        "(the pre-fix wiring silently ignored train.target_sigma)")

    # Prove this is not just a numeric-equality check on a config key: render
    # an actual target with the sigma the training loop would really use and
    # confirm its width matches sigma=2.0 (tight), not sigma=7.0 (diffuse) --
    # the same statistic test_transforms_sigma.py uses directly on
    # gaussian_heatmaps.
    xy = np.array([[112.0, 112.0]], np.float32)[None]     # (B=1,K=1,2)
    vis = np.array([1], np.int32)[None]                    # (B=1,K=1)
    hm = np.asarray(render_heatmaps(
        xy, vis, heatmap_size=HEATMAP_SIZE, sigma=captured["sigma"]))[0, :, :, 0]
    assert _concentration(hm) > 0.95, (
        "heatmap rendered with the sigma reaching the training loop is not "
        "sharply concentrated -- sigma did not actually reach 2.0")
