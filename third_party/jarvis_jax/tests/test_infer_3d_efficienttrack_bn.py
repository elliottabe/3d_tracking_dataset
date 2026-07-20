"""front_end='efficienttrack_bn': our production 4-channel EfficientTrackBN
detector wired into the 3-D inference/bout path (load_inference_model +
predict_batch), CPU-testable against real 'final' checkpoints when present.

Prior to this change, `load_inference_model`'s only non-default branch
(`front_end='efficienttrack'`) built a WRONG 3-channel HybridNet-port
EfficientTrack (via `load_efficienttrack_ckpt(..., in_channels=3)`) and set
`_frontend_channels=3` so `predict_batch` sliced away the SAM3 mask channel --
that is not our detector. Our detector is `EfficientTrackBN` (4 channels,
trained on `normalize_image`d RGB+mask input by the 2-D trainer), restored the
same way `scripts/precompute_repro_cache.py::_build_front_end` does for
`front_end='efficienttrack_bn'`: `eval_keypoints_2d.restore_model(ckpt,
"efficienttrack_bn", cfg)` + `HybridNet3D(..., normalize_frontend=True)`, with
`_frontend_channels=None` so the full 4-channel crop reaches the front-end.
"""
import os

import numpy as np
import pytest

ET_BN_CKPT = ("/gscratch/portia/eabe/data/Johnson_lab/jax_efficienttrack_runs/"
              "et2d_bn_imagenet/final")
V2V_ETBN = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v2v_etbn/final"
_HAVE_CKPTS = os.path.isdir(ET_BN_CKPT) and os.path.isdir(V2V_ETBN)


def _tiny_camera_fixture(B, nc):
    """Same non-degenerate, no-perspective-divide (W==1) camera fixture used
    by test_backend_selector.py / test_hybridnet_fusion.py -- avoids a Z==0
    grid-slice division-by-zero (and its NaN-spreading-through-V2VNet risk)
    that a bare jnp.eye(4, 3) camera would hit."""
    centerHM = np.broadcast_to(np.array([113.0, 113.0], dtype=np.float32),
                               (B, nc, 2)).copy()
    M = np.zeros((nc, 4, 3), dtype=np.float32)
    for c in range(nc):
        M[c, 0, 0] = 10.0
        M[c, 1, 1] = 10.0
        M[c, 3, 2] = 1.0
    cameraMatrices = np.broadcast_to(M[None], (B, nc, 4, 3)).copy()
    return centerHM, cameraMatrices


@pytest.mark.skipif(not _HAVE_CKPTS, reason="needs et2d_bn_imagenet + v2v_etbn 'final' checkpoints")
def test_load_inference_model_efficienttrack_bn_builds_correctly():
    """front_end='efficienttrack_bn' must restore a real EfficientTrackBN
    (not the 3-channel HybridNet-port EfficientTrack) with normalize_frontend
    threaded all the way to HybridNet3D, and must NOT slice the mask channel
    (frontend_channels stays None)."""
    from jarvis_jax.models.efficienttrack import EfficientTrackBN
    from jarvis_jax.predict.infer_3d import load_inference_model

    J = 50
    model = load_inference_model(
        None, V2V_ETBN,  # vitpose_ckpt unused for this front_end
        front_end="efficienttrack_bn", efficienttrack_ckpt=ET_BN_CKPT,
        num_keypoints=J, sharpen=3.0,
    )

    assert isinstance(model.front_end, EfficientTrackBN), type(model.front_end)
    assert model.normalize_frontend is True
    assert model.fusion_mode == "none"
    # Must NOT slice the mask channel away -- our detector consumes all 4.
    assert model._frontend_channels is None
    assert hasattr(model, "_mesh")


@pytest.mark.skipif(not _HAVE_CKPTS, reason="needs et2d_bn_imagenet + v2v_etbn 'final' checkpoints")
def test_predict_batch_efficienttrack_bn_finite():
    """End-to-end predict_batch (front-end -> reproject -> V2VNet -> soft-
    argmax) on a tiny synthetic 4-channel crop batch. Uses the SAME sharded
    predict_batch entry point production code calls (no bespoke device-mesh
    setup needed here -- load_inference_model already replicates the model
    and stashes model._mesh, and predict_batch reads that directly)."""
    from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch

    J = 50
    model = load_inference_model(
        None, V2V_ETBN,
        front_end="efficienttrack_bn", efficienttrack_ckpt=ET_BN_CKPT,
        num_keypoints=J, sharpen=3.0,
    )

    rng = np.random.RandomState(0)
    B, nc = 1, 7
    crops4 = np.zeros((B, nc, 448, 448, 4), dtype=np.uint8)
    crops4[..., :3] = rng.randint(0, 255, (B, nc, 448, 448, 3), dtype=np.uint8)
    # Leave the mask channel (index 3) all-zero: estimate_center3d_from_masks
    # then short-circuits to center3D=zeros per item (n_valid=0 < 2), same
    # convention as test_backend_selector.py's efficienttrack fixture -- this
    # exercises the real code path without needing a triangulation-friendly
    # multi-camera geometry fixture.

    centerHM, cameraMatrices = _tiny_camera_fixture(B, nc)

    kp3d, conf, center3D = predict_batch(model, crops4, centerHM, cameraMatrices)

    assert kp3d.shape == (B, J, 3), kp3d.shape
    assert conf.shape == (B, J), conf.shape
    assert center3D.shape == (B, 3), center3D.shape
    assert np.all(np.isfinite(kp3d))
    assert np.all(np.isfinite(conf))


def test_default_vitpose_path_unchanged(monkeypatch, tmp_path):
    """Cheap regression guard (no real checkpoints needed): the default
    front_end='vitpose' construction path must still dispatch exactly as
    before -- unaffected by the new efficienttrack_bn branch. Mirrors
    test_backend_selector.py's equivalent monkeypatched-vitpose check."""
    import jarvis_jax.predict.infer_3d as infer_3d
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from flax import nnx
    import orbax.checkpoint as ocp

    J = 6
    calls = []
    sentinel = object()

    def fake_load_vitpose(ckpt, cfg):
        calls.append((ckpt, cfg.num_keypoints))
        return sentinel

    monkeypatch.setattr(infer_3d, "load_vitpose", fake_load_vitpose)

    v2v = V2VNet(J, J, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    ckptr = ocp.StandardCheckpointer()
    v2v_dir = tmp_path / "v2v"
    ckptr.save(str(v2v_dir), state)
    ckptr.wait_until_finished()

    model = infer_3d.load_inference_model(
        "unused/vitpose/ckpt", str(v2v_dir), sharpen=3.0, num_keypoints=J,
    )

    assert calls == [("unused/vitpose/ckpt", J)]
    assert model.front_end is sentinel
    assert model.fusion_mode == "none"
    assert model.normalize_frontend is False
    assert model._frontend_channels is None
