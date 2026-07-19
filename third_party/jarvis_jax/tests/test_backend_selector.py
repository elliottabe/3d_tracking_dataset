# tests/test_backend_selector.py
"""Task 9: selectable 2-D front-end (+ mask fusion) in the 3-D inference path.

There is no separate "vitpose_dlt" triangulation path in this codebase --
`load_inference_model` has always built a single `HybridNet3D` (front-end ->
reproject -> V2VNet -> soft-argmax). This module adds a `front_end` selector
('vitpose' default vs opt-in 'efficienttrack') plus mask-fusion passthrough,
without disturbing the default construction/behavior.
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

HN = os.path.join(os.path.dirname(__file__), "..", "..", "JARVIS-HybridNet",
                   "projects", "unified_V3_masked", "models", "HybridNet",
                   "Run_20260620-173554", "HybridNet-large_final.pth")
_HAS_HN = os.path.exists(HN)


def test_default_frontend_dispatches_to_vitpose_unchanged(monkeypatch, tmp_path):
    """Cheap construction/signature smoke check for the default front_end='vitpose'
    path (Requirement: default must stay byte-identical to pre-Task-9 behavior).

    Building + running a REAL ViTPose (depth=12, embed_dim=768, 784 tokens) on
    CPU is expensive and additionally requires a real Orbax ViTPose checkpoint
    that may not be present in this environment -- so this test does not do a
    full predict_batch forward for the vitpose path (that full-stack, real-
    weights forward is exercised by test_hybridnet_efficienttrack_parity.py's
    test_vitpose_frontend_unchanged, which calls HybridNet3D directly with a
    freshly constructed ViTPose). Instead it monkeypatches
    infer_3d.load_vitpose to confirm load_inference_model's default args
    dispatch to EXACTLY the pre-existing vitpose construction call (same ckpt
    path, same cfg.num_keypoints), and that the returned model carries the
    unchanged default fusion_mode='none'/_frontend_channels=None (so
    predict_batch never slices crops4's channel axis for this front-end).
    """
    import jarvis_jax.predict.infer_3d as infer_3d
    from jarvis_jax.hybridnet.v2vnet import V2VNet

    J = 6
    calls = []
    sentinel = object()

    def fake_load_vitpose(ckpt, cfg):
        calls.append((ckpt, cfg.num_keypoints))
        return sentinel

    monkeypatch.setattr(infer_3d, "load_vitpose", fake_load_vitpose)

    # Real (but tiny-cost-to-save) V2VNet checkpoint -- saving state is just
    # writing arrays, no forward pass required.
    v2v = V2VNet(J, J, rngs=nnx.Rngs(0))
    gdef, state = nnx.split(v2v)
    import orbax.checkpoint as ocp
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
    assert model._frontend_channels is None
    assert hasattr(model, "_mesh")


@pytest.mark.skipif(not _HAS_HN, reason="needs HybridNet-large_final.pth weights")
def test_load_inference_model_efficienttrack_builds_and_predicts(tmp_path):
    """front_end='efficienttrack': convert real HybridNet effTrack+v2vNet
    weights into Orbax dirs, build via load_inference_model, and run
    predict_batch on a tiny synthetic crop batch -> (B, J, 3) finite points3D.
    """
    from jarvis_jax.convert.load_efficienttrack import convert_efficienttrack_pth
    from jarvis_jax.convert.load_v2vnet_torch import convert_v2vnet_pth
    from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch

    J = 50
    et_dir = tmp_path / "et"
    v2v_dir = tmp_path / "v2v"
    # HybridNet's effTrack sub-model is 3-channel (RGB only); in_channels is
    # inferred from the checkpoint's stem conv shape inside the converter.
    convert_efficienttrack_pth(str(HN), num_joints=J, out_dir=str(et_dir),
                               strip_prefix="effTrack.")
    convert_v2vnet_pth(str(HN), in_ch=J, out_ch=J, out_dir=str(v2v_dir))

    model = load_inference_model(
        None, str(v2v_dir),  # vitpose_ckpt unused for this front_end
        front_end="efficienttrack", efficienttrack_ckpt=str(et_dir),
        num_keypoints=J, sharpen=1.0,
    )
    assert model.fusion_mode == "none"
    assert model._frontend_channels == 3

    # Tiny synthetic crop batch: (B, nc, 448, 448, 4) uint8. The mask channel
    # (index 3) is left all-zero -- estimate_center3d_from_masks then reports
    # n_valid=0 per item and short-circuits to center3D=zeros (see its source:
    # the SVD-triangulated point is only used when n_valid>=2), so this
    # exercises the real code path without needing a triangulation-friendly
    # multi-camera geometry fixture. Channel 3 is sliced away before the
    # EfficientTrack forward regardless (see _frontend_channels handling).
    rng = np.random.RandomState(0)
    B, nc = 2, 3
    crops4 = np.zeros((B, nc, 448, 448, 4), dtype=np.uint8)
    crops4[..., :3] = rng.randint(0, 255, (B, nc, 448, 448, 3), dtype=np.uint8)

    # centerHM=113 (heatmap_size/2) + a W==1 (no-perspective-divide) camera --
    # matches test_hybridnet_fusion.py's proven-non-degenerate fixture. A bare
    # jnp.eye(4, 3) camera (X/Z, Y/Z with Z running through 0 across the
    # grid) risks NaN at the Z==0 grid slice (division by zero) that a deep
    # V2VNet can spread across the whole output volume -- avoided here by a
    # W==1 projection (M[:,3,2]=1, no other column-2 entries), which removes
    # the perspective divide entirely.
    centerHM = np.broadcast_to(np.array([113.0, 113.0], dtype=np.float32),
                               (B, nc, 2)).copy()
    M = np.zeros((nc, 4, 3), dtype=np.float32)
    for c in range(nc):
        M[c, 0, 0] = 10.0
        M[c, 1, 1] = 10.0
        M[c, 3, 2] = 1.0
    cameraMatrices = np.broadcast_to(M[None], (B, nc, 4, 3)).copy()

    kp3d, conf, center3D = predict_batch(model, crops4, centerHM, cameraMatrices)

    assert kp3d.shape == (B, J, 3), kp3d.shape
    assert conf.shape == (B, J), conf.shape
    assert center3D.shape == (B, 3), center3D.shape
    assert np.all(np.isfinite(kp3d))
    assert np.all(np.isfinite(conf))


def test_predict_batch_masks_threaded_for_carve_fusion(tmp_path):
    """Direct (no real-checkpoint dependency) coverage of predict_batch's new
    `masks` param: fusion_mode='carve' must actually change the output when
    masks differ, and fusion_mode='none' must ignore masks entirely (the
    byte-identical regression guard), matching test_hybridnet_fusion.py's
    HybridNet3D.__call__-level guarantees but exercised through the sharded
    predict_batch/step_vol path used in production.
    """
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.predict.infer_3d import predict_batch
    from jarvis_jax.sharding import data_parallel_mesh, replicate

    J = 4
    B, nc = 1, 3

    class ConstFrontEnd(nnx.Module):
        def predict_heatmaps(self, crops):  # (B,cam,H,W,C) -> constant heatmaps
            Bc, camc = crops.shape[0], crops.shape[1]
            return jnp.ones((Bc, camc, 224, 224, J), "float32")

    def _build(fusion_mode):
        class Cfg:
            num_keypoints = J
            sharpen = 1.0
            gate_temperature = 0.5
            gate_floor = 0.0

        cfg = Cfg()
        cfg.fusion_mode = fusion_mode
        m = HybridNet3D(ConstFrontEnd(), V2VNet(J, J, rngs=nnx.Rngs(0)), cfg)
        mesh = data_parallel_mesh()
        gd, st = nnx.split(m)
        m = nnx.merge(gd, replicate(st, mesh))
        m._mesh = mesh
        m._frontend_channels = None  # ConstFrontEnd ignores channel count
        return m

    # Fixture mirrors test_hybridnet_fusion.py's proven-non-degenerate camera
    # geometry (see that file's _inputs() docstring for why a bare jnp.eye(4,3)
    # camera reprojects to an all-zero pre-gate volume and is unusable here).
    # crops4's mask channel (index 3) is left all-zero, so
    # estimate_center3d_from_masks naturally yields center3D=zeros -- matching
    # that file's explicit c3=jnp.zeros((B,3)) -- without hand-deriving a
    # triangulation-friendly mask blob.
    crops4 = np.zeros((B, nc, 448, 448, 4), dtype=np.uint8)
    centerHM = np.broadcast_to(np.array([113.0, 113.0], dtype=np.float32),
                               (B, nc, 2)).copy()
    M = np.zeros((nc, 4, 3), dtype=np.float32)
    for c in range(nc):
        M[c, 0, 0] = 10.0
        M[c, 1, 1] = 10.0
        M[c, 3, 2] = 1.0
    cameraMatrices = np.broadcast_to(M[None], (B, nc, 4, 3)).copy()

    # fusion_mode='none': masks must be ignored (result identical with/without).
    m_none = _build("none")
    kp_a, conf_a, _ = predict_batch(m_none, crops4, centerHM, cameraMatrices, masks=None)
    full_masks = np.ones((B, nc, 226, 226), dtype=np.float32)
    kp_b, conf_b, _ = predict_batch(m_none, crops4, centerHM, cameraMatrices, masks=full_masks)
    np.testing.assert_array_equal(kp_a, kp_b)
    np.testing.assert_array_equal(conf_a, conf_b)

    # fusion_mode='carve': a partially-carved mask must change the prediction
    # relative to a full (all-ones) mask.
    m_carve = _build("carve")
    kp_full, _, _ = predict_batch(m_carve, crops4, centerHM, cameraMatrices, masks=full_masks)
    partial_masks = full_masks.copy()
    partial_masks[:, :, 120:, :] = 0.0
    kp_partial, _, _ = predict_batch(m_carve, crops4, centerHM, cameraMatrices, masks=partial_masks)

    assert kp_full.shape == (B, J, 3)
    assert np.all(np.isfinite(kp_full)) and np.all(np.isfinite(kp_partial))
    assert not np.allclose(kp_full, kp_partial)
