# tests/test_hybridnet_efficienttrack_parity.py
"""Full-stack parity: crops -> EfficientTrack -> reproject -> V2VNet ->
soft-argmax -> points3D, against the PyTorch ``HybridNetBackbone`` golden
fixture (Task 5).

Also carries a ViTPose regression guard: HybridNet3D's pluggable front-end
refactor must not change the (byte-identical) ViTPose construction/forward
path used by ``infer_3d.py``, ``train_3d.py``, etc.
"""
import os

import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                    "fixtures", "hybridnet_large.npz")
HN = os.path.join(os.path.dirname(__file__), "..", "..", "JARVIS-HybridNet",
                   "projects", "unified_V3_masked", "models", "HybridNet",
                   "Run_20260620-173554", "HybridNet-large_final.pth")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(FIX) and os.path.exists(HN)),
    reason="needs hybridnet fixture + HybridNet-large_final.pth weights",
)


def test_points3d_match_pytorch(tmp_path):
    """End-to-end points3D parity vs the PyTorch HybridNetBackbone oracle.

    Bound rationale (see task-5 brief, and the investigation recorded in
    task-5-report.md): points3D are world coordinates on a 48-unit cube
    (magnitudes ~O(10-220) here). The pipeline composes several
    already-parity-tested stages: EfficientTrack (heatmaps), reproject_heatmaps
    (documented ~1-ULP boundary-voxel residual), V2VNet, and soft_argmax_3d
    (exact, pure-JAX). A DIRECT isolation check (run once during development,
    not part of this test) loaded the SAME effTrack.* weights used here into
    the reference PyTorch ``EfficientTrackBackbone`` (CPU; unlike
    ``HybridNetBackbone``/``ReprojectionLayer`` it is not CUDA-hardcoded) and
    compared res2 heatmaps for the SAME fixture crops directly against this
    module's output: mean_abs=9.6e-5, max_abs=8.2e-3, corr=1.00000000 --
    i.e. EfficientTrack itself is exactly as tight as the already-validated
    standalone-checkpoint case (test_efficienttrack_parity.py). So the
    observed points3D-level residual (measured: mean_abs~0.084, max_abs~0.344,
    corr~0.9999992, no systematic per-axis bias, confidence stats match to
    ~1e-3) is NOT a wiring bug (a transpose/pad/scale/center3D-offset/sharpen
    bug would show up as a large systematic bias or corr well below 1, and
    would already be visible at the heatmap level checked above) -- it is
    float32-noise from EfficientTrack + reproject's documented 1-ULP residual,
    nonlinearly amplified by the soft-argmax weighted-centroid reduction over
    volumes that are peaky-but-not-extremely-so (max/mean ratio ~95-190) when
    driven by meaningless torch.rand() input images (this fixture's crops are
    NOT real fly images, so heatmap confidence is low/diffuse: mean conf
    ~0.26 of max), a harsher stress-test than production input would be. corr
    is the load-bearing wiring-bug guard here (an axis/offset bug collapses
    it far below 1); mean/max bound the compounded-noise floor and would flag
    if it silently grew.
    """
    from jarvis_jax.convert.load_efficienttrack import (
        convert_efficienttrack_pth, load_efficienttrack_ckpt,
    )
    from jarvis_jax.convert.load_v2vnet_torch import (
        convert_v2vnet_pth, load_v2vnet_ckpt,
    )
    from jarvis_jax.hybridnet.model import HybridNet3D

    z = np.load(FIX, allow_pickle=True)
    J = z["points3D"].shape[1]

    et_dir = tmp_path / "et"
    v2v_dir = tmp_path / "v2v"
    # HybridNet's effTrack sub-model is 3-channel (RGB only, no mask channel);
    # in_channels is inferred from the checkpoint's stem conv shape.
    convert_efficienttrack_pth(str(HN), num_joints=J, out_dir=str(et_dir),
                               strip_prefix="effTrack.")
    convert_v2vnet_pth(str(HN), in_ch=J, out_ch=J, out_dir=str(v2v_dir))

    front = load_efficienttrack_ckpt(str(et_dir), num_joints=J, in_channels=3)
    v2v = load_v2vnet_ckpt(str(v2v_dir), in_ch=J, out_ch=J)

    class Cfg:
        num_keypoints = J
        sharpen = 1.0  # PyTorch reference has no sharpen -- must be 1.0 for parity

    model = HybridNet3D(front, v2v, Cfg())

    # (B, cam, 3, 448, 448) -> (B, cam, 448, 448, 3) NHWC per camera
    crops = jnp.asarray(np.transpose(z["imgs_nchw"], (0, 1, 3, 4, 2)))
    center3D = jnp.asarray(z["center3D"])
    centerHM = jnp.asarray(z["centerHM"])
    cameraMatrices = jnp.asarray(z["cameraMatrices"])

    _, pts, conf = model(crops, center3D, centerHM, cameraMatrices,
                         use_running_average=True)

    pred = np.asarray(pts)
    ref = z["points3D"]
    assert pred.shape == ref.shape, (pred.shape, ref.shape)

    diff = pred - ref
    absdiff = np.abs(diff)
    mean_abs = absdiff.mean()
    max_abs = absdiff.max()
    signed_mean = diff.mean()
    corr = np.corrcoef(pred.ravel(), ref.ravel())[0, 1]
    print(f"[hybridnet points3D parity] mean_abs_diff={mean_abs:.4e} "
          f"max_abs_diff={max_abs:.4e} signed_mean={signed_mean:.4e} "
          f"corr={corr:.8f} (n_coords={absdiff.size})")

    # corr is the primary wiring-bug tripwire (see docstring): a transpose,
    # axis-order, scale, or center3D-offset bug would collapse this far below
    # 1, and would already be visible in the isolated EfficientTrack heatmap
    # check (corr=1.00000000) cited above.
    assert corr > 0.9999, f"correlation {corr:.8f} -- suspect wiring bug"
    assert abs(signed_mean) < 0.1, f"signed mean {signed_mean:.4e} (systematic bias?)"
    # Measured 0.084 (mean_abs) / 0.344 (max_abs); bounds set with headroom
    # above the measured, amplified-but-non-buggy noise floor (see docstring).
    assert mean_abs < 0.15, f"mean abs diff {mean_abs:.4e} (world units)"
    assert max_abs < 0.5, f"max abs diff {max_abs:.4e} (world units) -- suspect wiring bug"


def test_vitpose_frontend_unchanged():
    """Regression guard: constructing HybridNet3D with a ViTPose front-end via
    the legacy positional signature still works and produces correctly-shaped
    (vol, points3D, conf) outputs -- proves the pluggable-front-end refactor
    didn't disturb the existing ViTPose call path (normalize_image + ViTPose,
    use_running_average=True, hardcoded)."""
    from jarvis_jax.hybridnet.model import HybridNet3D
    from jarvis_jax.hybridnet.v2vnet import V2VNet
    from jarvis_jax.models.vitpose import ViTPose
    from jarvis_jax.config import ViTPoseConfig

    assert hasattr(HybridNet3D, "predict_heatmaps")

    cfg = ViTPoseConfig()
    rngs = nnx.Rngs(0)
    vitpose = ViTPose(cfg, rngs=rngs)
    v2vnet = V2VNet(cfg.num_keypoints, cfg.num_keypoints, rngs=rngs)
    model = HybridNet3D(vitpose, v2vnet, cfg)  # legacy positional signature

    # Backward-compat alias: code that reaches into `.vitpose` (e.g. train_3d
    # tests inspecting decoder params) must keep working.
    assert model.vitpose is vitpose

    B, num_cam, J = 1, 2, cfg.num_keypoints
    rng = np.random.RandomState(0)
    crops = jnp.asarray(rng.randint(0, 255, (B, num_cam, 448, 448, 4), dtype=np.uint8))
    hm = model.predict_heatmaps(crops)
    assert hm.shape == (B, num_cam, 224, 224, J), hm.shape

    center3D = jnp.zeros((B, 3), dtype=jnp.float32)
    centerHM = jnp.zeros((B, num_cam, 2), dtype=jnp.float32)
    cam_mat = jnp.broadcast_to(jnp.eye(4, 3, dtype=jnp.float32)[None, None],
                               (B, num_cam, 4, 3))
    vol, pts, conf = model(crops, center3D, centerHM, cam_mat, use_running_average=True)
    assert vol.shape == (B, J, 24, 24, 24), vol.shape
    assert pts.shape == (B, J, 3), pts.shape
    assert conf.shape == (B, J), conf.shape
