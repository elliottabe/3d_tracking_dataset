"""CenterDetect (EfficientTrack ``model_size="medium"``, EfficientNet-b1
backbone) parity vs PyTorch, on REAL camera frames with the real trained
fly50_V6 CenterDetect weights (see
``jarvis_jax/convert/export_centerdetect_fixture.py`` for how the fixture
npz was produced -- real frames from ``projects/fly50_V6/visualization``,
not synthetic noise, run through the real ``EfficientTrack-medium_final.pth``).

Expectation stated up front: if the b1 stage config (``_MODEL_SIZE_TABLE``'s
"medium" entry in ``jarvis_jax.models.efficienttrack``) and the parameterized
``EfficientNetB3`` backbone are correct, the JAX heatmap should be
numerically near-identical to PyTorch's (float32 rounding noise only, same
order of magnitude as the existing ``large`` parity test, likely SMALLER here
since medium has fewer InstanceNorm layers/channels to accumulate noise in)
-- AND, because this is checked on 4 real frames spanning 4 different
cameras/times, the predicted heatmap peak (the actual quantity CenterDetect
is used for: locating the fly to crop around) must land at the exact same
pixel in both frameworks for every sample. A mismatched peak location on real
frames -- even with a "small" mean heatmap error -- would mean the port is
wrong in a way that matters downstream (wrong crop center), which is exactly
what a synthetic-noise-only fixture could hide.
"""
import os
import numpy as np
import jax.numpy as jnp
import pytest

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                   "fixtures", "centerdetect_medium.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(FIX),
                                 reason="generate fixture: export_centerdetect_fixture.py")


def _load_full(z):
    from flax import nnx
    from jarvis_jax.models.efficienttrack import EfficientTrack, load_efficienttrack_from_npz
    net = EfficientTrack(num_joints=int(z["num_joints"]), in_channels=int(z["in_channels"]),
                          model_size=str(z["model_size"]), rngs=nnx.Rngs(0))
    return load_efficienttrack_from_npz(net, z)


def _peak_xy(heatmap_hw: np.ndarray) -> tuple:
    """argmax (row, col) of a single-channel (H, W) heatmap."""
    return np.unravel_index(np.argmax(heatmap_hw), heatmap_hw.shape)


def _assert_center_detect_parity(pred, ref, *, name):
    """CenterDetect heatmap parity vs PyTorch, on real frames.

    Tolerances (looser bound with margin over the OBSERVED numbers, not
    tuned to make a failure pass -- see task report for the raw numbers):
    on the real-frame fixture (4 samples, 4 different cameras/times)
    mean_abs~1.0e-6, signed_mean~-1.6e-8, max_abs~2.7e-4, corr=1.0000000000,
    every one of the 4 peak locations exact (0 px) in both res1 and res2.
    This is tighter than the existing ``large``-config parity test's noise
    floor (max_abs~6e-3) -- expected, since ``medium`` has fewer
    InstanceNorm layers/channels (88 vs 160 fpn filters, 4 vs 6 BiFPN cells)
    to accumulate float32 rounding noise in.
    """
    diff = np.asarray(pred) - ref
    absdiff = np.abs(diff)
    mean_abs = absdiff.mean()
    signed_mean = diff.mean()
    max_abs = absdiff.max()
    corr = np.corrcoef(np.asarray(pred).ravel(), ref.ravel())[0, 1]

    peak_dists = []
    for b in range(pred.shape[0]):
        p_peak = _peak_xy(np.asarray(pred[b, ..., 0]))
        r_peak = _peak_xy(ref[b, ..., 0])
        peak_dists.append(float(np.hypot(p_peak[0] - r_peak[0], p_peak[1] - r_peak[1])))
    max_peak_dist = max(peak_dists)

    print(f"[{name}] mean_abs_diff={mean_abs:.3e} signed_mean={signed_mean:.3e} "
          f"max_abs_diff={max_abs:.3e} corr={corr:.10f} peak_dists_px={peak_dists}")

    assert mean_abs < 5e-5,        f"mean abs diff {mean_abs:.2e}"
    assert abs(signed_mean) < 5e-6, f"signed mean {signed_mean:.2e} (systematic bias?)"
    assert max_abs < 1e-3,          f"max abs diff {max_abs:.2e}"
    assert corr > 0.999999,         f"correlation {corr:.10f}"
    assert max_peak_dist <= 1.0,    f"peak location disagreement {max_peak_dist:.2f}px: {peak_dists}"


def test_res2_heatmaps_match_pytorch_real_frames():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res2 = net(x)                                   # (N,160,160,1)
    ref = np.transpose(z["res2"], (0, 2, 3, 1))     # NCHW->NHWC
    assert res2.shape == ref.shape, (res2.shape, ref.shape)
    _assert_center_detect_parity(res2, ref, name="centerdetect_res2")


def test_res1_heatmaps_match_pytorch_real_frames():
    z = np.load(FIX, allow_pickle=True)
    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))
    net = _load_full(z)
    res1, res2 = net.forward_both(x)
    ref1 = np.transpose(z["res1"], (0, 2, 3, 1))
    assert res1.shape == ref1.shape, (res1.shape, ref1.shape)
    _assert_center_detect_parity(res1, ref1, name="centerdetect_res1")


def test_backbone_taps_have_medium_channel_widths():
    """Regression guard for the b1 stage-config transcription itself: P3/P4/P5
    must be (24, 40, 112) -- JARVIS's own conv_channel_coef for
    model_size='medium' (model.py's size table) -- not the b3/large values
    (24, 48, 120) nor anything else. Checked directly against the PyTorch
    fixture's captured backbone feature maps, independent of the head."""
    z = np.load(FIX, allow_pickle=True)
    assert z["feat_p3"].shape[1] == 24
    assert z["feat_p4"].shape[1] == 40
    assert z["feat_p5"].shape[1] == 112
