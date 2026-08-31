"""Convert EfficientTrack-medium_final.pth (CenterDetect) -> Orbax ckpt ->
load -> parity vs the real-frame fixture. Mirrors
``test_efficienttrack_convert.py`` (the ``large``/KeypointDetect sibling)."""
import os
import numpy as np
import jax.numpy as jnp
import pytest

from test_centerdetect_parity import _assert_center_detect_parity

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                    "fixtures", "centerdetect_medium.npz")
PTH = os.path.join(os.path.dirname(__file__), "..", "..", "JARVIS-HybridNet",
                    "projects", "fly50_V6", "models", "CenterDetect",
                    "Run_20260810-094955", "EfficientTrack-medium_final.pth")
pytestmark = pytest.mark.skipif(not (os.path.exists(FIX) and os.path.exists(PTH)),
                                 reason="needs fixture + .pth")


def test_convert_roundtrip_matches_fixture(tmp_path):
    from jarvis_jax.convert.load_efficienttrack import (
        convert_efficienttrack_pth, load_efficienttrack_ckpt)

    z = np.load(FIX, allow_pickle=True)
    num_joints = int(z["num_joints"])
    out = tmp_path / "cd_ckpt"

    convert_efficienttrack_pth(PTH, num_joints=num_joints, out_dir=str(out),
                                model_size="medium")
    net = load_efficienttrack_ckpt(str(out), num_joints=num_joints,
                                    in_channels=int(z["in_channels"]),
                                    model_size="medium")

    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))  # NCHW -> NHWC
    ref = np.transpose(z["res2"], (0, 2, 3, 1))

    res2 = net(x)
    assert res2.shape == ref.shape, (res2.shape, ref.shape)
    _assert_center_detect_parity(res2, ref, name="convert_centerdetect_res2")
