"""Convert EfficientTrack-large_final.pth -> Orbax ckpt -> load -> parity vs fixture."""
import os
import numpy as np
import jax.numpy as jnp
import pytest

from test_efficienttrack_parity import _assert_parity

FIX = os.path.join(os.path.dirname(__file__), "..", "jarvis_jax", "convert",
                    "fixtures", "efficienttrack_large.npz")
PTH = os.path.join(os.path.dirname(__file__), "..", "..", "JARVIS-HybridNet",
                    "projects", "unified_V3_masked", "models", "KeypointDetect",
                    "Run_20260619-185121", "EfficientTrack-large_final.pth")
pytestmark = pytest.mark.skipif(not (os.path.exists(FIX) and os.path.exists(PTH)),
                                 reason="needs fixture + .pth")


def test_convert_roundtrip_matches_fixture(tmp_path):
    from jarvis_jax.convert.load_efficienttrack import (
        convert_efficienttrack_pth, load_efficienttrack_ckpt)

    z = np.load(FIX, allow_pickle=True)
    num_joints = int(z["num_joints"])
    out = tmp_path / "et_ckpt"

    convert_efficienttrack_pth(PTH, num_joints=num_joints, out_dir=str(out))
    net = load_efficienttrack_ckpt(str(out), num_joints=num_joints)

    x = jnp.asarray(np.transpose(z["input_nchw"], (0, 2, 3, 1)))  # NCHW -> NHWC
    ref = np.transpose(z["res2"], (0, 2, 3, 1))

    res2 = net(x)
    assert res2.shape == ref.shape, (res2.shape, ref.shape)
    _assert_parity(res2, ref, name="convert_res2")
