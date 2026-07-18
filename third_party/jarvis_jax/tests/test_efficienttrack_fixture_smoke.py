"""Smoke test: the exported EfficientTrack golden fixture has the expected keys.

The fixture is produced on a compute node by
``jarvis_jax/convert/export_efficienttrack_fixture.py`` (PyTorch env). This test
SKIPs when the fixture is absent so the JAX-only CI stays green; it must PASS once
the fixture has been generated.
"""
import os

import numpy as np
import pytest

FIX = os.path.join(
    os.path.dirname(__file__), "..", "jarvis_jax", "convert", "fixtures",
    "efficienttrack_large.npz",
)


@pytest.mark.skipif(not os.path.exists(FIX), reason="run export_efficienttrack_fixture.py first")
def test_fixture_has_expected_keys():
    z = np.load(FIX, allow_pickle=True)
    for k in ["input_nchw", "feat_p3", "feat_p4", "feat_p5", "res1", "res2", "num_joints"]:
        assert k in z, f"missing {k}"
    assert z["res2"].shape[-2:] == (224, 224), z["res2"].shape
    assert z["feat_p3"].shape[1] == 24, z["feat_p3"].shape
    assert z["feat_p4"].shape[1] == 48, z["feat_p4"].shape
    assert z["feat_p5"].shape[1] == 120, z["feat_p5"].shape
    assert any(k.startswith("w::") for k in z.files), "no weights exported"
