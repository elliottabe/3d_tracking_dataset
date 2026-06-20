import os
import subprocess
import sys
import jax.numpy as jnp
import numpy as np
import pytest

from jarvis_jax import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose

NPZ = os.environ.get("MAE_NPZ", "/tmp/mae_vitb.npz")


@pytest.mark.skipif(not os.path.exists(NPZ), reason="needs Task 8 npz")
def test_roundtrip_checkpoint(tmp_path):
    ck = tmp_path / "ck"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jarvis_jax.convert.build_checkpoint",
            "--npz",
            NPZ,
            "--out",
            str(ck),
        ],
        check=True,
    )
    m = load_vitpose(str(ck), ViTPoseConfig())
    out = np.asarray(m(jnp.zeros((1, 448, 448, 4))))
    assert out.shape == (1, 224, 224, 50) and np.isfinite(out).all()
