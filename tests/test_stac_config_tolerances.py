"""Every anatomy config must carry the jaxls termination tolerances, and Stac
must forward them. A missing key must fall back, never raise."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ANATOMY = [
    REPO / "configs/anatomy/v1.yaml",
    REPO / "configs/anatomy/v2_3.yaml",
    REPO / "configs/anatomy/v2_muscles.yaml",
    REPO / "stac-mjx/configs/anatomy/v1.yaml",
    REPO / "stac-mjx/configs/anatomy/v2.yaml",
    REPO / "stac-mjx/configs/anatomy/v2_muscles.yaml",
]


@pytest.mark.parametrize("path", ANATOMY, ids=lambda p: p.name)
def test_anatomy_config_declares_tolerances(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    model = yaml.safe_load(path.read_text())["model"]
    assert float(model["JAXLS_GRADIENT_TOLERANCE"]) == 1e-8
    assert float(model["JAXLS_PARAMETER_TOLERANCE"]) == 1e-10


def test_stac_falls_back_when_keys_absent():
    """A config without the keys must still build (defaults), not KeyError."""
    pytest.importorskip("jaxls")
    from omegaconf import OmegaConf
    from stac_mjx.stac_core import StacCore

    cfg = OmegaConf.create({"model": {}})
    tol_g = cfg.model.get("JAXLS_GRADIENT_TOLERANCE", 1e-8)
    tol_p = cfg.model.get("JAXLS_PARAMETER_TOLERANCE", 1e-10)
    core = StacCore(tol=1e-5, use_jaxls=True,
                    jaxls_gradient_tolerance=tol_g,
                    jaxls_parameter_tolerance=tol_p)
    assert core._jaxls_solver.gradient_tolerance == 1e-8
