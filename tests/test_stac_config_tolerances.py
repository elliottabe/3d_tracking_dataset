"""Every anatomy config must carry the jaxls termination tolerances, and Stac
must forward them. A missing key must fall back, never raise."""
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
_STAC_MJX_ROOT = REPO / "stac-mjx"

# Pin `import stac_mjx` to *this* checkout's submodule. This repo is routinely
# checked out into multiple worktrees, each with its own stac-mjx submodule
# commit, and an ambient editable install of stac_mjx elsewhere on
# sys.path/PYTHONPATH would otherwise silently win and test the wrong code.
if str(_STAC_MJX_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAC_MJX_ROOT))

# Parent-repo configs are always present in a checkout of this repo: their
# absence means a rename/deletion, not an uninitialised submodule, so they
# must fail loudly rather than silently skip.
PARENT_ANATOMY = [
    REPO / "configs/anatomy/v1.yaml",
    REPO / "configs/anatomy/v2_3.yaml",
    REPO / "configs/anatomy/v2_muscles.yaml",
]
# Submodule configs may legitimately be absent if stac-mjx is uninitialised.
SUBMODULE_ANATOMY = [
    REPO / "stac-mjx/configs/anatomy/v1.yaml",
    REPO / "stac-mjx/configs/anatomy/v2.yaml",
    REPO / "stac-mjx/configs/anatomy/v2_muscles.yaml",
]
ANATOMY = PARENT_ANATOMY + SUBMODULE_ANATOMY


@pytest.mark.parametrize("path", ANATOMY, ids=lambda p: p.name)
def test_anatomy_config_declares_tolerances(path):
    if path in PARENT_ANATOMY:
        assert path.exists(), f"{path} missing -- parent-repo anatomy configs must always be present"
    elif not path.exists():
        pytest.skip(f"{path} not present (stac-mjx submodule may be uninitialised)")
    model = yaml.safe_load(path.read_text())["model"]
    assert float(model["JAXLS_GRADIENT_TOLERANCE"]) == 1e-8
    assert float(model["JAXLS_PARAMETER_TOLERANCE"]) == 1e-10
    assert float(model["JAXLS_COST_TOLERANCE"]) == 1e-5


def test_stac_forwards_tolerance_defaults_when_keys_absent():
    """Stac.__init__'s getattr(...) forwarding lines must supply the
    StacCore defaults (1e-8 / 1e-10 / 1e-5) when an anatomy config lacks
    JAXLS_GRADIENT_TOLERANCE / JAXLS_PARAMETER_TOLERANCE / JAXLS_COST_TOLERANCE.

    Builds a real `Stac` against stac-mjx's own rodent test fixtures (which
    predate this task and so genuinely lack both keys) so this exercises the
    actual forwarding lines in stac_mjx/stac.py, not a re-implementation of
    them. If the getattr(...) key names or defaults in stac.py regress, this
    test fails.
    """
    pytest.importorskip("jaxls")
    from omegaconf import OmegaConf
    from stac_mjx import main
    from stac_mjx.stac import Stac

    config_dir = _STAC_MJX_ROOT / "tests" / "configs"
    cfg = main.load_configs(str(config_dir))

    # Sanity: the fixture config must genuinely lack these keys, or this
    # test would not be exercising the fallback path at all.
    assert "JAXLS_GRADIENT_TOLERANCE" not in cfg.model
    assert "JAXLS_PARAMETER_TOLERANCE" not in cfg.model
    assert "JAXLS_COST_TOLERANCE" not in cfg.model

    # This fixture config predates the jaxls solver work: force USE_JAXLS on
    # (so StacCore actually builds a _jaxls_solver to inspect) and fill in
    # the `dt` key its own tests/configs/stac/test_stac.yaml is missing.
    # Neither of these touches the tolerance keys under test.
    OmegaConf.set_struct(cfg, False)
    cfg.model.USE_JAXLS = True
    if "dt" not in cfg.stac.mujoco:
        cfg.stac.mujoco.dt = 0.001

    xml_path = _STAC_MJX_ROOT / cfg.model.MJCF_PATH
    kp_names = list(cfg.model.KP_NAMES)

    stac = Stac(xml_path, cfg, kp_names)

    assert stac.stac_core_obj._jaxls_solver.gradient_tolerance == 1e-8
    assert stac.stac_core_obj._jaxls_solver.parameter_tolerance == 1e-10
    assert stac.stac_core_obj._jaxls_solver.cost_tolerance == 1e-5
