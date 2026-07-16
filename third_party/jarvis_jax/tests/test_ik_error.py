import numpy as np
import pytest
mujoco = pytest.importorskip("mujoco")
from jarvis_jax.tracking import ik_error as ike

# A planar 2-hinge arm (links length 1) with a site "tracking[tip]" at the end.
_XML = """
<mujoco>
  <worldbody>
    <body name="l1" pos="0 0 0"><joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size="0.02"/>
      <body name="l2" pos="1 0 0"><joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.02"/>
        <site name="tracking[tip]" pos="1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _mjx():
    import mujoco.mjx as mjx
    m = mujoco.MjModel.from_xml_string(_XML)
    mx = mjx.put_model(m)
    return m, mx, mjx.make_data(mx)


def test_marker_site_ids_finds_tracking_site():
    m, _, _ = _mjx()
    ids = ike.marker_site_ids(m, ["tip"])
    assert ids.shape == (1,)
    assert ids[0] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tracking[tip]")


def test_marker_jacobian_matches_analytic_planar_arm():
    import jax.numpy as jnp
    m, mx, d0 = _mjx()
    ids = ike.marker_site_ids(m, ["tip"])
    q = jnp.array([0.3, 0.4])                        # (nq=2)
    J = np.asarray(ike.marker_jacobian(mx, d0, q, ids))   # (3, 2)
    q1, q2 = 0.3, 0.4
    dx = np.array([-np.sin(q1) - np.sin(q1 + q2), -np.sin(q1 + q2)])   # d x / d[q1,q2]
    dy = np.array([np.cos(q1) + np.cos(q1 + q2), np.cos(q1 + q2)])
    assert np.allclose(J[0], dx, atol=1e-4)
    assert np.allclose(J[1], dy, atol=1e-4)
    assert np.allclose(J[2], 0, atol=1e-5)           # planar -> z insensitive


def test_qpos_sensitivity_shapes_and_isotropic_scaling():
    import jax.numpy as jnp
    J = jnp.asarray(np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]))   # (3,2)
    W = jnp.eye(3)
    out = ike.qpos_sensitivity(J, W, jnp.eye(3))          # iso unit noise
    assert out["transfer"].shape == (2,) and out["std"].shape == (2,)
    # joint 1 (marker-dim0 gain 1) more sensitive than joint 2 (gain 2)
    assert out["transfer"][0] > out["transfer"][1]


def test_dof_units_classifies_free_and_hinge():
    names = ["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz", "j_femur"]
    du = ike.dof_units(names)
    kinds = [d["kind"] for d in du]
    assert kinds[:3] == ["root_trans"] * 3
    assert kinds[3:7] == ["root_quat"] * 4
    assert kinds[7] == "hinge"
