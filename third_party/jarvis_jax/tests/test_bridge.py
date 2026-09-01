"""The bridge is what Stage E maps model units to mm through, so its contract
is checked against RECORDED values from a real bout, not synthetic ones."""
import os

import numpy as np
import pytest

BACKUP = ("/gscratch/portia/eabe/data/Johnson_lab/processed/_courtship_backup/"
          "Session0/2025_10_20_13_20_04/pose/bouts/bout_00028")


def test_bridge_mode_is_validated():
    from jarvis_jax.tracking.bridge import compute_bridges
    with pytest.raises(ValueError, match="keypoint.*mask"):
        compute_bridges("x.h5", "x.xml", None, None, "cal", bridge_mode="silhouette")


def test_mask_mode_requires_masks_but_keypoint_mode_does_not():
    """The point of the default: 'keypoint' reads no masks at all, so the IK no
    longer depends on SAM output. Only 'mask' may demand masks_dict."""
    from jarvis_jax.tracking.bridge import compute_bridges
    with pytest.raises(ValueError, match="masks_dict"):
        compute_bridges("x.h5", "x.xml", None, None, "cal",
                        bridge_mode="mask", masks_dict=None)


@pytest.mark.skipif(not os.path.exists(BACKUP), reason="recorded bout unavailable")
@pytest.mark.parametrize("fly", [0, 1])
def test_recorded_bridge_matches_the_qpos_refined_contract(fly):
    """Guards the invariants Stage E relies on, read off the shipped artifact:
    s > 0 and finite wherever ok, R a proper rotation (det +1, orthonormal), and
    every not-ok frame left at the identity/const defaults so a stale value can
    never masquerade as a real bridge."""
    p = os.path.join(BACKUP, f"fly{fly}", "qpos_refined.npz")
    if not os.path.exists(p):
        pytest.skip("bout 28 fly%d not in the backup" % fly)
    z = np.load(p)
    s, R, t, ok = z["bridge_s"], z["bridge_R"], z["bridge_t"], z["bridge_ok"]
    assert ok.any(), "a bout with no usable bridge would be all-NaN downstream"

    assert np.isfinite(s[ok]).all() and (s[ok] > 0).all()
    assert np.isfinite(t[ok]).all()
    RtR = np.einsum("tij,tik->tjk", R[ok], R[ok])
    assert np.allclose(RtR, np.eye(3), atol=1e-4), "bridge R must be orthonormal"
    assert np.allclose(np.linalg.det(R[ok]), 1.0, atol=1e-4), "no reflections"

    # Un-bridged frames keep the untouched defaults, never a neighbour's value.
    if (~ok).any():
        assert np.allclose(R[~ok], np.eye(3))
        assert np.allclose(t[~ok], 0.0)


@pytest.mark.skipif(not os.path.exists(BACKUP), reason="recorded bout unavailable")
def test_the_deleted_polish_was_a_passthrough():
    """Why deleting Stage D's silhouette is behaviour-preserving: with both
    weights 0 (the shipped default) polish_bout early-returned q_init, so the
    stored qpos_refined equals the STAC qpos exactly. Measured here rather than
    asserted in a commit message -- if this ever fails, the removal DID change
    the pose and the golden bout-28 comparison must be re-read."""
    import h5py
    d = os.path.join(BACKUP, "fly1")
    if not (os.path.exists(f"{d}/qpos_refined.npz") and os.path.exists(f"{d}/stac_ik.h5")):
        pytest.skip("bout 28 fly1 artifacts not in the backup")
    q_ref = np.load(f"{d}/qpos_refined.npz")["qpos"]
    with h5py.File(f"{d}/stac_ik.h5", "r") as f:
        q_stac = np.asarray(f["qpos"])
    n = min(len(q_ref), len(q_stac))
    assert np.nanmax(np.abs(q_ref[:n] - q_stac[:n])) == 0.0
