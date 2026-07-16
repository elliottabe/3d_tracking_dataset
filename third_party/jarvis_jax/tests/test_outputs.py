# tests/test_outputs.py
import os
import numpy as np
import pytest


def test_mesh_subset_indices_full_and_named():
    from jarvis_jax.tracking import outputs
    fake_anat = {"fps": {500: np.arange(0, 61666, 123, dtype=np.int64)[:500]},
                 "vlocal": np.zeros((61666, 3), np.float32)}
    idx = outputs.mesh_subset_indices(fake_anat, subset="fps_500")
    assert idx.dtype == np.int32
    assert idx.shape == (500,)
    full = outputs.mesh_subset_indices(fake_anat, subset="full")
    assert full.shape == (61666,)
    assert full[0] == 0 and full[-1] == 61665


def test_mesh_subset_indices_wing_union_and_dedup():
    """fps_500+wing unions the body subset with the wing FPS set and dedups.

    fps_500 alone lands only ~4/500 verts on the thin wing membranes, so wings
    are invisible in overlays without the wing set; the '+' union fixes that.
    """
    from jarvis_jax.tracking import outputs
    fake_anat = {"fps": {500: np.array([0, 5, 10, 27000], np.int64),
                         "wing": np.array([27000, 27406, 61138], np.int64)},
                 "vlocal": np.zeros((61666, 3), np.float32)}
    idx = outputs.mesh_subset_indices(fake_anat, subset="fps_500+wing")
    assert idx.dtype == np.int32
    # union of {0,5,10,27000} and {27000,27406,61138}, 27000 de-duplicated
    assert set(idx.tolist()) == {0, 5, 10, 27000, 27406, 61138}
    assert len(idx) == 6
    # the wing-only token resolves the wing set
    w = outputs.mesh_subset_indices(fake_anat, subset="wing")
    assert set(w.tolist()) == {27000, 27406, 61138}
    # a bad token raises
    with pytest.raises(ValueError):
        outputs.mesh_subset_indices(fake_anat, subset="bogus")


def test_write_outputs_h5_roundtrip(tmp_path):
    from jarvis_jax.tracking import outputs
    import stac_mjx.io_dict_to_hdf5 as ioh5
    T, nq, K, nkp = 3, 93, 5, 50
    qpos = np.random.default_rng(0).normal(size=(T, nq)).astype(np.float32)
    root = qpos[:, :7].copy()
    scale = np.full((T,), 0.9, np.float32)
    mesh_mm = np.zeros((T, K, 3), np.float32)
    kp3d = np.zeros((T, nkp, 3), np.float32)
    vidx = np.arange(K, dtype=np.int32)
    out = str(tmp_path / "fly_out.h5")
    p = outputs.write_outputs_h5(
        out, qpos=qpos, root_se3=root, scale=scale, mesh_mm=mesh_mm,
        kp3d_mm=kp3d, mesh_vert_idx=vidx, mesh_subset="fps_500",
        kp_names=[f"kp{i}" for i in range(nkp)])
    assert p == out and os.path.exists(out)
    d = ioh5.load(out)
    assert np.asarray(d["qpos"]).shape == (T, nq)
    assert np.asarray(d["mesh_mm"]).shape == (T, K, 3)
    assert np.asarray(d["kp3d_mm"]).shape == (T, nkp, 3)
    assert np.asarray(d["root_se3"]).shape == (T, 7)
    assert np.asarray(d["scale"]).shape == (T,)
    assert np.asarray(d["mesh_vert_idx"]).shape == (K,)
    assert str(np.asarray(d["mesh_subset"]).astype(str)) == "fps_500"


def test_fk_mesh_world_mm_applies_bridge_and_nan(tmp_path):
    from jarvis_jax.tracking import outputs
    # fk stub: identity model verts independent of qpos (K fixed points)
    K = 4
    base = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)

    def fk_stub(qpos, scale=1.0, indices=None):
        return base[indices] if indices is not None else base

    qpos = np.zeros((2, 7), np.float32)
    vidx = np.arange(K, dtype=np.int32)
    # frame0 bridge: scale 2, identity R, translate +10 in x; frame1: None
    bridges = [(2.0, np.eye(3), np.array([10.0, 0, 0])), None]
    out = outputs.fk_mesh_world_mm(fk_stub, qpos, vidx, bridges)
    assert out.shape == (2, K, 3)
    # frame0 vertex1 (1,0,0) -> 2*(1,0,0)+ (10,0,0) = (12,0,0)
    assert np.allclose(out[0, 1], [12.0, 0.0, 0.0])
    # frame1 all nan
    assert np.isnan(out[1]).all()


def test_build_fly_outputs_derives_root_scale(monkeypatch, tmp_path):
    """build_fly_outputs must set root_se3 = qpos[:, :7] and scale from the
    per-frame bridge s (nan where bridge is None), and write the h5."""
    from jarvis_jax.tracking import outputs
    import stac_mjx.io_dict_to_hdf5 as ioh5
    T, nq, nkp, K = 3, 93, 50, 4

    qpos = np.random.default_rng(1).normal(size=(T, nq)).astype(np.float32)
    bridges = [(1.5, np.eye(3), np.zeros(3)), None, (2.0, np.eye(3), np.zeros(3))]
    fake_anat = {"fps": {500: np.arange(K, dtype=np.int64)},
                 "vlocal": np.zeros((K, 3), np.float32)}
    base = np.eye(3, dtype=np.float32)[:K] if K <= 3 else np.zeros((K, 3), np.float32)

    def fk_stub(q, scale=1.0, indices=None):
        return base[indices] if indices is not None else base

    monkeypatch.setattr(outputs, "load_anatomy", lambda xml, npz: fake_anat)
    monkeypatch.setattr(outputs, "make_fk_repose", lambda anat: fk_stub)
    monkeypatch.setattr(outputs, "_load_solver_bits",
                        lambda ik_h5, xml: (None, None, np.arange(nkp), [f"kp{i}" for i in range(nkp)]))
    monkeypatch.setattr(outputs, "fk_sites_world_mm",
                        lambda *a, **k: np.zeros((T, nkp, 3), np.float32))

    out = str(tmp_path / "fly0.h5")
    rep = outputs.build_fly_outputs(
        "rec", ik_h5="x", model_xml="x", mesh_npz="x", qpos=qpos,
        bridges=bridges, out_path=out, mesh_subset="fps_500")
    assert rep["out_path"] == out
    d = ioh5.load(out)
    assert np.allclose(np.asarray(d["root_se3"]), qpos[:, :7])
    sc = np.asarray(d["scale"])
    assert np.isclose(sc[0], 1.5) and np.isnan(sc[1]) and np.isclose(sc[2], 2.0)
