# tests/test_silhouette_index_space.py
import os
import numpy as np
import pytest

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"


@pytest.mark.skipif(not os.path.exists(MESH), reason="mesh not present")
def test_silhouette_fk_indices_are_full_array_space_not_fps_relative():
    """The silhouette factor's vertex subset MUST be FULL-vertex-array indices
    (0..61665), not fps-relative (0..299). Regression guard mirroring
    _wing_fk_indices' fps[idx] bridge: fps-relative indices would (a) all be
    < 300 and (b) select the wrong vertices."""
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    z = np.load(MESH, allow_pickle=True)
    n_verts = z["vertices_local"].shape[0]      # 61666
    fps300 = z["fps_300"]

    idx = silhouette_fk_indices(MESH, subset="fps_300")
    assert idx.shape == fps300.shape
    # full-array space: exactly equals fps_300 (which itself holds full-array ids)
    np.testing.assert_array_equal(np.sort(idx), np.sort(fps300))
    # values span the FULL array, not just 0..299 (proves not fps-relative).
    assert idx.max() >= 300 and idx.max() < n_verts
    assert not (idx < 300).all()


@pytest.mark.skipif(not os.path.exists(MESH), reason="mesh not present")
def test_silhouette_fk_indices_index_geoms_correctly():
    """Indexing vertex_geom (a FULL-array (61666,) map) with the returned
    indices must be in-bounds and select real geoms -- a fps-relative index
    set would index the first 300 rows only (a different, wrong selection)."""
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    z = np.load(MESH, allow_pickle=True)
    vgeom = z["vertex_geom"]                      # (61666,)
    idx = silhouette_fk_indices(MESH, subset="fps_300")
    assert idx.max() < vgeom.shape[0]
    geoms = vgeom[idx]
    assert np.isfinite(geoms.astype(float)).all()
    # the subset must touch MORE than one geom (a whole-body silhouette subset)
    assert len(np.unique(geoms)) > 1
