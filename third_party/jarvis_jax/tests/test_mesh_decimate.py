import numpy as np, trimesh
from jarvis_jax.cse.mesh_decimate import decimate_mesh_npz


def test_decimation_reduces_faces_keeps_watertight_shape(tmp_path):
    s = trimesh.creation.icosphere(subdivisions=4)        # ~20480 faces
    src = tmp_path / "m.npz"
    np.savez(src, vertices=np.asarray(s.vertices, np.float32), faces=np.asarray(s.faces, np.int32))
    out = tmp_path / "d.npz"
    decimate_mesh_npz(str(src), str(out), target_faces=2000)
    z = np.load(out)
    assert z["faces"].shape[0] <= 3000 and z["faces"].shape[0] < 20480
    assert z["vertices"].shape[1] == 3
    # bounding box roughly preserved (shape not destroyed)
    assert np.allclose(z["vertices"].max(0), s.vertices.max(0), atol=0.1)
