"""Load the canonical CSE mesh and re-pose its vertices under any STAC qpos.

The canonical asset stores each vertex's coordinates in its *geom* frame
(``vertices_local``) plus the source ``vertex_geom`` id.  Given a MuJoCo model +
data advanced to a target qpos (``mj_forward``), every canonical vertex maps to
world coordinates by the rigid geom transform::

    world_v = geom_xpos[g] + geom_xmat[g] @ local_v

This is the bridge from a STAC pose to per-frame 3-D surface points, used to
render auto-labels (Workstream B) and to define dense IK targets (Workstream D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CanonicalMesh:
    vertices: np.ndarray        # (N,3) reference-pose world coords
    vertices_local: np.ndarray  # (N,3) geom-frame coords
    vertices_body: np.ndarray   # (N,3) body-frame coords (STAC marker offsets)
    faces: np.ndarray           # (F,3)
    vertex_segment: np.ndarray  # (N,) MuJoCo body id
    vertex_geom: np.ndarray     # (N,) MuJoCo geom id
    edges: np.ndarray           # (E,2)
    sym_index: np.ndarray       # (N,)
    seg_ids: np.ndarray
    seg_names: np.ndarray
    fps: dict                   # {M: (M,) indices}
    meta: dict

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def subset(self, m: int) -> np.ndarray:
        """Return the FPS index set of size ``m`` (must have been built)."""
        if m not in self.fps:
            raise KeyError(f"no FPS subset M={m}; available: {sorted(self.fps)}")
        return self.fps[m]


def load_canonical(npz_path: str) -> CanonicalMesh:
    z = np.load(npz_path, allow_pickle=True)
    fps = {}
    for k in z.files:                       # fps_<int> subsets; skip named ones (fps_wing)
        if k.startswith("fps_"):
            try:
                fps[int(k.split("_")[1])] = z[k]
            except ValueError:
                pass
    meta = {k: z[k] for k in ("keyframe", "units", "source", "max_edge") if k in z.files}
    return CanonicalMesh(
        vertices=z["vertices"], vertices_local=z["vertices_local"],
        vertices_body=z["vertices_body"], faces=z["faces"],
        vertex_segment=z["vertex_segment"], vertex_geom=z["vertex_geom"],
        edges=z["edges"], sym_index=z["sym_index"],
        seg_ids=z["seg_ids"], seg_names=z["seg_names"], fps=fps, meta=meta,
    )


def repose_vertices(model, data, mesh: CanonicalMesh, *, indices=None) -> np.ndarray:
    """World coords of canonical vertices for the model's *current* pose.

    Caller must have set ``data.qpos`` and run ``mujoco.mj_forward(model, data)``.

    Args:
        model, data: MuJoCo model/data already advanced to the target pose.
        mesh: loaded CanonicalMesh.
        indices: optional (K,) vertex indices to re-pose (e.g. an FPS subset);
                 default = all vertices.

    Returns:
        (K,3) world-space coordinates (same units as the model, cm).
    """
    idx = np.arange(mesh.n_vertices) if indices is None else np.asarray(indices)
    geoms = mesh.vertex_geom[idx]
    local = mesh.vertices_local[idx]
    xpos = data.geom_xpos[geoms]                       # (K,3)
    xmat = data.geom_xmat[geoms].reshape(-1, 3, 3)     # (K,3,3)
    return xpos + np.einsum("kij,kj->ki", xmat, local)


def repose_vertices_batch(model, mj_data_factory, qpos_seq, mesh: CanonicalMesh,
                          *, indices=None) -> np.ndarray:
    """Re-pose canonical vertices over a sequence of qpos frames (CPU FK loop).

    Args:
        model: MuJoCo model.
        mj_data_factory: callable returning a fresh MjData (e.g. ``lambda: mujoco.MjData(model)``).
        qpos_seq: (T, nq) array of poses.
        mesh: loaded CanonicalMesh.
        indices: optional vertex subset.

    Returns:
        (T, K, 3) world coords.
    """
    import mujoco
    data = mj_data_factory()
    out = []
    for q in np.asarray(qpos_seq):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        out.append(repose_vertices(model, data, mesh, indices=indices))
    return np.stack(out)
