"""Anatomy-agnostic differentiable FK + repose for silhouette/keypoint IK (JAX/MJX).

An "anatomy" is an (model_xml, canonical_mesh_npz) pair, where the canonical mesh
was baked from that same model's collision geoms (build_canonical_mesh.py), so its
``vertices_local`` / ``vertex_geom`` index the model directly.  Given a qpos:

    qpos -> mjx.kinematics -> geom_xpos/xmat -> world_v = xpos[g] + xmat[g] @ local_v

is a fully differentiable JAX function of qpos (and trivially of a global scale),
the basis for fitting qpos to keypoints + multi-view silhouettes.

This module is model-version agnostic: V1 now, V2/V2.1 later, just by passing a
different (xml, npz).  Run as __main__ to verify the JAX repose matches MuJoCo.
"""
from __future__ import annotations
import argparse


def load_anatomy(model_xml, mesh_npz):
    import numpy as np, mujoco
    from mujoco import mjx
    m = mujoco.MjModel.from_xml_path(model_xml)
    z = np.load(mesh_npz, allow_pickle=True)
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx)
    return dict(
        m=m, mx=mx, dx=dx,
        vlocal=np.asarray(z["vertices_local"], np.float32),
        vgeom=np.asarray(z["vertex_geom"], np.int32),
        faces=np.asarray(z["faces"], np.int32),
        fps={(int(k[4:]) if k[4:].isdigit() else k[4:]): z[k]
             for k in z.files if k.startswith("fps_")},  # int keys + "wing"
        seg_names=z["seg_names"], seg_ids=z["seg_ids"], vertex_segment=z["vertex_segment"],
        nq=int(m.nq), qpos0=np.asarray(m.qpos0, np.float32),
    )


def make_fk_repose(anat):
    """Return jitted fk_repose(qpos, scale=1.0, indices=None) -> (K,3) world verts."""
    import jax, jax.numpy as jnp
    from mujoco import mjx
    mx, dx = anat["mx"], anat["dx"]
    vlocal_all = jnp.asarray(anat["vlocal"]); vgeom_all = jnp.asarray(anat["vgeom"])

    def fk_repose(qpos, scale=1.0, indices=None):
        d = dx.replace(qpos=qpos)
        d = mjx.kinematics(mx, d)
        vg = vgeom_all if indices is None else vgeom_all[indices]
        vl = vlocal_all if indices is None else vlocal_all[indices]
        xpos = d.geom_xpos[vg]                      # (K,3)
        xmat = d.geom_xmat[vg].reshape(-1, 3, 3)    # (K,3,3)
        return scale * (xpos + jnp.einsum("kij,kj->ki", xmat, vl))

    return jax.jit(fk_repose, static_argnums=())


def _selftest(model_xml, mesh_npz):
    import numpy as np, jax, jax.numpy as jnp, mujoco
    from jarvis_jax.cse.mesh_assets import load_canonical, repose_vertices
    anat = load_anatomy(model_xml, mesh_npz)
    print(f"loaded anatomy: nq={anat['nq']} nverts={len(anat['vlocal'])} "
          f"nfaces={len(anat['faces'])} fps={sorted(anat['fps'], key=str)}")
    fk = make_fk_repose(anat)
    rng = np.random.default_rng(0)
    mesh = load_canonical(mesh_npz)
    m = anat["m"]; data = mujoco.MjData(m)
    maxdiff = 0.0
    for t in range(3):
        q = anat["qpos0"].copy()
        # perturb a few joints (skip free-joint quat for a clean test)
        q[7:] += rng.normal(0, 0.05, size=q.shape[0] - 7).astype(np.float32)
        # JAX/MJX repose
        wj = np.asarray(fk(jnp.asarray(q)))
        # MuJoCo repose (reference)
        data.qpos[:] = q; mujoco.mj_forward(m, data)
        wm = repose_vertices(m, data, mesh)
        d = np.abs(wj - wm).max(); maxdiff = max(maxdiff, d)
        print(f"  qpos[{t}] max|JAX-MuJoCo repose| = {d:.3e}")
    print(f"MAX repose diff over trials: {maxdiff:.3e}  ({'OK' if maxdiff < 1e-3 else 'MISMATCH'})")
    # gradient sanity
    g = jax.grad(lambda q: fk(q)[anat['vertex_segment'].shape[0] // 2].sum())(jnp.asarray(anat["qpos0"]))
    print(f"grad wrt qpos finite={bool(np.isfinite(np.asarray(g)).all())} ||g||={float(np.linalg.norm(np.asarray(g))):.3e}")
    print("SILHOUETTE_IK SELFTEST DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    a = ap.parse_args()
    _selftest(a.xml, a.mesh)
