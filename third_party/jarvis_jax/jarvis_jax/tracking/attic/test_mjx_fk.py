"""De-risk: can MJX load the STAC fly model and do differentiable forward kinematics?

If this works, qpos -> mjx.kinematics -> geom_xpos/xmat -> repose mesh -> project ->
soft silhouette is a fully-differentiable JAX chain for silhouette IK.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    a = ap.parse_args()
    import numpy as np, jax, jax.numpy as jnp, mujoco
    from mujoco import mjx

    m = mujoco.MjModel.from_xml_path(a.xml)
    print(f"MjModel: nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom}")
    nmesh = int((m.geom_type == mujoco.mjtGeom.mjGEOM_MESH).sum())
    wing_geoms = [g for g in range(m.ngeom)
                  if "wing" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()]
    wing_bodies = [b for b in range(m.nbody)
                   if "wing" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()]
    print(f"mesh_geoms={nmesh}  wing_geoms={len(wing_geoms)}  wing_bodies={wing_bodies}")

    try:
        mx = mjx.put_model(m)
        print("mjx.put_model: OK")
    except Exception as e:
        print("mjx.put_model FAILED:", repr(e)[:300]); return

    dx = mjx.make_data(mx)
    q0 = jnp.asarray(m.qpos0)

    @jax.jit
    def fk(qpos):
        d = dx.replace(qpos=qpos)
        d = mjx.kinematics(mx, d)
        return d.geom_xpos, d.geom_xmat

    try:
        gx, gm = fk(q0)
        gx.block_until_ready()
        print(f"mjx.kinematics OK: geom_xpos {gx.shape} geom_xmat {gm.shape}  finite={bool(np.isfinite(np.asarray(gx)).all())}")
    except Exception as e:
        print("mjx.kinematics FAILED:", repr(e)[:300]); return

    # differentiability: gradient of a wing geom position wrt qpos
    if wing_geoms:
        wg = wing_geoms[0]
        try:
            g = jax.grad(lambda q: fk(q)[0][wg].sum())(q0)
            g = np.asarray(g)
            nz = int((np.abs(g) > 0).sum())
            print(f"jax.grad wing geom {wg} wrt qpos OK: {nz}/{m.nq} qpos entries have nonzero grad, "
                  f"||g||={np.linalg.norm(g):.3e}")
        except Exception as e:
            print("jax.grad FAILED:", repr(e)[:300])
    print("MJX FK TEST DONE")


if __name__ == "__main__":
    main()
