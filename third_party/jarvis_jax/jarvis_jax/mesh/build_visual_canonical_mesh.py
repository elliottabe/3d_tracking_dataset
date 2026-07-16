"""Build a WATERTIGHT, shape-accurate VISUAL canonical mesh (drop-in for the
collision mesh) from a MuJoCo fly model's group=1 visual mesh geoms.

Motivation: the collision canonical mesh is a coarse union of tessellated
primitives (capsules/ellipsoids) -> silhouette IoU caps ~0.76 and it's a
1,154-piece triangle soup. The per-segment VISUAL OBJs are the true fly surface
but are non-watertight open shells. We make each part watertight with
trimesh.repair.fill_holes (verified to close the seam gaps WITHOUT losing
vertices -> full fidelity), repose per-geom, and emit the same npz schema
load_anatomy consumes. This is the finer silhouette signal the female-on-wall
kinematics recovery leans on (masks stay good when keypoints don't).

Output schema (matches fly_v1_collision_canonical_wings.npz where it matters):
  vertices (N,3), vertices_local (N,3), vertices_body (N,3), faces (F,3),
  vertex_segment (N,), vertex_geom (N,), sym_index (N,), seg_ids, seg_names,
  fps_100/200/300/500, fps_wing, n_bodyleg, keyframe, units, source.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import mujoco
import trimesh
from scipy.spatial import cKDTree

from jarvis_jax.mesh.build_canonical_mesh import _mesh_geom, _stratified_fps, _wing_membrane_geoms

GT = mujoco.mjtGeom


def _watertight_part(me: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close seam gaps -> watertight, preserving geometry (no voxel remesh).

    merge_vertices() FIRST: the MuJoCo visual geoms load as unwelded triangle
    'soups' (each triangle has its own copy of shared-edge vertices), so every
    edge is a boundary edge and fill_holes cannot stitch the surface (it fans
    degenerate triangles from a hub -> holey, e.g. the old wing blade had 11k
    open edges / 380 holes). Welding coincident vertices restores the shared
    topology, after which the part is genuinely closed (the wing blade welds to
    a watertight 252v/500f membrane) and fill_holes only closes real gaps.
    """
    m = me.copy()
    m.merge_vertices()
    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    return m


def build(xml_path, out_dir, *, name="fly_v1_visual_canonical_wings", keyframe=0,
          fps_sizes=(100, 200, 500), wing_fps=100, wing_substr="wing"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > keyframe:
        data.qpos[:] = model.key_qpos[keyframe]
    mujoco.mj_forward(model, data)
    bname = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    vis = [g for g in range(model.ngeom)
           if int(model.geom_group[g]) == 1 and model.geom_type[g] == GT.mjGEOM_MESH]
    # body/legs/head = all non-wing visual meshes. WINGS: use ONLY the low-poly
    # SOLID membrane blade (like build_canonical_mesh) -- the high-poly veined wing
    # rasterizes to hollow strips, under-filling the wing silhouette. Wings appended
    # last so fps_wing + body/leg fps are separable and body/leg indices are stable.
    body_geoms = [g for g in vis if wing_substr not in bname(int(model.geom_bodyid[g])).lower()]
    wing_geoms = list(_wing_membrane_geoms(model))

    Vw, Vl, Fl, seg, geo = [], [], [], [], []
    voff = 0
    n_watertight = 0

    def _add(g, n_subdiv=0):
        nonlocal voff, n_watertight
        mg = _mesh_geom(model, g)
        mg.merge_vertices()                   # weld the unwelded soup -> coherent surface
        for _ in range(int(n_subdiv)):        # densify low-poly wings so the projected
            mg = mg.subdivide()               # verts tile -> solid silhouette fill.
            # UNIFORM midpoint subdivide (not subdivide_to_size): it splits every edge
            # at its shared midpoint, so the thin blade stays WATERTIGHT. subdivide_to_size
            # splits per-triangle and re-introduces T-junctions (~4k open edges).
        me = _watertight_part(mg)
        n_watertight += int(me.is_watertight)
        R = data.geom_xmat[g].reshape(3, 3)
        p = data.geom_xpos[g]
        local = np.asarray(me.vertices, np.float32)
        Vw.append(local @ R.T + p)
        Vl.append(local)
        Fl.append(np.asarray(me.faces) + voff)
        seg.append(np.full(len(local), int(model.geom_bodyid[g]), np.int32))
        geo.append(np.full(len(local), g, np.int32))
        voff += len(local)

    for g in body_geoms:
        _add(g)
    n_bodyleg = voff
    for g in wing_geoms:
        _add(g, n_subdiv=2)     # 2x uniform subdivide: watertight blade, ~4k verts/wing,
                                # median edge ~0.0018 (fps_wing subsamples 100 for the solve)

    Vw = np.concatenate(Vw).astype(np.float32)
    Vl = np.concatenate(Vl).astype(np.float32)
    F = np.concatenate(Fl).astype(np.int32)
    seg = np.concatenate(seg)
    geo = np.concatenate(geo)
    N = len(Vw)

    # per-vertex body-frame coords (parity with the collision asset)
    bp = data.xpos[seg]
    bm = data.xmat[seg].reshape(-1, 3, 3)
    Vbody = np.einsum("nji,nj->ni", bm, Vw - bp).astype(np.float32)

    # L/R symmetry index via Y-mirror nearest neighbour (used by detector flip aug)
    ref = Vw.copy(); ref[:, 1] = -ref[:, 1]
    tree = cKDTree(Vw)
    fe = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    med_edge = float(np.median(np.linalg.norm(Vw[fe[:, 0]] - Vw[fe[:, 1]], axis=1)))
    dist, j = tree.query(ref, k=1)
    sym = np.where(dist < 0.5 * med_edge, j, -1).astype(np.int32)

    # body/leg FPS over the body/leg block only; wing FPS over the wing block
    bl = np.arange(N) < n_bodyleg
    fps_sets = {int(m): _stratified_fps(Vw[bl], seg[bl], m) for m in fps_sizes}
    extra_fps = {}
    if wing_geoms:
        widx = np.where(~bl)[0]
        wfps = widx[_stratified_fps(Vw[widx], seg[widx], wing_fps)]
        extra_fps["fps_wing"] = wfps.astype(np.int32)
        extra_fps[f"fps_{200 + len(wfps)}"] = np.concatenate(
            [fps_sets[200], wfps]).astype(np.int32)   # fps_300 = 200 body + 100 wing

    seg_ids = np.unique(seg)
    seg_names = np.array([bname(int(s)) for s in seg_ids])

    os.makedirs(out_dir, exist_ok=True)
    payload = dict(
        vertices=Vw, vertices_local=Vl, vertices_body=Vbody, faces=F,
        vertex_segment=seg, vertex_geom=geo, sym_index=sym,
        seg_ids=seg_ids, seg_names=seg_names,
        keyframe=np.int32(keyframe), units="cm",
        source=f"{os.path.basename(xml_path)} group=1 visual meshes, fill_holes-watertight",
        n_bodyleg=np.int32(n_bodyleg),
        **{f"fps_{m}": v for m, v in fps_sets.items()},
        **extra_fps,
    )
    npz = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz, **payload)
    trimesh.Trimesh(Vw, F, process=False).export(os.path.join(out_dir, f"{name}.obj"))

    print(f"[build_visual_canonical_mesh] {name}: {N} verts, {len(F)} faces, {len(seg_ids)} segments")
    print(f"  parts: {len(body_geoms)} body/leg + {len(wing_geoms)} wing = {len(vis)}; "
          f"per-part watertight: {n_watertight}/{len(vis)}")
    print(f"  n_bodyleg={n_bodyleg} (wing verts appended after) | median edge {med_edge:.4f} cm | "
          f"sym coverage {(sym >= 0).mean():.1%}")
    print(f"  FPS: {', '.join(f'{k}:{len(v)}' for k,v in {**fps_sets, **extra_fps}.items())}")
    print(f"  saved -> {npz}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml")
    ap.add_argument("--out-dir", default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse")
    ap.add_argument("--name", default="fly_v1_visual_canonical_wings")
    ap.add_argument("--keyframe", type=int, default=0)
    a = ap.parse_args()
    build(a.xml, a.out_dir, name=a.name, keyframe=a.keyframe)


if __name__ == "__main__":
    main()
