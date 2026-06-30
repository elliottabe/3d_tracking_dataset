"""Build the fruit-fly CSE canonical mesh from MuJoCo COLLISION geoms.

The fruit-fly body model stores its visual surface as 85 separate, non-watertight,
high-poly OBJs (~818k verts) in per-segment local frames.  For dense surface
correspondence we instead assemble the model's *collision* primitives
(capsule / ellipsoid / cylinder / sphere / box), which are:

* clean & low-poly (watertight per part, no decimation artefacts),
* one primitive per kinematic segment -> exact per-vertex segment assignment,
* the *same* geometry STAC-IK fits, so "predict canonical point k" and
  "constrain model point k" are self-consistent,
* rod-like on the legs, matching the real leg segments well.

Each primitive is tessellated and then ``subdivide_to_size``-d so capsule/cylinder
*shafts* get surface vertices (trimesh otherwise puts vertices only on the
hemispherical end-caps, which land at the joints and leave bare shafts).

The mesh is assembled at the model's reference keyframe.  Pose comes per-frame
from STAC (see ``mesh_assets.repose_vertices``); the canonical positions here are
only the rest reference + vertex identities.

Outputs ``<out_dir>/<name>.npz`` and ``.obj``.  NPZ fields:

    vertices        (N,3) f32  world coords at the reference keyframe
    vertices_local  (N,3) f32  coords in each vertex's geom frame (re-posable)
    faces           (F,3) i32
    vertex_segment  (N,)  i32  MuJoCo body id of the source segment
    vertex_geom     (N,)  i32  MuJoCo geom id of the source primitive
    edges           (E,2) i32  undirected mesh edges (+ kinematic-tree stitch)
    fps_<M>         (M,)  i32  farthest-point-sampled vertex subsets
    sym_index       (N,)  i32  bilateral (y->-y) partner per vertex, -1 if none
    seg_ids/seg_names          segment id <-> name table
    keyframe, units, source, max_edge   metadata

CLI:
    python -m jarvis_jax.cse.build_canonical_mesh \
        --xml /.../fruitfly_v1/fruitfly_v1_free.xml \
        --out /.../fruitfly_body_models/fruitfly_cse \
        --max-edge 0.01
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

GT = mujoco.mjtGeom
_PRIM = {GT.mjGEOM_SPHERE, GT.mjGEOM_CAPSULE, GT.mjGEOM_ELLIPSOID,
         GT.mjGEOM_CYLINDER, GT.mjGEOM_BOX}


def _tess(model, g):
    """Tessellate collision geom ``g`` into a unit-placed trimesh (geom frame)."""
    t, s = model.geom_type[g], model.geom_size[g]
    if t == GT.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * s[1], radius=s[0], count=[12, 12])
    if t == GT.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=16)
    if t == GT.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=s[0])
    if t == GT.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s[:3])
    if t == GT.mjGEOM_ELLIPSOID:
        sph = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sph.vertices *= s[:3]
        return sph
    return None


def _mesh_geom(model, g):
    """Extract a geom's baked mesh (geom-frame coords) as a trimesh."""
    mid = int(model.geom_dataid[g])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    v = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3)
    f = np.asarray(model.mesh_face[fa:fa + fn]).reshape(-1, 3)
    return trimesh.Trimesh(v, f, process=False)


def _wing_membrane_geoms(model):
    """Per wing body, the lowest-poly group-1 MESH geom (the membrane blade,
    ~1500 verts) rather than the high-poly veined 'brown' wing or the rounded
    collision ellipsoid."""
    bname = lambda b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()
    best = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != GT.mjGEOM_MESH or int(model.geom_group[g]) != 1:
            continue
        if "wing" not in bname(int(model.geom_bodyid[g])):
            continue
        b = int(model.geom_bodyid[g])
        nv = int(model.mesh_vertnum[int(model.geom_dataid[g])])
        if b not in best or nv < best[b][1]:
            best[b] = (g, nv)
    return [v[0] for v in best.values()]


def _fps(pts, m, seed=0):
    """Euclidean farthest-point sampling -> (m,) indices into pts."""
    m = min(m, len(pts))
    idx = np.zeros(m, np.int32)
    idx[0] = seed
    d2 = np.sum((pts - pts[seed]) ** 2, 1)
    for i in range(1, m):
        idx[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((pts - pts[idx[i]]) ** 2, 1))
    return idx


def _stratified_fps(verts, seg, m, floor=1):
    """Segment-stratified FPS for uniform surface density.

    Plain Euclidean FPS under-samples the thin legs (it spreads toward the large
    body volume), giving ringed/sparse legs.  Here we allocate each segment a
    budget proportional to its vertex count (~surface area) with a per-segment
    floor (so distal tarsus/claw segments are never dropped), then farthest-point
    sample *within* each segment so points spread along its length, not in rings.
    Result: leg/body split tracks the mesh's true surface area (~70/30) with even
    longitudinal coverage of every leg segment.
    """
    seg_ids = np.unique(seg)
    nseg = len(seg_ids)
    sizes = np.array([(seg == s).sum() for s in seg_ids], float)
    floor = min(floor, max(1, m // nseg))
    alloc = np.full(nseg, floor, int)
    rem = m - alloc.sum()
    if rem > 0:                                   # distribute by largest remainder
        share = rem * sizes / sizes.sum()
        add = np.floor(share).astype(int)
        for i in np.argsort(-(share - add))[: rem - int(add.sum())]:
            add[i] += 1
        alloc += add
    elif rem < 0:                                 # m < nseg: 1 each to biggest m
        alloc[:] = 0
        alloc[np.argsort(-sizes)[:m]] = 1
    idx = []
    for s, k in zip(seg_ids, alloc):
        si = np.where(seg == s)[0]
        k = int(min(k, len(si)))
        if k > 0:
            idx.extend(si[_fps(verts[si], k)].tolist())
    return np.array(idx[:m], np.int32)


def build(xml_path, out_dir, *, name="fly_v1_collision_canonical", max_edge=0.01,
          keyframe=0, include_wings=False, fps_sizes=(100, 200, 500), wing_fps=100,
          exclude_name_substrings=("wing", "haltere")):
    """Assemble + save the canonical mesh.  Returns the saved dict."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > keyframe:
        data.qpos[:] = model.key_qpos[keyframe]
    mujoco.mj_forward(model, data)
    bname = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    Vw, Vl, Fl, seg, geo = [], [], [], [], []
    voff = 0
    for g in range(model.ngeom):
        if model.geom_type[g] not in _PRIM:
            continue
        if int(model.geom_group[g]) in (3, 5):  # wing fluid / membrane
            continue
        bn = bname(int(model.geom_bodyid[g])).lower()
        if any(s in bn for s in exclude_name_substrings):   # body+legs only here
            continue
        me = _tess(model, g)
        if me is None:
            continue
        me = me.subdivide_to_size(max_edge)  # fill bare capsule shafts
        R = data.geom_xmat[g].reshape(3, 3)
        p = data.geom_xpos[g]
        local = np.asarray(me.vertices, np.float32)
        Vw.append(local @ R.T + p)
        Vl.append(local)
        Fl.append(np.asarray(me.faces) + voff)
        seg.append(np.full(len(local), int(model.geom_bodyid[g]), np.int32))
        geo.append(np.full(len(local), g, np.int32))
        voff += len(local)
    n_bodyleg = voff   # wings appended after this -> body/leg indices preserved

    # Wings: append each wing's low-poly membrane VISUAL mesh (true blade shape;
    # the collision ellipsoid is a rounded ~70% blob). Appended last so the
    # body/leg vertices (and their FPS subset) keep their indices for warm-start.
    if include_wings:
        for g in _wing_membrane_geoms(model):
            me = _mesh_geom(model, g).subdivide_to_size(max_edge)
            R = data.geom_xmat[g].reshape(3, 3)
            p = data.geom_xpos[g]
            local = np.asarray(me.vertices, np.float32)
            Vw.append(local @ R.T + p)
            Vl.append(local)
            Fl.append(np.asarray(me.faces) + voff)
            seg.append(np.full(len(local), int(model.geom_bodyid[g]), np.int32))
            geo.append(np.full(len(local), g, np.int32))
            voff += len(local)

    Vw = np.concatenate(Vw).astype(np.float32)
    Vl = np.concatenate(Vl).astype(np.float32)
    F = np.concatenate(Fl).astype(np.int32)
    seg = np.concatenate(seg)
    geo = np.concatenate(geo)
    N = len(Vw)

    # per-vertex coords in the BODY frame (for STAC marker offsets, Workstream D):
    # body_local = R_body^T @ (world - body_xpos)
    bp = data.xpos[seg]                              # (N,3)
    bm = data.xmat[seg].reshape(-1, 3, 3)           # (N,3,3) body rotation
    Vbody = np.einsum("nji,nj->ni", bm, Vw - bp).astype(np.float32)

    # edges: intra (faces) + inter (stitch each segment to nearest ancestor seg)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = np.unique(np.sort(e, 1), axis=0)
    present = {int(s): np.where(seg == s)[0] for s in np.unique(seg)}
    inter = []
    for b in np.unique(seg):
        b = int(b)
        par = int(model.body_parentid[b])
        while par > 0 and par not in present:
            par = int(model.body_parentid[par])
        if par in present and par != b:
            ci, pi = present[b], present[par]
            dist, j = cKDTree(Vw[pi]).query(Vw[ci], k=1)
            for k in np.argsort(dist)[:2]:
                inter.append((int(ci[k]), int(pi[j[k]])))
    edges = np.unique(np.sort(np.vstack([e, np.array(inter, np.int64).reshape(-1, 2)]), 1),
                      axis=0).astype(np.int32)

    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    ncomp, _ = connected_components(A + A.T, directed=False)

    # Body/leg FPS over the body/leg block only (its indices 0..n_bodyleg-1 are
    # preserved, so these subsets are byte-identical to the wingless v0 asset).
    bl = np.arange(N) < n_bodyleg
    fps_sets = {int(m): _stratified_fps(Vw[bl], seg[bl], m) for m in fps_sizes}
    extra_fps = {}
    if include_wings and (~bl).any():
        widx = np.where(~bl)[0]
        wfps = widx[_stratified_fps(Vw[widx], seg[widx], wing_fps)]   # full-array idx
        extra_fps["fps_wing"] = wfps.astype(np.int32)
        # deployment subset fps_300 = 200 body/leg (preserved order) + 100 wing, in
        # that order, so a 250-joint model warm-starts and only the appended wing
        # channels (250:300) are new.
        extra_fps[f"fps_{200 + len(wfps)}"] = np.concatenate(
            [fps_sets[200], wfps]).astype(np.int32)

    ref = Vw.copy()
    ref[:, 1] = -ref[:, 1]
    dist, j = cKDTree(Vw).query(ref, k=1)
    med_edge = float(np.median(np.linalg.norm(Vw[edges[:, 0]] - Vw[edges[:, 1]], axis=1)))
    sym = np.where(dist < 0.5 * med_edge, j, -1).astype(np.int32)

    seg_ids = np.unique(seg)
    seg_names = np.array([bname(int(s)) for s in seg_ids])

    os.makedirs(out_dir, exist_ok=True)
    payload = dict(
        vertices=Vw, vertices_local=Vl, vertices_body=Vbody, faces=F,
        vertex_segment=seg, vertex_geom=geo, edges=edges,
        sym_index=sym, seg_ids=seg_ids, seg_names=seg_names,
        keyframe=np.int32(keyframe), units="cm",
        source=f"{os.path.basename(xml_path)} collision geoms (groups!=3,5)",
        max_edge=np.float32(max_edge), n_bodyleg=np.int32(n_bodyleg),
        **{f"fps_{m}": v for m, v in fps_sets.items()},
        **extra_fps,
    )
    npz = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz, **payload)
    trimesh.Trimesh(Vw, F, process=False).export(os.path.join(out_dir, f"{name}.obj"))

    print(f"[build_canonical_mesh] {name}: {N} verts, {len(F)} faces, {len(edges)} edges, "
          f"{len(seg_ids)} segments")
    print(f"  graph components: {ncomp} (1 = fully stitched; >1 ok for Design 1)")
    print(f"  median edge {med_edge:.4f} cm | symmetry coverage {(sym >= 0).mean():.1%}")
    print(f"  FPS subsets: {', '.join(f'M={m}:{len(v)}' for m, v in fps_sets.items())}")
    print(f"  saved -> {npz}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="path to fruitfly_*_free.xml")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", default="fly_v1_collision_canonical")
    ap.add_argument("--max-edge", type=float, default=0.01, help="max surface edge len (cm)")
    ap.add_argument("--keyframe", type=int, default=0)
    ap.add_argument("--include-wings", action="store_true")
    args = ap.parse_args()
    build(args.xml, args.out, name=args.name, max_edge=args.max_edge,
          keyframe=args.keyframe, include_wings=args.include_wings)


if __name__ == "__main__":
    main()
