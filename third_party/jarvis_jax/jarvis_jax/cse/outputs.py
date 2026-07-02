"""Phase 7 / component 8: serialize per-fly IK outputs to h5.

Given a solved qpos trajectory + the V1 anatomy + a per-frame model->mm
Umeyama bridge (fit exactly as run_single_fly does, from the STAC-fitted
MODEL-frame marker_sites vs. the recording's triangulated coco keypoints),
FK-repose the posed mesh (a documented vertex SUBSET by default) and the 50
keypoint sites, map both to calibrated world mm, and write an h5 per fly:
{qpos, root_se3, scale, mesh_mm, kp3d_mm, mesh_vert_idx, mesh_subset, kp_names}.

MEMORY: the full posed mesh is T*61666*3*4 bytes (~0.74 MB/frame, ~22 GB for
a 30k-frame recording). The default is a `fps_500` vertex SUBSET (500 verts,
~6 KB/frame, float32). `mesh_subset="full"` writes the whole mesh and is only
for short clips -- documented tradeoff per the efficiency mandate. FK over T
frames is a single jax.vmap for the mesh subset (one XLA call).
"""
from __future__ import annotations
import numpy as np

from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose


def mesh_subset_indices(anat, subset="fps_500"):
    """Full-vertex-array indices for a named fps subset (or all verts).

    subset="full" -> np.arange(nverts). subset="fps_<N>" -> anat["fps"][N]
    (the FK-repose `indices` argument indexes the FULL vertex array, so the
    fps arrays -- which already hold full-array indices -- are used directly).
    """
    if subset == "full":
        nverts = len(anat["vlocal"])
        return np.arange(nverts, dtype=np.int32)
    if not subset.startswith("fps_"):
        raise ValueError(f"subset must be 'full' or 'fps_<N>', got {subset!r}")
    n = int(subset.split("_")[1])
    if n not in anat["fps"]:
        raise ValueError(f"fps subset {n} not in anatomy (have {sorted(anat['fps'])})")
    return np.asarray(anat["fps"][n], dtype=np.int32)


def fk_mesh_world_mm(fk_repose, qpos, vert_idx, bridges):
    """(T,K,3) world-mm mesh verts: vmapped model-frame FK of the subset, then
    per-frame model->mm via `bridges[t]=(s,R,t)`. None bridge -> nan frame."""
    import jax
    import jax.numpy as jnp
    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    vert_idx = np.asarray(vert_idx, np.int32)
    K = vert_idx.shape[0]

    def _one(q):
        return fk_repose(q, 1.0, vert_idx)  # (K,3) model frame

    verts_model = np.asarray(jax.vmap(_one)(jnp.asarray(qpos)))  # (T,K,3)
    out = np.full((T, K, 3), np.nan, np.float32)
    for t in range(T):
        br = bridges[t]
        if br is None:
            continue
        s, R, tr = br
        vm = s * (np.asarray(R) @ verts_model[t].T).T + np.asarray(tr)
        out[t] = vm.astype(np.float32)
    return out


def fk_sites_world_mm(mjx_model, mjx_data, site_idxs, qpos, bridges):
    """(T,n_site,3) world-mm keypoint sites: per-frame kinematics->com_pos->
    get_site_xpos (MODEL) then model->mm via bridges[t]. None -> nan frame."""
    import stac_mjx.utils as stac_utils
    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    site_idxs = np.asarray(site_idxs)
    n_site = site_idxs.shape[0]
    out = np.full((T, n_site, 3), np.nan, np.float32)
    for t in range(T):
        br = bridges[t]
        if br is None:
            continue
        data_t = mjx_data.replace(qpos=qpos[t])
        data_t = stac_utils.kinematics(mjx_model, data_t)
        data_t = stac_utils.com_pos(mjx_model, data_t)
        sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))
        s, R, tr = br
        out[t] = (s * (np.asarray(R) @ sites_model.T).T + np.asarray(tr)).astype(np.float32)
    return out


def write_outputs_h5(out_path, *, qpos, root_se3, scale, mesh_mm, kp3d_mm,
                     mesh_vert_idx, mesh_subset, kp_names):
    """Write the per-fly outputs h5 via stac_mjx.io_dict_to_hdf5.save."""
    import os
    import stac_mjx.io_dict_to_hdf5 as ioh5
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dic = {
        "qpos": np.asarray(qpos, np.float32),
        "root_se3": np.asarray(root_se3, np.float32),
        "scale": np.asarray(scale, np.float32),
        "mesh_mm": np.asarray(mesh_mm, np.float32),
        "kp3d_mm": np.asarray(kp3d_mm, np.float32),
        "mesh_vert_idx": np.asarray(mesh_vert_idx, np.int32),
        # h5py has no native conversion for numpy unicode ('<U..') dtypes, so
        # strings are stored as fixed-width bytes ('S'); `load(...).astype(str)`
        # decodes them back (see test_write_outputs_h5_roundtrip).
        "mesh_subset": np.asarray(str(mesh_subset)).astype("S"),
        "kp_names": np.asarray([str(n) for n in kp_names]).astype("S"),
    }
    ioh5.save(out_path, dic)
    return out_path


def _load_solver_bits(ik_h5, model_xml):
    """Return (mjx_model, mjx_data, site_idxs, kp_names) from build_solver_inputs."""
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    inp = build_solver_inputs(ik_h5, model_xml)
    return inp["mjx_model"], inp["mjx_data"], inp["site_idxs"], list(inp["kp_names"])


def build_fly_outputs(recording, *, ik_h5, model_xml, mesh_npz, qpos, bridges,
                      out_path, mesh_subset="fps_500"):
    """Glue: FK the mesh subset + sites to world mm and write the per-fly h5.

    root_se3 = qpos[:, :7] (free-joint SE3: xyz + wxyz quat, the model-frame
    root as stored by the solver). scale[t] = the per-frame bridge scale `s`
    (nan where bridges[t] is None). Efficiency: the mesh FK is one jax.vmap.
    """
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    vert_idx = mesh_subset_indices(anat, subset=mesh_subset)
    mjx_model, mjx_data, site_idxs, kp_names = _load_solver_bits(ik_h5, model_xml)

    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    mesh_mm = fk_mesh_world_mm(fk, qpos, vert_idx, bridges)
    kp3d_mm = fk_sites_world_mm(mjx_model, mjx_data, site_idxs, qpos, bridges)

    root_se3 = qpos[:, :7].copy()
    scale = np.array([br[0] if br is not None else np.nan for br in bridges],
                     dtype=np.float32)
    assert scale.shape[0] == T, "one bridge per frame required"

    write_outputs_h5(
        out_path, qpos=qpos, root_se3=root_se3, scale=scale, mesh_mm=mesh_mm,
        kp3d_mm=kp3d_mm, mesh_vert_idx=vert_idx, mesh_subset=mesh_subset,
        kp_names=kp_names)
    return {"out_path": out_path, "mesh_subset": mesh_subset,
            "shapes": {"qpos": tuple(qpos.shape), "mesh_mm": tuple(mesh_mm.shape),
                       "kp3d_mm": tuple(kp3d_mm.shape)}}
