"""Auto-label generator: dense canonical-vertex 2D targets from STAC fits.

Pipeline (per recording in a V3 COCO dataset)
--------------------------------------------
Stage 1 (CPU, login-safe)  ``build_bout``
    triangulate each frameset's 50 keypoints -> reorder to model order ->
    isotropic per-recording scale (mm -> model cm, rotation-invariant Umeyama) ->
    write a ``preprocessed_bout``-format HDF5 that the existing STAC runner reads.

Stage 2 (GPU, sbatch)  -- run STAC on each bout via run_stac_fly_model / run_stac_bout
    fit_offsets + ik_only -> Fruitfly_ik_*.h5 with per-frameset ``qpos`` (T,93).

Stage 3 (CPU, login-safe)  ``project_labels``
    re-pose the canonical mesh by qpos (FK), divide by the recording scale to
    return to the camera (mm) frame, project the M FPS vertices into every
    camera -> aux labels {image_id: (M,3) [x,y,visible]}.

Because a vertex is "just a joint", these aux labels extend each annotation's
keypoints (50 -> 50+M) for Design-1 training without touching the curated COCO.

Scale convention
----------------
``kp_stac = kp_mm * s`` (scale only; STAC root-optimisation finds global R,t), so
the STAC world equals the camera (mm) frame times ``s``.  Inverse for projection
is therefore ``vertex_mm = vertex_stac / s``.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import mujoco

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.mesh_assets import load_canonical, repose_vertices


# --------------------------------------------------------------------------- #
# model / keypoint helpers
# --------------------------------------------------------------------------- #
def model_kp_order(anatomy_yaml: str) -> list[str]:
    """Ordered model keypoint names (KEYPOINT_MODEL_PAIRS keys)."""
    import yaml
    m = yaml.safe_load(open(anatomy_yaml))["model"]
    return list(m["KEYPOINT_MODEL_PAIRS"].keys())


def model_rest_keypoints(model, kp_names, keyframe=0) -> np.ndarray:
    """(K,3) model-rest keypoint positions from ``aligned[<name>]`` sites (NaN if absent)."""
    data = mujoco.MjData(model)
    if model.nkey > keyframe:
        data.qpos[:] = model.key_qpos[keyframe]
    mujoco.mj_forward(model, data)
    pos = np.full((len(kp_names), 3), np.nan, np.float64)
    for i, kp in enumerate(kp_names):
        # tracking[<kp>] = model-rest keypoint reference (aligned[<kp>] is a
        # dynamic placeholder set during preprocessing, zero-extent at rest)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tracking[{kp}]")
        if sid >= 0:
            pos[i] = data.site_xpos[sid]
    return pos


def umeyama_scale(data_pts, model_pts, valid) -> float:
    """Rotation-invariant isotropic scale mapping data -> model (Umeyama 1991)."""
    v = valid & np.all(np.isfinite(model_pts), 1) & np.all(np.isfinite(data_pts), 1)
    d = data_pts[v] - data_pts[v].mean(0)
    mo = model_pts[v] - model_pts[v].mean(0)
    return float(np.sqrt((mo ** 2).sum() / max((d ** 2).sum(), 1e-12)))


# --------------------------------------------------------------------------- #
# COCO indexing
# --------------------------------------------------------------------------- #
def _index_coco(coco):
    id2img = {im["id"]: im for im in coco["images"]}
    id2ann = {}
    for a in coco["annotations"]:
        id2ann.setdefault(a["image_id"], a)
    by_rec = defaultdict(list)  # rec -> [(fs_key, frame_ids)]
    for k, v in coco["framesets"].items():
        by_rec[v["datasetName"]].append((k, v["frames"]))
    return id2img, id2ann, by_rec


def _reorder_index(src_names, dst_names):
    """Index array mapping dst order -> position in src (src[idx] == dst)."""
    pos = {n: i for i, n in enumerate(src_names)}
    return np.array([pos[n] for n in dst_names], np.int64)


def triangulate_recording(coco, rec, rt, kp_order, id2img, id2ann, by_rec):
    """Per-frameset triangulated 3D keypoints in model order.

    Returns
    -------
    kp3d   (n_fs, K, 3) float64   triangulated world (mm); 0 where invisible
    vis    (n_fs, K)    bool
    fs_keys list[str]
    fs_imgids list[list[int]]     per-frameset camera image ids (len num_cam)
    """
    src_names = coco["keypoint_names"]
    reorder = _reorder_index(src_names, kp_order)  # dst(model) -> src(V3)
    K = len(kp_order)
    nc = rt.num_cameras

    kp3d, vis, fs_keys, fs_imgids = [], [], [], []
    for fs_key, frame_ids in by_rec[rec]:
        ids = frame_ids[:nc]
        anns = [id2ann.get(i) for i in ids]
        if any(a is None for a in anns) or len(anns) != nc:
            continue
        kp2d = np.zeros((nc, K, 3), np.float64)
        for c, a in enumerate(anns):
            raw = np.asarray(a["keypoints"], np.float64).reshape(-1, 3)  # V3 order
            kp2d[c] = raw[reorder]
        p3d = np.zeros((K, 3)); vmask = np.zeros(K, bool)
        for j in range(K):
            cams = [c for c in range(nc) if kp2d[c, j, 2] > 0]
            if len(cams) >= 2:
                p3d[j] = rt.reconstruct_point(kp2d[:, j, :2], cams_to_use=cams)
                vmask[j] = True
        kp3d.append(p3d); vis.append(vmask)
        fs_keys.append(fs_key); fs_imgids.append(list(ids))
    return np.asarray(kp3d), np.asarray(vis), fs_keys, fs_imgids


# --------------------------------------------------------------------------- #
# Stage 1: build STAC bout
# --------------------------------------------------------------------------- #
def build_bout(coco_path, calib_root, rec, anatomy_yaml, model_xml, out_h5):
    """Write a preprocessed_bout HDF5 for one recording (scaled to model)."""
    import h5py
    coco = json.load(open(coco_path))
    id2img, id2ann, by_rec = _index_coco(coco)
    kp_order = model_kp_order(anatomy_yaml)
    rt = ReprojectionTool(os.path.join(calib_root, rec))
    kp3d, vis, fs_keys, fs_imgids = triangulate_recording(
        coco, rec, rt, kp_order, id2img, id2ann, by_rec)
    if len(kp3d) == 0:
        raise RuntimeError(f"no complete framesets for {rec}")

    model = mujoco.MjModel.from_xml_path(model_xml)
    rest = model_rest_keypoints(model, kp_order)
    # one scale for the recording from the per-frameset median keypoint cloud
    med = np.nanmedian(np.where(vis[..., None], kp3d, np.nan), axis=0)  # (K,3)
    s = umeyama_scale(med, rest, np.all(np.isfinite(med), 1))
    kp_scaled = (kp3d * s).astype(np.float32)            # (n, K, 3) model cm
    kp_scaled[~vis] = 0.0

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("keypoints", data=kp_scaled)
        f.create_dataset("kp_names", data=np.array(kp_order, dtype="S20"))
        f.create_dataset("vis", data=vis)
        f.attrs["scale"] = s
        f.attrs["recording"] = rec
        # frameset bookkeeping for stage 3 (qpos row i <-> frameset i)
        f.create_dataset("fs_keys", data=np.array(fs_keys, dtype="S64"))
        f.create_dataset("fs_imgids", data=np.asarray(fs_imgids, np.int64))
    print(f"[build_bout] {rec}: {len(kp_scaled)} framesets, scale={s:.4f}, "
          f"kp extent~{np.ptp(kp_scaled[vis], 0).max():.3f} cm -> {out_h5}")
    return out_h5, s


# --------------------------------------------------------------------------- #
# Stage 3: project canonical vertices using STAC qpos
# --------------------------------------------------------------------------- #
def project_labels(bout_h5, stac_ik_h5, calib_root, rec, model_xml, mesh_npz,
                   out_npz, M=200):
    """Project M canonical vertices into every camera per frameset -> aux labels."""
    import h5py
    from jarvis_jax.cse.mesh_assets import load_canonical, repose_vertices

    with h5py.File(bout_h5, "r") as f:
        s = float(f.attrs["scale"])
        fs_imgids = f["fs_imgids"][:]                    # (n, num_cam)
    import stac_mjx.io_dict_to_hdf5 as ioh5  # qpos lives in STAC output
    stac = ioh5.load(stac_ik_h5)
    qpos = np.asarray(stac["qpos"])                      # (n, 93)
    assert len(qpos) == len(fs_imgids), (len(qpos), len(fs_imgids))

    rt = ReprojectionTool(os.path.join(calib_root, rec))
    nc = rt.num_cameras
    mesh = load_canonical(mesh_npz)
    idx = mesh.subset(M)
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    aux = {}                                  # image_id -> (M,3) [x,y,vis]  (2D, ViTPose)
    verts3d = np.zeros((len(qpos), M, 3), np.float32)   # per-frameset 3D GT (mm world)
    for fi in range(len(qpos)):
        data.qpos[:] = qpos[fi]
        mujoco.mj_forward(model, data)
        vw = repose_vertices(model, data, mesh, indices=idx)   # (M,3) stac frame
        vmm = vw / s                                           # camera (mm) frame
        verts3d[fi] = vmm                                      # 3D GT for HybridNet
        for c in range(nc):
            P = rt._camera_list[c].cameraMatrix               # (3,4)
            ph = np.concatenate([vmm, np.ones((M, 1))], 1) @ P.T   # (M,3)
            z = ph[:, 2]
            uv = ph[:, :2] / z[:, None]
            lab = np.concatenate([uv, (z > 0)[:, None].astype(np.float32)], 1)
            aux[int(fs_imgids[fi, c])] = lab.astype(np.float32)

    order = sorted(aux)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        image_ids=np.array(order),                       # (n_img,)
        labels2d=np.stack([aux[k] for k in order]),      # (n_img, M, 3) [x,y,vis]
        labels=np.stack([aux[k] for k in order]),        # back-compat alias
        fs_image_ids=fs_imgids.astype(np.int64),         # (n_fs, num_cam)
        verts3d=verts3d,                                 # (n_fs, M, 3) mm world
        fps_indices=idx, M=M, recording=rec, scale=s)
    print(f"[project_labels] {rec}: {len(aux)} imgs (2D) + {len(qpos)} framesets (3D) "
          f"x {M} verts -> {out_npz}")
    return out_npz


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-bout", help="Stage 1: triangulate+scale -> bout h5")
    b.add_argument("--coco", required=True)
    b.add_argument("--calib-root", required=True)
    b.add_argument("--rec", required=True)
    b.add_argument("--anatomy", required=True)
    b.add_argument("--model-xml", required=True)
    b.add_argument("--out", required=True)

    p = sub.add_parser("project", help="Stage 3: qpos -> projected vertex labels")
    p.add_argument("--bout", required=True)
    p.add_argument("--stac-ik", required=True)
    p.add_argument("--calib-root", required=True)
    p.add_argument("--rec", required=True)
    p.add_argument("--model-xml", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--M", type=int, default=200)

    a = ap.parse_args()
    if a.cmd == "build-bout":
        build_bout(a.coco, a.calib_root, a.rec, a.anatomy, a.model_xml, a.out)
    elif a.cmd == "project":
        project_labels(a.bout, a.stac_ik, a.calib_root, a.rec, a.model_xml,
                       a.mesh, a.out, M=a.M)


if __name__ == "__main__":
    main()
