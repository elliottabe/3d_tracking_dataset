"""Workstream D: feed dense canonical vertices into STAC-IK as extra markers.

STAC builds one marker site per KEYPOINT_MODEL_PAIRS entry at its
KEYPOINT_INITIAL_OFFSETS position.  We add 200 vertex markers (vtx_i -> its
segment body, at the canonical body-frame offset mesh.vertices_body[fps[i]]), so
IK is solved against 50 keypoints + 200 surface vertices = an over-determined,
occlusion-robust fit.  Vertex offsets are regularised toward the canonical values
(we trust them) rather than freely fit.

Observations (kp_data) are scaled by the per-recording factor `s` from the bout
(same convention as cse_labels.build_bout: model_cm = raw * s).

Robustness harness: fit kp-only vs kp+verts, optionally dropping a fraction of
keypoint observations (simulated occlusion), and compare the resulting qpos to the
full-data kp-only fit.  Run on GPU (sbatch).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def augment_cfg_with_vertices(cfg, mesh, M):
    """Add M vertex markers to a STAC cfg (in place); return ordered vtx names."""
    from omegaconf import OmegaConf, open_dict
    id2name = {int(s): str(n.decode() if isinstance(n, bytes) else n)
               for s, n in zip(mesh.seg_ids, mesh.seg_names)}
    fps = mesh.subset(M)
    vtx_names = [f"vtx_{i}" for i in range(M)]
    with open_dict(cfg):
        kmp = cfg.model.KEYPOINT_MODEL_PAIRS
        kio = cfg.model.KEYPOINT_INITIAL_OFFSETS
        reg = list(cfg.model.get("SITES_TO_REGULARIZE", []) or [])
        for i, vi in enumerate(fps):
            name = vtx_names[i]
            kmp[name] = id2name[int(mesh.vertex_segment[vi])]
            off = mesh.vertices_body[vi]
            kio[name] = f"{off[0]} {off[1]} {off[2]}"
            reg.append(name)                       # anchor vertex offsets to canonical
        cfg.model.SITES_TO_REGULARIZE = reg
    return vtx_names


def assemble_kp_data(bout_h5, labels_npz, M, *, occlude_frac=0.0, seed=0, pred_npz=None):
    """Return (kp_data (T,50+M,3) scaled, kp_names_250, vis (T,50+M)).

    Vertex observations are the STAC-GT verts3d by default; if ``pred_npz`` is a
    predict_full output, the model-PREDICTED vertices (pred3d[:,50:]) are used
    instead (real-world robustness, not the GT upper bound).  Both are in the raw
    mm frame and scaled by the per-recording factor s.
    """
    import numpy as np, h5py
    with h5py.File(bout_h5, "r") as f:
        kp_scaled = f["keypoints"][:]                 # (T,50,3) already * s
        kp_vis = f["vis"][:]                          # (T,50)
        s = float(f.attrs["scale"])
        kp_names = [n.decode() if isinstance(n, bytes) else n for n in f["kp_names"]]
    if pred_npz is not None:
        verts3d = np.load(pred_npz, allow_pickle=True)["pred3d"][:, 50:]   # (T,M,3) raw mm
    else:
        z = np.load(labels_npz, allow_pickle=True)
        verts3d = z["verts3d"]                        # (T,M,3) raw mm
    assert len(verts3d) == len(kp_scaled), (len(verts3d), len(kp_scaled))
    vert_obs = (verts3d * s).astype(np.float32)       # -> model cm
    vert_vis = np.ones((len(verts3d), M), bool)
    if occlude_frac > 0:                              # drop a fraction of KEYPOINT obs
        rng = np.random.default_rng(seed)
        drop = rng.random(kp_vis.shape) < occlude_frac
        kp_vis = kp_vis & ~drop
    kp_data = np.concatenate([kp_scaled, vert_obs], axis=1)         # (T,50+M,3)
    kp_data[..., :][~np.concatenate([kp_vis, vert_vis], 1)] = 0.0   # zero invisible
    names = list(kp_names) + [f"vtx_{i}" for i in range(M)]
    return kp_data, names, np.concatenate([kp_vis, vert_vis], 1)


def run_ik(cfg, kp_data, kp_names, out_h5):
    """fit_offsets + ik_only with the (augmented) marker set -> qpos h5."""
    import stac_mjx
    from pathlib import Path
    cfg.stac.n_frames_per_clip = kp_data.shape[0]
    cfg.stac.n_fit_frames = min(int(cfg.stac.get("n_fit_frames", kp_data.shape[0])),
                                kp_data.shape[0])
    cfg.stac.continuous = False
    cfg.stac.infer_qvels = False
    cfg.stac.fit_offsets_path = Path(out_h5).stem + "_fit.h5"
    cfg.stac.ik_only_path = Path(out_h5).name
    kp_flat = kp_data.reshape(kp_data.shape[0], -1) * cfg.model["MOCAP_SCALE_FACTOR"]
    return stac_mjx.run_stac(cfg, kp_flat, kp_names, save_path=Path(out_h5).parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--stac-config-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["kp", "dense"], required=True)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--occlude-frac", type=float, default=0.0)
    ap.add_argument("--pred-npz", default=None,
                    help="predict_full output; use PREDICTED vertices (real-world) vs GT")
    ap.add_argument("--overrides", nargs="*",
                    default=["paths=hyak", "anatomy=v1", "dataset="])
    a = ap.parse_args()

    import hydra
    from stac_mjx.path_utils import register_custom_resolvers, convert_dict_to_path
    from jarvis_jax.cse.mesh_assets import load_canonical
    register_custom_resolvers()
    with hydra.initialize_config_dir(config_dir=a.stac_config_dir, version_base=None):
        cfg = hydra.compose(config_name="config", overrides=list(a.overrides) + [
            "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"])
    cfg.paths = convert_dict_to_path(cfg.paths)

    kp_data, names, vis = assemble_kp_data(a.bout, a.labels, a.M,
                                           occlude_frac=a.occlude_frac, pred_npz=a.pred_npz)
    if a.mode == "kp":
        kp_data, names = kp_data[:, :50], names[:50]   # keypoints only
    else:
        mesh = load_canonical(a.mesh)
        augment_cfg_with_vertices(cfg, mesh, a.M)

    os.makedirs(a.out_dir, exist_ok=True)
    src = "pred" if a.pred_npz else "gt"
    tag = f"{a.mode}_{src}_occ{int(a.occlude_frac*100)}"
    out = os.path.join(a.out_dir, f"Fruitfly_ik_{tag}.h5")
    run_ik(cfg, kp_data, names, out)
    print(f"[stac_dense] {tag}: {kp_data.shape} -> {out}")


if __name__ == "__main__":
    main()
