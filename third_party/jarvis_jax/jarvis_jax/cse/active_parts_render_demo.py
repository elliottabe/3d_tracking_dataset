"""Reprojection-overlay viz for Phase-4 active-parts IK results (amputee/headless).

Renders the ACTIVE-PARTS-solved qpos (not the raw STAC qpos) into every camera,
overlaid on the real camera image, to visually confirm (1) the fit is correct and
(2) the missing body part is locked at rest (not hallucinated/reaching).

Geometry chain (mirrors silhouette_render_demo.py, see its module docstring):
  qpos --(mjx FK)--> model-frame mesh verts --(Umeyama)--> calibrated mm --(P_c)--> px

The model->mm bridge is made CONSISTENT with the active-parts qpos (rather than
reusing the ik_h5's STAC-qpos marker_sites as-is, which would be geometrically
inconsistent for frames where the active-parts solve differs from STAC, especially
near the off-part): for each frame, `build_solver_inputs` gives the mjx model +
`site_idxs` (STAC marker-site ids) + `kp_names`; running FK from the active-parts
qpos through those sites (mirroring `run_single_fly`'s reproj_px computation in
silhouette_ik_solve.py: `mjx_data.replace(qpos=...)` -> `stac_utils.kinematics` ->
`stac_utils.com_pos` -> `stac_utils.get_site_xpos`) gives "fitted markers" in MODEL
frame that are consistent with the qpos being rendered. Those are Umeyama-matched
(by NAME) to that frame's triangulated coco keypoints (present/visible only) to get
(s, R, t), which is then applied to the FK mesh vertices (also from the active-parts
qpos) for projection into each camera.

Coloring (mesh_npz's vertex_segment, same field the template uses):
  body/leg -> cyan, wing -> red, off-part (build_active_mask's excluded_seg_ids) ->
  magenta (overrides cyan/red).

Also prints, per recording, the mean reprojection error (px) of the fitted markers
(present markers only) against the coco 2-D keypoints, across the rendered frames'
visible cameras -- this is the same "reproj_px" convention as
silhouette_ik_solve.run_single_fly, but recomputed here against the active-parts
qpos + qpos-consistent bridge (rather than the solved-but-STAC-bridge value that
run_single_fly reports internally).
"""
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import stac_mjx.io_dict_to_hdf5 as ioh5
from stac_mjx import utils as stac_utils

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs, _umeyama, _model_to_mm
from jarvis_jax.cse.active_parts import derive_active_parts, build_active_mask

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"

CONDITIONS = {
    "amputee": dict(
        cond_root="/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/S8_male_R_amp",
        rec="2026_02_13_13_44_49",
        split="train",
        qpos_npz="/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad/p4_viz/amputee/2026_02_13_13_44_49_qpos.npz",
        frames=[40, 120],
        expected_off=["legT1R"],
        out_dir="/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad/p4_viz/amputee",
    ),
    "headless": dict(
        cond_root="/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model/headless_22_50",
        rec="2026_06_09_15_46_55",
        split="train",
        qpos_npz="/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad/p4_viz/headless/2026_06_09_15_46_55_qpos.npz",
        frames=[5, 12],
        expected_off=["head"],
        out_dir="/tmp/claude-398823/-mmfs1-gscratch-portia-eabe-Research-MyRepos-3d-tracking-dataset/3d38bebe-eeae-4891-b119-71d70de6e331/scratchpad/p4_viz/headless",
    ),
}


def _cam2img_for_frame(fs_imgids_row, id2file, cam_names):
    cam2img = {}
    for iid in fs_imgids_row:
        fn = id2file.get(int(iid), "")
        cam = fn.split("/")[1] if "/" in fn else ""
        if cam in cam_names:
            cam2img[cam_names.index(cam)] = int(iid)
    return cam2img


def _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann):
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    n = len(kp_names)
    kp_mm = np.zeros((n, 3))
    kok = np.zeros(n, dtype=bool)
    kpx_by_cam = {}  # (kp_idx, cam) -> (u, v)  for reproj-error bookkeeping
    for j, nm in enumerate(kp_names):
        ci = name2coco.get(nm)
        if ci is None:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        cams = []
        for c, iid in cam2img.items():
            ann = id2ann.get(iid)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
            if kp[ci, 2] > 0:
                obs[c] = kp[ci, :2]
                cams.append(c)
                kpx_by_cam[(j, c)] = kp[ci, :2]
        if len(cams) >= 2:
            kp_mm[j] = rt.reconstruct_point(obs, cams_to_use=cams)
            kok[j] = True
    return kp_mm, kok, kpx_by_cam


def _load_mask(root, split, fn, ann_id):
    p = os.path.join(root, "sam3_masks", split, os.path.splitext(fn)[0] + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    if z["masks"].shape[0] == 0:
        return None
    sel = np.where((z["ann_ids"] == ann_id) & z["matched"])[0]
    if not len(sel):
        sel = np.where(z["ann_ids"] == ann_id)[0]
    return z["masks"][sel[0]].astype(bool) if len(sel) else None


def process_condition(name, cfg):
    print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
    root, rec, split = cfg["cond_root"], cfg["rec"], cfg["split"]
    ik_h5 = os.path.join(root, "cse_work", rec, "Fruitfly_ik_v1_cse.h5")
    bout_h5 = os.path.join(root, "cse_work", f"{rec}_bout.h5")
    calib_dir = os.path.join(root, "calib_params", rec)
    coco_path = os.path.join(root, "annotations", f"instances_{split}.json")

    # --- load active-parts qpos (the pose we're rendering) ---
    qpos_ap = np.load(cfg["qpos_npz"])["qpos"]
    print(f"active-parts qpos shape: {qpos_ap.shape}")

    # --- bout h5 frame -> per-cam image_id mapping ---
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()]
    print(f"bout fs_imgids shape: {fs_imgids.shape}")
    assert fs_imgids.shape[0] == qpos_ap.shape[0], (
        f"frame-count mismatch: bout has {fs_imgids.shape[0]} rows, "
        f"qpos.npz has {qpos_ap.shape[0]} -- frame correspondence assumption violated!"
    )
    print("frame-index correspondence verified: qpos.npz row t == bout h5 row t")

    coco = json.load(open(coco_path))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    coco_kpnames = coco["keypoint_names"]

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    ncam = rt.num_cameras
    print(f"n cameras: {ncam} -> {cam_names}")

    # --- off-parts: derive (don't hardcode), sanity-check vs. expectation ---
    off_derived = derive_active_parts(coco_kpnames)["off"]
    if sorted(off_derived) != sorted(cfg["expected_off"]):
        print(f"WARNING: derived off_parts {off_derived} != expected {cfg['expected_off']}")
    else:
        print(f"off_parts (derived, matches expectation): {off_derived}")

    # --- qpos-consistent bridge setup: build_solver_inputs -> mjx model + site_idxs ---
    inputs = build_solver_inputs(ik_h5, XML)
    mjx_model, mjx_data_template = inputs["mjx_model"], inputs["mjx_data"]
    site_idxs = inputs["site_idxs"]
    kp_names = list(inputs["kp_names"])  # STAC-order kp names matching site_idxs

    mask = build_active_mask(kp_names, XML, MESH, off_derived)
    excluded_seg_ids = set(mask["excluded_seg_ids"])
    print(f"excluded_seg_ids (off-part mesh segs): {sorted(excluded_seg_ids)}")

    # --- mesh + FK-repose for the active-parts qpos (model frame) ---
    anat = load_anatomy(XML, MESH)
    fk = make_fk_repose(anat)
    seg_names = [(n.decode() if isinstance(n, bytes) else n).lower() for n in anat["seg_names"]]
    id2nm = {int(s): seg_names[i] for i, s in enumerate(anat["seg_ids"])}
    vseg = anat["vertex_segment"]
    iswing = np.array([("wing" in id2nm[int(s)]) for s in vseg])
    isoff = np.array([int(s) in excluded_seg_ids for s in vseg])

    def fitted_markers_from_qpos(q_row):
        """qpos (nq,) -> (n_kp,3) MODEL-frame marker-site positions.

        Mirrors run_single_fly's reproj_px computation in silhouette_ik_solve.py:
        mjx_data.replace(qpos=...) -> stac_utils.kinematics -> stac_utils.com_pos ->
        stac_utils.get_site_xpos(., site_idxs).
        """
        data_t = mjx_data_template.replace(qpos=np.asarray(q_row, dtype=np.float32))
        data_t = stac_utils.kinematics(mjx_model, data_t)
        data_t = stac_utils.com_pos(mjx_model, data_t)
        return np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))  # (n_kp, 3)

    import jax.numpy as jnp

    all_frame_reproj = []  # per-frame mean px (for the printed summary)
    png_paths = []

    for fr in cfg["frames"]:
        if fr >= qpos_ap.shape[0]:
            print(f"skip frame {fr}: out of range (T={qpos_ap.shape[0]})")
            continue
        q_row = qpos_ap[fr]

        fitted_markers = fitted_markers_from_qpos(q_row)  # (n_kp,3) MODEL frame

        cam2img = _cam2img_for_frame(fs_imgids[fr], id2file, cam_names)
        kp_mm, kok, kpx_by_cam = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann)

        n_present = int(kok.sum())
        if n_present < 3:
            print(f"frame {fr}: only {n_present} triangulated markers, skipping (need >=3 for Umeyama)")
            continue

        s, R, t = _umeyama(fitted_markers[kok], kp_mm[kok])
        print(f"frame {fr}: model->mm scale={s:.4f}  n_kp_present={n_present}")

        # mesh verts (active-parts qpos, MODEL frame) -> mm via this frame's transform
        vmodel = np.asarray(fk(jnp.asarray(q_row.astype(np.float32))))
        vmm = s * (R @ vmodel.T).T + t

        # fitted markers -> mm, for the reprojection-error number
        markers_mm = _model_to_mm(fitted_markers, s, R, t)
        frame_px_errs = []
        for j, nm in enumerate(kp_names):
            if not kok[j]:
                continue
            uv_pred_all = rt.reproject_point(markers_mm[j])  # (n_cam, 2)
            for c in cam2img:
                obs = kpx_by_cam.get((j, c))
                if obs is None:
                    continue
                frame_px_errs.append(float(np.linalg.norm(uv_pred_all[c] - obs)))
        if frame_px_errs:
            fpx = float(np.mean(frame_px_errs))
            print(f"frame {fr}: mean reproj_px = {fpx:.4f}  (n_obs={len(frame_px_errs)})")
            all_frame_reproj.extend(frame_px_errs)
        else:
            print(f"frame {fr}: no valid reproj observations")

        # --- figure: 7-camera grid, mesh overlay colored cyan/red/magenta ---
        cols = min(4, ncam)
        rowsN = (ncam + cols - 1) // cols
        fig, ax = plt.subplots(rowsN, cols, figsize=(5.2 * cols, 3.8 * rowsN), squeeze=False)

        sub = np.arange(0, len(vmm), max(1, len(vmm) // 4000))
        uv_all = np.stack([rt.reproject_point(vmm[k]) for k in sub], 1)  # (ncam, K, 2)
        isw_sub = iswing[sub]
        isoff_sub = isoff[sub]
        isbody_sub = ~isw_sub & ~isoff_sub
        iswing_only_sub = isw_sub & ~isoff_sub

        for c in range(ncam):
            aa = ax.flat[c]
            if c in cam2img:
                fn = id2file[cam2img[c]]
                try:
                    aa.imshow(mpimg.imread(os.path.join(root, split, fn)))
                except Exception as e:
                    print(f"  warn: could not load image for cam {c}: {e}")
                ann = id2ann.get(cam2img[c])
                if ann is not None:
                    m = _load_mask(root, split, fn, ann["id"])
                    if m is not None:
                        aa.imshow(np.ma.masked_where(~m, m), alpha=0.3, cmap="autumn")
            uv = uv_all[c]
            aa.scatter(uv[isbody_sub, 0], uv[isbody_sub, 1], s=2, c="cyan", linewidths=0)
            aa.scatter(uv[iswing_only_sub, 0], uv[iswing_only_sub, 1], s=3, c="red", linewidths=0)
            aa.scatter(uv[isoff_sub, 0], uv[isoff_sub, 1], s=6, c="magenta", linewidths=0)
            x0, y0 = uv.min(0) - 60
            x1, y1 = uv.max(0) + 60
            aa.set_xlim(x0, x1)
            aa.set_ylim(y1, y0)
            aa.set_title(cam_names[c], fontsize=9)
            aa.axis("off")
        for c in range(ncam, rowsN * cols):
            ax.flat[c].axis("off")

        off_str = ",".join(off_derived) if off_derived else "none"
        fig.suptitle(
            f"Active-parts IK mesh -> all cameras | {name} {rec} frame {fr} | off-part: {off_str}\n"
            f"cyan=body/leg, red=wing, MAGENTA=off-part ({off_str}), orange=SAM mask"
        )
        os.makedirs(cfg["out_dir"], exist_ok=True)
        out_path = os.path.join(cfg["out_dir"], f"overlay_frame{fr}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=115)
        plt.close(fig)
        print(f"wrote {out_path}")
        png_paths.append(out_path)

    mean_reproj = float(np.mean(all_frame_reproj)) if all_frame_reproj else float("nan")
    print(f"\n{name}: MEAN reproj_px across rendered frames = {mean_reproj}")
    return dict(png_paths=png_paths, mean_reproj_px=mean_reproj, off_parts=off_derived)


def main():
    results = {}
    for name, cfg in CONDITIONS.items():
        results[name] = process_condition(name, cfg)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, r in results.items():
        print(f"{name}: off_parts={r['off_parts']}  mean_reproj_px={r['mean_reproj_px']}")
        for p in r["png_paths"]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
