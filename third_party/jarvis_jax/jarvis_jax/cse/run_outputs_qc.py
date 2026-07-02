"""Phase 7 / component 8 DEPLOY driver: wrap a scenario's solved qpos into
outputs (Task 1) + QC report (Task 2) + optional overlay videos (Task 3).

Consumes scenario qpos (single/multi/amputation/headless) -- NEVER changes any
solver. Resolves calibration PER RECORDING (refined cse_work/calib_refined/<rec>
if present, else factory <root>/calib_params/<rec>): the hardcoded single-recording
_DEFAULT_*_CALIB_DIR in silhouette_ik_solve is a deploy blocker and is not used
here.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


def resolve_calib_dir(recording, *, root, cse_work_dir):
    """Refined calib if present, else factory; raise if neither exists."""
    refined = os.path.join(cse_work_dir, "calib_refined", recording)
    factory = os.path.join(root, "calib_params", recording)
    if os.path.isdir(refined):
        return refined
    if os.path.isdir(factory):
        return factory
    raise FileNotFoundError(
        f"no calib for {recording}: neither {refined} nor {factory} exists")


def _load_coco(root, split):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return coco, id2file, id2ann_multi


def compute_bridges(recording, *, ik_h5, model_xml, root, split, calib_dir,
                    fs_imgids, T, ann_id_by_image=None):
    """Per-frame model->mm Umeyama bridges, exactly as run_single_fly."""
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik_solve import (
        build_solver_inputs, _umeyama, _triangulate_kp_mm, _cam2img_for_frame)

    inputs = build_solver_inputs(ik_h5, model_xml)
    kp_names = list(inputs["kp_names"])
    marker_sites = np.asarray(ioh5.load(ik_h5)["marker_sites"])[:T]

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco, id2file, id2ann_multi = _load_coco(root, split)
    coco_kpnames = coco["keypoint_names"]

    bridges = []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img,
                                        id2ann_multi, ann_id_by_image)
        if kok.sum() < 3:
            bridges.append(None)
            continue
        s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
        bridges.append((s, R, tr))
    return bridges


def gather_qc_frame_inputs(recording, *, root, split, calib_dir, fs_imgids, T,
                           ik_kpnames, ann_id_by_image=None):
    """Per-frame QC inputs aligned to ik_kpnames order: kp2d/vis/masks by cam."""
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik_solve import (
        _cam2img_for_frame, _ann_for_image, _load_sam_mask)

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco, id2file, id2ann_multi = _load_coco(root, split)
    coco_kpnames = coco["keypoint_names"]
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    # map each ik keypoint slot -> its coco index (or -1 if absent)
    coco_idx = np.array([name2coco.get(n, -1) for n in ik_kpnames], int)
    n_kp = len(ik_kpnames)

    kp2d_by_frame, vis_by_frame, masks_by_frame = [], [], []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp2d, vis, masks = {}, {}, {}
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            uv = np.zeros((n_kp, 2)); vv = np.zeros(n_kp, bool)
            for j, ci in enumerate(coco_idx):
                if ci >= 0 and kp[ci, 2] > 0:
                    uv[j] = kp[ci, :2]; vv[j] = True
            kp2d[c] = uv; vis[c] = vv
            m = _load_sam_mask(root, split, id2file[iid], ann["id"])
            if m is not None:
                masks[c] = m
        kp2d_by_frame.append(kp2d); vis_by_frame.append(vis); masks_by_frame.append(masks)
    return {"kp2d_by_frame": kp2d_by_frame, "vis_by_frame": vis_by_frame,
            "masks_by_frame": masks_by_frame}


def run_outputs_qc(recording, *, ik_h5, model_xml, mesh_npz, root, split="val",
                   qpos_npz=None, calib_dir=None, cse_work_dir, out_dir,
                   mesh_subset="fps_500", max_frames=0, make_video=False,
                   video_fps=30, ann_id_by_image=None):
    """Deploy entrypoint: outputs h5 + QC json (+ optional per-cam mp4s)."""
    import h5py
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse import outputs as outmod
    from jarvis_jax.cse import qc as qcmod
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs

    if calib_dir is None:
        calib_dir = resolve_calib_dir(recording, root=root, cse_work_dir=cse_work_dir)

    # qpos: from the passed npz else the ik h5's stored trajectory.
    if qpos_npz is not None:
        qpos = np.asarray(np.load(qpos_npz)["qpos"], np.float32)
    else:
        qpos = np.asarray(ioh5.load(ik_h5)["qpos"], np.float32)
    T_full = qpos.shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)
    qpos = qpos[:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    inputs = build_solver_inputs(ik_h5, model_xml)
    ik_kpnames = list(inputs["kp_names"])

    bridges = compute_bridges(
        recording, ik_h5=ik_h5, model_xml=model_xml, root=root, split=split,
        calib_dir=calib_dir, fs_imgids=fs_imgids, T=T, ann_id_by_image=ann_id_by_image)

    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, f"{recording}_outputs.h5")
    schema = outmod.build_fly_outputs(
        recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
        qpos=qpos, bridges=bridges, out_path=out_h5, mesh_subset=mesh_subset)

    # QC: read back the FK'd mesh/kp from the h5 (world mm) for the report.
    d = ioh5.load(out_h5)
    mesh_mm = np.asarray(d["mesh_mm"]); kp3d_mm = np.asarray(d["kp3d_mm"])
    qi = gather_qc_frame_inputs(
        recording, root=root, split=split, calib_dir=calib_dir,
        fs_imgids=fs_imgids, T=T, ik_kpnames=ik_kpnames,
        ann_id_by_image=ann_id_by_image)

    rt = ReprojectionTool(calib_dir)
    out_json = os.path.join(out_dir, f"{recording}_qc.json")
    report = qcmod.qc_report(
        rt, kp3d_by_frame=[kp3d_mm[t] for t in range(T)],
        mesh_by_frame=[mesh_mm[t] for t in range(T)],
        kp2d_by_frame=qi["kp2d_by_frame"], vis_by_frame=qi["vis_by_frame"],
        masks_by_frame=qi["masks_by_frame"], out_json=out_json)

    videos = []
    if make_video:
        from jarvis_jax.cse.reproj_video import write_camera_video
        import matplotlib.image as mpimg
        cam_names = list(rt.cameras.keys())
        _coco, id2file, _m = _load_coco(root, split)
        from jarvis_jax.cse.silhouette_ik_solve import _cam2img_for_frame
        for c, cam in enumerate(cam_names):
            # per-frame raw path + projected mesh/kp for this camera
            def _frames():
                for t in range(T):
                    cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
                    iid = cam2img.get(c)
                    if iid is None:
                        yield np.zeros((8, 8, 3), np.uint8)
                    else:
                        yield mpimg.imread(os.path.join(root, split, id2file[iid]))
            mesh2d = [np.stack([rt.reproject_point(mesh_mm[t][k])[c]
                                for k in range(mesh_mm.shape[1])], 0)
                      if not np.isnan(mesh_mm[t]).all() else np.zeros((0, 2))
                      for t in range(T)]
            kp2d = [np.stack([rt.reproject_point(kp3d_mm[t][j])[c]
                              for j in range(kp3d_mm.shape[1])], 0)
                    if not np.isnan(kp3d_mm[t]).all() else np.zeros((0, 2))
                    for t in range(T)]
            vpath = os.path.join(out_dir, f"{recording}_{cam}_reproj.mp4")
            write_camera_video(vpath, frames_rgb_iter=_frames(),
                               mesh2d_by_frame=mesh2d, kp2d_by_frame=kp2d, fps=video_fps)
            videos.append(vpath)

    return {"out_h5": out_h5, "qc_json": out_json, "calib_dir": calib_dir,
            "videos": videos, "n_frames": T, "qc": report, "schema": schema}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ik-h5", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--qpos-npz", default=None)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--cse-work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh-subset", default="fps_500")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=30)
    a = ap.parse_args()
    rep = run_outputs_qc(
        a.recording, ik_h5=a.ik_h5, model_xml=a.xml, mesh_npz=a.mesh, root=a.root,
        split=a.split, qpos_npz=a.qpos_npz, calib_dir=a.calib_dir,
        cse_work_dir=a.cse_work_dir, out_dir=a.out_dir, mesh_subset=a.mesh_subset,
        max_frames=a.max_frames, make_video=a.make_video, video_fps=a.video_fps)
    print("PHASE7 OUTPUTS+QC REPORT")
    print(f"  out_h5: {rep['out_h5']}")
    print(f"  qc_json: {rep['qc_json']}")
    print(f"  calib_dir: {rep['calib_dir']}")
    print(f"  n_frames: {rep['n_frames']}")
    print(f"  qc: {json.dumps(rep['qc'], indent=2)}")


if __name__ == "__main__":
    main()
