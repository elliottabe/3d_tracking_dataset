"""CLI: per-recording affine-camera bundle adjustment from coco keypoints.

    python -m jarvis_jax.cse.run_bundle_adjust \
        --root /.../red_data_unified_V3 --recording 2026_03_18_15_31_22 --split val \
        --calib-out /.../calib_refined/2026_03_18_15_31_22 \
        --report-out /.../calib_refined/2026_03_18_15_31_22/ba_report.json \
        --refine pose --max-frames 300
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import cv2
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.bundle_adjust import assemble_observations, refine_calibration


def load_kp2d_from_coco(root, split, recording, max_frames=0):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann = {an["image_id"]: an for an in coco["annotations"]}
    K = len(coco["keypoint_names"])
    rt = ReprojectionTool(os.path.join(root, "calib_params", recording))
    cam_names = list(rt.cameras.keys())
    fs_items = [(k, v) for k, v in coco["framesets"].items()
                if v.get("datasetName") == recording]
    if max_frames and len(fs_items) > max_frames:
        idx = np.linspace(0, len(fs_items) - 1, max_frames).astype(int)
        fs_items = [fs_items[i] for i in idx]
    F, C = len(fs_items), len(cam_names)
    kp2d = np.zeros((F, C, K, 3), np.float64)
    for f, (_, fv) in enumerate(fs_items):
        for iid in fv["frames"]:
            fn = id2file.get(int(iid), "")
            cam = fn.split("/")[1] if "/" in fn else ""
            if cam not in cam_names:
                continue
            ann = id2ann.get(int(iid))
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            kp2d[f, cam_names.index(cam)] = kp
    return kp2d, cam_names


def write_calibration(out_dir, cam_names, mats):
    os.makedirs(out_dir, exist_ok=True)
    for nm, P in zip(cam_names, mats):
        fs = cv2.FileStorage(os.path.join(out_dir, f"{nm}.yaml"), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", np.asarray(P, np.float64)); fs.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--recording", required=True)
    ap.add_argument("--split", default="val"); ap.add_argument("--calib-out", required=True)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--refine", default="pose", choices=["pose", "pose+K", "full"])
    # calibration BA is over-determined with a few dozen frames; dense_cholesky does not scale to thousands of points (use fewer frames or a CG solver for larger problems).
    ap.add_argument("--max-frames", type=int, default=20)
    a = ap.parse_args()
    rt = ReprojectionTool(os.path.join(a.root, "calib_params", a.recording))
    cam_mats = [c.cameraMatrix for c in rt._camera_list]
    kp2d, cam_names = load_kp2d_from_coco(a.root, a.split, a.recording, a.max_frames)
    obs = assemble_observations(kp2d, min_cams=2)
    refined, report = refine_calibration(obs, cam_mats, refine=a.refine)
    write_calibration(a.calib_out, cam_names, refined)
    report["recording"] = a.recording
    print(f"BA {a.recording}: err {report['err_before']:.3f} -> {report['err_after']:.3f} px "
          f"(improved={report['improved']}, n_pts={report['n_points']})")
    if a.report_out:
        os.makedirs(os.path.dirname(a.report_out), exist_ok=True)
        json.dump(report, open(a.report_out, "w"), indent=2)


if __name__ == "__main__":
    main()
