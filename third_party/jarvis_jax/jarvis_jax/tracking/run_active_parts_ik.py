"""Phase-4 driver: active-parts silhouette-IK on a general_model recording.

Given a general_model condition (amputee / headless), read the recording's coco
keypoint_names, auto-derive the off body-part(s), and run the Phase-2/3
run_single_fly with the active-parts mask (markers dropped + joints locked +
post-solve clamp + off-part geoms excluded from the silhouette). Reports the
present-marker reprojection error and the no-phantom check (locked joints pinned
to rest).

Per-frame: the general_model recordings are sparse, individually-labeled
annotation frames with NO temporal continuity between them, so temporal
smoothness is meaningless (it would wrongly couple non-adjacent frames).
``smooth_weight`` therefore DEFAULTS TO 0.0 here (unlike run_single_fly's 0.1,
which is correct for the contiguous bouts of Phases 2-3).
"""
from __future__ import annotations

import json
import os

from jarvis_jax.tracking.active_parts import derive_active_parts
from jarvis_jax.tracking.ik_solve import run_single_fly


def run_active_parts_ik(recording, *, cond_root, model_xml, mesh_npz, split="train",
                        calib_dir=None, use_silhouette=True, max_frames=0, n_iter=50,
                        smooth_weight=0.0, out_dir, override=None):
    # smooth_weight defaults to 0.0: general_model recordings are sparse,
    # non-contiguous annotation frames with no temporal continuity, so the
    # temporal smoothness term is meaningless and must be disabled.
    coco = json.load(open(os.path.join(cond_root, "annotations", f"instances_{split}.json")))
    off_parts = derive_active_parts(coco["keypoint_names"], override=override)["off"]
    ik_h5 = os.path.join(cond_root, "cse_work", recording, "Fruitfly_ik_v1_cse.h5")
    if calib_dir is None:
        calib_dir = os.path.join(cond_root, "calib_params", recording)
    report = run_single_fly(
        recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
        root=cond_root, split=split, calib_dir=calib_dir,
        use_silhouette=use_silhouette, max_frames=max_frames, n_iter=n_iter,
        smooth_weight=smooth_weight,
        out_dir=out_dir, active_parts=(override if override is not None else off_parts))
    return {"off_parts": off_parts, "report": report}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-root", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-xml",
                    default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml")
    ap.add_argument("--mesh",
                    default="/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-silhouette", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--smooth-weight", type=float, default=0.0,
                    help="0.0 for sparse annotation frames (no temporal continuity)")
    ap.add_argument("--override", nargs="*", default=None)
    a = ap.parse_args()
    out = run_active_parts_ik(
        a.rec, cond_root=a.cond_root, model_xml=a.model_xml, mesh_npz=a.mesh,
        split=a.split, use_silhouette=not a.no_silhouette, max_frames=a.max_frames,
        n_iter=a.n_iter, smooth_weight=a.smooth_weight, out_dir=a.out_dir, override=a.override)
    print(f"[run_active_parts_ik] off={out['off_parts']} "
          f"reproj_px={out['report']['reproj_px']:.2f} "
          f"locked={out['report']['locked_qpos_idx']}")


if __name__ == "__main__":
    main()
