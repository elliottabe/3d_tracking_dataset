#!/usr/bin/env python3
"""SAM3 masks for the explainer clip.

The retrained ViTPose is 4-channel: session_frameset.build_frameset writes the
SAM3 target mask into channel 3 of every 448x448 crop, so masks are a hard
prerequisite for the 2D stage, not an optional QC aid.

num_animals=1 and distractor_masks stay None -- predict_bout_2d's docstring
records that distractor gray-fill is "None for single-animal assays".

PROJECT RESOLUTION: the sam3 config default ("red_data_unified") does not
exist under any local `projects/` dir. Of the projects at
`/home/eabe/Research/MyRepos/JARVIS-HybridNet/projects`
(Example_Project, unified_V2_masked, unified_V2_maskfree), only
`unified_V2_masked` matches this pipeline: 50 KEYPOINT_NAMES equal to the
detector's fly keypoints (T1L_FeTi, ...), NUM_CAMERAS: 7, and
`KEYPOINTDETECT.INSTANCE_MASK_INPUT: true` -- the mask-aware variant, which is
the one that needs SAM3 masks at all. `unified_V2_maskfree` is the same
keypoint set but INSTANCE_MASK_INPUT: false (no use for masks);
Example_Project is an unrelated 23-joint/12-camera hand-tracking project. So
`unified_V2_masked` is the default `--project` here.

Also: `third_party/JARVIS-HybridNet` (this repo's submodule, used only for
`import jarvis.*`) has NO `projects/` dir, so `jarvis_root` must point at the
separate checkout that owns `projects/` -- passed explicitly below rather than
relying on the `JARVIS_ROOT` env var.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, prepare_clip   # noqa: E402

# third_party/JARVIS-HybridNet (jarvis_jax's code_root) has no projects/ dir;
# this is the separate checkout that does. See module docstring.
JARVIS_ROOT_DEFAULT = "/home/eabe/Research/MyRepos/JARVIS-HybridNet"
PROJECT_DEFAULT = "unified_V2_masked"


def run_masks(clip: str, *, limit_frames: int = 0, jarvis_root: str | None = None,
              project: str = PROJECT_DEFAULT):
    """Run SAM3 over the clip's single bout. limit_frames>0 => pilot slice."""
    from jarvis_jax.predict.sam3_driver import run_sam3_masks

    d = clip_io.out_dirs(clip)
    # A pilot MUST use a truncated bout and its OWN output dir: run_sam3_masks
    # has no frame limit (`limit` counts bouts), and reuse_masks=True would
    # otherwise let a 40-frame pilot npz stand in for the full run.
    if limit_frames:
        bouts = prepare_clip.write_bouts_csv(clip, n_override=limit_frames,
                                             name="bouts_pilot.csv")
        out_dir = d["predictions"] / "sam3_pilot"
    else:
        bouts = prepare_clip.write_bouts_csv(clip)
        out_dir = d["predictions"] / "sam3"
    t0 = time.time()
    manifest = run_sam3_masks(
        project=project,
        session_dir=str(clip_io.stage_session_dir(clip)),
        bouts_csv=str(bouts),
        out=str(out_dir),
        num_animals=1,
        reuse_masks=True,
        jarvis_root=jarvis_root or os.environ.get("JARVIS_ROOT", JARVIS_ROOT_DEFAULT),
        sam3={"text_prompt": "insect", "sam3_version": "sam3.1", "gpu_id": 0,
              "compile": False},
        overlay=True, overlay_cams=3, overlay_frames=min(300, limit_frames or 300),
    )
    dt = time.time() - t0
    n = limit_frames or n_frames_full(clip)
    print(f"[timing] SAM3 {n} frames x 7 cams in {dt:.1f}s "
          f"({dt / max(n, 1):.3f} s/frame)")
    if limit_frames:
        full = prepare_clip.n_frames(clip)
        print(f"[timing] extrapolated full run ({full} frames): "
              f"{dt / limit_frames * full / 60:.1f} min")
    return manifest, dt


def n_frames_full(clip: str) -> int:
    return prepare_clip.n_frames(clip)


def mask_npz_path(clip: str) -> Path:
    d = clip_io.out_dirs(clip)
    hits = sorted((d["predictions"] / "sam3").rglob("sam3_masks.npz"))
    if not hits:
        raise FileNotFoundError(f"no sam3_masks.npz under {d['predictions'] / 'sam3'}")
    return hits[0]


def qc_masks(clip: str, frames=(120, 450, 780)):
    """Mask QC montage.

    EXPECTATION if masks are good: exactly ONE connected fly-sized blob per
    camera per frame, covering the fly and not the arena wall or its
    reflection, present in all 7 views at every sampled frame.
    FALSIFICATION: blobs on arena features, multiple competing blobs, or an
    empty view -> the 2D stage will read garbage from crop channel 3.
    """
    import cv2
    from jarvis_jax.tracking.bout_masks import load_bout_masks

    d = clip_io.out_dirs(clip)
    _mats, names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    # load_bout_masks(npz_path, fly, *, expected_cameras=None) -> DICT with keys
    # masks (T,C,H,W) bool, valid (T,C), centroids (T,C,2). Passing
    # expected_cameras reorders the C axis BY NAME into our DLT order -- without
    # it the mask camera axis can silently disagree with the calibration (see
    # scripts/fix_mask_camera_order.py and tests/test_mask_camera_order.py).
    bm = load_bout_masks(str(mask_npz_path(clip)), 0, expected_cameras=names)
    panels = []
    for f in frames:
        row = []
        for ci, cam in enumerate(names):
            img = clip_io.read_frames(clip_io.video_path(clip, cam), [f])[0]
            m = np.asarray(bm["masks"][f, ci], bool)
            n_blobs, _lab = cv2.connectedComponents(m.astype(np.uint8))
            over = img.copy()
            over[m] = (0.5 * over[m] + 0.5 * np.array([200, 200, 200])).astype(np.uint8)
            cv2.putText(over, f"{cam} f{f} blobs={n_blobs - 1} px={int(m.sum())}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row.append(cv2.resize(over, (484, 112)))
        panels.append(np.hstack(row))
    out = d["qc"] / "01_masks.png"
    cv2.imwrite(str(out), np.vstack(panels))
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, run only this many frames and report timing")
    ap.add_argument("--jarvis-root", default=None)
    ap.add_argument("--project", default=PROJECT_DEFAULT)
    ap.add_argument("--qc", action="store_true", help="write the mask QC montage")
    a = ap.parse_args()
    if a.qc:
        qc_masks(a.clip)
        return 0
    run_masks(a.clip, limit_frames=a.pilot, jarvis_root=a.jarvis_root,
              project=a.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
