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

REQUIRED INTERPRETER: run this module with
`/home/eabe/miniconda3/envs/sam3/bin/python`, NOT the `3d_tracking` env's
python that runs every other stage of this plan. `3d_tracking`'s torch
(2.11.0+cu130) cannot use CUDA on this workstation at all -- the installed
driver (570.207) has a CUDA-12.8 ceiling, one major version short of what a
cu130 wheel needs (`torch.cuda.is_available()` is False there, confirmed).
`sam3`'s torch (2.7.0+cu126) is driver-compatible and `cuda_available=True`.
(The `jarvis` env's torch is also cu128/driver-compatible, but its `cv2`
fails to import there on a `CXXABI_1.3.15` mismatch, so it isn't an option.)
This split is natural, not a hack: SAM3 is a separate PyTorch process, while
every other stage in this plan is JAX, which works fine in `3d_tracking`
because it uses `jax-cuda12-plugin` rather than torch's own CUDA. See
`_require_cuda_torch` below, which fails fast rather than silently falling
back to a CPU run that could take days.

No `LD_PRELOAD`/`LD_LIBRARY_PATH` overrides were needed to run this in the
`sam3` env (verified): torch there already finds its CUDA libraries under
each `nvidia/<component>/lib` (e.g. `nvidia/cudnn/lib`, `nvidia/cublas/lib`)
without help. This differs from the `3d_tracking` env's documented prelude
(`LD_LIBRARY_PATH=.../nvidia/cu13/lib`), which exists because that env's
nvidia wheels are the newer combined per-CUDA-major package layout -- a
different packaging convention, not a sign that `sam3` needs the same fix.
"""
import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
# `import jarvis.*` (jarvis.utils.reprojection, for _repro_tool_from_staged_calib
# below) needs this on sys.path too -- sam3_driver.run_sam3_masks normally adds
# it lazily on its own entry (sam3_driver.py:1340), but our shim needs `jarvis`
# importable BEFORE that call, since it builds the repro tool up front.
sys.path.insert(0, str(_REPO / "third_party" / "JARVIS-HybridNet"))

from scripts.viz.ik_explainer import clip_io, prepare_clip   # noqa: E402

# third_party/JARVIS-HybridNet (jarvis_jax's code_root) has no projects/ dir;
# this is the separate checkout that does. See module docstring.
JARVIS_ROOT_DEFAULT = "/home/eabe/Research/MyRepos/JARVIS-HybridNet"
PROJECT_DEFAULT = "unified_V2_masked"
SAM3_PYTHON = "/home/eabe/miniconda3/envs/sam3/bin/python"


def _require_cuda_torch():
    """Fail fast rather than let a CUDA-less torch silently fall back to a
    CPU run that could take days -- see the module docstring for why this
    module must run under the `sam3` conda env's interpreter, not
    `3d_tracking`'s."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"SAM3 needs CUDA-capable torch; this interpreter "
            f"({sys.executable}) has torch {torch.__version__} with "
            f"cuda_available=False. Run this stage with "
            f"{SAM3_PYTHON} -- the 3d_tracking env "
            f"ships a cu130 build the 570.207 driver (CUDA 12.8 ceiling) "
            f"cannot use."
        )


@contextlib.contextmanager
def _repro_tool_from_staged_calib(calib_dir, device="cuda"):
    """Supply SAM3's ReprojectionTool from THIS clip's calibration.

    jarvis.utils.reprojection.load_reprojection_tools (reprojection.py:189)
    unconditionally reads the TRAINING dataset's annotations/instances_val.json
    and iterates data['calibrations'] -- a coupling to training data that this
    clip has no part in. The local projects point at red_data_unified_V2, which
    is not on this machine, and V4 (which is) has no 'calibrations' key.

    We already hold this clip's exact calibration as staged Cam*.yaml, so we
    build JARVIS's own torch ReprojectionTool directly from it and patch it in
    for the duration of the SAM3 call. Nothing about SAM3's behaviour changes --
    it receives the same class it always would, just constructed from this
    clip's calibration instead of a training set's.

    ReprojectionTool.__init__(root_dir, calib_paths, device, use_dlt=True)
    (jarvis/utils/reprojection.py:17) loads each camera from
    `os.path.join(root_dir, calib_paths[camera])` -- so calib_paths values are
    bare filenames ("Cam2012630.yaml"), root_dir is calib_dir itself.
    """
    from jarvis.utils.reprojection import ReprojectionTool
    import jarvis.utils.reprojection as _rp
    names = sorted(p.stem for p in Path(calib_dir).glob("Cam*.yaml"))
    if not names:
        raise FileNotFoundError(f"no Cam*.yaml under {calib_dir}")
    calib_paths = {n: f"{n}.yaml" for n in names}
    tool = ReprojectionTool(str(calib_dir), calib_paths, device)
    with mock.patch.object(_rp, "get_repro_tool", lambda cfg, ds, device="cuda": tool):
        yield tool


def run_masks(clip: str, *, limit_frames: int = 0, jarvis_root: str | None = None,
              project: str = PROJECT_DEFAULT, device: str = "cuda"):
    """Run SAM3 over the clip's single bout. limit_frames>0 => pilot slice.

    The repro tool JARVIS's cfg/project would normally supply is replaced,
    for the duration of the call, with one built directly from this clip's
    staged calibration -- see `_repro_tool_from_staged_calib`.
    """
    _require_cuda_torch()
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
    session_dir = clip_io.stage_session_dir(clip)
    t0 = time.time()
    with _repro_tool_from_staged_calib(str(session_dir / "calibration"), device=device):
        manifest = run_sam3_masks(
            project=project,
            session_dir=str(session_dir),
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
    ap.add_argument("--device", default="cuda",
                    help="device for the injected repro tool (matches SAM3's own device)")
    ap.add_argument("--qc", action="store_true", help="write the mask QC montage")
    a = ap.parse_args()
    if a.qc:
        qc_masks(a.clip)
        return 0
    run_masks(a.clip, limit_frames=a.pilot, jarvis_root=a.jarvis_root,
              project=a.project, device=a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
