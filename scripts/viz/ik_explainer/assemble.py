#!/usr/bin/env python3
"""Assemble the four acts into the deliverable: mp4, README, manifest.

Concatenates `frames/act{1..4}_*/f%05d.png` in story order with 30-frame
(1 s) crossfades at each of the 3 act boundaries, encodes 1920x1080 @ 30 fps
H.264 (silent) via `viz.core.io.write_video` (imageio + ffmpeg, the repo's
shared encoder), then writes `README.md` and `manifest.json` beside it.

EXPECTED TOTAL FRAMES: 330 + 300 + 90 + 900 - 3*30 = 1530 (51.0 s @ 30 fps).
(Task-14 shortened Act 1 from 450 to 270 frames -- 15s to 9s -- and
re-sourced it from a contiguous ~40% sub-range of the clip rather than the
whole 921 frames; see `act1_views.py`'s module docstring. A later task-14
round lengthened the crossfades themselves from 15 to 30 frames per the
user's "smoother transitions" request -- see `N_CROSS` below. Task-15
shortened Act 3 from 600 to 360 frames -- 20s to 12s -- and dropped its
rest-pose fade-in phase; a direct mid-task user follow-up ("the scaling
doesn't look good... just have it do the root alignment... and can we have
it shorter") then retired the merged scale+align design and shortened Act 3
again, 360 -> 180 frames (12s -> 6s). Task-16 then cut Act 3 a further time,
180 -> 90 frames (6s -> 3s), keeping only the swing-in-and-align beat per a
second direct user pivot ("just the second half... where it just swings in
and aligns"); see `act3_align.py`'s module docstring's TASK-16 PIVOT
section. Task-20 then EXTENDED Act 1 again, 270 -> 330 frames, adding a
60-frame closing beat that crossfades the raw detector 2D into the
reprojection of the triangulated 3D across all seven panels -- see
`act1_views.py`'s module docstring's TASK-20 section; total 1470 -> 1530.)
This is asserted TWICE: once per-act (`_list_frames` requires each act directory to
hold EXACTLY its expected count -- neither short nor padded with extras),
and once on the assembled edit plan before any frame is written to ffmpeg.
Act 4 previously died mid-render at 700/900 and had to be resumed; this
check exists so a truncated act fails loudly here rather than shipping a
silently-short video.

CROSSFADE CONSTRUCTION: a transition between act A (length n_A) and act B
consumes A's last `N_CROSS` frames and B's first `N_CROSS` frames and
produces `N_CROSS` blended output frames (not 2x) via `draw.fade` -- so each
of the 3 transitions nets the sequence `-N_CROSS` frames overall, which is
where the "-3*N_CROSS" in the total comes from. Concretely, for output
position i in 0..N_CROSS-1 of a transition: `draw.fade(A[n_A-N_CROSS+i],
B[i], i/(N_CROSS-1))`, so alpha ramps from an exact copy of A's tail (t=0)
to an exact copy of B's head (t=1) with N_CROSS-2 genuine blends in between
-- no plain cut, no black frame (fade() never multiplies to zero; it is a
straight `cv2.addWeighted` cross-dissolve of two real frames).

Everything this script writes lands under `<clip>/ik_explainer/`; the raw
clip inputs (calibration/, Cam*.mp4, enhanced/, preview/, data3D_*.csv) are
never opened for writing here.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import clip_io, draw   # noqa: E402
from viz.core.io import write_video                  # noqa: E402

# --- edit plan ---------------------------------------------------------
# (directory name under frames/, expected frame count) in story order.
ACT_SPECS = [
    ("act1_views", 330),
    ("act2_triangulate", 300),
    ("act3_align", 90),
    ("act4_solve", 900),
]
N_CROSS = 30   # 1 s @ 30 fps; lengthened from 15 (task-14 round 4, "smoother transitions")
FPS = 30
CANVAS_W, CANVAS_H = 1920, 1080
EXPECTED_TOTAL = sum(n for _, n in ACT_SPECS) - (len(ACT_SPECS) - 1) * N_CROSS  # 1530


def _list_frames(act_dir: Path, act_name: str, expected: int) -> list:
    """Sorted `f*.png` frames in `act_dir`; raises if the count is not exactly
    `expected` -- both a short (truncated render) and a long (stray/duplicate
    files from a bad resume) directory are refused rather than silently
    reshaping the edit."""
    files = sorted(act_dir.glob("f*.png"))
    if len(files) != expected:
        raise RuntimeError(
            f"{act_name}: expected exactly {expected} frames in {act_dir}, "
            f"found {len(files)} -- refusing to assemble a truncated/malformed "
            f"act (act4_solve previously died mid-render at 700/900; this "
            f"check exists so that failure mode ships loudly, not silently)."
        )
    return files


def build_edit_plan(frame_lists: dict, order: list, n_cross: int = N_CROSS) -> list:
    """Return the ordered list of draw ops for the assembled video: each op is
    either a `Path` (a plain, unblended frame) or an `(path_a, path_b, t)`
    tuple (a crossfade blend, resolved by `draw.fade`). Exactly
    `EXPECTED_TOTAL` ops long for `ACT_SPECS`."""
    plan = []
    n_acts = len(order)
    for ai, name in enumerate(order):
        files = frame_lists[name]
        n = len(files)
        lo = n_cross if ai > 0 else 0
        hi = n - n_cross if ai < n_acts - 1 else n
        plan.extend(files[lo:hi])
        if ai < n_acts - 1:
            next_files = frame_lists[order[ai + 1]]
            for i in range(n_cross):
                t = i / (n_cross - 1)
                plan.append((files[n - n_cross + i], next_files[i], t))
    return plan


def _render_op(op):
    if isinstance(op, tuple):
        path_a, path_b, t = op
        img_a, img_b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
        if img_a is None or img_b is None:
            raise IOError(f"failed to read crossfade source frames {path_a}, {path_b}")
        return draw.fade(img_a, img_b, t)
    img = cv2.imread(str(op))
    if img is None:
        raise IOError(f"failed to read frame {op}")
    return img


def assemble_mp4(clip: str = clip_io.CLIP_DEFAULT, fps: int = FPS) -> Path:
    dirs = clip_io.out_dirs(clip)
    frame_root = dirs["frames"]
    order = [name for name, _ in ACT_SPECS]
    frame_lists = {}
    for name, expected in ACT_SPECS:
        frame_lists[name] = _list_frames(frame_root / name, name, expected)

    plan = build_edit_plan(frame_lists, order, N_CROSS)
    if len(plan) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"assembled edit plan has {len(plan)} frames, expected exactly "
            f"{EXPECTED_TOTAL} -- edit-plan arithmetic bug, refusing to encode."
        )
    print(f"[assemble] edit plan: {len(plan)} frames "
          f"({[f'{n}={len(frame_lists[n])}' for n, _ in ACT_SPECS]}, "
          f"{N_CROSS}-frame crossfades x{len(ACT_SPECS) - 1})")

    out_path = dirs["root"] / "ik_explainer.mp4"
    frames_iter = (_render_op(op) for op in plan)
    # macro_block_size=1: the default (16) would pad 1080 -> 1088 (1080 is not
    # a multiple of 16), silently violating the "exactly 1920x1080" contract
    # this script's own ffprobe check enforces below.
    write_video(str(out_path), frames_iter, fps=fps, macro_block_size=1)
    print(f"[assemble] wrote {out_path}")
    return out_path


# --- provenance ----------------------------------------------------------

def _git_commit() -> tuple:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=_REPO,
                                 capture_output=True, text=True, check=True).stdout.strip())
    return commit, dirty


def _detector_facts() -> dict:
    cfg = clip_io._yaml(_REPO / "configs" / "detector" / "vitpose_v3.yaml")
    ckpt_template = cfg["ckpt"]                      # "${paths.vit_runs_root}/v4_8gpu_20260808/final"
    ckpt_suffix = ckpt_template.split("}", 1)[1].lstrip("/")
    # The paths.* hydra group interpolates to a per-machine base_dir that
    # this workstation's checked-in configs/paths/*.yaml profiles do not
    # capture (none resolve to /data2/.../3d_tracking); the concrete path is
    # therefore recorded directly and cross-checked two ways instead of
    # resolved through hydra: it must end with the config's own ckpt suffix,
    # and it must exist on disk.
    ckpt_real = ("/data2/users/eabe/datasets/3d_tracking/jax_vitpose_runs/"
                 "v4_8gpu_20260808/final")
    if not ckpt_real.endswith(ckpt_suffix):
        raise RuntimeError(
            f"detector ckpt path {ckpt_real!r} does not end with the "
            f"configured suffix {ckpt_suffix!r} (configs/detector/vitpose_v3.yaml "
            f"ckpt template {ckpt_template!r}) -- config and recorded path have "
            f"drifted apart.")
    if not Path(ckpt_real).exists():
        raise RuntimeError(f"detector ckpt does not exist on disk: {ckpt_real}")
    return {
        "ckpt": ckpt_real,
        "ckpt_config_template": ckpt_template,
        "config": "configs/detector/vitpose_v3.yaml",
        "num_keypoints": cfg["num_keypoints"],
        "crop": cfg["crop"],
        "conf_thresh": cfg["conf_thresh"],
        "view_conf_thresh": cfg["view_conf_thresh"],
        "decode_sharpen": cfg["decode_sharpen"],
    }


def _anatomy_facts() -> dict:
    cfg = clip_io._yaml(_REPO / "configs" / "anatomy" / "v1.yaml")
    model = cfg["model"]
    trunk_kps = model.get("TRUNK_OPTIMIZATION_KEYPOINTS")
    return {
        "config": "configs/anatomy/v1.yaml",
        "mjcf": "models/fruitfly_v1/fruitfly_v1_free.xml",
        "root_optimization_keypoint": model.get("ROOT_OPTIMIZATION_KEYPOINT"),
        "trunk_optimization_keypoints": trunk_kps,
        "note": (
            "TRUNK_OPTIMIZATION_KEYPOINTS is empty -> root_optimization has no "
            "keypoint loss constraining rotation and performs TRANSLATION ONLY "
            "on this anatomy (measured 0.0000 deg change). See README correction #1."
        ),
    }


def _sam3_facts() -> dict:
    masks_src = (_REPO / "scripts" / "viz" / "ik_explainer" / "masks.py").read_text()
    m_prompt = re.search(r'"text_prompt":\s*"([^"]+)"', masks_src)
    m_version = re.search(r'"sam3_version":\s*"([^"]+)"', masks_src)
    if not m_prompt or not m_version:
        raise RuntimeError(
            "could not extract text_prompt/sam3_version from masks.py source "
            "-- has the sam3={...} call shape changed?")
    sam3_version = m_version.group(1)
    return {
        "hf_repo": f"facebook/{sam3_version}",
        "sam3_version": sam3_version,
        "text_prompt": m_prompt.group(1),
        "python_env": "/home/eabe/miniconda3/envs/sam3/bin/python",
        "note": ("run under the `sam3` conda env, not `3d_tracking`: 3d_tracking "
                  "ships a cu130 torch that cannot see this machine's CUDA-12.8 "
                  "driver; JAX (every other stage) is unaffected."),
    }


def _stage_facts(clip: str) -> dict:
    dirs = clip_io.out_dirs(clip)
    with np.load(dirs["predictions"] / "06_stages.npz", allow_pickle=True) as z:
        stage_names = [str(s) for s in z["stage_names"]]
        residual_mm = np.asarray(z["residual_mm"], np.float64)
        shared_scale = float(z["shared_scale"])
        frame_for_stills = int(z["frame_for_stills"])
    if len(stage_names) != len(residual_mm):
        raise RuntimeError("06_stages.npz: stage_names/residual_mm length mismatch")
    residuals = {name: float(r) for name, r in zip(stage_names, residual_mm)}
    if list(residuals.values()) != sorted(residuals.values(), reverse=True):
        raise RuntimeError(
            f"stage residuals are not monotonically decreasing: {residuals} "
            f"-- IK stages should only ever tighten the fit.")
    with np.load(dirs["predictions"] / "02_kp2d.npz", allow_pickle=True) as z:
        n_frames = int(z["kp2d"].shape[0])
        cam_names = [str(c) for c in z["cam_names"]]
    return {
        "stage_residuals_mm": residuals,
        "shared_scale": shared_scale,
        "frame_for_stills": frame_for_stills,
        "n_frames": n_frames,
        "cameras": cam_names,
    }


def ffprobe_video_info(path) -> dict:
    """ffprobe's `-of csv` does NOT honour the field order given to
    `-show_entries`; it emits the stream's own internal field order instead
    (verified directly: requesting width,height,nb_frames,r_frame_rate,codec_name
    still comes back as codec_name,width,height,r_frame_rate,nb_frames). JSON
    output keys by name instead of position, so it is parsed here rather than
    relying on any positional convention."""
    fields = "width,height,nb_frames,r_frame_rate,codec_name"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", f"stream={fields}", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "n_frames": int(stream["nb_frames"]),
            "r_frame_rate": stream["r_frame_rate"], "codec_name": stream["codec_name"]}


def build_manifest(clip: str, mp4_path: Path) -> dict:
    commit, dirty = _git_commit()
    video = ffprobe_video_info(mp4_path)
    expected_video = {"width": CANVAS_W, "height": CANVAS_H,
                      "n_frames": EXPECTED_TOTAL, "r_frame_rate": f"{FPS}/1",
                      "codec_name": "h264"}
    if video != expected_video:
        raise RuntimeError(
            f"assembled mp4 does not match the expected stream properties: "
            f"got {video}, expected {expected_video}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip": str(clip),
        "repo_commit": commit,
        "repo_dirty": dirty,
        "detector": _detector_facts(),
        "anatomy": _anatomy_facts(),
        "sam3": _sam3_facts(),
        "stages": _stage_facts(clip),
        "video": {
            **video,
            "path": str(mp4_path),
            "acts": {name: n for name, n in ACT_SPECS},
            "crossfade_frames": N_CROSS,
            "expected_total_frames": EXPECTED_TOTAL,
        },
    }


# --- README ----------------------------------------------------------------

README_TEMPLATE = """\
# IK explainer -- generated artifacts for {clip}

This directory holds every intermediate and final artifact for the "IK
explainer" talk video: SAM3 masks -> 4ch ViTPose 2D -> DLT triangulation ->
temporal filtering -> staged STAC IK -> four rendered acts -> `ik_explainer.mp4`.
Raw clip inputs one level up (`calibration/`, `Cam*.mp4`, `enhanced/`,
`preview/`, `data3D_*.csv`) are read-only inputs and are never modified by
anything under this directory. See `docs/specs/2026-08-13-ik-explainer-animation-design.md`
for the full design; `manifest.json` beside this file records the provenance
(commit, checkpoint, configs, residuals) that produced the artifacts below.

## Layout

```
ik_explainer/
  README.md              this file
  manifest.json          provenance: repo commit, ckpt/config paths, residuals
  bouts.csv              single-bout summary this clip's nonstandard format needs
  predictions/           the re-run pipeline, in stage order (numeric prefixes
                         make the dependency order readable at a glance)
    01_sam3_masks.npz    SAM3 masks + centroids (also under predictions/sam3/)
    02_kp2d.npz          (7, {n_frames}, 50, 2) 2D keypoints + conf, MODEL order
    03_kp3d.npz          (N, 50, 3) mm triangulated 3D + conf3d
    04_kp3d_filt.npz     post temporal-filter 3D (fed to STAC)
    05_stac_ik.h5        STAC output (per-frame qpos)
    06_stages.npz        per-stage qpos snapshots (default/scaled/root/pose)
                         + residual_mm + shared_scale -- Acts 3-4's data source
  qc/                    verification figures + numbers (the CLAUDE.md gate)
    01_masks.png  02_kp2d_overlay.png  03_kp3d_vs_shipped.png
    04_filter_effect.png  05_ik_stages.png  qc.json
  frames/                per-act PNG sequences (f%05d.png, 1920x1080),
                         independently re-renderable without repaying a GPU pass
    act1_views/          {n1} frames -- on-screen title "2D Keypoint tracking":
                         seven camera views, 2D keypoints fade in; closing
                         60-frame beat crossfades the raw detector 2D into
                         the reprojection of the triangulated 3D across all
                         seven panels (resolves the near-edge-on cameras'
                         leg-assignment jitter)
    act2_triangulate/    {n2} frames -- on-screen title "3D triangulation":
                         views converge into the 3D keypoint cloud
    act3_align/          {n3} frames -- on-screen title "Root alignment":
                         already-scaled keypoint skeleton starts modestly
                         offset from the (static) model and swings onto it,
                         settling at the solver's true fitted
                         root_optimization position
    act4_solve/          {n4} frames -- on-screen title "Solve joint angles":
                         0-239 the joints solve (+ orientation), 3D only;
                         240-899 side-by-side for the REST of the act (task-19:
                         previously only the closing 120 frames) -- left is
                         the MuJoCo IK render, right is the real camera frame
                         with the detector's own 2D keypoints, coloured to
                         match the left panel's per-limb-chain colours
  ik_explainer.mp4       the deliverable: {total} frames, 1920x1080 @ 30 fps,
                         H.264, silent (assembled by scripts/viz/ik_explainer/assemble.py)
```

## Two corrections to the original plan (read this before the figures)

The original design document described Act 3 as the mesh "translating and
rotating" onto the keypoint cloud. Both of the following were measured after
Task 7's real solve ran and are recorded here because a future reader
re-deriving the story from the original plan wording alone would get it wrong:

1. **`root_optimization` performs TRANSLATION ONLY on this anatomy.**
   `configs/anatomy/v1.yaml` sets `TRUNK_OPTIMIZATION_KEYPOINTS = {{}}` (empty),
   so nothing constrains rotation during that stage. Measured directly on
   `06_stages.npz`: the root quaternion is **unchanged (0.0000 deg)** and joint
   DOF are unchanged (0.000000) between `qpos_scaled` and `qpos_root`. All
   **34.73 deg** of rotation happens in the *next* stage, `pose_optimization`,
   simultaneously with every joint angle -- Act 4 carries the orienting beat,
   not Act 3. (Root heading does vary 0-33 deg across the 921-frame sequence,
   so this is a genuine per-frame solve, not a lucky rest pose.)
2. **Body scale is applied to the KEYPOINTS, not by morphing the mesh.** STAC's
   `compute_shared_scale` scales the triangulated 3D keypoints by
   `shared_scale = {shared_scale:.4f}` (Umeyama-style) before fitting; the
   MuJoCo mesh geometry itself never changes size. Act 3's "mesh grows to meet
   the cloud" framing is a camera-staging choice (a live wide camera would
   otherwise hide the cloud's real contraction), not evidence the mesh scaled.

## Verified facts (measured, not assumed)

- Clip: 7 cameras, N = {n_frames} frames, 800 fps -> 1.15 s of fly time.
- Cameras: {cameras}
- Per-stage marker residual (mm), monotonically decreasing:
  {residual_line}
- `shared_scale` = {shared_scale:.6f}
- Detector checkpoint: `{ckpt}`
- Anatomy: `configs/anatomy/v1.yaml` -> `models/fruitfly_v1/fruitfly_v1_free.xml`
- SAM3: `{hf_repo}`, text prompt `"{text_prompt}"`, run under `{sam3_env}`
- Keypoint colours: JARVIS per-limb-chain scheme -- `scripts/viz/ik_explainer/
  kp_colors.py` calls `third_party/JARVIS-HybridNet`'s own `get_skeleton`
  against `data/fly50.json`'s skeleton graph, so each leg (and the head/thorax
  loop, and each wing) gets its own colour instead of one flat per-group
  colour. Task-19: Act 4's side-by-side (240-899, the whole rest of the act)
  colour-matches its two panels -- the right panel draws the detector's own
  2D keypoints in this SAME per-limb-chain scheme, not the old FK-reprojected
  fit -- so `viz.core.colors.PALETTE`'s fit=green/observed=cyan convention no
  longer applies there.
- This is the EASY case (one fly, open arena). CLAUDE.md is explicit that the
  female fly on walls / in occlusion is where this pipeline actually fails;
  do not cite this video as general QC evidence.

## Display source: `enhanced/` (presentation only)

Every place the rendered video shows real camera footage (Act 1's seven-panel
grid, Act 4's side-by-side right panel, and the `act1_data3d*_test.py`
diagnostics) reads from `<clip>/enhanced/Cam*.mp4` -- a brightness/contrast
-lifted copy of the same recording -- instead of the raw `<clip>/Cam*.mp4`,
via `clip_io.video_path`'s `enhanced=True` default (with a documented
fallback to the raw file, and a load-time assertion that the enhanced copy's
frame count/dimensions match before it's ever read). Verified before
switching: all 7 files, same filenames, dimensions 1936x448, rate 800/1, and
921 frames as the raw copies; normalised cross-correlation against the raw
frames is 0.9984-0.9986 at frames 0/400/900 with lag 0 (frame-aligned, no
offset); grey mean 22.1 -> 40.5, std 32.3 -> 47.1.

**This is display-only.** SAM3 masks and the ViTPose detector were run
against the RAW videos (`detect2d.py`'s detector-input load and
`clip_io.stage_session_dir`'s SAM3 symlinks both pin `enhanced=False`
explicitly, so a future re-run of those stages keeps reading the exact
frames the shipped predictions were derived from) -- none of the 2D
keypoints, 3D triangulation, or STAC/IK in `predictions/` is re-derived by
this swap; only the video pixels drawn underneath the overlays changed.

## Regenerating from scratch

All commands run from the repo root
(`{repo}`) against `--clip {clip}`
(the default for every script below, so `--clip` may be omitted). Stages 1-4
and the IK solve are **JAX** and run in the `3d_tracking` conda env; the SAM3
mask stage is **PyTorch** and must run in the separate `sam3` env (see the
correction above -- `3d_tracking`'s torch cannot see this workstation's CUDA
driver). Renders (Acts 1-4, `stage_ik.py --qc`) need `MUJOCO_GL=egl` and a
GPU; run directly on a GPU node, never via sbatch and never on a login node.

```bash
cd {repo}

# 0. bout-summary scaffold for this clip's nonstandard (no-bout-CSV) format
python scripts/viz/ik_explainer/prepare_clip.py --clip {clip}

# 1. SAM3 masks -- PYTORCH env (not 3d_tracking); ~22 min for the full clip
/home/eabe/miniconda3/envs/sam3/bin/python scripts/viz/ik_explainer/masks.py \\
    --clip {clip}
python scripts/viz/ik_explainer/masks.py --clip {clip} --qc   # qc/01_masks.png

# 2. 2D keypoints -- 4-channel ViTPose, written in MODEL order
python scripts/viz/ik_explainer/detect2d.py --clip {clip}
python scripts/viz/ik_explainer/detect2d.py --clip {clip} --qc   # qc/02_kp2d_overlay.png

# 3. Triangulate to 3D (mm) + temporal filter; diffs vs the shipped CSV baseline
python scripts/viz/ik_explainer/triangulate3d.py --clip {clip}
python scripts/viz/ik_explainer/triangulate3d.py --clip {clip} --qc   # qc/03_*, 04_*

# 4. Staged STAC IK -- snapshots qpos after each stage (default/scaled/root/pose)
python scripts/viz/ik_explainer/stage_ik.py --clip {clip} --frame {frame_for_stills}
python scripts/viz/ik_explainer/stage_ik.py --clip {clip} --qc   # qc/05_ik_stages.png

# 5. Render the four acts (MUJOCO_GL=egl needed for 3-4; GPU node; not sbatch)
python scripts/viz/ik_explainer/acts/act1_views.py --clip {clip}
python scripts/viz/ik_explainer/acts/act2_triangulate.py --clip {clip}
MUJOCO_GL=egl python scripts/viz/ik_explainer/acts/act3_align.py --clip {clip}
MUJOCO_GL=egl python scripts/viz/ik_explainer/acts/act4_solve.py --clip {clip}
#   (act4_solve.py --start-frame N resumes a render that already wrote
#    frames 0..N-1, e.g. after an interrupted GPU run -- it died mid-render
#    at 700/900 once and had to be resumed this way.)

# 6. Assemble: crossfade the four act sequences into ik_explainer.mp4,
#    then write this README.md and manifest.json
python scripts/viz/ik_explainer/assemble.py --clip {clip}
```

## Tests

Pure-function tests (no GPU, no clip data) live at the repo root:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_ik_explainer_*.py -q
```
"""


def write_readme(clip: str, mp4_path: Path, manifest: dict) -> Path:
    dirs = clip_io.out_dirs(clip)
    frame_lists_counts = {name: n for name, n in ACT_SPECS}
    stages = manifest["stages"]
    sam3 = manifest["sam3"]
    residual_line = "  ".join(
        f"{name} {r:.3f}" for name, r in stages["stage_residuals_mm"].items())
    text = README_TEMPLATE.format(
        clip=clip,
        repo=str(_REPO),
        n_frames=stages["n_frames"],
        n1=frame_lists_counts["act1_views"],
        n2=frame_lists_counts["act2_triangulate"],
        n3=frame_lists_counts["act3_align"],
        n4=frame_lists_counts["act4_solve"],
        total=EXPECTED_TOTAL,
        shared_scale=stages["shared_scale"],
        cameras=", ".join(stages["cameras"]),
        residual_line=residual_line,
        ckpt=manifest["detector"]["ckpt"],
        hf_repo=sam3["hf_repo"],
        text_prompt=sam3["text_prompt"],
        sam3_env=sam3["python_env"],
        frame_for_stills=stages["frame_for_stills"],
    )
    out = dirs["root"] / "README.md"
    out.write_text(text)
    print(f"[assemble] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    mp4_path = assemble_mp4(args.clip, fps=args.fps)
    manifest = build_manifest(args.clip, mp4_path)
    dirs = clip_io.out_dirs(args.clip)
    manifest_path = dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[assemble] wrote {manifest_path}")
    write_readme(args.clip, mp4_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
