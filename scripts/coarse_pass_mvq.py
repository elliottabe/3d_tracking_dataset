#!/usr/bin/env python3
"""Mask-free coarse pass over a whole recording: CenterDetect + mvq at stride 16.

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §4.3. The
drop-in replacement for `scripts/coarse_pass.py` (SAM3 + JARVIS project) that
needs no masks and no JARVIS project: it reads the mp4s directly, localises
the flies with CenterDetect, lifts each window with the mvq checkpoint and
writes `coarse_tracks.npz` in the SAME schema (plus the mvq fields), so
`scripts/coarse_pass_gates.py` opens either file. Everything numeric lives in
`jarvis_jax.tracking.coarse_track`; this file is argparse, frame IO,
chunking/resume and progress.

Run (GPU node; the render/CUDA preamble is the same as the other jax jobs):

    module load cuda/12.9.1
    export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
    unset LD_LIBRARY_PATH JAX_PLATFORMS
    PYTHONPATH=third_party/jarvis_jax:. python scripts/coarse_pass_mvq.py \
      --session-dir .../Session0/2025_10_20_13_20_04 \
      --calib-dir   .../Session0/2025_10_20_13_20_04/calibration \
      --run  .../jax_mvq_runs/<run>/final \
      --centerdetect .../jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004 \
      --stride 16 --out .../coarse_mvq/coarse_tracks.npz --resume

CAMERA ORDER (CLAUDE.md's camera-order trap). `--cameras` defaults to
`configs/recording/session0.yaml`'s `recording.cameras`, which is the
CANONICAL order and, by construction, the calibration glob order.
`MVQRunner` refuses a calibration directory whose glob order differs, and
every per-camera array here -- the frames from `read_window`, CenterDetect's
peaks, `cam_mats`, the written `centroid` -- is in that one order. Passing a
`--cameras` list in any other order would plot one camera's fly on another
camera's image, which still looks almost plausible.

FRAME IO. `jarvis_jax.predict.synced_reader` owns the canonical-slot ->
mp4-position mapping (`load_plan` + `slot_positions`); Session0 has no
`sync_plan.json`, so the mapping is positional. This script uses those
helpers with the `cv2.VideoCapture`s KEPT OPEN across frames rather than
calling `read_window(..., T=1)` per sampled frame: at stride 16 that would
open and close 7 captures ~31k times per recording, which costs more than
the model does. The frames read are byte-identical either way.

RESUME. The pass runs in chunks of `--partial-every` coarse frames; after
each chunk the accumulated tracks are written to
`<out>.partial.npz` (a COMPLETE, gates-readable file, just shorter). With
`--resume`, an existing partial (or a complete `<out>`) is loaded and the
pass continues at the frame after its last one. The floor plane and all the
features are recomputed over the whole concatenated track at the end, so a
resumed run and a single-shot run produce the same file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "third_party", "jarvis_jax"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CAMERAS_YAML = os.path.join(_REPO, "configs", "recording", "session0.yaml")


def default_cameras(path=DEFAULT_CAMERAS_YAML):
    """`recording.cameras` from a recording config, WITHOUT hydra/omegaconf
    resolution (the rest of the file is full of `${...}` interpolations this
    script has no use for -- and no composed config to resolve them against).
    """
    with open(path) as f:
        for line in f:
            if line.strip().startswith("cameras:"):
                body = line.split(":", 1)[1].strip()
                return [c.strip() for c in body.strip("[]").split(",") if c.strip()]
    raise ValueError(f"no `cameras:` list in {path}")


def video_size(session_dir, camera):
    import cv2
    cap = cv2.VideoCapture(os.path.join(str(session_dir), f"{camera}.mp4"))
    if not cap.isOpened():
        raise FileNotFoundError(os.path.join(str(session_dir), f"{camera}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


class SlotReader:
    """`coarse_pass`'s reader over one recording, with the captures kept open.

    `__call__(slot)` returns `(frames (C,H,W,3) uint8 RGB, present (C,) bool)`
    for one CANONICAL slot, in `cameras` order -- the same bytes
    `synced_reader.read_window(session_dir, cameras, plan, slot, 1)` yields
    (it uses the same `slot_positions` mapping and the same `_read_at` seek),
    without reopening the captures on every sampled frame. A camera that
    dropped the slot, or whose read fails, yields a black frame and
    `present=False`, which the model gates out through `cam_valid`.
    """

    def __init__(self, session_dir, cameras, plan):
        import cv2
        from jarvis_jax.predict.synced_reader import slot_positions
        self._cv2 = cv2
        self._slot_positions = slot_positions
        self.session_dir, self.cameras, self.plan = str(session_dir), list(cameras), plan
        self.caps, self.cursors = [], []
        for c in self.cameras:
            cap = cv2.VideoCapture(os.path.join(self.session_dir, f"{c}.mp4"))
            if not cap.isOpened():
                raise FileNotFoundError(os.path.join(self.session_dir, f"{c}.mp4"))
            self.caps.append(cap)
            self.cursors.append(None)
        self.H = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    def __call__(self, slot):
        from jarvis_jax.predict.synced_reader import _read_at
        out = np.zeros((len(self.cameras), self.H, self.W, 3), np.uint8)
        present = np.zeros(len(self.cameras), bool)
        for ci, cam in enumerate(self.cameras):
            pos, pres = self._slot_positions(self.plan, cam, int(slot), 1)
            if not pres[0]:
                continue
            if self.cursors[ci] is None:
                self.caps[ci].set(self._cv2.CAP_PROP_POS_FRAMES, pos[0])
                self.cursors[ci] = pos[0]
            fr, self.cursors[ci] = _read_at(self.caps[ci], self.cursors[ci], pos[0])
            if fr is None:
                continue
            out[ci] = self._cv2.cvtColor(fr, self._cv2.COLOR_BGR2RGB)
            present[ci] = True
        return out, present

    def close(self):
        for cap in self.caps:
            cap.release()


def load_partial(path, num_animals):
    """Read a previously written coarse_tracks(.partial).npz back into a
    `coarse_pass`-shaped tracks dict so the run can continue from it.

    `kp3d` comes back as float16 (that is what the file stores); it is widened
    to float32 here so a resumed run's array dtypes match a fresh one's.
    """
    with np.load(path, allow_pickle=True) as z:
        tr = {"frame": np.asarray(z["coarse_frame"], np.int64),
              "kp3d": np.asarray(z["kp3d"], np.float32),
              "centroid": np.asarray(z["X3d"], np.float32),
              "exist": np.asarray(z["exist"], np.float32),
              "sex_prob": np.asarray(z["sex_prob"], np.float32),
              "slot": np.asarray(z["slot"], np.int8),
              "centre_source": np.asarray(z["centre_source"], np.int8),
              "n_windows": np.asarray(z["n_windows"], np.int8),
              "kp_names": [str(n) for n in z["kp_names"]]}
    meta_path = path.rsplit(".npz", 1)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    tr["W"], tr["H"] = meta.get("W"), meta.get("H")
    tr["last_centres"] = None            # the reuse chain does not survive a restart
    tr["floor"] = None
    if tr["exist"].shape[0] != num_animals:
        raise ValueError(f"{path} has {tr['exist'].shape[0]} flies, --num-animals is "
                         f"{num_animals}; refusing to append rows of a different shape")
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--calib-dir", default=None,
                    help="default: <session-dir>/calibration")
    ap.add_argument("--cameras", default=None,
                    help=f"comma list; default: recording.cameras of {DEFAULT_CAMERAS_YAML}")
    ap.add_argument("--run", required=True,
                    help="mvq `final/` dir, or the run dir when --step is given")
    ap.add_argument("--step", default=None, help="checkpoint step (int); omit for a final/ dir")
    ap.add_argument("--centerdetect", required=True, help="CenterDetect ckpt dir (epoch_XXX)")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None,
                    help="exclusive; default = the video's frame count")
    ap.add_argument("--out", required=True, help="output coarse_tracks.npz path")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--min-score", type=float, default=0.2, help="CenterDetect peak threshold")
    ap.add_argument("--min-views", type=int, default=3)
    ap.add_argument("--max-resid-px", type=float, default=25.0)
    ap.add_argument("--merge-dist-units", type=float, default=30.0)
    ap.add_argument("--partial-every", type=int, default=2000,
                    help="coarse frames per chunk / partial write")
    ap.add_argument("--progress-every", type=int, default=500, help="coarse frames per log line")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing partial (or complete) output")
    ap.add_argument("--attn-impl", default=None, help='"xla" to run a cuDNN-trained run on CPU')
    args = ap.parse_args()

    from jarvis_jax.predict.synced_reader import load_plan
    from jarvis_jax.tracking.coarse_centres import CenterDetector
    from jarvis_jax.tracking.coarse_track import coarse_pass
    from jarvis_jax.tracking.lift_mvq import MVQRunner

    cameras = ([c.strip() for c in args.cameras.split(",") if c.strip()]
               if args.cameras else default_cameras())
    calib_dir = args.calib_dir or os.path.join(args.session_dir, "calibration")
    n_frames, W, H = video_size(args.session_dir, cameras[0])
    end = int(args.end) if args.end is not None else n_frames
    all_frames = list(range(int(args.start), int(end), int(args.stride)))

    out = args.out
    partial = out.rsplit(".npz", 1)[0] + ".partial.npz"
    done = None
    if args.resume:
        src = partial if os.path.isfile(partial) else (out if os.path.isfile(out) else None)
        if src:
            done = load_partial(src, args.num_animals)
            last = int(done["frame"][-1])
            all_frames = [f for f in all_frames if f > last]
            print(f"[resume] {src}: {done['frame'].shape[0]} coarse frames "
                  f"(last {last}); {len(all_frames)} to go", flush=True)

    print(f"[coarse] {args.session_dir}\n[coarse] cameras {cameras}\n"
          f"[coarse] frames {args.start}..{end} stride {args.stride} -> "
          f"{len(all_frames)} coarse frames ({W}x{H})", flush=True)
    if not all_frames and done is None:
        raise SystemExit("no frames to process")

    t0 = time.time()
    runner = MVQRunner(args.run, step=(int(args.step) if args.step is not None else None),
                       attn_impl=args.attn_impl, calib_dir=calib_dir, cameras=cameras,
                       batch=args.batch)
    detector = CenterDetector(args.centerdetect, min_score=args.min_score)
    t_load = time.time() - t0
    print(f"[coarse] models loaded in {t_load:.1f}s (mvq step {runner.step_label}, "
          f"K={runner.K}, I={runner.I})", flush=True)

    reader = SlotReader(args.session_dir, cameras, load_plan(args.session_dir))
    chunks = [done] if done is not None else []
    n_done = 0 if done is None else int(done["frame"].shape[0])
    t_pass = time.time()
    state = {"next_log": args.progress_every}

    def progress(i, n, _fps):
        if i >= state["next_log"]:
            state["next_log"] += args.progress_every
            k = n_done + i
            el = time.time() - t_pass
            rate = (k - (0 if done is None else int(done["frame"].shape[0]))) / max(el, 1e-9)
            todo = len(all_frames) - i
            print(f"[coarse] {k}/{n_done + len(all_frames)} coarse frames  "
                  f"{rate:.2f} frames/s  eta {todo / max(rate, 1e-9) / 60:.1f} min", flush=True)

    try:
        last_centres = None
        for c0 in range(0, len(all_frames), args.partial_every):
            block = all_frames[c0:c0 + args.partial_every]
            tr = coarse_pass(reader, runner, detector, frames=block,
                             num_animals=args.num_animals,
                             merge_dist_units=args.merge_dist_units,
                             min_views=args.min_views, max_resid_px=args.max_resid_px,
                             progress=lambda i, n, f, off=c0: progress(off + i, n, f),
                             init_centres=last_centres)
            last_centres = tr["last_centres"]
            chunks.append(tr)
            _write(partial, chunks, cameras, runner, args, W, H, t_load, t_pass, final=False)
            print(f"[coarse] partial written: {partial} "
                  f"({sum(int(c['frame'].shape[0]) for c in chunks)} coarse frames)", flush=True)
    finally:
        reader.close()

    meta = _write(out, chunks, cameras, runner, args, W, H, t_load, t_pass, final=True)
    if os.path.isfile(partial):
        os.remove(partial)
        pm = partial.rsplit(".npz", 1)[0] + ".meta.json"
        if os.path.isfile(pm):
            os.remove(pm)
    el = time.time() - t_pass
    print(f"[coarse] done: {meta['n_coarse']} coarse frames in {el / 60:.1f} min "
          f"({meta['n_coarse'] / max(el, 1e-9):.2f} frames/s)\n"
          f"[coarse] centre_source {meta['centre_source_counts']}  "
          f"frac_trackable {meta['frac_trackable']}\n[coarse] -> {out}", flush=True)


def _write(path, chunks, cameras, runner, args, W, H, t_load, t_pass, *, final):
    from jarvis_jax.tracking.coarse_track import (coarse_features, concat_tracks, fit_floor,
                                                  write_coarse_tracks)
    tr = concat_tracks(chunks)
    tr["W"], tr["H"] = tr.get("W") or W, tr.get("H") or H
    try:
        floor = fit_floor(tr["centroid"])
    except ValueError as e:                 # no finite centroid yet (early partial)
        print(f"[coarse] floor not fit ({e}); heights are NaN in this write", flush=True)
        floor = None
    tr["floor"] = floor
    if floor is None:
        from jarvis_jax.tracking.coarse_track import FloorPlane
        feats = coarse_features(tr, tr["kp_names"],
                                floor=FloorPlane(np.array([np.nan] * 3), float("nan")))
    else:
        feats = coarse_features(tr, tr["kp_names"], floor=floor)
    extra = {"checkpoint": runner.checkpoint, "mvq_step": runner.step_label,
             "exist_thresh": runner.exist_thresh, "centerdetect": args.centerdetect,
             "min_score": args.min_score, "min_views": args.min_views,
             "max_resid_px": args.max_resid_px, "merge_dist_units": args.merge_dist_units,
             "batch": args.batch, "start": args.start, "end": args.end,
             "complete": bool(final),
             "timing_s": {"model_load": round(t_load, 1),
                          "pass": round(time.time() - t_pass, 1)}}
    return write_coarse_tracks(path, tr, feats, cameras, session_dir=args.session_dir,
                               stride=args.stride, num_animals=args.num_animals,
                               cam_mats=runner.cam_mats, meta_extra=extra)


if __name__ == "__main__":
    main()
