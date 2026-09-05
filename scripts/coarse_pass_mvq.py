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
`sync_plan.json`, so the mapping is positional. `SlotReader` uses those
helpers, but NOT via `read_window(..., T=1)` per sampled frame and NOT via a
kept-open capture whose cursor is re-seeked every call (that was measured at
0.39 coarse frames/s / ~22h ETA on a real 498k-frame recording -- CPU-bound
on repeated keyframe-seek + GOP redecode, ~470% CPU / ~0% GPU;
`.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/real-run-wave-report.md`
Step 2). Instead each camera gets its own thread (`_CamStream`) that decodes
the mp4 FORWARD ONLY -- `grab()` (decode, discard) through the frames the
stride skips, `retrieve()` only at the stride hit -- so after the one
initial seek (to the run's start/resume slot) every mp4 frame is decoded at
most once, ever, and decode for the next slot overlaps whatever the main
thread is doing with the current one. The frames read are byte-identical to
`read_window`'s, for the SAME strictly-increasing slot sequence
`SlotReader` was started with (see its docstring); it is not a general
random-access reader.

RESUME. The pass runs in chunks of `--partial-every` coarse frames; after
each chunk the accumulated tracks are written to
`<out>.partial.npz` (a COMPLETE, gates-readable file, just shorter). With
`--resume`, an existing partial (or a complete `<out>`) is loaded and the
pass continues at the frame after its last one, restoring the last
CENTRE_DETECTED/REUSED window centres (`last_centres`) so a blank frame right
after the resume boundary still REUSES them (`centre_source == 1`) instead of
going NaN as if there were no history. The floor plane and all the features
are recomputed over the whole concatenated track at the end, so a resumed run
and a single-shot run have IDENTICAL FRAME COVERAGE AND CENTRE REUSE -- not a
byte-identical file: `load_partial` reads the earlier chunks' `kp3d` back from
the npz's `float16` storage, so `wing_angle_deg` (and anything else derived
from `kp3d`) on a resumed chunk is recomputed from that f16-quantised value
rather than the original f32 model output. This is an inherent, harmless
(~0.05 mm) rounding difference from a true single-shot run, not a resume bug.
"""
import argparse
import json
import os
import queue
import sys
import threading
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


def parse_up_hint(s):
    """`--up-hint`'s `'x,y,z'` string -> a `(3,)` float array, or `None` if
    `s` is `None`/empty. Passed straight through to `fit_floor`'s `up_hint`
    (world units; only the DIRECTION matters, not the magnitude)."""
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--up-hint must be 'x,y,z' (3 comma-separated numbers), got {s!r}")
    return np.array([float(p) for p in parts], np.float64)


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


class _CamStream:
    """One camera's forward-only decode, owned by its own thread.

    `coarse_pass` calls the reader with a STRICTLY INCREASING sequence of
    canonical slots, `start_slot, start_slot+stride, start_slot+2*stride, ...`
    (see `jarvis_jax.tracking.coarse_track.coarse_pass`'s frame loop and
    `main`'s `all_frames = range(start, end, stride)`). The old reader (kept
    captures open, but still called `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES,
    ...)` on every sampled frame because its cursor only ever advanced by 1
    per read while the request jumped by `stride`) turned every coarse frame
    into a keyframe seek + a decode of the whole GOP back up to the target --
    measured at 0.39 coarse frames/s / ~22h ETA on a real 498k-frame
    recording (`.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/
    real-run-wave-report.md` Step 2), with ps showing ~470% CPU and ~0% GPU,
    i.e. CPU-bound in decode, not GPU-bound in the mvq forward (36ms/window,
    `scripts/benchmark/mvq_window_cost.py`).

    This class instead walks the mp4 FORWARD ONLY: `cap.grab()` (decode,
    discard) through every frame the stride skips, `cap.retrieve()` only at
    the stride hit that is actually wanted -- so each mp4 frame is decoded
    at most once for the whole pass, and after the ONE initial seek (to
    `start_slot`, which on `--resume` is the resume boundary -- "a resume
    seeks once to the boundary then streams") there is no seek and no
    redundant GOP re-decode ever again. It runs in its own thread (cv2
    releases the GIL around `grab`/`retrieve`/`read`) so up to `cameras`-many
    decodes proceed in parallel and decode for slot N+1 overlaps whatever the
    main thread (CenterDetect peaks, mvq batching/forward) is doing with slot
    N's frames -- feeding a small bounded queue rather than the main thread
    blocking on one camera at a time.

    A camera whose plan drops a slot (only possible with a real
    `sync_plan.json`; Session0 has none, so `plan is None` and every slot is
    positional and present) yields `frame=None, present=False` for that slot
    without touching the decode cursor -- the NEXT present slot's absolute
    target position naturally catches the cursor up across the gap, exactly
    as the old per-call seek did, just via `grab()` instead of `set()`.
    """

    def __init__(self, session_dir, cam, plan, start_slot, stride, queue_size=4):
        from jarvis_jax.predict.synced_reader import slot_positions
        self._slot_positions = slot_positions
        self.session_dir, self.cam, self.plan = str(session_dir), cam, plan
        self.start_slot, self.stride = int(start_slot), int(stride)
        self.q = queue.Queue(maxsize=int(queue_size))
        self.H = self.W = None
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camstream-{cam}")
        self._thread.start()
        self._ready.wait()    # blocks only until H/W (or a startup error) are known
        if self.error is not None:
            raise self.error

    def _run(self):
        import cv2
        cv2.setNumThreads(1)  # this thread decodes one stream; don't fan out internally
        path = os.path.join(self.session_dir, f"{self.cam}.mp4")
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise FileNotFoundError(path)
            self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except Exception as e:
            self.error = e
            self._ready.set()
            cap.release()
            return
        self._ready.set()
        cursor = None
        slot = self.start_slot
        try:
            while not self._stop.is_set():
                pos, pres = self._slot_positions(self.plan, self.cam, slot, 1)
                pos_i, present_i = pos[0], pres[0]
                frame = None
                if present_i:
                    if cursor is None or pos_i < cursor:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_i)  # the ONE seek (start/resume)
                        cursor = pos_i
                    ok = True
                    while cursor < pos_i:            # discard the stride-skipped frames
                        ok = cap.grab()
                        if not ok:
                            break
                        cursor += 1
                    if ok:
                        ok = cap.grab()               # decode the WANTED frame (not yet retrieved)
                    if not ok:
                        cursor = None                 # EOF
                    if cursor is not None:
                        ok, fr = cap.retrieve()
                        if ok:
                            # Checked on EVERY stride hit, not just the first
                            # few: a drift that only appears later in the
                            # decode (e.g. after a dropped/corrupt GOP) would
                            # otherwise run silently past whatever window the
                            # check used to stop at. Cheap (one more
                            # `cap.get` per already-decoded frame) next to the
                            # grab/retrieve it follows.
                            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            if actual != pos_i:
                                raise RuntimeError(
                                    f"{self.cam}: frame-index drift at slot {slot} -- "
                                    f"expected mp4 frame {pos_i}, CAP_PROP_POS_FRAMES "
                                    f"reports {actual} (sequential grab/retrieve landed "
                                    f"on the wrong frame)")
                            frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                            cursor += 1
                        else:
                            cursor = None
                self._put((slot, frame, bool(present_i and frame is not None)))
                slot += self.stride
        except Exception as e:
            self.error = e
            self._put((slot, None, None))  # unblock a waiting `get` with the error
        finally:
            cap.release()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def get(self, slot):
        got_slot, frame, present = self.q.get()
        if self.error is not None:
            raise RuntimeError(f"{self.cam} reader thread failed") from self.error
        assert got_slot == slot, (
            f"{self.cam}: reader produced slot {got_slot}, caller asked for {slot} -- "
            f"the driver must call in the same strictly-increasing stride sequence "
            f"the stream was started with")
        return frame, present

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)


class SlotReader:
    """`coarse_pass`'s reader over one recording: one `_CamStream` thread per
    camera, each decoding sequentially FORWARD ONLY (see `_CamStream`).

    `__call__(slot)` returns `(frames (C,H,W,3) uint8 RGB, present (C,) bool)`
    for one CANONICAL slot, in `cameras` order -- the same bytes
    `synced_reader.read_window(session_dir, cameras, plan, slot, 1)` yields
    for the SAME strictly-increasing slot sequence this was constructed with
    (`start_slot, start_slot+stride, ...`; see `_CamStream`'s docstring for
    why: unlike `read_window`, which can answer any single slot via a seek,
    this reader trades that generality for never re-decoding a GOP).
    """

    def __init__(self, session_dir, cameras, plan, start_slot=0, stride=1, queue_size=4):
        self.session_dir, self.cameras, self.plan = str(session_dir), list(cameras), plan
        self.streams = []
        try:
            for c in self.cameras:
                self.streams.append(_CamStream(self.session_dir, c, plan, start_slot, stride,
                                               queue_size=queue_size))
        except Exception:
            for s in self.streams:            # one camera failed to open -- stop the rest
                s.close()
            raise
        self.H = self.streams[0].H
        self.W = self.streams[0].W

    def __call__(self, slot):
        slot = int(slot)
        out = np.zeros((len(self.cameras), self.H, self.W, 3), np.uint8)
        present = np.zeros(len(self.cameras), bool)
        for ci, stream in enumerate(self.streams):
            frame, pres = stream.get(slot)
            if pres:
                out[ci] = frame
                present[ci] = True
        return out, present

    def close(self):
        for stream in self.streams:
            stream.close()


def load_partial(path, num_animals, *, cameras=None, calib_dir=None, checkpoint=None,
                 step_label=None, stride=None, start=None, fallback_wh=None):
    """Read a previously written coarse_tracks(.partial).npz back into a
    `coarse_pass`-shaped tracks dict so the run can continue from it.

    `kp3d` comes back as float16 (that is what the file stores); it is widened
    to float32 here so a resumed run's array dtypes match a fresh one's.
    `last_centres` is restored (NaN-unpadded) so a blank frame right after the
    resume boundary still REUSES it instead of going NaN as if there were no
    history (see `coarse_track.unpad_last_centres`).

    Beyond the fly-count check, every keyword below (when given, i.e. not
    `None`) must match the value the file was WRITTEN with, or this raises --
    a `--resume` with a different `--cameras`, `--calib-dir`, `--run`/`--step`,
    `--stride` or `--start` would otherwise silently splice two different
    geometries into one file (different camera order, different calibration,
    a different checkpoint's keypoint semantics, a different sample grid).
    `concat_tracks`'s own `kp_names`/frame-size check catches a DIFFERENT
    failure mode (a bad splice that already happened in memory, within one
    process); this one catches the input to a splice across process
    boundaries, before it happens.

    Args:
        cameras: this run's camera list (canonical order).
        calib_dir: this run's calibration directory.
        checkpoint: this run's mvq run dir (`args.run`, any form -- compared
            after `os.path.abspath`).
        step_label: this run's mvq step ("final" or an int step), matching
            `MVQRunner.step_label`/the meta's `mvq_step`.
        stride, start: this run's `--stride`/`--start`.
        fallback_wh: `(W, H)` to use if the sibling `.meta.json` is missing
            or lacks `W`/`H` (MINOR 6: a kill between the npz write and the
            meta write, on an OLDER file written before meta-first ordering,
            could leave exactly that).

    Raises:
        ValueError: the fly count, or any given field above, disagrees with
            what the file was written with -- named in the message so the
            fix is obvious from the error alone.
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
              # "collapsed" postdates some on-disk partials/finals; a file
              # written before it existed has no second-slot collapse guard
              # to report, so default to all-False rather than KeyError.
              "collapsed": (np.asarray(z["collapsed"], bool) if "collapsed" in z.files
                           else np.zeros((int(num_animals), len(z["coarse_frame"])), bool)),
              "kp_names": [str(n) for n in z["kp_names"]]}
        from jarvis_jax.tracking.coarse_track import unpad_last_centres
        tr["last_centres"] = (unpad_last_centres(z["last_centres"])
                              if "last_centres" in z.files else None)
    meta_path = path.rsplit(".npz", 1)[0] + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    tr["W"] = meta.get("W", (fallback_wh[0] if fallback_wh else None))
    tr["H"] = meta.get("H", (fallback_wh[1] if fallback_wh else None))
    tr["floor"] = None
    if tr["exist"].shape[0] != num_animals:
        raise ValueError(f"{path} has {tr['exist'].shape[0]} flies, --num-animals is "
                         f"{num_animals}; refusing to append rows of a different shape")

    def _refuse(field, got, want):
        if got is not None and want is not None and got != want:
            raise ValueError(f"{path} was written with {field}={got!r}, but this run has "
                             f"{field}={want!r}; refusing to resume -- a --resume with a "
                             f"different {field} would splice two different geometries into "
                             f"one file")

    _refuse("cameras", meta.get("cameras"), list(cameras) if cameras is not None else None)
    _refuse("stride", meta.get("stride"), int(stride) if stride is not None else None)
    _refuse("start", meta.get("start"), int(start) if start is not None else None)
    _refuse("calib_dir", meta.get("calib_dir"),
           os.path.abspath(str(calib_dir)) if calib_dir is not None else None)
    _refuse("checkpoint", meta.get("checkpoint"),
           os.path.abspath(str(checkpoint)) if checkpoint is not None else None)
    _refuse("mvq_step", meta.get("mvq_step"), step_label)
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
    ap.add_argument("--up-hint", default=None,
                    help="'x,y,z' world-unit up direction for fit_floor, e.g. hand-picked from "
                         "a floor-majority stretch of this recording; skips the bottom-heaviness "
                         "skew heuristic entirely (see fit_floor's round-2 addendum) -- use this "
                         "when the notes' floor-block check says the skew was marginal")
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

    step_label = "final" if args.step is None else int(args.step)

    out = args.out
    partial = out.rsplit(".npz", 1)[0] + ".partial.npz"
    done = None
    if args.resume:
        src = partial if os.path.isfile(partial) else (out if os.path.isfile(out) else None)
        if src:
            done = load_partial(src, args.num_animals, cameras=cameras, calib_dir=calib_dir,
                                checkpoint=args.run, step_label=step_label,
                                stride=args.stride, start=args.start,
                                fallback_wh=(W, H))
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

    # `_CamStream` decodes forward-only from ONE start slot and expects every
    # later call at exactly `start_slot + k*stride` -- `all_frames` (already
    # resume-filtered above) IS that sequence, so its first element (or
    # `--start` if there is nothing left to do) is the one seek.
    reader_start = all_frames[0] if all_frames else args.start
    reader = SlotReader(args.session_dir, cameras, load_plan(args.session_dir),
                       start_slot=reader_start, stride=args.stride)
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
        # IMPORTANT 3: on a fresh run there is no history; on --resume, restore
        # the LAST written chunk's centres (`load_partial`'s `last_centres`) so
        # a blank frame right after the resume boundary still REUSES them
        # (`centre_source == 1`) instead of reading as no-history NaN, which a
        # single-shot run would not have produced.
        last_centres = None if done is None else done["last_centres"]
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
            _write(partial, chunks, cameras, runner, args, W, H, t_load, t_pass,
                   calib_dir=calib_dir, final=False)
            print(f"[coarse] partial written: {partial} "
                  f"({sum(int(c['frame'].shape[0]) for c in chunks)} coarse frames)", flush=True)
    finally:
        reader.close()

    meta = _write(out, chunks, cameras, runner, args, W, H, t_load, t_pass,
                  calib_dir=calib_dir, final=True)
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


def _write(path, chunks, cameras, runner, args, W, H, t_load, t_pass, *, calib_dir, final):
    from jarvis_jax.tracking.coarse_track import (coarse_features, concat_tracks, fit_floor,
                                                  write_coarse_tracks)
    tr = concat_tracks(chunks)
    tr["W"], tr["H"] = tr.get("W") or W, tr.get("H") or H
    up_hint = parse_up_hint(args.up_hint)
    try:
        floor = fit_floor(tr["centroid"], exist=tr["exist"], up_hint=up_hint)
    except ValueError as e:                 # no finite, trackable centroid yet (early partial)
        print(f"[coarse] floor not fit ({e}); heights are NaN in this write", flush=True)
        floor = None
    tr["floor"] = floor
    if floor is None:
        from jarvis_jax.tracking.coarse_track import FloorPlane
        feats = coarse_features(tr, tr["kp_names"],
                                floor=FloorPlane(np.array([np.nan] * 3), float("nan"),
                                                orientation="none", skew=float("nan")))
    else:
        feats = coarse_features(tr, tr["kp_names"], floor=floor)
    extra = {"checkpoint": runner.checkpoint, "mvq_step": runner.step_label,
             "calib_dir": os.path.abspath(str(calib_dir)),
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
