#!/usr/bin/env python3
"""Figure gate 1 for the mask-free coarse pass: are the fly centres real?

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §8.1.

EXPECTATION (write before looking, verbatim from the spec): "every visible
fly has a centre within its body in both cameras; when the flies touch, one
centre between them is acceptable; no centre on the wall or reflection." A
large circle sitting on the arena wall, a reflection, or clearly off both
fly bodies means `coarse_track.coarse_pass`'s CenterDetect->3D lift is
placing windows on the wrong thing -- the mvq lifter would then be run on a
crop that does not contain the fly it claims to.

Reads `coarse_tracks.npz` (`jarvis_jax.tracking.coarse_track.write_coarse_tracks`
schema -- mvq only: this script needs `exist`/`slot`/`kp_names`, which a
SAM3-schema file does not have) and, for 12 sampled coarse frames, crops a
700-px-wide window around the reprojected centres in TWO NAMED cameras (one
overhead, one side -- never by integer axis position, see CLAUDE.md), and
draws:

  - large circles: the reprojected 3D centroid (`tracks["centroid"]`,
    already in this camera's full-image px, canonical camera order by
    NAME) -- female (fly row 0) cyan, male (fly row 1) orange
    (`viz.core.colors.PALETTE["fly0"]`/`["fly1"]`).
  - small yellow squares: CenterDetect's own peaks for that frame/camera,
    RECOMPUTED on demand with `jarvis_jax.tracking.coarse_centres.
    CenterDetector` when `--centerdetect <ckpt_dir>` is given (peaks are not
    stored in `coarse_tracks.npz`). Omitted (title says so) otherwise --
    CenterDetect needs a GPU/JAX and a checkpoint restore, which is not
    always available (e.g. all GPUs busy).

The 12 frames are chosen to cover three cases named in the title/JSON:
INSIDE a human-reviewed bout, OUTSIDE any reviewed bout, and at a bout
BOUNDARY (within `--boundary-tol-frames` real frames of a reviewed bout's
start/end) -- 4 of each, spread across the recording -- because "does the
centre lift work" during courtship contact (inside), during ordinary
locomotion (outside) and right at a hand-drawn cut (boundary) are three
different failure modes. `--frames` overrides this selection with an
explicit, comma-separated list of real video frame indices (must each be a
value in the tracks file's own `coarse_frame` array).

Video frames: read through `jarvis_jax.predict.synced_reader`
(`load_plan` + `slot_positions` + the same `_read_at` seek
`scripts/coarse_pass_mvq.py::SlotReader` uses), which maps a real video
frame index to each NAMED camera's own mp4 position -- Session0 has no
`sync_plan.json`, so the mapping is positional, but a recording that has one
would silently read the wrong frame from a positional index. `--video-dir`
is optional: with it omitted (or a frame unreadable) the crop is a flat grey
placeholder, so the geometry (peaks/centres) is still checked even with no
video handy; the title and JSON say plainly when a panel has no real pixels
under it -- never claim a placeholder is a real frame.

Output: `<out-dir>/coarse_centres_check.png` + a `.json` beside it recording,
per chosen frame/camera, the category, bout id (if any), and every drawn
point in full-image px.

Usage (see docs/benchmark/2026-09-mvq/p4-maskfree-notes.md for the exact
20_04 command line):

    python scripts/viz/coarse_centres_check.py \\
      --tracks .../coarse_mvq/coarse_tracks.npz \\
      --video-dir .../Video_recordings/courtship/Session0/2025_10_20_13_20_04 \\
      --reviewed-bouts .../2025_10_20_13_20_04/courtship_bouts_fly0_summary.csv \\
      --out-dir figures/2026-09-mvq/p4_maskfree
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)
# NOTE: `scripts/` is deliberately NOT put on sys.path here. `scripts/viz/` has its
# own (empty) `__init__.py`, so with `scripts/` on the path `import viz` resolves to
# THAT namesake package instead of the real `viz/` at the repo root -- the same trap
# `scripts/viz/coarse_pass_timeline.py` avoids by not importing `viz.core` at all.
# `coarse_pass_mvq.default_cameras` is loaded straight from its file instead (the
# `importlib.util.spec_from_file_location` idiom `tests/test_viz_compare_args.py`
# already uses for a `scripts/`-only module).
for _p in (os.path.join(_REPO, "third_party", "jarvis_jax"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_cpm_spec = importlib.util.spec_from_file_location(
    "coarse_pass_mvq", os.path.join(_SCRIPTS, "coarse_pass_mvq.py"))
_cpm = importlib.util.module_from_spec(_cpm_spec)
_cpm_spec.loader.exec_module(_cpm)
default_cameras = _cpm.default_cameras

from jarvis_jax.tracking.bout_gates import load_ground_truth  # noqa: E402
from viz.core.colors import PALETTE  # noqa: E402

CROP_W = 700
BOUNDARY_TOL_FRAMES_DEFAULT = 400   # real video frames (~0.5s at 800fps): "at a bout boundary"


def _rgb01(bgr):
    """`viz.core.colors.PALETTE` is BGR (cv2); matplotlib wants RGB 0-1."""
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


FEMALE_RGB = _rgb01(PALETTE["fly0"])   # cyan
MALE_RGB = _rgb01(PALETTE["fly1"])     # orange
PEAK_RGB = (1.0, 1.0, 0.0)             # yellow, small squares


def load_tracks_and_meta(tracks_path, meta_path=None):
    meta_path = meta_path or (tracks_path.rsplit(".npz", 1)[0] + ".meta.json")
    z = np.load(tracks_path, allow_pickle=True)
    if "exist" not in z:
        raise ValueError(f"{tracks_path} has no `exist` array -- this script reads the mvq "
                         f"coarse-track schema (jarvis_jax.tracking.coarse_track."
                         f"write_coarse_tracks), not the SAM3 one")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    return z, meta


def categorize_frames(coarse_frame, reviewed, boundary_tol):
    """(inside_mask, outside_mask, boundary_mask) over `coarse_frame` (T,), real
    video frame indices, against `reviewed` = [(bout_idx, start, end), ...].
    `boundary` = within `boundary_tol` real frames of a bout's start/end but
    not required to be inside it (a boundary frame just outside the bout is
    exactly the case a hand-drawn cut is most likely to be wrong about)."""
    coarse_frame = np.asarray(coarse_frame)
    inside = np.zeros(coarse_frame.shape, bool)
    boundary = np.zeros(coarse_frame.shape, bool)
    bout_of = np.full(coarse_frame.shape, -1, np.int64)
    for bidx, s, e in reviewed:
        this_inside = (coarse_frame >= s) & (coarse_frame <= e)
        inside |= this_inside
        bout_of[this_inside & (bout_of < 0)] = bidx
        near = (np.abs(coarse_frame - s) <= boundary_tol) | (np.abs(coarse_frame - e) <= boundary_tol)
        boundary |= near
        bout_of[near & ~this_inside & (bout_of < 0)] = bidx
    outside = ~inside & ~boundary
    return inside, outside, boundary, bout_of


def spread_pick(mask, k):
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    n = min(k, len(idx))
    picks = np.unique(np.round(np.linspace(0, len(idx) - 1, n)).astype(int))
    return sorted(int(idx[p]) for p in picks)


def pick_frames(coarse_frame, reviewed, n_each=4, boundary_tol=BOUNDARY_TOL_FRAMES_DEFAULT):
    inside, outside, boundary, bout_of = categorize_frames(coarse_frame, reviewed, boundary_tol)
    chosen = []
    for cat, mask in (("inside", inside), ("boundary", boundary), ("outside", outside)):
        for t in spread_pick(mask, n_each):
            chosen.append((int(coarse_frame[t]), t, cat, int(bout_of[t])))
    chosen.sort(key=lambda r: r[0])
    return chosen


class _TwoCamReader:
    """Two NAMED cameras, captures kept open across non-contiguous frame
    reads (see `scripts/coarse_pass_mvq.py::SlotReader`, same idiom, just for
    2 cameras instead of the canonical C)."""

    def __init__(self, video_dir, cameras, plan):
        import cv2
        from jarvis_jax.predict.synced_reader import slot_positions
        self._cv2, self._slot_positions = cv2, slot_positions
        self.video_dir, self.cameras, self.plan = str(video_dir), list(cameras), plan
        self.caps, self.cursors = {}, {}
        for c in self.cameras:
            path = os.path.join(self.video_dir, f"{c}.mp4")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise FileNotFoundError(path)
            self.caps[c] = cap
            self.cursors[c] = None

    def read(self, camera, frame_idx):
        from jarvis_jax.predict.synced_reader import _read_at
        pos, present = self._slot_positions(self.plan, camera, int(frame_idx), 1)
        if not present[0]:
            return None
        cap = self.caps[camera]
        if self.cursors[camera] is None:
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, pos[0])
            self.cursors[camera] = pos[0]
        fr, self.cursors[camera] = _read_at(cap, self.cursors[camera], pos[0])
        if fr is None:
            return None
        return self._cv2.cvtColor(fr, self._cv2.COLOR_BGR2RGB)

    def close(self):
        for cap in self.caps.values():
            cap.release()


def crop_window(img, cx, w=CROP_W):
    """`w`-px-wide crop (full frame height) centred on `cx`, clipped to stay
    inside the image -- a "full-frame crop", not the model's 448-px window."""
    H, W = img.shape[:2]
    w = min(w, W)
    x0 = int(np.clip(round(cx - w / 2), 0, max(W - w, 0)))
    x1 = x0 + w
    return img[:, x0:x1], x0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--meta", default=None, help="default: <tracks without .npz>.meta.json")
    ap.add_argument("--video-dir", default=None,
                    help="recording dir with the per-camera mp4s; omit for a grey placeholder")
    ap.add_argument("--cameras", default=None,
                    help="comma list, canonical order; default: configs/recording/session0.yaml")
    ap.add_argument("--cam-overhead", default="Cam2012630")
    ap.add_argument("--cam-side", default="Cam2012861")
    ap.add_argument("--reviewed-bouts", required=True, help="human-reviewed bout-summary CSV")
    ap.add_argument("--session-tag", default=None, help="fly_id filter for --reviewed-bouts")
    ap.add_argument("--centerdetect", default=None,
                    help="CenterDetect ckpt dir; omit to skip peaks (title says so)")
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--frames", default=None,
                    help="comma list of real video frame indices, overriding the 12-frame pick")
    ap.add_argument("--n-each", type=int, default=4)
    ap.add_argument("--boundary-tol-frames", type=int, default=BOUNDARY_TOL_FRAMES_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    z, meta = load_tracks_and_meta(args.tracks, args.meta)
    cameras = ([c.strip() for c in args.cameras.split(",") if c.strip()]
               if args.cameras else default_cameras())
    if args.cam_overhead not in cameras or args.cam_side not in cameras:
        raise ValueError(f"--cam-overhead/--cam-side must be in --cameras {cameras}, got "
                         f"{args.cam_overhead!r}/{args.cam_side!r}")
    cam_idx = {"overhead": cameras.index(args.cam_overhead), "side": cameras.index(args.cam_side)}
    cam_name = {"overhead": args.cam_overhead, "side": args.cam_side}

    coarse_frame = np.asarray(z["coarse_frame"], np.int64)
    centroid_px = np.asarray(z["centroid"], np.float32)   # (F,C,T,2)
    valid = np.asarray(z["valid"], bool)                  # (F,C,T)
    exist = np.asarray(z["exist"], np.float32)            # (F,T)

    reviewed = load_ground_truth(args.reviewed_bouts, fly_id=args.session_tag)

    if args.frames:
        want = [int(v) for v in args.frames.split(",") if v.strip()]
        t_of = {int(f): i for i, f in enumerate(coarse_frame)}
        missing = [f for f in want if f not in t_of]
        if missing:
            raise ValueError(f"--frames {missing} not in this tracks file's coarse_frame array")
        _, _, _, bout_of_all = categorize_frames(coarse_frame, reviewed, args.boundary_tol_frames)
        chosen = [(f, t_of[f], "requested", int(bout_of_all[t_of[f]])) for f in want]
    else:
        chosen = pick_frames(coarse_frame, reviewed, n_each=args.n_each,
                             boundary_tol=args.boundary_tol_frames)
    if not chosen:
        raise ValueError("no frames chosen -- reviewed-bouts CSV and tracks file may not overlap")

    detector = None
    if args.centerdetect:
        from jarvis_jax.tracking.coarse_centres import CenterDetector
        detector = CenterDetector(args.centerdetect, min_score=args.min_score)

    from jarvis_jax.predict.synced_reader import load_plan
    reader = None
    if args.video_dir:
        reader = _TwoCamReader(args.video_dir, [args.cam_overhead, args.cam_side],
                               load_plan(args.video_dir))

    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.1 * n), squeeze=False)
    json_frames = []
    try:
        for row, (frame, t, cat, bidx) in enumerate(chosen):
            row_rec = {"frame": frame, "t": t, "category": cat,
                      "bout_idx": bidx if bidx >= 0 else None, "cameras": {}}
            for col, which in enumerate(("overhead", "side")):
                ax = axes[row, col]
                cam, ci = cam_name[which], cam_idx[which]
                img = None
                if reader is not None:
                    img = reader.read(cam, frame)
                had_real_pixels = img is not None
                if img is None:
                    img = np.full((448, 1936, 3), 200, np.uint8)   # flat grey placeholder

                centres_this_cam = []
                for f_idx, (label, rgb) in enumerate((("female", FEMALE_RGB), ("male", MALE_RGB))):
                    cx, cy = centroid_px[f_idx, ci, t]
                    v = bool(bool(valid[f_idx, ci, t]) and bool(np.isfinite([cx, cy]).all()))
                    centres_this_cam.append({"fly": label, "px": [float(cx), float(cy)] if v else None,
                                             "valid": v, "exist": float(exist[f_idx, t])})

                finite_cx = [c["px"][0] for c in centres_this_cam if c["px"] is not None]
                cx0 = float(np.mean(finite_cx)) if finite_cx else img.shape[1] / 2.0
                crop, x0 = crop_window(img, cx0)
                ax.imshow(crop)

                for c in centres_this_cam:
                    if c["px"] is None:
                        continue
                    rgb = FEMALE_RGB if c["fly"] == "female" else MALE_RGB
                    ax.scatter([c["px"][0] - x0], [c["px"][1]], s=260, facecolors="none",
                              edgecolors=[rgb], linewidths=2.2, label=c["fly"])

                peaks_this_cam = None
                if detector is not None and img is not None and had_real_pixels:
                    peaks, scores = detector.peaks(img[None])
                    peaks, scores = peaks[0], scores[0]      # this one camera
                    peaks_this_cam = []
                    for p in range(peaks.shape[0]):
                        px, py = peaks[p]
                        if np.isfinite(px) and np.isfinite(py):
                            ax.scatter([px - x0], [py], s=36, marker="s", facecolors=PEAK_RGB,
                                      edgecolors="black", linewidths=0.5, zorder=5)
                        peaks_this_cam.append({"px": [float(px), float(py)] if np.isfinite(px) else None,
                                               "score": float(scores[p])})

                ax.set_xlim(0, crop.shape[1])
                ax.set_ylim(crop.shape[0], 0)
                ax.set_xticks([]); ax.set_yticks([])
                px_note = "" if had_real_pixels else " [no video frame]"
                ax.set_title(f"{cam} ({which}) -- frame {frame} [{cat}"
                             f"{'' if bidx < 0 else f' bout {bidx}'}]{px_note}", fontsize=8)
                if row == 0 and col == 0:
                    ax.legend(loc="upper right", fontsize=6, framealpha=0.6)

                row_rec["cameras"][cam] = {"centres": centres_this_cam,
                                           "peaks": peaks_this_cam,
                                           "crop_x0": x0, "had_real_pixels": had_real_pixels}
            json_frames.append(row_rec)
    finally:
        if reader is not None:
            reader.close()

    peaks_note = ("CenterDetect peaks (yellow squares) drawn" if detector is not None
                 else "CenterDetect peaks SKIPPED (--centerdetect not given)")
    fig.suptitle(f"Coarse centres check -- {meta.get('session_dir', args.tracks)}\n"
                f"circles = reprojected 3D centre (cyan female / orange male); {peaks_note}",
                fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, "coarse_centres_check.png")
    json_path = os.path.join(args.out_dir, "coarse_centres_check.json")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    with open(json_path, "w") as f:
        json.dump({"tracks": args.tracks, "video_dir": args.video_dir,
                   "cameras": {"overhead": args.cam_overhead, "side": args.cam_side},
                   "centerdetect": args.centerdetect, "reviewed_bouts": args.reviewed_bouts,
                   "frames": json_frames}, f, indent=2)
    print(f"wrote {png_path}\nwrote {json_path}")
    return png_path, json_path


if __name__ == "__main__":
    main()
