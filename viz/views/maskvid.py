"""maskvid view: a stacked SAM-mask overlay video for one bout.

For a single bout, overlays the SAM3 masks of BOTH flies -- fly0 in
PALETTE["fly0"] (cyan), fly1 in PALETTE["fly1"] (orange) -- directly onto the
raw camera frames, picks the few most-informative cameras (most mask pixels
over the segment) and STACKS them VERTICALLY into one mp4 (plus a middle-frame
still PNG). Because fly0/fly1 keep the same colour on every camera, identity
consistency across views is visible at a glance.

Much simpler than viz/views/sidebyside.py: SAM masks are already per-camera 2D
rasters, so there is NO reprojection and NO mujoco/jax/stac render path -- this
module imports only cv2/numpy + viz.core + load_bout_masks (+ a lazy
bout_start_frame purely for video-frame alignment), so it stays importable and
runnable in the PyTorch SAM3 env.

A FIXED crop window is computed once over the whole segment (union of both
flies' mask bboxes across the chosen cameras) so every camera panel keeps a
constant size for every frame -- a per-frame crop would change panel
dimensions and break the video encoder (same rationale as sidebyside).

DictConfig / lazy scripts.run_bout import: identical rationale to
viz/views/sidebyside.py -- bout_start_frame (which pulls in the pipeline's
CSV parsing) is imported lazily inside run() so `import viz.views.maskvid` is
cheap. Reusing bout_start_frame keeps the seeked video frames aligned with the
mask frames exactly as sidebyside does.
"""
import os
import sys

import cv2
import numpy as np
from hydra import initialize_config_dir, compose

from viz.core import colors as vcolors
from viz.core import io as vio
from viz.core import layout
from viz.core import overlays
from viz.config import courtship_recording, _CFG_DIR

_CAM_BANNER_COLOR = (0, 255, 255)  # yellow camera-name label (matches legskel)


def _compose_cfg():
    """Re-compose the raw `pipeline` DictConfig, needed only for
    scripts.run_bout.bout_start_frame. Reuses viz.config._CFG_DIR
    (single source of truth for the configs/ path); mirrors sidebyside."""
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        return compose(config_name="pipeline")


def _load_fly_masks(run_root, bout, fly, cameras):
    """Masks for one fly, or None if the fly is absent / all-invalid for this
    bout. A bout can hold 1 or 2 flies, so a missing fly index (or a fully
    empty validity mask) is a normal condition, not an error."""
    try:
        m = vio.load_masks(run_root, bout, fly, cameras)
    except Exception as e:
        print(f"[maskvid] fly{fly}: masks unavailable ({e}); skipping this fly")
        return None
    if not np.asarray(m["valid"]).any():
        print(f"[maskvid] fly{fly}: no valid mask frames; skipping this fly")
        return None
    return m


def run(args):
    # cwd-independence: repo_root is 3 levels up from this file
    # (viz/views/maskvid.py -> viz/views -> viz -> repo root); needed so the
    # lazy `scripts.run_bout` import below resolves.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    rec = courtship_recording()
    all_cameras = list(rec["cameras"])
    # Recording override: the raw videos live in the recording being processed,
    # NOT necessarily the default (Session0) courtship_recording() returns.
    session_dir = getattr(args, "session_dir", None) or rec["session_dir"]
    bout = int(args.bout)
    n_cap = int(args.n) if getattr(args, "n", None) else 300
    n_cams = int(getattr(args, "n_cams", 3) or 3)
    panel_h = int(getattr(args, "panel_h", 320) or 320)
    fps = int(getattr(args, "fps", 30) or 30)

    # --- SAM masks (T,C,H,W) for each fly, indexed against the full camera list ---
    flies = {0: _load_fly_masks(args.run, bout, 0, all_cameras),
             1: _load_fly_masks(args.run, bout, 1, all_cameras)}
    present = {f: m for f, m in flies.items() if m is not None}
    if not present:
        raise FileNotFoundError(
            f"no usable SAM masks for bout {bout} under {args.run} "
            f"(bout_{bout:05d}/sam3_masks.npz missing or all-invalid for both flies)")

    ref = next(iter(present.values()))
    H, W = int(ref["H"]), int(ref["W"])
    T = min(int(m["T"]) for m in present.values())
    # start_t offsets BOTH the mask index and the video seek; shifting only one
    # would silently draw each mask on a different frame than it came from.
    t0 = max(0, int(getattr(args, "start_t", 0) or 0))
    if t0 >= T:
        raise ValueError(f"--start-t {t0} is beyond bout {bout} (T={T})")
    N = min(n_cap, T - t0)
    if N <= 0:
        raise ValueError(f"no frames to render for bout {bout} (usable T={T})")
    seg = slice(t0, t0 + N)

    # --- choose cameras: explicit --cams, else top n_cams by total mask pixels ---
    want = list(args.cams) if getattr(args, "cams", None) else None
    if want is not None:
        chosen = [c for c in want if c in all_cameras]
        if not chosen:
            raise ValueError(
                f"none of --cams {want} are recording cameras {all_cameras}")
    else:
        pix = np.zeros(len(all_cameras), dtype=np.int64)
        for ci in range(len(all_cameras)):
            for m in present.values():
                v = np.asarray(m["valid"])[seg, ci]
                if v.any():
                    pix[ci] += int(np.asarray(m["masks"])[seg, ci][v].sum())
        order = [i for i in np.argsort(pix)[::-1] if pix[i] > 0]
        if not order:
            order = list(range(len(all_cameras)))  # nothing had pixels; fall back
        chosen = ([c for c in all_cameras] if n_cams >= len(all_cameras)
                  else [all_cameras[i] for i in order[:max(1, n_cams)]])
    cam_idx = {c: all_cameras.index(c) for c in chosen}
    print(f"[maskvid] bout {bout}: flies {sorted(present)}; cams {chosen}; "
          f"segment t[{t0}:{t0 + N}] of T={T}")

    # --- ONE fixed crop window over the segment (union of both flies' mask
    #     bboxes across the chosen cams) so every panel is a constant size ---
    bx0, by0, bx1, by1 = W, H, 0, 0
    for c in chosen:
        ci = cam_idx[c]
        for m in present.values():
            v = np.asarray(m["valid"])[seg, ci]
            if not v.any():
                continue
            um = np.asarray(m["masks"])[seg, ci][v].any(axis=0)
            if um.any():
                ys, xs = np.where(um)
                bx0, bx1 = min(bx0, int(xs.min())), max(bx1, int(xs.max()))
                by0, by1 = min(by0, int(ys.min())), max(by1, int(ys.max()))
    if bx1 <= bx0 or by1 <= by0:  # no mask pixels anywhere in the segment
        bx0, by0, bx1, by1 = 0, 0, W, H
    pad = 45
    cx0, cy0 = max(bx0 - pad, 0), max(by0 - pad, 0)
    cx1, cy1 = min(bx1 + pad, W), min(by1 + pad, H)
    panel_w = int(round((cx1 - cx0) / (cy1 - cy0) * panel_h))
    print(f"[maskvid] crop x[{cx0}:{cx1}] y[{cy0}:{cy1}] -> {panel_w}x{panel_h} per cam")

    # --- absolute start frame of this bout (to seek the raw video) ---
    if getattr(args, "start_frame", None) is not None:
        start_abs = int(args.start_frame)
    else:
        from scripts.run_bout import bout_start_frame  # lazy: pulls in pipeline deps
        start_abs = bout_start_frame(_compose_cfg(), bout)

    def _panel(bgr, cam, t):
        t = t0 + t                       # stream index -> absolute bout index
        for f, m in present.items():
            ci = cam_idx[cam]
            if np.asarray(m["valid"])[t, ci]:
                overlays.draw_mask(bgr, np.asarray(m["masks"])[t, ci],
                                   vcolors.PALETTE[f"fly{f}"])
        panel = cv2.resize(bgr[cy0:cy1, cx0:cx1], (panel_w, panel_h))
        return np.vstack([layout.banner(panel_w, [(cam, _CAM_BANNER_COLOR)]), panel])

    legend_items = [(f"bout{bout} SAM masks", (255, 255, 255)),
                    ("fly0", vcolors.PALETTE["fly0"]),
                    ("fly1", vcolors.PALETTE["fly1"])]
    top_band = layout.banner(panel_w, legend_items)

    frames_out = []
    # Sync-aware: cameras drop frames independently, so a positional read
    # overlays masks on the wrong frame for any camera that dropped one before
    # this bout. Mask GENERATION is already sync-aware (positions_per_cam), so
    # a positional read here would disagree with the very masks it is drawing.
    # 25 courtship bouts start after a recorded drop slot. Falls back to
    # positional when no plan exists, so clean recordings are unaffected.
    stream = vio.read_frames_synced(session_dir, chosen, start_abs + t0, N)
    for k, imgs in enumerate(stream):
        blocks = [top_band]
        for cam, rgb in zip(chosen, imgs):
            bgr = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None
                   else np.zeros((H, W, 3), np.uint8))
            blocks.append(_panel(bgr, cam, k))
        frames_out.append(np.vstack(blocks))

    if not frames_out:
        raise RuntimeError(
            f"no frames rendered for bout {bout} (cams {chosen}); check the "
            f"session videos under {session_dir}")

    out_path = args.out or os.path.join(
        args.run, f"bout_{bout:05d}", f"maskvid_bout{bout}.mp4")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    vio.write_video(out_path, frames_out, fps=fps)
    still_path = os.path.splitext(out_path)[0] + "_still.png"
    cv2.imwrite(still_path, frames_out[len(frames_out) // 2])
    print(f"[maskvid] wrote {out_path} ({len(frames_out)} frames) + {still_path}")
    return 0
