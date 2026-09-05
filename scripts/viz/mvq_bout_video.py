#!/usr/bin/env python
"""Run the mvq multi-view lifter over a whole courtship bout and render it.

Figure gate 2 for the mvq lifter (gate 1 is `scripts/viz/mvq_overlay.py`, which
scores single labelled val WINDOWS). This one takes the model out of the
labelled-window world entirely: raw session video + SAM3 masks -> per-frame
multi-view inference -> pipeline-format `kp2d.npz`/`kp3d.npz` -> one composite
mp4 of the whole bout, seven camera panels plus a 3D panel, with the existing
ViTPose+DLT pipeline result drawn underneath as the comparison arm.

EXPECTATION (written before rendering, so the figure can disagree with it).
If the typed-slot decoder works on real, unlabelled courtship video:

  * the FEMALE slot's points stay on fly0 and the MALE slot's points stay on
    fly1 in every camera for the whole bout, INCLUDING the mounting frames --
    no swaps, and no points sitting on the other animal. A slot whose dots
    jump between the two bodies (or whose 50 points straddle both) is the
    cross-fly mixing failure this render exists to see; it will be most
    visible in the cameras where the two flies overlap.
  * the mvq track is visibly SMOOTHER than the white DLT baseline -- fewer
    single-frame spikes, especially on the female's legs -- and agrees with
    it to within a few px on easy (well-separated, unoccluded) frames. mvq
    dots systematically offset from the white circles by the same vector in
    every camera would be a center3D/crop-origin bookkeeping error, not a
    model error; right in some cameras and mirrored/rotated in others would
    be a camera-order error.
  * the 3D panel shows two PLAUSIBLE FLY SHAPES at fly scale: a ~2.5 mm body
    (Scutellum->Abd_tip is ~1.3 mm on the DLT baseline), six legs radiating
    from one thorax, wings behind. A tangle, a collapsed point cloud, or a
    skeleton an order of magnitude off the grey baseline means the world
    assembly (`center3D` + ROI-local xyz) is wrong even if the 2D panels look
    fine.

The hard case is deliberately in frame: fly0 is the FEMALE (walls, occlusion,
OOD poses -- where this pipeline fails), all seven cameras are shown, and the
render covers the whole bout rather than a flattering mid-bout frame.

ORDER DISCIPLINE (CLAUDE.md; both traps were live bugs here):
  * CAMERA axis is the canonical `cfg.recording.cameras` order everywhere --
    it is asserted equal to `ReprojectionTool`'s own key order, the SAM3 mask
    axis is permuted into it BY NAME (the same `_camera_permutation`
    `load_bout_masks` uses), and the baseline artifacts are loaded through
    `viz.core.bout_artifacts.load_bout_kp(..., cameras=...)`.
  * KEYPOINT axis: mvq speaks the v12 detector order (`meta["keypoint_names"]`)
    and the pipeline artifacts speak MODEL order. They are the same 50 names
    in DIFFERENT orders, so the baseline is permuted into mvq order BY NAME
    through the named accessors, never by integer index.

Usage (CPU smoke on 4 frames, then the real thing on a GPU node):

    PYTHONPATH=third_party/jarvis_jax:. JAX_PLATFORMS=cpu \\
    python scripts/viz/mvq_bout_video.py --run <run_dir> --step latest \\
        --attn_impl xla --n 4

    python scripts/viz/mvq_bout_video.py --run <run_dir> --step latest \\
        --render-mode unprompted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (registers 3d proj)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from jarvis_jax.data.transforms import crop_origin  # noqa: E402
from jarvis_jax.data.v12_windows import CROP, _affine_np  # noqa: E402
from jarvis_jax.geometry.center3d import triangulate_dlt_batched  # noqa: E402
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool  # noqa: E402
from jarvis_jax.models.mvq.checkpoint import load_mvq_model  # noqa: E402
from jarvis_jax.models.mvq.model import assemble  # noqa: E402
from jarvis_jax.models.mvq.policy import EXIST_THRESH, policy_instance  # noqa: E402
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for  # noqa: E402
from jarvis_jax.predict.synced_reader import load_plan, read_window  # noqa: E402
from jarvis_jax.tracking.bout_masks import _camera_permutation, unpack_one  # noqa: E402
from jarvis_jax.train.matching import SLOT_FEMALE, SLOT_MALE, SLOT_PROMPTED  # noqa: E402
from jarvis_jax.train.train_mvq import MM_PER_UNIT, _fwd, normalize_crops  # noqa: E402
from viz.core.bout_artifacts import load_bout_kp  # noqa: E402
from viz.core.colors import PALETTE, leg_chains  # noqa: E402
from viz.core.io import write_video  # noqa: E402

# ---------------------------------------------------------------- defaults
DEF_SESSION = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
               "courtship/Session0/2025_10_20_13_20_04")
DEF_PROCESSED = ("/gscratch/portia/eabe/data/Johnson_lab/processed/"
                 "courtship/Session0/2025_10_20_13_20_04")
DEF_BASELINE = ("/gscratch/portia/eabe/data/Johnson_lab/processed/_courtship_backup/"
                "Session0/2025_10_20_13_20_04/pose/bouts")
DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "session0.yaml")

PANEL_W, PANEL_H = 600, 448          # one grid cell
HEADER_H = 40
GRID_COLS, GRID_ROWS = 4, 2
BASELINE_CONF_THRESH = 0.3           # pipeline kp2d conf gate for the white circles
VIS_THRESH = 0.5                     # mvq per-view visibility sigmoid gate

# Body skeleton BY NAME (leg chains come from viz.core.colors.leg_chains, which
# only knows about legs). Only T1 has a ThxCx landmark, so T2/T3 hang off the
# thorax at their trochanter.
BODY_SEGMENTS = [
    ("EyeL", "Antenna_Base"), ("EyeR", "Antenna_Base"), ("Antenna_Base", "Scutellum"),
    ("Scutellum", "Abd_A4"), ("Abd_A4", "Abd_tip"),
    ("Scutellum", "WingL_base"), ("WingL_base", "WingL_V12"), ("WingL_base", "WingL_V13"),
    ("Scutellum", "WingR_base"), ("WingR_base", "WingR_V12"), ("WingR_base", "WingR_V13"),
    ("Scutellum", "T1L_ThxCx"), ("Scutellum", "T2L_Tro"), ("Scutellum", "T3L_Tro"),
    ("Scutellum", "T1R_ThxCx"), ("Scutellum", "T2R_Tro"), ("Scutellum", "T3R_Tro"),
]

_bgr = lambda key: tuple(int(c) for c in PALETTE[key])            # cv2 wants BGR
_rgb = lambda key: tuple(c / 255.0 for c in reversed(PALETTE[key]))  # matplotlib wants RGB
FLY_BGR = {0: _bgr("fly0"), 1: _bgr("fly1")}                       # cyan / orange
FLY_RGB = {0: _rgb("fly0"), 1: _rgb("fly1")}
BASELINE_BGR = (255, 255, 255)
BASELINE_RGB = (0.55, 0.55, 0.55)   # readable grey on the 3D panel's white ground


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ---------------------------------------------------------------- geometry / IO helpers
def skeleton_edges(kp_names):
    """(i, j) index pairs in `kp_names` order: leg chains + body segments.

    Built BY NAME so the SAME edge list is correct for the mvq keypoint order
    and, after the name permutation below, for the MODEL-order baseline.
    """
    idx = {n: i for i, n in enumerate(kp_names)}
    edges = []
    for chain in leg_chains(kp_names).values():
        edges.extend(zip(chain[:-1], chain[1:]))
    for a, b in BODY_SEGMENTS:
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))
    return edges


def resolve_bout_frames(session_dir, bout_idx):
    """(start_frame, end_frame, n_frames) for `bout_idx` from the session's bouts CSVs.

    The unified CSV is a symlink into a `Predictions_3D_*` dir that no longer
    exists on this session, so the per-fly CSVs (whose `fly_id` carries a
    `_fly<f>` suffix the unified one does not) are the fallback. The result is
    cross-checked against the mask npz's own frame count by the caller.
    """
    tag = session_tag_for(str(session_dir))
    tries = [(os.path.join(session_dir, "courtship_bouts_unified_summary.csv"), tag)]
    tries += [(os.path.join(session_dir, f"courtship_bouts_fly{f}_summary.csv"), f"{tag}_fly{f}")
              for f in (0, 1)]
    errs = []
    for path, want in tries:
        try:
            rows = parse_bouts(path, want, bout_ids=[bout_idx])
        except OSError as e:                                   # broken symlink / absent
            errs.append(f"{os.path.basename(path)}: {e}"); continue
        if rows:
            r = rows[0]
            return int(r["start"]), int(r["end"]), int(r["n"])
        errs.append(f"{os.path.basename(path)}: no row for bout {bout_idx} / fly_id {want!r}")
    raise KeyError(f"bout {bout_idx} not found in any bouts CSV under {session_dir}: {errs}")


class BoutMaskStore:
    """SAM3 masks for one bout, camera axis reordered BY NAME, unpacked LAZILY.

    `tracking.bout_masks.load_bout_masks` unpacks the WHOLE bout eagerly:
    (2007, 7, 448, 1936) bool is 12.2 GB for ONE fly and 24 GB for the pair,
    which does not fit in the job's memory alongside the model. This holds the
    packed array instead (3 GB) and unpacks one (fly, camera, frame) on demand
    with that module's OWN `unpack_one`, permuting the camera axis with that
    module's OWN `_camera_permutation` -- the same two functions
    `load_bout_masks` composes, so the by-name identity guarantee is identical.
    A legacy npz with no `cameras` name array is REFUSED rather than assumed
    positional (that assumption is the camera-scramble bug).
    """

    def __init__(self, npz_path, cameras):
        z = np.load(npz_path)
        if "cameras" not in z.files:
            raise RuntimeError(
                f"{npz_path} predates the `cameras` name array, so its camera axis "
                f"cannot be verified by name -- a permutation would silently crop "
                f"each fly out of the wrong camera. Re-run SAM3 for this bout.")
        self.npz_cameras = [str(c) for c in np.asarray(z["cameras"]).tolist()]
        self.cameras = [str(c) for c in cameras]
        self.perm = _camera_permutation(self.npz_cameras, self.cameras)
        self.packed = z["packed"]                                    # (A,C,T,H,Wb)
        self.valid = np.asarray(z["valid"])[:, self.perm]            # (A,C,T) canonical
        self.centroids = np.asarray(z["centroids"], np.float32)[:, self.perm]  # (A,C,T,2)
        self.H, self.W = int(z["shape"][0]), int(z["shape"][1])
        self.n_flies, self.T = self.packed.shape[0], self.packed.shape[2]

    def valid_at(self, fly, t):
        return np.asarray(self.valid[fly, :, t], bool)               # (C,)

    def centroid_at(self, fly, t):
        return np.asarray(self.centroids[fly, :, t], np.float64)     # (C,2)

    def mask_at(self, fly, cam_i, t):
        """(H,W) bool for canonical camera index `cam_i`."""
        return unpack_one(self.packed, fly, int(self.perm[cam_i]), t, self.W)


def centers_3d(store, cam_mats, n_frames, t0):
    """(A, n_frames, 3) DLT of each fly's valid mask centroids + (A, n_frames) ok.

    One batched `triangulate_dlt_batched` call for the whole bout rather than
    a jit call per frame. A frame with fewer than 2 valid mask views has no
    usable center3D and is marked not-ok (its window is skipped -> NaN output).
    """
    A = store.n_flies
    pts = np.zeros((A * n_frames, len(store.cameras), 2), np.float32)
    val = np.zeros((A * n_frames, len(store.cameras)), bool)
    for fly in range(A):
        for i in range(n_frames):
            r = fly * n_frames + i
            pts[r] = store.centroid_at(fly, t0 + i)
            val[r] = store.valid_at(fly, t0 + i)
    ok = val.sum(axis=1) >= 2
    cm = np.broadcast_to(np.asarray(cam_mats, np.float32)[None], (A * n_frames,) + cam_mats.shape)
    xyz = np.asarray(triangulate_dlt_batched(jnp.asarray(pts), jnp.asarray(cm), jnp.asarray(val)))
    xyz = np.where((ok & np.isfinite(xyz).all(axis=1))[:, None], xyz, np.nan)
    ok = ok & np.isfinite(xyz).all(axis=1)
    return xyz.reshape(A, n_frames, 3), ok.reshape(A, n_frames)


# ---------------------------------------------------------------- inference
def build_window(frames, present, store, fly, t, center, M, tvec, img_wh):
    """One (fly, frame) inference window -- mirrors V12WindowDataset._build's
    inference half exactly: crop every camera at the projection of the SAME
    window-level center3D, carry the local offset of that projection, and hand
    the fly's own SAM3 mask (cropped identically) in as the prompt."""
    W, H = img_wh
    C = M.shape[0]
    origin = np.zeros((C, 2), np.int32)
    crops = np.zeros((C, CROP, CROP, 3), np.uint8)
    prompt = np.zeros((C, CROP, CROP), bool)
    for c in range(C):
        u, v = M[c] @ center + tvec[c]
        x0, y0 = crop_origin([u, v, 0, 0], W, H, CROP)
        origin[c] = (x0, y0)
        crops[c] = frames[c][y0:y0 + CROP, x0:x0 + CROP]
        prompt[c] = store.mask_at(fly, c, t)[y0:y0 + CROP, x0:x0 + CROP]
    t_local = (M @ center + tvec - origin).astype(np.float32)        # (C,2)
    return dict(crops=crops, cam_valid=np.asarray(present, bool), t_local=t_local,
                prompt_mask=prompt, origin=origin, center=center.astype(np.float32),
                fly=fly, t=t)


def _empty_mode_arrays(n, C, K, I):
    """`I` is the checkpoint's OWN slot count (`n_instances`), not the P3a
    constant: a legacy 3-slot run must not be silently padded to 4."""
    return dict(kp2d=np.full((n, C, K, 2), np.nan, np.float32),
                conf=np.zeros((n, C, K), np.float32),
                kp3d=np.full((n, K, 3), np.nan, np.float32),
                conf3d=np.zeros((n, K), np.float32),
                slot=np.full(n, -1, np.int32),
                fallback=np.zeros(n, bool),
                has_mask=np.zeros(n, bool),
                exist=np.full((n, I), np.nan, np.float32),
                sex_prob=np.full((n, I), np.nan, np.float32))


def run_inference(model, store, args, cameras, cam_mats, M, tvec, sex_slot, K, I, t0, n):
    """Both passes (unprompted + prompted) over `n` frames x every fly.

    Returns `{mode: {fly: arrays}}`, arrays as in `_empty_mode_arrays`.
    """
    C = len(cameras)
    plan = load_plan(args.session_dir)
    abs_start, _, _ = args._frames
    centers, ok3d = centers_3d(store, cam_mats, n, t0)
    res = {m: {f: _empty_mode_arrays(n, C, K, I) for f in range(store.n_flies)}
           for m in ("unprompted", "prompted")}
    B = int(args.batch)
    Mb = np.broadcast_to(M.astype(np.float32)[None], (B, C, 2, 3))
    buf, done, t_start = [], 0, time.time()

    def flush(buf):
        if not buf:
            return
        B0 = len(buf)
        pad = [buf[-1]] * (B - B0)
        rows = buf + pad
        crops = np.stack([r["crops"] for r in rows])[:, None]         # (B,1,C,448,448,3)
        cam_valid = np.stack([r["cam_valid"] for r in rows])[:, None]  # (B,1,C)
        t_local = np.stack([r["t_local"] for r in rows])[:, None]      # (B,1,C,2)
        prompt = np.stack([r["prompt_mask"] for r in rows])[:, None]   # (B,1,C,448,448)
        origin = np.stack([r["origin"] for r in rows]).astype(np.float32)   # (B,C,2)
        center = np.stack([r["center"] for r in rows])                 # (B,3)
        j = dict(crops=jnp.asarray(crops), cam_valid=jnp.asarray(cam_valid),
                 M=jnp.asarray(Mb), t_local=jnp.asarray(t_local),
                 prompt_mask=jnp.asarray(prompt))
        # same has_mask definition as train_mvq.evaluate: a prompt only counts
        # in a camera that is itself valid
        has_mask = np.asarray((prompt.any((3, 4)) & cam_valid).any((1, 2)))
        for mode, prompted in (("unprompted", False), ("prompted", True)):
            on = jnp.asarray(np.full(B, prompted) & has_mask)
            out = _fwd(model, normalize_crops(j["crops"]), j["cam_valid"], j["M"],
                       j["t_local"], j["prompt_mask"], on)
            kp3d, conf3d, kp2d, sex_prob = assemble(out, center, origin,
                                                    cam_valid=np.asarray(cam_valid))
            xyz = np.asarray(out["xyz"])                               # (B,I,T,K,3) ROI-local
            exist = sigmoid(np.asarray(out["exist_logit"]))            # (B,I)
            vis = sigmoid(np.asarray(out["vis_logit"]))                # (B,I,T,C,K)
            for b in range(B0):
                r = rows[b]
                fly, i = r["fly"], r["t"] - t0
                a = res[mode][fly]
                a["exist"][i] = exist[b]
                a["sex_prob"][i] = sex_prob[b]
                a["has_mask"][i] = bool(has_mask[b])
                if prompted:
                    # the SHARED policy: slot 0 when there is a prompt to follow,
                    # and its own typed-candidate rule when there is not
                    slot = policy_instance(exist[b], xyz[b], prompted=True,
                                           has_mask=bool(has_mask[b]))
                    a["fallback"][i] = slot != SLOT_PROMPTED
                else:
                    slot = sex_slot[fly]
                    if exist[b][slot] < EXIST_THRESH:
                        a["fallback"][i] = True
                        slot = policy_instance(exist[b], xyz[b], prompted=False, has_mask=False)
                if slot is None:                                       # policy MISS -> NaN frame
                    continue
                a["slot"][i] = int(slot)
                a["kp3d"][i] = kp3d[b, slot, 0]
                a["conf3d"][i] = conf3d[b, slot, 0]
                a["kp2d"][i] = kp2d[b, slot, 0]
                a["conf"][i] = vis[b, slot, 0]
        return

    reader = read_window(args.session_dir, cameras, plan, abs_start + t0, n)
    img_wh = (store.W, store.H)
    for i, (frames, present) in enumerate(reader):
        t = t0 + i
        for fly in range(store.n_flies):
            if not ok3d[fly, i]:
                continue
            buf.append(build_window(frames, present, store, fly, t, centers[fly, i],
                                    M, tvec, img_wh))
            if len(buf) == B:
                flush(buf); buf = []; done += B
                el = time.time() - t_start
                print(f"[mvq] {done} windows, frame {t - t0 + 1}/{n}, "
                      f"{el:.0f}s ({el / max(done, 1):.2f}s/window)", flush=True)
    flush(buf)
    return res


# ---------------------------------------------------------------- rendering
class Panel3D:
    """The eighth cell: both flies' mvq skeletons in world mm, fixed axes.

    Limits are the BOUT's 1st..99th percentile of the mvq kp3d (so the camera
    does not chase a spike), the box aspect is the true mm extent (isotropic
    scale), and the grey skeleton is the ViTPose+DLT baseline for the same
    frame -- the comparison arm, in the same units.
    """

    def __init__(self, lims_mm, edges):
        self.edges = list(edges)
        self.fig = Figure(figsize=(PANEL_W / 100.0, PANEL_H / 100.0), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        # the axes fill the whole cell (a default subplot leaves ~40% of a
        # 600x448 panel as white margin, which is the cell this render can
        # least afford to waste)
        self.ax = self.fig.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
        self.lims = np.asarray(lims_mm, float)                        # (3,2)
        self.proxies = [Line2D([], [], color=FLY_RGB[0], lw=1.4, marker="o", ms=2.5,
                               label="fly0 female — mvq"),
                        Line2D([], [], color=FLY_RGB[1], lw=1.4, marker="o", ms=2.5,
                               label="fly1 male — mvq"),
                        Line2D([], [], color=BASELINE_RGB, lw=1.2, label="ViTPose+DLT baseline")]

    def _skeleton(self, ax, xyz_mm, keep, color, lw, ms):
        if xyz_mm is None:
            return
        good = keep & np.isfinite(xyz_mm).all(axis=1)
        if not good.any():
            return
        segs = [[xyz_mm[i], xyz_mm[j]] for i, j in self.edges if good[i] and good[j]]
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=[color] * len(segs), linewidths=lw))
        p = xyz_mm[good]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=ms, c=[color], depthshade=False)

    def draw(self, mvq, base, frame_label):
        """mvq: {fly: (xyz_mm (K,3), keep (K,) bool)}; base: {fly: xyz_mm|None}."""
        ax = self.ax
        ax.cla()
        for fly, (xyz, keep) in sorted(mvq.items()):
            self._skeleton(ax, xyz, keep, FLY_RGB[fly], 1.4, 6.0)
        for fly, xyz in sorted(base.items()):
            if xyz is not None:
                self._skeleton(ax, xyz, np.ones(len(xyz), bool), BASELINE_RGB, 0.8, 2.0)
        ax.set_xlim(*self.lims[0]); ax.set_ylim(*self.lims[1]); ax.set_zlim(*self.lims[2])
        # box aspect = the true mm extents, so a mm is a mm on every axis
        ax.set_box_aspect(tuple(np.diff(self.lims, axis=1).ravel()), zoom=1.25)
        ax.set_xlabel("x (mm)", fontsize=6, labelpad=-8)
        ax.set_ylabel("y (mm)", fontsize=6, labelpad=-8)
        ax.set_zlabel("z (mm)", fontsize=6, labelpad=-8)
        ax.tick_params(labelsize=5, pad=-3)
        ax.set_title(f"3D world (mm) — {frame_label}", fontsize=7, y=0.96)
        ax.legend(handles=self.proxies, fontsize=5.5, loc="lower left", framealpha=0.5)
        self.canvas.draw()
        rgb = np.asarray(self.canvas.buffer_rgba())[..., :3]
        return np.ascontiguousarray(rgb[:, :, ::-1])                  # -> BGR


def panel_window(centroids, valid, img_w):
    """x0 of the PANEL_W-wide crop centred on the mean of the two flies'
    mask centroids in this camera (clamped into the frame)."""
    pts = [c for c, v in zip(centroids, valid) if v and np.isfinite(c).all()]
    cx = float(np.mean([p[0] for p in pts])) if pts else img_w / 2.0
    return int(np.clip(round(cx - PANEL_W / 2.0), 0, max(0, img_w - PANEL_W)))


def draw_camera_panel(panel, x0, cam_name, mvq_per_fly, base_per_fly, edges):
    """mvq_per_fly: {fly: (uv (K,2), keep (K,) bool)}; base_per_fly: {fly: (uv, keep)}."""
    h, w = panel.shape[:2]

    def _pt(uv):
        return (int(round(uv[0] - x0)), int(round(uv[1])))

    def _in(p):
        return 0 <= p[0] < w and 0 <= p[1] < h

    for fly, (uv, keep) in sorted(base_per_fly.items()):             # baseline UNDER mvq
        for k in np.where(keep)[0]:
            p = _pt(uv[k])
            if _in(p):
                cv2.circle(panel, p, 3, BASELINE_BGR, 1, cv2.LINE_AA)
    for fly, (uv, keep) in sorted(mvq_per_fly.items()):
        col = FLY_BGR[fly]
        for i, j in edges:
            if keep[i] and keep[j]:
                a, b = _pt(uv[i]), _pt(uv[j])
                if _in(a) and _in(b):
                    cv2.line(panel, a, b, col, 1, cv2.LINE_AA)
        for k in np.where(keep)[0]:
            p = _pt(uv[k])
            if _in(p):
                cv2.circle(panel, p, 3, col, -1, cv2.LINE_AA)
    cv2.putText(panel, cam_name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, cam_name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def header_strip(width, lines):
    strip = np.zeros((HEADER_H, width, 3), np.uint8)
    for i, txt in enumerate(lines[:2]):
        cv2.putText(strip, txt, (8, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def render_mode(args, mode, res, store, cameras, base, mvq_names, edges, step, out_mp4,
                t0, n, header_json):
    """Stream the composite mp4 for one mode; also dumps 4 evenly spaced PNGs."""
    plan = load_plan(args.session_dir)
    abs_start = args._frames[0]
    flies = sorted(res[mode])
    # fixed 3D axis limits: the BOUT's 1..99 percentile over both flies, in mm
    allxyz = np.concatenate([res[mode][f]["kp3d"].reshape(-1, 3) for f in flies], axis=0)
    allxyz = allxyz[np.isfinite(allxyz).all(axis=1)] * MM_PER_UNIT
    if allxyz.shape[0] < 2:
        raise RuntimeError(f"[{mode}] no finite mvq kp3d in this window -- nothing to render; "
                           f"check the mask/center3D coverage of frames "
                           f"{abs_start + t0}..{abs_start + t0 + n - 1}")
    lo, hi = np.percentile(allxyz, 1, axis=0), np.percentile(allxyz, 99, axis=0)
    pad = 0.1 * np.maximum(hi - lo, 1e-3)
    lims = np.stack([lo - pad, hi + pad], axis=1)
    panel3d = Panel3D(lims, edges)
    idxs = list(range(0, n, max(1, int(args.every))))
    png_at = {idxs[k] for k in
              np.linspace(0, len(idxs) - 1, min(4, len(idxs))).round().astype(int).tolist()}
    fig_dir = os.path.dirname(out_mp4)
    headers = []

    def frames_iter():
        reader = read_window(args.session_dir, cameras, plan, abs_start + t0, n)
        sel = set(idxs)
        for i, (frames, present) in enumerate(reader):
            if i not in sel:
                continue
            t = t0 + i
            absf = abs_start + t
            canvas = np.zeros((HEADER_H + GRID_ROWS * PANEL_H, GRID_COLS * PANEL_W, 3), np.uint8)
            for c, cam in enumerate(cameras):
                cen = [store.centroid_at(f, t)[c] for f in flies]
                val = [bool(store.valid_at(f, t)[c]) for f in flies]
                x0 = panel_window(cen, val, store.W)
                panel = np.ascontiguousarray(frames[c][:, x0:x0 + PANEL_W, ::-1])  # RGB->BGR
                if panel.shape[1] < PANEL_W:                       # narrow frame: pad right
                    panel = np.pad(panel, ((0, 0), (0, PANEL_W - panel.shape[1]), (0, 0)))
                if not present[c]:
                    cv2.putText(panel, "camera absent", (6, 42), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2, cv2.LINE_AA)
                mvq_p, base_p = {}, {}
                for f in flies:
                    a = res[mode][f]
                    keep = (a["conf"][i, c] > VIS_THRESH) & np.isfinite(a["kp3d"][i]).all(axis=1)
                    mvq_p[f] = (a["kp2d"][i, c], keep)
                    if base.get(f) is not None:
                        b2, bc = base[f]["kp2d"], base[f]["conf"]
                        bkeep = (bc[t, c] > BASELINE_CONF_THRESH) & np.isfinite(b2[t, c]).all(axis=1)
                        base_p[f] = (b2[t, c], bkeep)
                draw_camera_panel(panel, x0, cam, mvq_p, base_p, edges)
                r, cc = divmod(c, GRID_COLS)
                canvas[HEADER_H + r * PANEL_H:HEADER_H + (r + 1) * PANEL_H,
                       cc * PANEL_W:(cc + 1) * PANEL_W] = panel
            mvq3, base3 = {}, {}
            for f in flies:
                a = res[mode][f]
                x = a["kp3d"][i] * MM_PER_UNIT
                mvq3[f] = (x, np.isfinite(x).all(axis=1))
                base3[f] = (base[f]["kp3d"][t] * MM_PER_UNIT) if base.get(f) is not None else None
            cell = panel3d.draw(mvq3, base3, f"frame {absf}")
            r, cc = divmod(len(cameras), GRID_COLS)
            canvas[HEADER_H + r * PANEL_H:HEADER_H + (r + 1) * PANEL_H,
                   cc * PANEL_W:(cc + 1) * PANEL_W] = cell
            fly_bits = []
            hdr = {"frame_abs": int(absf), "frame_local": int(t), "mode": mode, "flies": {}}
            for f in flies:
                a = res[mode][f]
                sl = int(a["slot"][i])
                e = float(a["exist"][i, sl]) if sl >= 0 else float("nan")
                sx = float(a["sex_prob"][i, sl]) if sl >= 0 else float("nan")
                fly_bits.append(f"fly{f} {args._fly_sex[f]}: slot{sl if sl >= 0 else '-'} "
                                f"exist={e:.2f} P(female)={sx:.2f}"
                                f"{' FALLBACK' if a['fallback'][i] else ''}")
                hdr["flies"][f"fly{f}"] = {"sex": args._fly_sex[f], "slot": sl,
                                           "exist": e, "sex_prob": sx,
                                           "fallback": bool(a["fallback"][i])}
            lines = [f"bout {args.bout}  frame {absf} (local {t}/{store.T - 1})  "
                     f"ckpt step {step}  mode {mode}  "
                     f"cyan=fly0 female mvq  orange=fly1 male mvq  white=ViTPose+DLT",
                     "   |   ".join(fly_bits)]
            canvas[:HEADER_H] = header_strip(canvas.shape[1], lines)
            headers.append(hdr)
            if i in png_at:
                png = os.path.join(fig_dir, f"frame_{absf}.png")
                cv2.imwrite(png, canvas)
                print("wrote", png, flush=True)
            yield canvas

    write_video(out_mp4, frames_iter(), fps=float(args.fps))
    with open(header_json, "w") as f:
        json.dump({"mode": mode, "bout": args.bout, "step": step,
                   "axis_limits_mm": lims.tolist(), "frames": headers}, f, indent=1)
    print("wrote", out_mp4, "and", header_json, flush=True)


# ---------------------------------------------------------------- main
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True,
                   help="mvq RUN dir (with --step) or a final/ dir (without)")
    p.add_argument("--step", default=None, help="load ckpt/<step> or 'latest' instead of final/")
    p.add_argument("--attn_impl", default=None, help="override the run's attn_impl (use 'xla' on CPU)")
    p.add_argument("--session-dir", default=DEF_SESSION)
    p.add_argument("--recording-cfg", default=DEF_RECORDING_CFG,
                   help="hydra recording config that defines the CANONICAL camera order")
    p.add_argument("--bout", type=int, default=28)
    p.add_argument("--masks-npz", default=None,
                   help="default <processed>/sam3_masks/bout_<idx>/sam3_masks.npz")
    p.add_argument("--processed-dir", default=DEF_PROCESSED)
    p.add_argument("--baseline-bouts", default=DEF_BASELINE,
                   help="ViTPose+DLT pipeline bouts dir (the comparison arm)")
    p.add_argument("--anatomy-cfg", default=os.path.join(ROOT, "configs", "anatomy", "v1.yaml"),
                   help="defines model.KP_NAMES -- the baseline artifacts' keypoint order")
    p.add_argument("--sex-json", default=None,
                   help="default <baseline-bouts>/bout_<idx>/sex.json (male_fly)")
    p.add_argument("--out-dir", default=None,
                   help="default <processed>/pose_mvq/bouts/bout_<idx>")
    p.add_argument("--fig-dir", default=None,
                   help="default figures/2026-09-mvq/bout<idx>_p3a")
    p.add_argument("--start", type=int, default=0, help="local frame offset into the bout")
    p.add_argument("--n", type=int, default=0, help="frames (0 = to the end of the bout)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--every", type=int, default=1, help="render every k-th frame (inference is every frame)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--render-mode", default="unprompted",
                   choices=("unprompted", "prompted", "both"))
    p.add_argument("--no-render", action="store_true", help="write the npz artifacts only")
    return p


def main():
    args = build_parser().parse_args()
    cameras = [str(c) for c in OmegaConf.load(args.recording_cfg).cameras]
    rt = ReprojectionTool(os.path.join(args.session_dir, "calibration"))
    assert list(rt.cameras.keys()) == cameras, (
        f"calibration glob order {list(rt.cameras.keys())} != canonical "
        f"{cameras}; every camera axis in this script assumes they are the same")
    cam_mats = np.asarray(rt.camera_matrices, np.float32)             # (C,4,3) canonical
    M, tvec = _affine_np(rt.camera_matrices)                          # (C,2,3),(C,2) float64

    masks_npz = args.masks_npz or os.path.join(
        args.processed_dir, "sam3_masks", f"bout_{args.bout:05d}", "sam3_masks.npz")
    store = BoutMaskStore(masks_npz, cameras)
    abs_start, abs_end, n_csv = resolve_bout_frames(args.session_dir, args.bout)
    if n_csv != store.T:
        raise RuntimeError(f"bout {args.bout}: CSV says {n_csv} frames "
                           f"({abs_start}..{abs_end}) but the mask npz has {store.T}")
    args._frames = (abs_start, abs_end, n_csv)
    t0 = int(args.start)
    n = store.T - t0 if int(args.n) <= 0 else min(int(args.n), store.T - t0)
    if n <= 0:
        raise SystemExit(f"--start {t0} leaves no frames (bout has {store.T})")

    sex_json = args.sex_json or os.path.join(args.baseline_bouts, f"bout_{args.bout:05d}", "sex.json")
    male_fly = 1
    if os.path.exists(sex_json):
        male_fly = int(json.load(open(sex_json))["male_fly"])
    else:
        print(f"[mvq] WARNING: no {sex_json}; assuming male_fly=1 (post-canonicalization default)")
    args._fly_sex = {f: ("male" if f == male_fly else "female") for f in range(store.n_flies)}
    sex_slot = {f: (SLOT_MALE if f == male_fly else SLOT_FEMALE) for f in range(store.n_flies)}
    args._sex_slot = sex_slot
    print(f"[mvq] bout {args.bout} frames {abs_start + t0}..{abs_start + t0 + n - 1} "
          f"({n} of {store.T}); {args._fly_sex}; slots {sex_slot}", flush=True)

    # "latest" is RESOLVED TO A CONCRETE STEP HERE, before the load, and that int
    # is what is both loaded and stamped into mvq_meta.json / the video name. The
    # run dir is a LIVE training job, so resolving it twice (once inside
    # load_mvq_model, once to report it) can name a step that is not the one in
    # the model -- provenance that is wrong exactly when it matters.
    step = args.step
    if step == "latest":
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager(os.path.abspath(os.path.join(args.run, "ckpt")),
                                    options=ocp.CheckpointManagerOptions(read_only=True))
        step = int(mgr.latest_step())
    elif step is not None:
        step = int(step)
    model, meta = load_mvq_model(args.run, step=step, attn_impl=args.attn_impl)
    mvq_names = list(meta["keypoint_names"])
    K = len(mvq_names)
    edges = skeleton_edges(mvq_names)
    step_used = step if step is not None else "final"
    print(f"[mvq] checkpoint {args.run} step {step_used}; K={K}; "
          f"unrestored={meta.get('_unrestored_leaves', [])}", flush=True)

    I = int(meta["model"]["n_instances"])
    res = run_inference(model, store, args, cameras, cam_mats, M, tvec, sex_slot, K, I, t0, n)

    out_dir = args.out_dir or os.path.join(args.processed_dir, "pose_mvq", "bouts",
                                           f"bout_{args.bout:05d}")
    os.makedirs(out_dir, exist_ok=True)
    per_frame = {}
    for mode in ("unprompted", "prompted"):
        per_frame[mode] = {}
        for fly in sorted(res[mode]):
            a = res[mode][fly]
            d = os.path.join(out_dir, mode, f"fly{fly}")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, "kp2d.npz"), kp2d=a["kp2d"], conf=a["conf"],
                                cameras=np.array(cameras), kp_names=np.array(mvq_names))
            np.savez_compressed(os.path.join(d, "kp3d.npz"), kp3d=a["kp3d"], conf3d=a["conf3d"],
                                kp_names=np.array(mvq_names))
            per_frame[mode][f"fly{fly}"] = {
                "slot": a["slot"].tolist(),
                "fallback": a["fallback"].astype(int).tolist(),
                "has_mask": a["has_mask"].astype(int).tolist(),
                "exist": np.round(a["exist"], 4).tolist(),
                "sex_prob": np.round(a["sex_prob"], 4).tolist(),
                "n_missing_frames": int((a["slot"] < 0).sum()),
            }
    meta_out = {
        "checkpoint": os.path.abspath(args.run), "step": step_used,
        "keypoint_names": mvq_names, "cameras": cameras,
        "bout": args.bout, "frame_start": int(abs_start + t0), "n_frames": int(n),
        "bout_frame_range": [int(abs_start), int(abs_end)],
        "male_fly": male_fly, "fly_sex": args._fly_sex,
        "sex_slot": {f"fly{f}": int(s) for f, s in sex_slot.items()},
        "exist_thresh": EXIST_THRESH, "vis_thresh": VIS_THRESH,
        "mm_per_unit": MM_PER_UNIT,
        "unrestored_leaves": meta.get("_unrestored_leaves", []),
        "per_frame": per_frame,
    }
    with open(os.path.join(out_dir, "mvq_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=1)
    for mode in ("unprompted", "prompted"):
        miss = {f"fly{f}": per_frame[mode][f"fly{f}"]["n_missing_frames"] for f in sorted(res[mode])}
        print(f"[mvq] {mode}: missing frames {miss} of {n}", flush=True)
    print("wrote", out_dir, flush=True)
    if args.no_render:
        return

    # ---- baseline arm, permuted into mvq keypoint order BY NAME
    base = {}
    bout_dir = os.path.join(args.baseline_bouts, f"bout_{args.bout:05d}")
    for fly in sorted(res["unprompted"]):
        try:
            bk = load_bout_kp(bout_dir, fly, anatomy_cfg=args.anatomy_cfg, cameras=cameras)
        except (OSError, FileNotFoundError) as e:
            print(f"[mvq] no ViTPose+DLT baseline for fly{fly}: {e}")
            base[fly] = None
            continue
        base[fly] = {
            "kp2d": np.stack([bk.kp2d(nm) for nm in mvq_names], axis=2),      # (T,C,K,2)
            "conf": np.stack([bk.conf2d(nm) for nm in mvq_names], axis=2),    # (T,C,K)
            "kp3d": np.stack([bk.kp3d(nm) for nm in mvq_names], axis=1),      # (T,K,3)
        }

    fig_dir = args.fig_dir or os.path.join(ROOT, "figures", "2026-09-mvq",
                                           f"bout{args.bout}_p3a")
    os.makedirs(fig_dir, exist_ok=True)
    modes = ("unprompted", "prompted") if args.render_mode == "both" else (args.render_mode,)
    for mode in modes:
        stem = f"bout{args.bout}_mvq_{mode}_step{step_used}"
        render_mode(args, mode, res, store, cameras, base, mvq_names, edges, step_used,
                    os.path.join(fig_dir, stem + ".mp4"), t0, n,
                    os.path.join(fig_dir, stem + "_header.json"))


if __name__ == "__main__":
    main()
