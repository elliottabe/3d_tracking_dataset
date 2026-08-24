"""Build the Figure 4 data bundle from the courtship analysis pipeline.

Split in two deliberately:

* `build_fig4_panels` is pure — it takes already-computed analysis structures
  and shapes them into `PanelData`. It is unit-testable with synthetic input.
* `main` does the I/O: loads the combined h5s, runs `analyze_all_pairs`,
  decodes video, drives MuJoCo, and calls `build_fig4_panels`.

`main` requires the `/data2` mounts and a GPU node (MuJoCo needs
`MUJOCO_GL=egl`). It cannot run on the Hyak login node.

This file replaces the notebook's Figure 4 cell as the source of truth; the
notebook should import from here rather than duplicating the logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# scripts/figures/ -> repo root, so `figbuilder`/`utils` imports work when run
# as a script (sys.path[0] is this file's directory, not the repo root).
# Same bootstrap as scripts/export/pack_reference_clips.py (commit e030c61).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from figbuilder.bundle import PanelData, segments_to_array, write_bundle


def _mean_z_by_label(results: List[dict], label: str) -> np.ndarray:
    out: List[float] = []
    for r in results:
        v = np.asarray(r["male_valid"], dtype=bool)
        lab = np.asarray(r["male_labels"])
        z = np.asarray(r["com_z"], dtype=float)
        m = v & np.isfinite(z) & (lab == label)
        if m.any():
            out.append(float(np.mean(z[m])))
    return np.asarray(out, dtype=float)


def build_fig4_panels(results: List[dict], ex: dict,
                      extras: Dict[str, Any]) -> Dict[str, PanelData]:
    """Shape analysis structures into bundle panels. Pure; no I/O."""
    fs = float(extras["fs"])
    s0, s1 = int(extras["start_frame"]), int(extras["end_frame"])
    sel = slice(s0, s1)
    t_ms = (np.arange(s0, s1) / fs) * 1000.0

    song = ex["song0"]
    wd = song["wing_data"]
    zL = np.asarray(wd["WingL_V13"]["z"], dtype=float)
    zR = np.asarray(wd["WingR_V13"]["z"], dtype=float)
    seg_L = segments_to_array(song["sides"]["L"]["segments"])
    seg_R = segments_to_array(song["sides"]["R"]["segments"])

    angle_L = np.asarray(song["angle_L"], dtype=float)
    angle_R = np.asarray(song["angle_R"], dtype=float)
    ext_is_L = angle_L > angle_R
    ext_z = np.where(ext_is_L, zL, zR)[sel]
    fold_z = np.where(ext_is_L, zR, zL)[sel]

    panels: Dict[str, PanelData] = {
        "wing": PanelData(type="courtship.wing_z", data={
            "t_ms": t_ms, "wingL_z": zL[sel], "wingR_z": zR[sel],
            "seg_L": seg_L, "seg_R": seg_R}, attrs={"fs": fs}),
        "scut": PanelData(type="courtship.scutellum_z", data={
            "t_ms": t_ms, "scutellum_z": np.asarray(ex["com_z"], float)[sel],
            "segments": seg_L}, attrs={"fs": fs}),
        "sine_phase": PanelData(type="courtship.sine_inphase", data={
            "t_ms": t_ms - t_ms[0], "ext_z": ext_z, "fold_z": fold_z,
            "sine_segments": segments_to_array(
                [s for s in song["sides"]["L"]["segments"]
                 if s.get("type") == "sine"])}, attrs={"fs": fs}),
        "wing_phase_polar": PanelData(type="courtship.wing_polar", data={
            "phase_diffs": np.asarray(extras["phase_diffs"], float)}),
        "angle_2d": PanelData(type="courtship.angle_density", data={
            "ext_pulse": np.asarray(extras.get("ext_pulse", np.zeros(0)), float),
            "ext_sine": np.asarray(extras.get("ext_sine", np.zeros(0)), float)}),
        "pulse_class": PanelData(
            type="courtship.pulse_class",
            data={
                # Flat per-type arrays; the adapter re-nests them into the
                # {'Pslow': ..., 'Pfast': ...} dicts the panel function wants.
                **{f"centroid_{t}": np.asarray(
                    extras.get("pulse_centroids", {}).get(t, np.zeros(0)), float)
                   for t in ("Pslow", "Pfast")},
                **{f"pooled_{t}": np.asarray(
                    extras.get("pulse_pooled", {}).get(t, np.zeros((0, 0))), float)
                   for t in ("Pslow", "Pfast")},
            },
            attrs={"fs": fs}),
        "zheight": PanelData(type="courtship.zheight", data={
            "pulse_z": _mean_z_by_label(results, "pulse"),
            "sine_z": _mean_z_by_label(results, "sine"),
            "walking_z": np.asarray(extras["walking_z"], float)}),
        "pitch": PanelData(type="courtship.male_pitch", data={
            "t_ms": np.asarray(extras.get("t_ms_full", t_ms), float),
            "male_pitch": np.asarray(extras["male_pitch"], float),
            "target_pitch": np.asarray(extras["target_pitch"], float)},
            attrs={"fs": fs}),
        "align_violin": PanelData(type="courtship.pitch_violin", data={
            "per_bout": np.asarray(extras["per_bout_align"], float)},
            attrs={"exemplar_idx": int(extras.get("exemplar_bout_idx", -1))}),
    }

    for i, img in enumerate(extras.get("video_frames", [])):
        panels[f"video_{i}"] = PanelData(type="image",
                                         assets={"img": np.asarray(img, np.uint8)})
    for i, img in enumerate(extras.get("render_frames", [])):
        panels[f"render_{i}"] = PanelData(type="image",
                                          assets={"img": np.asarray(img, np.uint8)})
    return panels


#: All verified present on this node (Ruling 14). Override with CLI flags.
#:
#: These MUST stay mutually consistent: the combined h5 holds 11 Session1
#: recordings, and the exemplar bout, its camera video, its calibration and its
#: SAM3 masks all come from 2026_04_02_16_21_32. Pointing any one of them at a
#: different recording silently produces a figure whose traces and video frames
#: describe different flies.
DEFAULT_H5 = ("/gscratch/portia/eabe/data/Johnson_lab/courtship/Data_analysis/"
              "analysis/v1/ik_output_combined_v1_courtship_both.h5")
DEFAULT_MODEL = "models/fruitfly_v1/fruitfly_v1_free.xml"
DEFAULT_SESSION = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/"
                   "courtship/Session1/2026_04_02_16_21_32")
DEFAULT_SAM3_ROOT = DEFAULT_SESSION + "/Predictions_3D_34662592"
DEFAULT_CAM = "Cam2012630"
DEFAULT_FREE_RUN_H5 = ("/gscratch/portia/eabe/data/Johnson_lab/processed/"
                       "free_running/NewBouts/v1/ik_output_combined_v1_free_running.h5")
#: The exemplar recording; bouts are matched by this substring of info/fly_ids.
DEFAULT_EXEMPLAR_RECORDING = "2026_04_02_16_21_32"
#: Root globbed by `find_courtship_sessions` for `**/sam3_aligned.h5` (the
#: pooled alignment violin's population). Independent of `--session`/
#: `--sam3-root`: those point at the ONE recording with per-bout
#: `sam3_masks.npz` (used by the pitch trace + video strip); this root may
#: resolve to a DIFFERENT session that has the pooled `sam3_aligned.h5`
#: instead. Which session(s) actually got pooled is recorded in the bundle
#: meta's `align_sessions`, never silently cross-sourced.
DEFAULT_COURTSHIP_VIDEO_ROOT = ("/gscratch/portia/eabe/data/Johnson_lab/"
                                "Video_recordings/courtship")

# NOTE: the FREE-RUNNING-not-free-walking tick-label fix (Ruling 15) is
# implemented for real in figbuilder.panels.courtship.ZHeightPanel.draw,
# which relabels after drawing. There is no constant to plumb through here.

#: Two-fly courtship-pair render (requirement change, 2026-08-24): the panel
#: is about courtship — two interacting flies — so the render strip must show
#: the styled PAIR (red fly0 / teal fly1), not one vanilla fly.
DEFAULT_FLOOR = "models/fruitfly_v1/floor.xml"
VIZ_SETTINGS = ("Earthy_V1_courtship_fly0", "Earthy_V1_courtship_fly1")
VIZ_CAMERA = "track1_fly0"

#: Camera for the pair render strip. Chosen by sweeping distance against the
#: fraction of frame the fly occupies: 0.30 -> 51% (clipped), 0.60 -> 18%
#: (whole fly, wings and eye legible), 1.00 -> 5% (too small). The model's
#: stat.extent is 0.647, so panel_render_strip's own 0.03 default is ~20x too
#: close and puts the camera inside the animal. azimuth=85.0 is the pair view
#: verified to show fly0 (red) and fly1 (teal) with an extended wing visible.
RENDER_CAM = {"distance": 0.6, "azimuth": 85.0, "elevation": -20.0}


def _pair_qpos(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """Concatenate two single-fly qpos arrays into the pair layout
    ``[fly0_qpos | fly1_qpos]`` that `build_courtship_pair_visualizer`'s model
    expects (nq = 2 * fly_nq). Pure; split out so the concatenation logic is
    unit-testable without MuJoCo."""
    q0 = np.asarray(q0, dtype=float)[:n]
    q1 = np.asarray(q1, dtype=float)[:n]
    return np.concatenate([q0, q1], axis=-1)


def _pair_center_xyz(data: dict, ex: dict, kp_names: List[str], T: int) -> np.ndarray:
    """Per-frame midpoint of the two flies' Scutellum, in the same world
    frame as `kp_xyz_per_frame` — feeds `panel_video_strip_with_kp`'s
    `center_xyz`/`crop_wh` auto-centred crop instead of a fixed `roi` copied
    from a different recording (Finding 2: the notebook's SESSION0 crop is
    the wrong window for a SESSION1 recording; a static crop is wrong for
    every new session). Pure; split out so the midpoint logic is
    unit-testable without video/DLT. Clamps to the shorter of the two bouts
    (and `T`), same as `_pair_qpos`."""
    scut_i = kp_names.index("Scutellum")
    kp0 = np.asarray(data[ex["key0"]]["kp_data"]).reshape(-1, len(kp_names), 3)
    kp1 = np.asarray(data[ex["key1"]]["kp_data"]).reshape(-1, len(kp_names), 3)
    n = min(len(kp0), len(kp1), T)
    return 0.5 * (kp0[:n, scut_i, :] + kp1[:n, scut_i, :])


def _render_frames(flybody_xml, floor_xml, qpos_pair, frame_idx,
                   camera=VIZ_CAMERA, track_midpoint=True, size=256):
    """Bake two-fly courtship-pair MuJoCo frames to uint8 RGB via the styled
    visualizer (`Earthy_V1_courtship_fly0`/`fly1` presets: red fly0, teal
    fly1), floor-aligned.

    `rig_pos` MUST stay None: `floor_xml`'s only geom is `floor`, and the
    rig-pose override (`rig_geom_name='Happy_house'` by default) would raise
    `ValueError` looking for a geom that isn't there. The override only
    aligns cosmetic chamber walls, so omitting it is free.

    Mirrors panel_render_strip's own `track_midpoint` + `viz.render_frame`
    path (verified: red fly0, teal fly1, extended wing visible), inlined here
    to return raw arrays without a matplotlib axes round-trip. When
    `track_midpoint` (default) a free camera tracks the fly0/fly1 midpoint
    every frame using `RENDER_CAM`; set it False to use the named `camera`
    (e.g. a model-defined `track1_fly0`) unmodified instead.
    """
    import mujoco

    from utils.courtship_figure_panels import (
        build_courtship_pair_visualizer, floor_align_qpos_pair)

    viz = build_courtship_pair_visualizer(
        flybody_xml=str(flybody_xml), floor_xml=str(floor_xml),
        settings_fly0=VIZ_SETTINGS[0], settings_fly1=VIZ_SETTINGS[1],
        rig_pos=None)
    qpos_pair = np.asarray(qpos_pair, dtype=float)
    if qpos_pair.shape[1] != viz.model.nq:
        raise ValueError(f"qpos_pair has {qpos_pair.shape[1]} dof but "
                         f"pair model nq={viz.model.nq}")
    qpos_pair = floor_align_qpos_pair(viz.model, qpos_pair)
    fly_nq = qpos_pair.shape[1] // 2

    out = []
    for fi in frame_idx:
        q_row = qpos_pair[int(fi)]
        if track_midpoint:
            mid = 0.5 * (q_row[0:3] + q_row[fly_nq:fly_nq + 3])
            cam_arg = mujoco.MjvCamera()
            cam_arg.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam_arg.lookat[:] = mid
            cam_arg.distance = RENDER_CAM["distance"]
            cam_arg.azimuth = RENDER_CAM["azimuth"]
            cam_arg.elevation = RENDER_CAM["elevation"]
        else:
            cam_arg = camera
        pixels = viz.render_frame(q_row, camera=cam_arg, height=size, width=size)
        out.append(np.asarray(pixels, dtype=np.uint8))
    return out


def main(argv=None) -> int:
    """Build the bundle from whatever inputs are present, and say what is not.

    Requires MUJOCO_GL=egl and a GPU node for the render strip. Never run heavy
    work on the Hyak login node.
    """
    import argparse

    import h5py
    from scipy.signal import hilbert

    from utils.courtship_loader import load_courtship_h5, pair_bouts, analyze_all_pairs
    from utils.song_analysis import SongAnalysisConfig
    from utils.sex_id import SexIdConfig
    from utils.locomotion import LocomotionConfig
    from utils.pair_validity import PairValidityConfig
    from utils.pulse_type_cache import get_pulse_type_labels
    from utils import courtship_figure_panels as cfp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--model-xml", default=DEFAULT_MODEL)
    ap.add_argument("--floor-xml", default=DEFAULT_FLOOR)
    ap.add_argument("--viz-camera", default=VIZ_CAMERA)
    ap.add_argument("--free-run-h5", default=DEFAULT_FREE_RUN_H5)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--sam3-root", default=DEFAULT_SAM3_ROOT)
    ap.add_argument("--sam3-bout", default="bout_00006")
    ap.add_argument("--cam", default=DEFAULT_CAM)
    ap.add_argument("--recording", default=DEFAULT_EXEMPLAR_RECORDING,
                    help="substring of info/fly_ids selecting the exemplar")
    ap.add_argument("--courtship-video-root", default=DEFAULT_COURTSHIP_VIDEO_ROOT,
                    help="root globbed for **/sam3_aligned.h5 (align_violin population)")
    ap.add_argument("--n-video", type=int, default=4)
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[400, 400],
                    help="(w, h) of the per-frame auto-centred video crop")
    ap.add_argument("--roi", type=int, nargs=4, default=None,
                    help="fixed (x, y, w, h) crop; OVERRIDES the auto-centred "
                         "--crop-wh when explicitly given")
    ap.add_argument("--kp-scale", type=float, default=0.1)
    ap.add_argument("--out", default="figures/paper_figures/fig4_bundle.h5")
    ap.add_argument("--width-mm", type=float, default=183.0)
    ap.add_argument("--height-mm", type=float, default=140.0)
    ap.add_argument("--n-render", type=int, default=4)
    ap.add_argument("--exemplar", type=int, default=0,
                    help="index into the filtered results list")
    args = ap.parse_args(argv)

    skipped: List[str] = []

    data, info, kp_names, bout_keys = load_courtship_h5(args.h5)
    pairs = pair_bouts(bout_keys, info)

    # info/fly_ids is an h5 GROUP keyed by STRING INTEGERS ("0", "1", "10"...),
    # not by bout name. Iterating it yields LEXICOGRAPHIC order, so entry 3 is
    # "100", not 3 — reading it that way silently attributes bouts to the wrong
    # recording. Index it numerically. This is the same defect class as commit
    # bd085f7 ("bout keys past 999 sorted out of order against info arrays").
    # Verified: lexicographic gives 2026_04_02_15_25_51 at index 3 where
    # numeric correctly gives 2026_04_02_11_52_43. Ruling 16.
    with h5py.File(args.h5, "r") as _f:
        _g = _f["info"]["fly_ids"]
        fly_ids = [_g[str(i)][()].decode() for i in range(len(_g))]

    def recording_of(res) -> str:
        """Recording id for a pair result, via its bout's numeric index."""
        return fly_ids[bout_keys.index(res["key0"])]
    song = SongAnalysisConfig(); song.pipeline = "both"
    results = analyze_all_pairs(
        data, pairs, kp_names, song_cfg=song, sex_cfg=SexIdConfig(),
        loc_cfg=LocomotionConfig(), pair_cfg=PairValidityConfig())
    if not results:
        raise SystemExit("no pairs survived filtering; nothing to bundle")
    # Prefer an exemplar from the recording that ALSO has video + SAM3, so the
    # traces and the video frames describe the same flies.
    cands = [r for r in results if args.recording in recording_of(r)]
    if not cands:
        skipped.append(f"exemplar from {args.recording!r} (no surviving pair); "
                       f"falling back to all recordings")
        cands = results

    def song_balance(r) -> float:
        """min(frac_pulse, frac_sine) over valid male frames.

        The figure's point is that wing kinematics distinguish song TYPES, so
        the exemplar must contain BOTH in usable proportion. Ranking by
        duration instead (Ruling 17) picked a 3.06 s bout that was 63% sine and
        23% pulse: one long sine region, nothing like the published figure.
        Maximising the WEAKER fraction selects genuinely mixed song.
        """
        lab = np.asarray(r["male_labels"])
        v = np.asarray(r["male_valid"], bool)
        if not v.any():
            return -1.0
        sel = lab[v]
        return min(float((sel == "pulse").mean()), float((sel == "sine").mean()))

    ex = max(cands, key=song_balance)
    print(f"exemplar {ex['key0']}/{ex['key1']} from {recording_of(ex)} "
          f"(T={int(ex['T'])}, {len(cands)} candidates)")
    fs = float(song.fs)
    T = int(ex["T"])

    # --- pooled aggregates -------------------------------------------------
    phase_diffs = []
    for r in results:
        wd = r["song0"]["wing_data"]
        zL = np.asarray(wd["WingL_V13"]["z"], float)
        zR = np.asarray(wd["WingR_V13"]["z"], float)
        for seg in r["song0"]["sides"]["L"]["segments"]:
            if seg.get("type") != "sine":
                continue
            i0, i1 = int(seg["start"]), int(seg["end"]) + 1
            if i1 - i0 < 16:
                continue
            a, b = zL[i0:i1], zR[i0:i1]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            R = np.mean(np.exp(1j * (np.angle(hilbert(a - a.mean()))
                                     - np.angle(hilbert(b - b.mean())))))
            if np.isfinite(R):
                phase_diffs.append(np.angle(R))

    ext_pulse, ext_sine = [], []
    for r in results:
        hL, hR = r["song0"].get("horiz_angle_L"), r["song0"].get("horiz_angle_R")
        if hL is None or hR is None:
            continue
        yL, yR = np.asarray(hL, float), np.asarray(hR, float)
        ext = np.abs(np.where(np.abs(yL) > np.abs(yR), yL, yR))
        lab = np.asarray(r["male_labels"])
        base = np.asarray(r["male_valid"], bool) & np.isfinite(ext)
        if (base & (lab == "pulse")).any():
            ext_pulse.append(ext[base & (lab == "pulse")])
        if (base & (lab == "sine")).any():
            ext_sine.append(ext[base & (lab == "sine")])

    ptr = get_pulse_type_labels(results, fs=fs)

    walking_z = np.zeros(0)
    try:
        from utils.free_walking_loader import load__scutellum_z
        walking_z = load__scutellum_z(args.free_run_h5, per_bout=True)
    except Exception as e:                       # noqa: BLE001 - report, don't die
        skipped.append(f"zheight free-running arm ({type(e).__name__}: {e})")

    # --- render strip (styled two-fly courtship pair) -----------------------
    render_frames = []
    try:
        q0 = np.asarray(data[ex["key0"]]["qpos"])
        q1 = np.asarray(data[ex["key1"]]["qpos"])
        n = min(len(q0), len(q1), T)
        qpos_pair = _pair_qpos(q0, q1, n)
        idx = np.linspace(0, n - 1, args.n_render, dtype=int)
        render_frames = _render_frames(
            args.model_xml, args.floor_xml, qpos_pair, idx,
            camera=args.viz_camera)
    except Exception as e:                       # noqa: BLE001 - report, don't die
        skipped.append(f"render strip ({type(e).__name__}: {e})")

    # --- video strip -------------------------------------------------------
    # Bake the strip by driving the EXISTING, tested panel function into an
    # offscreen figure and grabbing the rasterized axes. The video strip is an
    # image panel by design, so baking pre-drawn frames (keypoints + SAM3 mask
    # overlay included) reuses verified code rather than reimplementing DLT
    # projection here.
    video_frames = []
    try:
        import matplotlib.pyplot as plt
        from utils.sam3_female_com import sam3_camera_index, unpack_sam3_masks_for_frames

        calib_dir = Path(args.session) / "calibration"
        dlt_csv = calib_dir / f"{args.cam}_dlt.csv"
        mp4 = Path(args.session) / f"{args.cam}.mp4"
        sam3_npz = Path(args.sam3_root) / args.sam3_bout / "sam3_masks.npz"
        dlt = cfp._dlt_load(dlt_csv)
        cam_idx = sam3_camera_index(calib_dir, dlt_csv.name)
        vidx = np.linspace(0, T - 1, args.n_video, dtype=int)
        masks = unpack_sam3_masks_for_frames(
            sam3_npz, cam_idx, fly_indices=[1, 0],
            frame_indices=[int(f) for f in vidx])
        kp_xyz = np.asarray(data[ex["key0"]]["kp_data"]).reshape(T, -1, 3)
        figv, axv = plt.subplots(1, args.n_video, figsize=(args.n_video * 2, 2), dpi=200)
        axv = np.atleast_1d(axv)
        # A fixed roi is a crop copied from whatever recording it was tuned
        # on; a per-frame crop centred on the pair's Scutellum midpoint is
        # correct for ANY recording. --roi, when explicitly given, overrides
        # this and forces the fixed window instead (Finding 2).
        if args.roi is not None:
            roi_kwargs = {"roi": tuple(args.roi)}
        else:
            center_xyz = _pair_center_xyz(data, ex, kp_names, T)
            roi_kwargs = {"center_xyz": center_xyz, "crop_wh": tuple(args.crop_wh)}
        cfp.panel_video_strip_with_kp(
            list(axv), mp4, vidx, kp_xyz_per_frame=kp_xyz, kp_names=kp_names,
            dlt_coeffs=dlt, fs=fs, kp_scale=args.kp_scale,
            video_frame_offset=0, masks_per_fly=masks,
            mask_colors=["#e74c3c", "#3a7bff"], mask_alpha=0.35,
            **roi_kwargs)
        figv.canvas.draw()
        for a in axv:
            a.set_position(a.get_position())      # freeze before extraction
            bb = a.get_window_extent()
            buf = np.asarray(figv.canvas.buffer_rgba())
            y0, y1 = int(figv.bbox.height - bb.y1), int(figv.bbox.height - bb.y0)
            video_frames.append(np.ascontiguousarray(
                buf[y0:y1, int(bb.x0):int(bb.x1), :3], dtype=np.uint8))
        plt.close(figv)
    except Exception as e:                       # noqa: BLE001 - report, don't die
        skipped.append(f"video strip ({type(e).__name__}: {e})")

    # --- male pitch vs target pitch (exemplar traces) -----------------------
    # Independent try/except from the pooled violin below (Finding 1b): a
    # failure here must not be reported as (or hide) a failure of the
    # violin, and vice versa — `male_pitch`/`target_pitch` are assigned
    # incrementally in this block, so if THIS block's own exception fires
    # they simply keep whatever partial value they had, same as before.
    male_pitch = target_pitch = np.zeros(0)
    try:
        from utils.sam3_female_com import triangulate_sam3_female_com

        female = triangulate_sam3_female_com(
            str(Path(args.sam3_root) / args.sam3_bout / "sam3_masks.npz"),
            str(Path(args.session) / "calibration"),
            fly_idx=0, min_cams=2, verbose=False) / args.kp_scale
        qm = np.asarray(data[ex["key0"]]["qpos"])
        n = min(T, qm.shape[0], female.shape[0])
        male_pitch = cfp.body_pitch_deg_from_quat(qm[:n, 3:7])
        scut = np.asarray(data[ex["key0"]]["kp_data"]).reshape(-1, len(kp_names), 3)
        scut = scut[:n, kp_names.index("Scutellum"), :]
        vec = female[:n] - scut
        nrm = np.linalg.norm(vec, axis=-1)
        target_pitch = np.degrees(np.arcsin(np.divide(
            vec[..., 2], nrm, out=np.full_like(nrm, np.nan), where=nrm > 0)))
    except Exception as e:                       # noqa: BLE001 - report, don't die
        skipped.append(f"male pitch trace ({type(e).__name__}: {e})")

    # --- pooled alignment violin (population, possibly a DIFFERENT session) -
    # `compute_pitch_alignment_all_sessions` wants a list of
    # (sam3_aligned_h5, bouts_root) TUPLES, discovered via
    # `find_courtship_sessions` — NOT a bare path string (Finding 1a: passing
    # `[args.sam3_root]` made its internal `for h5_path, bouts_root in
    # sessions` unpack the string itself and raise
    # `ValueError: too many values to unpack`, on every run, permanently
    # emptying this panel behind a message that looked like ordinary missing
    # data). Split into its own try/except (Finding 1b) so this failure can
    # never be blamed on / hide behind the pitch-trace block above.
    per_bout_align = np.zeros(0)
    align_sessions: List[tuple] = []
    try:
        from utils.sam3_aligned_bouts import (
            compute_pitch_alignment_all_sessions, find_courtship_sessions)

        align_sessions = find_courtship_sessions(args.courtship_video_root)
        if not align_sessions:
            skipped.append(
                f"align_violin (no sam3_aligned.h5 found under "
                f"{args.courtship_video_root})")
        else:
            al = compute_pitch_alignment_all_sessions(
                align_sessions, kp_scale=args.kp_scale)
            per_bout_align = np.asarray(al["median_abs_alignment_deg"], float)
    except Exception as e:                       # noqa: BLE001 - report, don't die
        skipped.append(f"align_violin ({type(e).__name__}: {e})")

    extras = {
        "fs": fs, "start_frame": 0, "end_frame": T,
        "walking_z": walking_z,
        "phase_diffs": np.asarray(phase_diffs, float),
        "ext_pulse": np.concatenate(ext_pulse) if ext_pulse else np.zeros(0),
        "ext_sine": np.concatenate(ext_sine) if ext_sine else np.zeros(0),
        "pulse_centroids": ptr.get("centroids", {}),
        "pulse_pooled": ptr.get("pooled_waveforms", {}),
        "pulse_counts": ptr.get("counts", {}),
        "per_bout_align": per_bout_align,
        "male_pitch": male_pitch, "target_pitch": target_pitch,
        "t_ms_full": (np.arange(male_pitch.size) / fs) * 1000.0,
        "video_frames": video_frames, "render_frames": render_frames,
    }
    panels = build_fig4_panels(results, ex, extras)
    # Drop panels with no data rather than bundling empty ones.
    panels = {k: v for k, v in panels.items()
              if v.data or v.assets or k in ("wing", "scut")}

    # Finding 1d: record which session(s) the pooled violin actually pooled.
    # Scientifically load-bearing — on this node `find_courtship_sessions`
    # resolves to a DIFFERENT session than the exemplar recording (the
    # exemplar has per-bout sam3_masks.npz for the pitch trace + video strip,
    # but no sam3_aligned.h5 for the pooled violin), so this must be recorded
    # rather than silently cross-sourced.
    align_sessions_str = "; ".join(str(h5) for h5, _ in align_sessions)

    write_bundle(args.out,
                 meta={"fig_width_mm": args.width_mm,
                       "fig_height_mm": args.height_mm,
                       "source_h5": args.h5,
                       "n_pairs": len(results),
                       "align_sessions": align_sessions_str,
                       "skipped": "; ".join(skipped)},
                 panels=panels)
    print(f"wrote {args.out}: {len(panels)} panels from {len(results)} pairs")
    for sk in skipped:
        print(f"  SKIPPED: {sk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
