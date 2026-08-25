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
import json
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
#: Body models live in the sibling Brunton-Lab/fruitfly_body_models checkout
#: (== paths.body_model_dir), not under models/ in this repo.
_BODY_MODELS = _PROJECT_ROOT.parent / "fruitfly_body_models"
DEFAULT_MODEL = str(_BODY_MODELS / "fruitfly_v1" / "fruitfly_v1_free.xml")
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
#: `--sam3-root`: those point at the raw video + calibration for the ONE
#: exemplar recording; this root may resolve to a DIFFERENT session that has
#: the pooled `sam3_aligned.h5` instead. Which session(s) actually got
#: pooled is recorded in the bundle meta's `align_sessions`, never silently
#: cross-sourced.
DEFAULT_COURTSHIP_VIDEO_ROOT = ("/gscratch/portia/eabe/data/Johnson_lab/"
                                "Video_recordings/courtship")

#: Round 6: the combined h5's `kp_data` is body-model-rescaled/re-centred
#: and must NEVER be projected through DLT (measured: its Scutellum
#: midpoint projects to uv (13.6, 426.6), the frame's bottom-left corner).
#: The video strip's keypoint overlay/crop-centring and the pitch block's
#: male-Scutellum position now come from THIS processed tree's per-bout
#: `kp3d.npz` (true DLT world frame) instead — see `_load_kp3d`,
#: `_pair_center_xyz`. `--session`/`--sam3-root` remain the source for the
#: raw mp4 + camera calibration, which this tree does not have.
DEFAULT_PROCESSED_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"

# NOTE: the FREE-RUNNING-not-free-walking tick-label fix (Ruling 15) is
# implemented for real in figbuilder.panels.courtship.ZHeightPanel.draw,
# which relabels after drawing. There is no constant to plumb through here.

#: Two-fly courtship-pair render (requirement change, 2026-08-24): the panel
#: is about courtship — two interacting flies — so the render strip must show
#: the styled PAIR (red fly0 / teal fly1), not one vanilla fly.
DEFAULT_FLOOR = str(_BODY_MODELS / "fruitfly_v1" / "floor.xml")
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


def _free_running_com_z(h5_path) -> np.ndarray:
    """Per-bout mean scutellum height ABOVE THE FLOOR for a free-running h5.

    ``utils.free_walking_loader.load__scutellum_z`` returns RAW z with no
    floor subtraction, while the courtship arms (``pulse_z``/``sine_z``,
    from ``r["com_z"]``) use ``utils.locomotion.compute_com_height``
    (scutellum z minus the 5th-percentile ground-keypoint z for that bout).
    Plotting the two together compared different quantities and overstated
    the courtship-vs-walking height difference roughly 2x (measured: raw
    free-running mean 0.250 vs floor-corrected 0.147, against courtship's
    0.127). Use the same estimator for both sides — this is a figbuilder-
    layer fix (not `utils/`, which is consumed unmodified): a local
    per-bout floor correction that mirrors `compute_com_height` exactly,
    using its own default `LocomotionConfig` so the two sides genuinely
    agree, rather than a single global floor across all bouts (which would
    reintroduce the same class of error `compute_com_height` exists to
    avoid — the floor can differ bout to bout).

    Returns the per-bout array, matching the shape
    ``load__scutellum_z(..., per_bout=True)`` returned so nothing
    downstream changes. Bouts with no finite `com_z` samples are skipped.
    """
    from utils.io_dict_to_hdf5 import load as h5_load
    from utils.locomotion import LocomotionConfig, compute_com_height
    from utils.stac_data_utils import sorted_bout_keys

    data = h5_load(str(h5_path))
    info = data.get("info", {}) or {}
    raw = info.get("kp_names", info.get("site_names_egocentric", []))
    if isinstance(raw, dict):
        kp_names = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    else:
        kp_names = list(raw)

    cfg = LocomotionConfig()
    keys = sorted_bout_keys(k for k in data.keys() if k != "info")
    means: List[float] = []
    for k in keys:
        kp = np.asarray(data[k]["kp_data"])
        if kp.ndim == 2:
            kp = kp.reshape(kp.shape[0], -1, 3)
        com_z, _floor_z = compute_com_height(kp, kp_names, cfg)
        com_z = com_z[np.isfinite(com_z)]
        if com_z.size == 0:
            continue
        means.append(float(np.nanmean(com_z)))
    return np.asarray(means, dtype=float)


def _pair_center_xyz(kp0: np.ndarray, kp1: np.ndarray, scut_idx: int, T: int) -> np.ndarray:
    """Per-frame midpoint of the two flies' Scutellum, given two already-
    loaded ``(T, n_kp, 3)`` kp3d arrays in the TRUE DLT world frame — feeds
    `panel_video_strip_with_kp`'s `center_xyz`/`crop_wh` auto-centred crop
    instead of a fixed `roi` copied from a different recording (Finding 2:
    the notebook's SESSION0 crop is the wrong window for a SESSION1
    recording; a static crop is wrong for every new session).

    Round 6: this used to reach into the combined h5's `kp_data`, which is
    body-model-RESCALED and RE-CENTERED and is NOT in the DLT world frame
    (measured: its Scutellum midpoint projects to uv (13.6, 426.6), the
    frame's bottom-left corner) — feeding it here centred every crop on
    empty chamber. Callers must now load `kp3d.npz` from the processed pose
    tree (`pose/bouts/<bout>/fly{0,1}/kp3d.npz`, key `"kp3d"`) and pass
    those arrays in; this function stays pure and only does the midpoint
    arithmetic, unit-testable without video/DLT. Clamps to the shorter of
    the two arrays (and `T`), same as `_pair_qpos`.
    """
    kp0 = np.asarray(kp0, dtype=float)
    kp1 = np.asarray(kp1, dtype=float)
    n = min(len(kp0), len(kp1), T)
    return 0.5 * (kp0[:n, scut_idx, :] + kp1[:n, scut_idx, :])


def _load_kp3d(npz_path, expected_n_kp=None) -> np.ndarray:
    """Load the ``(T, 50, 3)`` ``kp3d`` array (TRUE DLT world frame) from a
    processed-pose-tree bout npz (``pose/bouts/<bout>/fly{0,1}/kp3d.npz``).
    Raises a clear error naming the path and the keys actually present when
    ``kp3d`` is missing, rather than letting a bare KeyError propagate.

    Round 11 finding D: every caller indexes this array by
    ``kp_names.index(<name>)`` from the COMBINED h5's keypoint order,
    assuming `kp3d`'s own second axis shares that order. `kp3d.npz` carries
    no keypoint-name list of its own to check against (verified: its only
    keys are `kp3d`/`conf3d`) — the order match was instead verified
    empirically (round 6): projecting `kp3d[:, 0, :]` (`Scutellum`, index 0
    in both the combined h5's `info/kp_names` and `configs/anatomy/
    v1.yaml`'s `model.KP_NAMES` MODEL order) through the Cam2012630 DLT
    landed at uv (1123.6, 118.6), matching `kp2d` ground truth to within
    3.3 px. `configs/detector/vitpose_v3.yaml`'s DETECTOR order is
    DIFFERENT (Scutellum at index 3) — an unreordered array would silently
    substitute a nearby keypoint (~1 mm off) and still look plausible. When
    `expected_n_kp` is given, assert the array's keypoint axis matches it,
    since that is the one cheap check available without a real per-file
    name list.
    """
    with np.load(npz_path) as z:
        if "kp3d" not in z.files:
            raise KeyError(f"{npz_path} has no 'kp3d' key (found: {z.files})")
        arr = np.asarray(z["kp3d"])
    if expected_n_kp is not None:
        assert arr.shape[1] == expected_n_kp, (
            f"{npz_path}: kp3d has {arr.shape[1]} keypoints, expected "
            f"{expected_n_kp} (len(kp_names)) — ORDER between this array "
            f"and kp_names is ASSUMED, not verified per-file; a keypoint-"
            f"count mismatch means it cannot be trusted at all")
    return arr


def _processed_fly_dir(fly_id: str) -> str:
    """``'Session1/2026_04_02_16_21_32_fly1'`` -> ``'fly1'``.

    Round 7: the combined h5's key0/key1 ordering is the PAIR ordering and
    does NOT correspond to the processed tree's fly0/fly1 directory names —
    verified: for the exemplar, key0=bout_183 has fly_id
    ``..._fly1`` and key1=bout_182 has fly_id ``..._fly0`` (INVERTED).
    `analyze_pair`'s `male_id='fly0'` refers to the pair's first element
    (key0), not to this fly_id suffix. Always derive the directory from the
    fly_id suffix; never assume key0 -> `fly0` / key1 -> `fly1`.
    """
    tail = fly_id.rsplit("_", 1)[-1]
    if not tail.startswith("fly"):
        raise ValueError(f"cannot derive fly dir from fly_id {fly_id!r}")
    return tail


def _resolve_sam3_camera_index(cameras, calib_dir, cam: str) -> tuple:
    """Return ``(cam_idx, source)`` for `cam` (e.g. ``"Cam2012630"``) against
    a SAM3 mask npz's own camera axis.

    Round 10 finding: SAM3 mask npzs in the PROCESSED tree self-label their
    camera axis via a ``cameras`` array, in an order that is NOT guaranteed
    to match ``sorted(glob('Cam*_dlt.csv'))`` — verified on the exemplar's
    own npz: ``cameras`` puts `Cam2012630` at index 5, while glob-sorted
    calibration order puts it at index 0. The OLD `Video_recordings` tree's
    npzs (``['packed','valid','centroids','shape']``, no ``cameras`` array)
    are the convention `utils/sam3_female_com.py`'s `sam3_camera_index` was
    written against; the round-6 repoint to the processed tree carried that
    glob-order assumption forward, silently permuting the camera axis.

    ``cameras`` is the npz's own array (or `None` when absent — the legacy
    convention, still supported as a fallback). When present, `source` is
    ``"npz cameras"`` and a `cam` absent from it raises `ValueError` naming
    what the npz DOES contain (never silently falls back to index 0). When
    `cameras` is `None`, falls back to `sam3_camera_index` against
    ``calib_dir``'s glob-sorted order, `source` is ``"glob fallback"`` — the
    caller should print which path was taken.
    """
    if cameras is not None:
        names = [str(c) for c in cameras]
        if cam not in names:
            raise ValueError(f"camera {cam!r} not in npz cameras: {names}")
        return names.index(cam), "npz cameras"
    from utils.sam3_female_com import sam3_camera_index
    return sam3_camera_index(calib_dir, f"{cam}_dlt.csv"), "glob fallback"


def _slot_to_raw_frame(sync_plan, camera: str, slot: int) -> tuple:
    """Convert a CANONICAL slot number to a raw mp4 frame position for
    `camera`. Returns ``(raw_frame, corrected)``; `corrected` is False (and
    `raw_frame == slot`, the pre-fix positional behaviour) when `sync_plan`
    is falsy/`None` or its `status` is not `"reindex"`.

    Round 11 finding A: this recording's own `sync_plan.json` has
    `status="reindex"` with one interior drop
    (`Cam2012630: gaps=[{"slot": 21, "lost": 26}]`), long before this
    exemplar's bout (canonical slot ~380781). Reading raw mp4 position ==
    canonical slot directly (the pre-fix behaviour, `utils/
    courtship_figure_panels.py`'s `_read_frame(cap, fidx + video_frame_
    offset, ...)` does exactly this) silently reads a DIFFERENT real
    instant than the one the masks/kp3d describe.

    Reuses the repo's OWN canonical slot<->position mapping —
    `jarvis_jax.predict.frame_sync.SyncCam.pos()`/`.has()` (vendored from
    JohnsonLabJanelia/cluster_pose's `check_sync.py`; also the engine
    behind `viz/core/io.py`'s `sync_positions`/`read_frames_synced`) —
    rather than re-deriving the gap-accumulation arithmetic here, per
    `SyncCam`'s own docstring: "pos(t) = mp4 frame position that delivers
    slot t (= count of present frames before t)" — i.e. `pos(t) = t -
    (frames lost before t)`, confirmed against the pre-existing repo test
    `third_party/jarvis_jax/tests/test_synced_reader.py::
    test_slot_positions_maps_around_gap` (gap at slot 41 losing 3: slot 44
    -> pos 41, `# pos = slot-3`). `SyncCam.has(t)` reports `False` — the
    camera dropped that exact slot entirely — for a slot inside a gap; this
    raises `ValueError` rather than silently reading an adjacent frame.
    """
    if not sync_plan or getattr(sync_plan, "status", None) != "reindex":
        return int(slot), False
    from jarvis_jax.predict.synced_reader import slot_positions
    positions, present = slot_positions(sync_plan, camera, int(slot), 1)
    if not present[0] or positions[0] is None:
        raise ValueError(
            f"camera {camera!r} dropped canonical slot {slot} entirely "
            f"(no raw mp4 frame delivers it)")
    return int(positions[0]), True


#: Historically-hardcoded fly_indices=[1, 0] / fly_idx=0 convention, used
#: only when an npz has no `sex_meta` to derive the real slots from.
_DEFAULT_MALE_SLOT = 1
_DEFAULT_FEMALE_SLOT = 0


def _resolve_male_female_slots(sex_meta, sex_json=None) -> tuple:
    """Return ``(male_slot, female_slot)``.

    Round 10 finding: the mask SLOT is a FOURTH id scheme — distinct from
    the combined h5's key0/key1 pairing (round 7) and the processed tree's
    fly0/fly1 DIRECTORY naming — and was hardcoded (`fly_indices=[1, 0]`,
    `fly_idx=0`).

    Two independent sexers can disagree on which mask slot is male:
    ``sex_meta`` (the SAM3 npz's own JSON string / dict, a mask-area VOTE —
    on the real exemplar a weak one, agreement 0.429) and ``sex_json``
    (``pose/bouts/<bout>/sex.json``, a dict — sometimes a HUMAN-CONFIRMED
    review). Verified on the real exemplar: `sex_meta` says `male_slot=1`
    and `sex_json` says `male_fly=1` (`confidence="user"`,
    `method="manual-gui"`) — they AGREE here (an earlier draft of this
    docstring wrongly claimed `sex_json` said `male_fly=0`/"INVERTED"; that
    was a false comment, corrected — always re-read the real file rather
    than trust a stale claim in code). Round 10's hardcode trusted only the
    weaker `sex_meta` vote; on any bout where the two sexers disagree, that
    could triangulate the MALE as the "female" COM (a degenerate male→male
    vector) and swap the panel-A mask colours — exactly the dir-index/
    mask-slot decoupling the repo's own sexing-canonicalization memory
    warns about (round 11 finding B).

    ``sex_json``'s `male_fly` is treated as authoritative ONLY when its
    `confidence` is `"user"` or its `method` mentions "manual" (a human
    review), never for a lower-confidence/automated `sex_json`. When BOTH a
    human-confirmed `sex_json` and a `sex_meta` vote are present, they must
    AGREE — a disagreement means neither can be trusted silently, so this
    raises `ValueError` naming both values rather than picking one (the
    figure would be meaningless with a degenerate male→male vector). When
    only one source is present, that source's value is used. Falls back to
    the documented default (`_DEFAULT_MALE_SLOT`, `_DEFAULT_FEMALE_SLOT`) —
    matching the historical hardcode — only when NEITHER is present; the
    caller should print when this fallback fires.
    """
    human_slot = None
    if sex_json:
        confidence = str(sex_json.get("confidence", "")).lower()
        method = str(sex_json.get("method", "")).lower()
        if confidence == "user" or "manual" in method:
            human_slot = int(sex_json["male_fly"])

    auto_slot = None
    if sex_meta:
        meta = json.loads(sex_meta) if isinstance(sex_meta, (str, bytes)) else dict(sex_meta)
        auto_slot = int(meta["male_slot"])

    if human_slot is not None and auto_slot is not None:
        if human_slot != auto_slot:
            raise ValueError(
                f"sexing sources disagree on the male mask slot: "
                f"sex.json male_fly={human_slot} (human-confirmed) vs sam3 "
                f"sex_meta male_slot={auto_slot} (mask-area vote) — "
                f"refusing to guess")
        male_slot = human_slot
    elif human_slot is not None:
        male_slot = human_slot
    elif auto_slot is not None:
        male_slot = auto_slot
    else:
        return _DEFAULT_MALE_SLOT, _DEFAULT_FEMALE_SLOT
    return male_slot, 1 - male_slot


def _resolve_session_bout(session_dir, sam3_root, recording: str, clip_len: int,
                          tol: int = 1) -> tuple:
    """Map a combined-h5 exemplar onto its session bout: `start_frame` from
    the recording's ``courtship_bout_summary.csv``, but the SAM3
    directory name by SCANNING ``sam3_root`` for the one whose own mask
    frame count matches — never by deriving it from the CSV's ``bout_idx``.

    Returns ``(bout_dir_name, start_frame)``.

    SAM3 bout directory numbering is NOT guaranteed to match the CSV's
    `bout_idx` — verified on the ``Video_recordings`` SAM3 tree (round 5):
    CSV bout_idx 5 (n=778, the exemplar) lived in `bout_00004`, while
    `bout_00005` held CSV bout_idx 6 (n=569); almost certainly parallel
    SAM3 shards writing their outputs in completion order. The processed
    tree this function is now pointed at (round 6:
    `<processed_root>/<recording>/{courtship_bout_summary.csv,sam3_masks}`)
    verifiably does NOT have this permutation (dir `bout_0000N` <-> CSV
    `bout_idx N` exactly, for every row) — but matching by mask frame count
    is kept regardless, since it is correct whether or not the numbering
    happens to line up, and the first three rows of the round-5 permutation
    also lined up before row 4 broke it. Do NOT "simplify" this back to a
    `bout_idx`-derived directory name.

    ``recording`` (e.g. from ``recording_of(ex)``, which reads
    ``info/fly_ids``) carries a trailing ``_flyN`` suffix that the CSV's
    ``fly_id`` column never has, so the suffix is stripped before matching
    and compared by EQUALITY, not substring containment (round-4 fix: the
    prior ``fly_id`` substring-of-``recording`` check ran backwards — the
    CSV id is a substring of the suffixed recording id, never the reverse,
    so it matched zero rows on every real run). A recording id with no
    ``_fly`` suffix is left unchanged by the strip, so the same code path
    handles both forms without a conditional.

    Reads ONLY each npz's tiny ``valid`` array (shape ``(2, 7, N)``) to get
    its frame count ``N`` — never ``packed`` (``(2, 7, N, 448, 242)``);
    decompressing all of them would be very slow.
    """
    import pandas as pd

    csv_path = Path(session_dir) / "courtship_bout_summary.csv"
    df = pd.read_csv(csv_path)
    base = str(recording).rsplit("_fly", 1)[0]
    rows = df[df["fly_id"].astype(str) == base]
    n = rows["end_frame"] - rows["start_frame"] + 1
    matches = rows[(n - int(clip_len)).abs() <= tol]
    if len(matches) == 0:
        raise ValueError(
            f"no bout in {csv_path} for recording {recording!r} with "
            f"clip_len={clip_len} (tol={tol})")
    if len(matches) > 1:
        idxs = sorted(int(v) for v in matches["bout_idx"])
        raise ValueError(
            f"ambiguous bout match in {csv_path} for recording {recording!r} "
            f"with clip_len={clip_len} (tol={tol}): candidate bout_idx "
            f"{idxs} — refusing to guess")
    start_frame = int(matches.iloc[0]["start_frame"])

    sam3_root = Path(sam3_root)
    counts = []
    for npz_path in sorted(sam3_root.glob("bout_*/sam3_masks.npz")):
        with np.load(npz_path) as z:
            counts.append((npz_path.parent.name, int(z["valid"].shape[-1])))
    dir_matches = [(d, c) for d, c in counts if abs(c - int(clip_len)) <= tol]
    if len(dir_matches) == 0:
        raise ValueError(
            f"no sam3 bout dir under {sam3_root} with mask frame count "
            f"matching clip_len={clip_len} (tol={tol}); available: {counts}")
    if len(dir_matches) > 1:
        dirs = sorted(d for d, _ in dir_matches)
        raise ValueError(
            f"ambiguous sam3 bout dir under {sam3_root} for clip_len="
            f"{clip_len} (tol={tol}): candidate dirs {dirs} — refusing to guess")
    bout_dir = dir_matches[0][0]
    return (bout_dir, start_frame)


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
    ap.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT,
                    help="root of the processed pose/sam3 tree "
                         "(<processed-root>/<recording>/{courtship_bout_summary.csv,"
                         "sam3_masks,pose/bouts}); kp_data in the combined h5 is "
                         "re-centred and must never be projected")
    ap.add_argument("--sam3-bout", default=None,
                    help="explicit SAM3 bout dir (e.g. bout_00006); OVERRIDES "
                         "the automatic session-CSV mapping when given")
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

    # --- resolve the exemplar's SAM3 bout + frame offset (round-3 finding) -
    # Ordinal position in the combined h5 does NOT match a bout_idx (round 5:
    # verified as an outright PERMUTATION on the Video_recordings SAM3 tree),
    # so the prior hardcoded --sam3-bout="bout_00006" + video_frame_offset=0
    # silently overlaid an UNRELATED bout's masks starting at the wrong video
    # frame — a scientific-correctness defect (panel A didn't show its own
    # traces' bout; panel I's target_pitch triangulated a different bout's
    # female). `--sam3-bout` stays an explicit override for a user who wants
    # to force a specific bout; otherwise this is resolved against the
    # PROCESSED tree (round 6: `kp_data`/masks under `--session`/`--sam3-root`
    # cannot be used at all -- see `DEFAULT_PROCESSED_ROOT`), whose
    # `courtship_bout_summary.csv` refuses an ambiguous match rather than
    # silently picking one. `recording_base` strips the `_flyN` suffix
    # `_resolve_session_bout` would otherwise strip internally, because it is
    # ALSO needed here to build the processed-tree recording directory path
    # (which, unlike the CSV's `fly_id` column, is not itself suffixed).
    recording_base = str(recording_of(ex)).rsplit("_fly", 1)[0]
    processed_recording_dir = Path(args.processed_root) / recording_base
    processed_sam3_root = processed_recording_dir / "sam3_masks"
    # Round 7: key0/key1's processed fly0/fly1 directory is NOT positionally
    # fixed either (same "don't assume, derive it" lesson as round 5's sam3
    # bout numbering) -- resolve each from its OWN fly_id suffix rather than
    # assuming key0 -> fly0 / key1 -> fly1. Verified inverted for the
    # exemplar: key0=bout_183 is `_fly1`, key1=bout_182 is `_fly0`.
    key0_fly_id = fly_ids[bout_keys.index(ex["key0"])]
    key1_fly_id = fly_ids[bout_keys.index(ex["key1"])]
    key0_dir = _processed_fly_dir(key0_fly_id)
    key1_dir = _processed_fly_dir(key1_fly_id)
    print(f"exemplar fly dirs -> key0={key0_dir} key1={key1_dir}")
    if args.sam3_bout is not None:
        sam3_bout, video_frame_offset = args.sam3_bout, 0
        print(f"sam3 bout mapping OVERRIDDEN by --sam3-bout={sam3_bout!r} "
              f"(video_frame_offset=0)")
    else:
        sam3_bout = video_frame_offset = None
        try:
            sam3_bout, video_frame_offset = _resolve_session_bout(
                processed_recording_dir, processed_sam3_root,
                recording_of(ex), T)
            print(f"exemplar -> {sam3_bout} @ start_frame {video_frame_offset}")
        except Exception as e:                   # noqa: BLE001 - report, don't die
            skipped.append(
                f"sam3 bout MAPPING failed, not absent data "
                f"({type(e).__name__}: {e}) -- video strip and "
                f"male-pitch/target_pitch will also be skipped")

    # --- resolve SAM3 camera axis + male/female mask slots (rounds 10-11) --
    # Both the video overlay and the female-COM triangulation read the SAME
    # sam3_masks.npz's own `cameras`/`sex_meta` arrays (plus this bout's
    # human-reviewed `pose/bouts/<bout>/sex.json`), so this is resolved
    # ONCE and shared; a failure here is its own declared skip, and both
    # consuming blocks guard on the sentinel the same way they already guard
    # on `sam3_bout is None`.
    cam_idx = cam_idx_source = None
    male_slot = female_slot = None
    sex_slot_source = None
    npz_cameras = None
    if sam3_bout is not None:
        try:
            _sam3_npz_path = processed_sam3_root / sam3_bout / "sam3_masks.npz"
            with np.load(_sam3_npz_path, allow_pickle=True) as _z:
                npz_cameras = _z["cameras"] if "cameras" in _z.files else None
                npz_sex_meta = (_z["sex_meta"].item()
                               if "sex_meta" in _z.files else None)
            calib_dir_for_cam = Path(args.session) / "calibration"
            cam_idx, cam_idx_source = _resolve_sam3_camera_index(
                npz_cameras, calib_dir_for_cam, args.cam)
            print(f"sam3 camera index: {cam_idx} (source: {cam_idx_source})")

            # Round 11 finding B: prefer the HUMAN-CONFIRMED sex.json over
            # the sam3 npz's own sex_meta (a mask-area VOTE -- weak on this
            # exemplar, agreement 0.429), and cross-check rather than
            # silently trust either.
            _sex_json_path = (processed_recording_dir / "pose" / "bouts"
                              / sam3_bout / "sex.json")
            npz_sex_json = (json.loads(_sex_json_path.read_text())
                            if _sex_json_path.exists() else None)
            if npz_sex_meta is None and npz_sex_json is None:
                print("sam3 sex_meta and sex.json both absent; using "
                      "default male/female mask slots (1, 0)")
            male_slot, female_slot = _resolve_male_female_slots(
                npz_sex_meta, npz_sex_json)
            sex_slot_source = ("sex.json (human-confirmed)"
                               if (npz_sex_json
                                   and str(npz_sex_json.get("confidence", "")).lower() == "user")
                               else ("sex_meta (mask-area vote)" if npz_sex_meta
                                    else "default"))
            print(f"sam3 mask slots -> male={male_slot} female={female_slot} "
                  f"(source: {sex_slot_source})")
        except Exception as e:                   # noqa: BLE001 - report, don't die
            skipped.append(
                f"sam3 camera/slot resolution ({type(e).__name__}: {e}) -- "
                f"video strip and male-pitch/target_pitch will also be skipped")

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

    # Floor-corrected (round 9 finding): the courtship arms (pulse_z/sine_z,
    # via r["com_z"]) already subtract each bout's own floor
    # (utils.locomotion.compute_com_height); the free-running arm must use
    # the SAME estimator or panel G compares two different quantities (was:
    # raw z ~0.25 vs courtship's floor-corrected ~0.127, overstating the
    # gap ~2x). See `_free_running_com_z`.
    walking_z = np.zeros(0)
    try:
        walking_z = _free_running_com_z(args.free_run_h5)
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
    #
    # Round 6: keypoints and the crop centre come from the PROCESSED tree's
    # `kp3d.npz` (true DLT world frame), never the combined h5's `kp_data`
    # (body-model-rescaled/re-centred; measured Scutellum uv (13.6, 426.6) --
    # the frame's bottom-left corner, which is why the centred crop showed
    # empty chamber). The raw mp4 + camera calibration still come from
    # `--session` (the processed tree has neither).
    kp3d_source = ""
    video_frame_offset_raw = video_frame_offset
    video_sync_correction = 0
    video_frames = []
    try:
        if sam3_bout is None:
            raise RuntimeError(
                "no resolved sam3 bout (see the sam3 bout mapping skip above)")
        if cam_idx is None or male_slot is None:
            raise RuntimeError(
                "no resolved sam3 camera index / mask slots (see the sam3 "
                "camera/slot resolution skip above)")
        import matplotlib.pyplot as plt
        from utils.sam3_female_com import unpack_sam3_masks_for_frames

        calib_dir = Path(args.session) / "calibration"
        dlt_csv = calib_dir / f"{args.cam}_dlt.csv"
        mp4 = Path(args.session) / f"{args.cam}.mp4"
        sam3_npz = processed_sam3_root / sam3_bout / "sam3_masks.npz"
        pose_bout_dir = processed_recording_dir / "pose" / "bouts" / sam3_bout
        # key0/key1 -> fly0/fly1 is per-exemplar, resolved above (round 7);
        # never hardcode fly0=male here.
        male_kp3d = _load_kp3d(pose_bout_dir / key0_dir / "kp3d.npz",
                               expected_n_kp=len(kp_names))
        female_kp3d = _load_kp3d(pose_bout_dir / key1_dir / "kp3d.npz",
                                 expected_n_kp=len(kp_names))
        kp3d_source = str(pose_bout_dir)
        dlt = cfp._dlt_load(dlt_csv)
        # cam_idx / male_slot / female_slot resolved once, shared with the
        # pitch block below (round 10) -- never re-derive via glob order or
        # a hardcoded [1, 0] here.
        #
        # Round 11 finding A: `panel_video_strip_with_kp` reads the mp4 at
        # RAW POSITION `fidx + video_frame_offset` -- using the CANONICAL
        # slot directly as a raw position assumes no camera ever dropped a
        # frame before this bout. Convert once: masks/kp3d are indexed by
        # canonical slot (`video_frame_offset` = this bout's first slot);
        # the raw mp4 position that actually delivers that slot can differ
        # when `sync_plan.json` reports interior drops. A single scalar
        # offset is only valid because this recording's one drop sits at
        # slot 21, long before any bout in it -- so the correction is
        # constant across this bout's whole frame range; see
        # `_slot_to_raw_frame`.
        from jarvis_jax.predict.synced_reader import load_plan
        sync_plan = load_plan(args.session)
        video_frame_offset_raw, video_sync_correction_applied = _slot_to_raw_frame(
            sync_plan, args.cam, video_frame_offset)
        video_sync_correction = video_frame_offset - video_frame_offset_raw
        if video_sync_correction_applied:
            print(f"video sync: canonical slot {video_frame_offset} -> raw "
                  f"mp4 position {video_frame_offset_raw} for {args.cam} "
                  f"({video_sync_correction} frame(s) corrected)")
        else:
            print(f"video sync: no reindexing applied for {args.cam} "
                  f"(sync_plan.json absent or status != 'reindex')")
        vidx = np.linspace(0, T - 1, args.n_video, dtype=int)
        masks = unpack_sam3_masks_for_frames(
            sam3_npz, cam_idx, fly_indices=[male_slot, female_slot],
            frame_indices=[int(f) for f in vidx])
        figv, axv = plt.subplots(1, args.n_video, figsize=(args.n_video * 2, 2), dpi=200)
        axv = np.atleast_1d(axv)
        # A fixed roi is a crop copied from whatever recording it was tuned
        # on; a per-frame crop centred on the pair's Scutellum midpoint is
        # correct for ANY recording. --roi, when explicitly given, overrides
        # this and forces the fixed window instead (Finding 2).
        if args.roi is not None:
            roi_kwargs = {"roi": tuple(args.roi)}
        else:
            center_xyz = _pair_center_xyz(
                male_kp3d, female_kp3d, kp_names.index("Scutellum"), T)
            roi_kwargs = {"center_xyz": center_xyz, "crop_wh": tuple(args.crop_wh)}
        cfp.panel_video_strip_with_kp(
            list(axv), mp4, vidx, kp_xyz_per_frame=male_kp3d, kp_names=kp_names,
            dlt_coeffs=dlt, fs=fs, kp_scale=args.kp_scale,
            video_frame_offset=video_frame_offset_raw, masks_per_fly=masks,
            kp_xyz_fly1_per_frame=female_kp3d,
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
    #
    # Round 6: `scut` (the male's Scutellum, used against the SAM3-
    # triangulated `female` COM below, which IS in world frame) must come
    # from the processed tree's `kp3d.npz`, not the combined h5's `kp_data`
    # — same re-centering defect as the video block above, and for the same
    # reason: mixing a re-centred position against a world-frame one puts
    # `target_pitch` in no coherent frame at all.
    male_pitch = target_pitch = np.zeros(0)
    try:
        if sam3_bout is None:
            raise RuntimeError(
                "no resolved sam3 bout (see the sam3 bout mapping skip above)")
        if female_slot is None:
            raise RuntimeError(
                "no resolved sam3 camera index / mask slots (see the sam3 "
                "camera/slot resolution skip above)")
        from utils.sam3_female_com import triangulate_sam3_female_com

        # camera_order built from the SAME npz `cameras` array the shared
        # resolution block above read (round 10) -- an explicit order, not
        # triangulate_sam3_female_com's own glob-sorted default, since the
        # npz's camera axis is not guaranteed to match glob order either.
        camera_order = ([f"{c}_dlt.csv" for c in [str(x) for x in npz_cameras]]
                        if npz_cameras is not None else None)
        female = triangulate_sam3_female_com(
            str(processed_sam3_root / sam3_bout / "sam3_masks.npz"),
            str(Path(args.session) / "calibration"),
            fly_idx=female_slot, camera_order=camera_order,
            min_cams=2, verbose=False) / args.kp_scale
        qm = np.asarray(data[ex["key0"]]["qpos"])
        # key0's processed directory, resolved above (round 7) -- never
        # hardcode fly0=male here either.
        scut_all = _load_kp3d(
            processed_recording_dir / "pose" / "bouts" / sam3_bout / key0_dir / "kp3d.npz",
            expected_n_kp=len(kp_names))
        n = min(T, qm.shape[0], female.shape[0], len(scut_all))
        male_pitch = cfp.body_pitch_deg_from_quat(qm[:n, 3:7])
        scut = scut_all[:n, kp_names.index("Scutellum"), :]
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
                       "sam3_bout": sam3_bout or "",
                       "video_frame_offset": int(video_frame_offset or 0),
                       "kp3d_source": kp3d_source,
                       "kp3d_fly_dirs": f"key0={key0_dir};key1={key1_dir}",
                       "zheight_free_running": "com_z (floor-corrected)",
                       "sam3_camera_index": int(cam_idx) if cam_idx is not None else -1,
                       "sam3_camera_index_source": cam_idx_source or "",
                       "sam3_mask_slots": (f"male={male_slot};female={female_slot}"
                                          if male_slot is not None else ""),
                       "sex_slot_source": sex_slot_source or "",
                       "video_frame_offset_raw": int(video_frame_offset_raw or 0),
                       "video_sync_correction": int(video_sync_correction or 0),
                       "skipped": "; ".join(skipped)},
                 panels=panels)
    print(f"wrote {args.out}: {len(panels)} panels from {len(results)} pairs")
    for sk in skipped:
        print(f"  SKIPPED: {sk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
