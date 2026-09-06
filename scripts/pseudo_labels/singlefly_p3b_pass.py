#!/usr/bin/env python3
"""Single-fly mask-free P3b pass -> P3b-SHAPED bout dirs for the v2 pseudo-labels.

Spec: `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §3.4 (single-fly
data). The v2 pseudo-label set is starved of female-host windows; the
free-running Session11 recordings are 7 whole recordings of ONE female fly on
the SAME 7-camera rig and calibration as courtship, so a mask-free P3b pass
over them adds female-host framesets, a different arena/lighting, climbing and
resting poses, and natural "one fly only" windows.

WHAT THIS WRITES, and why it is shaped like the masked campaign's output:

    <out>/bouts/bout_<idx:05d>/fly0/kp2d.npz    kp2d (T,C,K,2) full-frame px,
                                                conf (T,C,K), cameras, kp_names
    <out>/bouts/bout_<idx:05d>/fly0/kp3d.npz    kp3d (T,K,3) world units,
                                                conf3d, conf3d_mvq_raw,
                                                kp_names, gates
    <out>/bouts/bout_<idx:05d>/mvq_meta.json

`scripts/pseudo_labels/extract_p3b_pseudolabels.py --num-animals 1
--no-identity-gate` then reads these dirs UNCHANGED: `pseudo_gates.admit_bout`
already accepts `store=None` (no SAM3 masks exist for these recordings, so the
containment gate cannot and must not run), and every other gate -- existence,
per-keypoint step, reprojection self-consistency -- is a property of the
written arrays alone.

HOW IT DIFFERS FROM `lift_masked_bout` (which this deliberately does not reuse):
that lifter hardcodes TWO typed slots (female fly0, male fly1) and takes its
window centres from SAM3 mask centroids. Here there is one fly, no mask, and
the window centre comes from CenterDetect exactly as in the mask-free coarse
pass -- `coarse_centres.lift_peaks_to_centres(..., max_animals=1)` ->
`cluster_centres` -> `plan_windows`, and a frame whose peaks lift to nothing
REUSES the previous centre (`centre_source`; see `coarse_track`'s docstring
for why a zero centre would be worse than a miss). The slot is read with
`MVQRunner.read_typed(want_sex=SEX_UNKNOWN)`, the SINGLE-FLY rule: whichever
typed slot exists, REPORTING its sex rather than assuming one.

`coarse_track.coarse_pass` runs that same loop but keeps only kp3d/centroid --
no kp2d, no per-view visibility -- and the P3b bout-dir format (and the
extractor's reprojection gate, and the export's reprojected 2D labels) needs
both, so the accumulate/flush loop is repeated here over the FULL `slot_read`
payload. Everything numeric (`windows`, `infer`, `read_typed`, `to_pipeline`,
`lift_peaks_to_centres`, `cluster_centres`, `plan_windows`) is the shipped
code, and the frame IO is `coarse_pass_mvq.SlotReader` -- the threaded
forward-only decoder, not a per-frame seek.

IDENTITY. There is nothing to identify: one fly, one slot. `mvq_meta.json`
says `identity_resolved: "single_typed"` and `per_frame.identity_source`
`["single_typed"] * T`, which is NEITHER of the two shipped modes -- so the
extractor's identity gate (which admits only `"mask"`) rejects every frame
unless it is run with `--no-identity-gate`, exactly as intended. The Stage-B
`gates` string stamped into kp3d.npz has no vocabulary for a single-fly read,
so the runner is built with `identity="sex"` (the closest shipped mode, and
the one whose typed slots the read actually used) and the meta names the real
rule beside it. `collapsed` is all-zero and `containment` is false, both by
construction: a single fly has no second slot to collapse against and no mask
to be contained by.

ORDER DISCIPLINE (CLAUDE.md). The written keypoint axis is
`cfg.model.KP_NAMES` (`--anatomy`), permuted from the model's own detector
order BY NAME inside `MVQRunner.to_pipeline` and checked against the rigid
EyeL-EyeR spacing. The camera axis is the recording config's canonical order,
which `MVQRunner` asserts equals the calibration glob order; the contact sheet
picks its cameras BY NAME out of the npz's own `cameras` array.

THREE MODES:

  (pass, the default) lift the given spans and write the bout dirs
  --scan            CenterDetect-only trackability probe of candidate spans,
                    to choose them before paying for the full pass
  --contact-sheet   render an already-written bout dir (no model, no GPU)

EXPECTATION for the contact sheet, stated before it is generated (CLAUDE.md):
ONE fly per crop, with the keypoints sitting ON her body -- head points at the
head end, abdomen points at the abdomen end, leg points on the legs -- at the
right SCALE (a fly is ~2.4 mm = 24 units from Antenna_Base to Abd_tip), in
BOTH cameras, and NO second skeleton anywhere in the crop. A second marker set
would be the phantom fly this pass exists to rule out; a small huddled cloud
near the crop centre would be the clip_A scale collapse
(`docs/benchmark/2026-09-mvq/p3b-notes.md`, "P3b on single-fly recordings").

Run (GPU node; the CUDA preamble is the other jax jobs'):

    module load cuda/12.9.1
    export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
    unset LD_LIBRARY_PATH JAX_PLATFORMS
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=third_party/jarvis_jax:. \\
    python scripts/pseudo_labels/singlefly_p3b_pass.py \\
      --session-dir .../free_running/Session11/2026_03_03_14_32_16 \\
      --run  .../jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/final \\
      --centerdetect .../jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004 \\
      --bout 25000:27000 --bout 90200:92200 --num-animals 1 \\
      --out OutFiles/v2_singlefly/2026_03_03_14_32_16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(ROOT, "third_party", "jarvis_jax"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# `scripts/` goes LAST, and only for the sibling-module import
# (`coarse_pass_mvq.SlotReader`). Ahead of the repo root it would bind
# `scripts/viz` as `viz`, and `viz.core.colors` -- which the contact sheet
# needs -- would not exist: the exact failure that killed Session0 bout 28
# after a 36-minute STAC solve (test_lift_masked_bout.py::
# test_import_keypoint_groups_recovers_when_pythonpath_holds_the_repo_root).
_SCRIPTS = os.path.join(ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)

DEF_RECORDING_CFG = os.path.join(ROOT, "configs", "recording", "free_running_session11.yaml")
DEF_ANATOMY_CFG = os.path.join(ROOT, "configs", "anatomy", "v1.yaml")

# `mvq_meta.json`'s identity name for this route. Deliberately NOT one of
# `lift_mvq.IDENTITY_MODES`: the extractor's identity gate admits only "mask",
# so a single-fly bout can never be admitted by accident.
SINGLE_TYPED = "single_typed"

CENTRE_DETECTED, CENTRE_REUSED, CENTRE_NONE = 0, 1, 2   # == coarse_track's codes


# --------------------------------------------------------------------- CLI bits
def parse_span(s):
    """`"start:end"` -> `(start, end)` with end EXCLUSIVE, or ValueError."""
    parts = str(s).split(":")
    if len(parts) != 2:
        raise ValueError(f"--bout/--scan takes 'start:end' (end exclusive), got {s!r}")
    a, b = int(parts[0]), int(parts[1])
    if b <= a:
        raise ValueError(f"empty span {s!r}: end must be greater than start")
    return a, b


def default_cameras(path=DEF_RECORDING_CFG):
    """`recording.cameras` from a recording config, without hydra resolution
    (the rest of the file is `${...}` interpolations this script cannot and
    need not resolve). Same reader as `scripts/coarse_pass_mvq.py`."""
    from coarse_pass_mvq import default_cameras as _dc
    return _dc(path)


def link_cameras(session_dir, link_dir, cameras):
    """`<link_dir>/<Cam>.mp4` symlinks for a recording whose mp4s carry a
    suffix (Clip's `Cam2012630_frames_1161383_1162303.mp4`), because every
    reader here resolves a camera's video as `<session_dir>/<Cam>.mp4`.

    `calibration/` and `sync_plan.json` are linked too when the source has
    them, so the link dir can be used as the session dir end to end -- the
    written `mvq_meta.json` names it as `session_dir`, and the pseudo-label
    extractor reads BOTH the videos and `<session_dir>/calibration` from it.

    Raises:
        ValueError: a camera name matches no video, or more than one -- two
            videos mapping to the same `<Cam>.mp4` would silently give that
            camera whichever the glob happened to sort first, which is the
            camera-order trap one level down (the frames would be a different
            span of time from the calibration's camera).
    """
    session_dir, link_dir = str(session_dir), str(link_dir)
    os.makedirs(link_dir, exist_ok=True)
    made = []
    for cam in cameras:
        hits = sorted(glob.glob(os.path.join(session_dir, f"{cam}*.mp4")))
        if not hits:
            raise ValueError(f"--link-cameras: no video matching {cam}*.mp4 in {session_dir}")
        if len(hits) > 1:
            raise ValueError(
                f"--link-cameras: {len(hits)} videos map to camera {cam}: "
                f"{[os.path.basename(h) for h in hits]} -- refusing to pick one, since "
                f"the reader would then read a different span of time than the rest of "
                f"the cameras")
        dst = os.path.join(link_dir, f"{cam}.mp4")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(hits[0]), dst)
        made.append(dst)
    for extra in ("calibration", "sync_plan.json"):
        src = os.path.join(session_dir, extra)
        dst = os.path.join(link_dir, extra)
        if os.path.exists(src) and not (os.path.islink(dst) or os.path.exists(dst)):
            os.symlink(os.path.abspath(src), dst)
    return made


# ------------------------------------------------------------------- the pass
def _bout_is_current(out_dir, gates_string):
    """True when `fly0/kp3d.npz` carries `gates_string` AND `mvq_meta.json`
    exists -- the same staleness contract as `lift_mvq.bout_lift_is_current`,
    minus its `sex.json` requirement (this route writes none: `sex.json` is the
    two-fly `canonicalize_bout` handshake, and these dirs never enter
    `run_bout.py`). The meta is written LAST, so its presence is what proves a
    bout finished rather than dying between the npz and the meta.
    """
    p = os.path.join(str(out_dir), "fly0", "kp3d.npz")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return False
    try:
        with np.load(p) as z:
            if "gates" not in z.files or str(z["gates"]) != str(gates_string):
                return False
    except (OSError, ValueError):
        return False
    m = os.path.join(str(out_dir), "mvq_meta.json")
    return os.path.exists(m) and os.path.getsize(m) > 0


def _frame_centre(detector, runner, imgs, prev_centres, *, min_views, max_resid_px,
                  merge_dist_units):
    """One frame's window plan: CenterDetect peaks -> at most ONE 3D centre ->
    window centres, with the coarse pass's reuse rule.

    Returns `(window_centres (W,3), centre_source, new_prev)`. `max_animals=1`
    is what makes this the single-fly path: a second CenterDetect candidate is
    never lifted, so a duplicate window on the one real fly (seen once on
    fr_b4 with `max_animals=2`, p3b-notes.md) cannot reach the output.
    """
    from jarvis_jax.tracking.coarse_centres import (cluster_centres, lift_peaks_to_centres,
                                                    plan_windows)
    peaks, scores = detector.peaks(imgs)
    centres, _n_views, _score = lift_peaks_to_centres(
        peaks, scores, runner.cam_mats, min_views=min_views,
        max_resid_px=max_resid_px, max_animals=1)
    centres = cluster_centres(centres)
    wc, _assign = plan_windows(centres, merge_dist_units=merge_dist_units)
    if wc.shape[0]:
        return wc, CENTRE_DETECTED, wc
    if prev_centres is not None and prev_centres.shape[0]:
        return prev_centres, CENTRE_REUSED, prev_centres
    return np.zeros((0, 3), np.float32), CENTRE_NONE, prev_centres


def lift_singlefly_bout(runner, detector, frames_iter, *, frames, out_dir, model_names,
                        min_views=3, max_resid_px=25.0, merge_dist_units=30.0,
                        fly_sex=None, meta_extra=None, progress_every=0, verbose=True):
    """Lift one span into `<out_dir>/fly0/{kp2d,kp3d}.npz` + `mvq_meta.json`.

    Args:
        runner: `MVQRunner` (or the tests' fake with the same
            `windows`/`infer`/`read_typed`/`to_pipeline`/`gates_string`
            surface), built with `identity="sex"`, `containment=False`.
        detector: anything with `.peaks(frames) -> ((C,2,2) px, (C,2) scores)`
            -- `coarse_centres.CenterDetector` in production. `peaks[c]` must
            be the SAME camera as `runner.cam_mats[c]` (canonical order).
        frames_iter: iterable of `(frames (C,H,W,3) uint8 RGB, present (C,))`,
            one item per frame of `frames`, in the runner's camera order.
        frames: the ABSOLUTE video frame indices (canonical slots) of the span.
        model_names: `cfg.model.KP_NAMES` -- the written keypoint order.
        fly_sex: "female"/"male" to record as ground truth, or None to take
            the majority of the typed slots the model actually read (recorded
            either way, with the read's own agreement fraction beside it).

    Returns the per-frame bookkeeping plus the `single_fly_check` stats block.
    """
    from jarvis_jax.tracking.lift_mvq import (COLLAPSE_DIST_UNITS, _check_eye_invariant,
                                              concat_windows, mvq_gate_string)
    from jarvis_jax.train.matching import SEX_UNKNOWN, SLOT_FEMALE, SLOT_MALE
    from jarvis_jax.tracking.resume import atomic_save_json, atomic_save_npz

    model_names = [str(n) for n in model_names]
    frames = [int(f) for f in frames]
    T, C, K, I = len(frames), len(runner.cameras), runner.K, runner.I
    out_dir = str(out_dir)

    kp3d = np.full((T, K, 3), np.nan, np.float32)
    kp2d = np.full((T, C, K, 2), np.nan, np.float32)
    vis = np.zeros((T, C, K), np.float32)        # 0 => the pipeline's conf gate drops it
    conf_raw = np.zeros((T, K), np.float32)
    exist = np.full(T, np.nan, np.float32)
    sex_prob = np.full(T, np.nan, np.float32)
    slot = np.full(T, -1, np.int8)
    all_exist = np.full((T, I), np.nan, np.float32)
    n_windows = np.zeros(T, np.int8)
    centre_source = np.full(T, CENTRE_NONE, np.int8)
    # THE phantom-fly quantity. When BOTH typed slots clear the threshold in
    # the window the fly was read from, this is the median per-keypoint 3D
    # distance between them: ~0 means the two slots are the SAME animal read
    # twice (the model hedging on sex), while anything past
    # `COLLAPSE_DIST_UNITS` (3 units = 0.3 mm, `pick_typed_pair`'s own rule)
    # means the second slot is asserting a fly SOMEWHERE ELSE -- which on a
    # single-fly recording would be a hallucination. Counting frames with two
    # live slots alone cannot tell those apart.
    second_dist = np.full(T, np.nan, np.float32)

    pend, batch, rows = [], [], 0
    t_start = time.time()

    def _flush():
        nonlocal pend, batch, rows
        if not pend:
            return
        out = runner.infer(concat_windows(batch))
        for t, off, nb in pend:
            # `max_animals=1` means one window per frame, so slot existences
            # of window `off` ARE the frame's; the loop below is still written
            # over `nb` so a future merge rule cannot silently read window 0.
            all_exist[t] = out["exist"][off]
            best, best_b = None, -1
            for b in range(off, off + nb):
                r = runner.read_typed(out, b, want_sex=SEX_UNKNOWN)
                if r is not None and (best is None or r["exist"] > best["exist"]):
                    best, best_b = r, b
            if best is None:
                continue
            ef = float(out["exist"][best_b, SLOT_FEMALE])
            em = float(out["exist"][best_b, SLOT_MALE])
            if min(ef, em) >= runner.exist_thresh:
                d = np.linalg.norm(out["kp3d"][best_b, SLOT_FEMALE]
                                   - out["kp3d"][best_b, SLOT_MALE], axis=-1)
                with np.errstate(invalid="ignore"):
                    second_dist[t] = np.nanmedian(d)
            kp3d[t] = best["kp3d"]
            kp2d[t] = best["kp2d"]
            vis[t] = best["vis"]
            conf_raw[t] = best["conf_raw"]
            exist[t] = best["exist"]
            sex_prob[t] = best["sex_prob"]
            slot[t] = best["slot"]
        pend, batch, rows = [], [], 0

    prev_centres, n_seen = None, 0
    for t, (imgs, present) in enumerate(frames_iter):
        if t >= T:
            raise ValueError(f"frames_iter yielded more than the {T} frames of this span")
        n_seen = t + 1
        imgs = np.asarray(imgs)
        wc, src, prev_centres = _frame_centre(
            detector, runner, imgs, prev_centres, min_views=min_views,
            max_resid_px=max_resid_px, merge_dist_units=merge_dist_units)
        centre_source[t] = src
        if wc.shape[0] == 0:
            continue
        if rows + wc.shape[0] > runner.batch:
            _flush()
        batch.append(runner.windows(imgs, present, wc))
        pend.append((t, rows, int(wc.shape[0])))
        rows += int(wc.shape[0])
        n_windows[t] = int(wc.shape[0])
        if progress_every and (t + 1) % int(progress_every) == 0:
            el = time.time() - t_start
            print(f"[singlefly] frame {t + 1}/{T}  {(t + 1) / max(el, 1e-9):.1f} frames/s  "
                  f"written {int((slot >= 0).sum())}", flush=True)
    _flush()
    if n_seen != T:
        raise ValueError(f"frames_iter yielded {n_seen} frames but the span has {T}; "
                         f"kp3d.npz must have exactly the span's T or every downstream "
                         f"stage silently mis-indexes time")

    # ---- the single-fly field-check quantities (p3b-notes.md's table)
    wrote = slot >= 0
    live = np.isfinite(all_exist)
    windowed = live.any(axis=1)              # frames the model was actually run on
    over = live & (np.nan_to_num(all_exist, nan=-1.0) >= 0.5)
    n_windowed = int(windowed.sum())
    exactly_one = (float((over.sum(axis=1)[windowed] == 1).mean()) if n_windowed else None)
    # descending slot existences; -1 stands in for a slot the frame has no
    # value for, and is dropped rather than averaged in as a zero
    srt = np.sort(np.where(live, all_exist, -1.0), axis=1)[:, ::-1]
    second = srt[windowed, 1] if n_windowed else np.zeros(0)
    second = second[second >= 0.0]
    # the same statistic over the TWO TYPED slots only (1 female, 2 male) --
    # slots 0 (prompted) and 3 (other) are never read, so counting them makes
    # the number stricter than the risk it stands for
    typed = all_exist[:, [SLOT_FEMALE, SLOT_MALE]]
    typed_over = np.isfinite(typed) & (np.nan_to_num(typed, nan=-1.0) >= 0.5)
    typed_second = (np.sort(np.where(np.isfinite(typed), typed, -1.0), axis=1)[windowed, 0]
                    if n_windowed else np.zeros(0))
    typed_second = typed_second[typed_second >= 0.0]
    step = np.linalg.norm(np.diff(kp3d, axis=0), axis=-1)     # (T-1,K) units
    far = np.nan_to_num(second_dist, nan=0.0) >= COLLAPSE_DIST_UNITS
    check = {
        "n_frames": int(T),
        "n_windowed": n_windowed,
        "frac_written": round(float(wrote.mean()), 4) if T else None,
        "frac_centre_detected": round(float((centre_source == CENTRE_DETECTED).mean()), 4),
        "frac_centre_reused": round(float((centre_source == CENTRE_REUSED).mean()), 4),
        "frac_no_centre": round(float((centre_source == CENTRE_NONE).mean()), 4),
        "frac_exactly_one_slot_over_0.5": None if exactly_one is None else round(exactly_one, 4),
        "frac_exactly_one_typed_slot_over_0.5": (
            round(float((typed_over.sum(1)[windowed] == 1).mean()), 4) if n_windowed else None),
        "second_slot_exist_mean": (round(float(second.mean()), 4) if second.size else None),
        "second_slot_exist_max": (round(float(second.max()), 4) if second.size else None),
        "second_typed_slot_exist_mean": (round(float(typed_second.mean()), 4)
                                         if typed_second.size else None),
        "exist_mean_per_slot": [None if not np.isfinite(all_exist[:, s]).any()
                                else round(float(np.nanmean(all_exist[:, s])), 4)
                                for s in range(I)],
        # both typed slots live: the same fly read twice, or a phantom?
        "n_both_typed_slots": int(np.isfinite(second_dist).sum()),
        "second_typed_slot_median_dist_units": (round(float(np.nanmedian(second_dist)), 3)
                                                if np.isfinite(second_dist).any() else None),
        "n_second_typed_slot_beyond_collapse": int(far.sum()),
        "collapse_dist_units": float(COLLAPSE_DIST_UNITS),
        "exist_mean": (round(float(np.nanmean(exist)), 4) if wrote.any() else None),
        "median_step_units": (round(float(np.nanmedian(step)), 4)
                              if np.isfinite(step).any() else None),
        "p99_step_units": (round(float(np.nanpercentile(step, 99)), 4)
                           if np.isfinite(step).any() else None),
    }

    # ---- which sex the typed read reported, and what we record as truth
    n_f = int((slot[wrote] == SLOT_FEMALE).sum())
    n_m = int((slot[wrote] == SLOT_MALE).sum())
    read_sex = "female" if n_f >= n_m else "male"
    sex_read = {"n_female_slot": n_f, "n_male_slot": n_m,
                "frac_majority": round(max(n_f, n_m) / max(n_f + n_m, 1), 4),
                "mean_sex_prob_female": (round(float(np.nanmean(sex_prob)), 4)
                                         if wrote.any() else None),
                "read": read_sex}
    if fly_sex is None:
        fly_sex, sex_source = read_sex, "mvq_sex_head_majority"
    else:
        fly_sex, sex_source = str(fly_sex), "recording_ground_truth"
        if fly_sex != read_sex and verbose:
            print(f"[singlefly] NOTE {out_dir}: recorded sex {fly_sex!r} but the model's "
                  f"typed read is {read_sex!r} on {sex_read['frac_majority']:.1%} of "
                  f"written frames", flush=True)

    gates_string = mvq_gate_string(runner.checkpoint, step=runner.step,
                                   exist_thresh=runner.exist_thresh,
                                   identity=getattr(runner, "identity", "sex"),
                                   containment=False, window_pref=None)

    # ---- write, keypoint axis permuted BY NAME
    p = runner.to_pipeline(kp3d, kp2d, vis, conf_raw, model_names)
    if list(p["kp_names"]) != model_names:
        raise RuntimeError(
            f"to_pipeline returned keypoint order {list(p['kp_names'])[:4]}... which is not "
            f"cfg.model.KP_NAMES -- the export would read every landmark as a different "
            f"body part")
    _check_eye_invariant(kp3d, p["kp3d"], runner.kp_names, model_names)
    d = os.path.join(out_dir, "fly0")
    os.makedirs(d, exist_ok=True)
    atomic_save_npz(os.path.join(d, "kp2d.npz"), kp2d=p["kp2d"], conf=p["conf"],
                    cameras=p["cameras"], kp_names=p["kp_names"])
    atomic_save_npz(os.path.join(d, "kp3d.npz"), kp3d=p["kp3d"], conf3d=p["conf3d"],
                    conf3d_mvq_raw=p["conf3d_mvq_raw"], kp_names=p["kp_names"],
                    gates=np.asarray(gates_string))

    meta = {
        "checkpoint": runner.checkpoint,
        "step": runner.step_label,
        "gates": json.loads(gates_string),
        "exist_thresh": float(runner.exist_thresh),
        "merge_dist_units": float(merge_dist_units),
        "min_views": int(min_views),
        "max_resid_px": float(max_resid_px),
        "cameras": list(runner.cameras),
        "keypoint_names_mvq": list(runner.kp_names),
        "keypoint_names_written": model_names,
        # `identity` is what the gates string names (there is no single-fly
        # mode in `IDENTITY_MODES`); `identity_resolved` is the rule this pass
        # actually ran -- `read_typed(want_sex=-1)`, whichever typed slot
        # exists -- and it is what the extractor's identity gate sees.
        "identity": getattr(runner, "identity", "sex"),
        "identity_resolved": SINGLE_TYPED,
        "single_fly": True,
        "want_sex": -1,
        "window_pref": "any",
        "containment": False,
        "containment_report": {"enabled": False,
                               "reason": "single-fly pass: no SAM3 masks exist"},
        "fly_slots": {"fly0": int(np.bincount(slot[wrote].astype(int)).argmax())
                      if wrote.any() else -1},
        "fly_sex": {"fly0": fly_sex},
        "sex_source": sex_source,
        "sex_read": sex_read,
        "n_frames": int(T),
        "n_missing": {"fly0": int((~wrote).sum())},
        "n_no_centre": int((centre_source == CENTRE_NONE).sum()),
        "n_collapsed": {"fly0": 0},
        "collapsed_frac": 0.0,
        "single_fly_check": check,
        "per_frame": {
            # (F=1, T) -- the shape `pseudo_gates.existence_gate` reads
            "exist": [np.round(np.nan_to_num(exist, nan=-1.0), 4).tolist()],
            "sex_prob": [np.round(np.nan_to_num(sex_prob, nan=-1.0), 4).tolist()],
            "slot": [slot.astype(int).tolist()],
            "slot_used": [slot.astype(int).tolist()],
            # per FRAME, like the campaign's: a single fly has no second slot
            # to collapse against, so this is all-zero by construction.
            "collapsed": [0] * int(T),
            "identity_source": [SINGLE_TYPED] * int(T),
            "n_windows": n_windows.astype(int).tolist(),
            "centre_source": centre_source.astype(int).tolist(),
            "no_centre": (centre_source == CENTRE_NONE).astype(int).tolist(),
            "exist_all_slots": np.round(np.nan_to_num(all_exist, nan=-1.0), 4).tolist(),
            "second_typed_slot_dist_units": np.round(
                np.nan_to_num(second_dist, nan=-1.0), 3).tolist(),
        },
    }
    meta.update(meta_extra or {})
    atomic_save_json(os.path.join(out_dir, "mvq_meta.json"), meta)
    if verbose:
        el = time.time() - t_start
        print(f"[singlefly] {out_dir}: {T} frames, written {int(wrote.sum())} "
              f"({100 * float(wrote.mean()):.1f}%), sex {fly_sex} ({sex_source}), "
              f"exactly-one-slot {check['frac_exactly_one_slot_over_0.5']} "
              f"(typed {check['frac_exactly_one_typed_slot_over_0.5']}), "
              f"2nd-slot exist mean {check['second_slot_exist_mean']}, "
              f"2nd typed slot {check['n_both_typed_slots']} frames at median "
              f"{check['second_typed_slot_median_dist_units']} units "
              f"({check['n_second_typed_slot_beyond_collapse']} beyond collapse), "
              f"median step {check['median_step_units']} units, "
              f"centre detected/reused/none "
              f"{check['frac_centre_detected']}/{check['frac_centre_reused']}/"
              f"{check['frac_no_centre']}, {T / max(el, 1e-9):.1f} frames/s", flush=True)
    return {"out_dir": out_dir, "gates": gates_string, "kp3d_mvq": kp3d, "kp2d_mvq": kp2d,
            "vis": vis, "exist": exist, "sex_prob": sex_prob, "slot": slot,
            "centre_source": centre_source, "n_windows": n_windows,
            "single_fly_check": check, "sex_read": sex_read, "fly_sex": fly_sex,
            "meta": meta}


# ------------------------------------------------------------------- the scan
def scan_spans(session_dir, cameras, cam_mats, detector, spans, *, n_probe=20,
               min_views=3, max_resid_px=25.0):
    """CenterDetect-only trackability probe: for each `(start, end)` span,
    the fraction of `n_probe` evenly spaced frames whose peaks LIFT to a 3D
    centre. No mvq forward, so it costs one decode sweep per span.

    Controller rule 2026-09-05: a span below 0.90 here is not used as a bout.
    """
    from coarse_pass_mvq import SlotReader
    from jarvis_jax.predict.synced_reader import load_plan
    from jarvis_jax.tracking.coarse_centres import cluster_centres, lift_peaks_to_centres

    plan = load_plan(session_dir)
    out = []
    for a, b in spans:
        stride = max(1, (b - a) // int(n_probe))
        slots = list(range(a, b, stride))[:int(n_probe)]
        reader = SlotReader(session_dir, cameras, plan, start_slot=slots[0], stride=stride)
        got, npeak = 0, []
        try:
            for s in slots:
                imgs, present = reader(s)
                peaks, scores = detector.peaks(imgs)
                npeak.append(int(np.isfinite(peaks).all(-1).sum()))
                centres, _nv, _sc = lift_peaks_to_centres(
                    peaks, scores, cam_mats, min_views=min_views,
                    max_resid_px=max_resid_px, max_animals=1)
                got += int(cluster_centres(centres).shape[0] > 0)
        finally:
            reader.close()
        row = {"span": [int(a), int(b)], "n_probe": len(slots), "stride": int(stride),
               "frac_trackable": round(got / max(len(slots), 1), 4),
               "mean_peaks_per_frame": round(float(np.mean(npeak)), 2)}
        print(f"[scan] {a}:{b}  trackable {row['frac_trackable']:.2f} "
              f"({got}/{len(slots)} probe frames)  peaks/frame "
              f"{row['mean_peaks_per_frame']:.1f}", flush=True)
        out.append(row)
    return out


# ----------------------------------------------------------- the contact sheet
def contact_sheet(bout_dir, out_png, *, cameras=None, n_frames=12, cell=256,
                  session_dir=None, cols=6, worst=False, vis_min_frac=0.33):
    """Read an already-written bout dir back and draw its 2D keypoints on the
    video: `n_frames` frames wrapped `cols` per row-block, one row per camera
    within a block.

    Frames are chosen from the ones the pass WROTE and where the first chosen
    camera sees at least `vis_min_frac` of the keypoints (per-view visibility
    >= 0.5) -- which is the same "is anything visible" test
    `write_pseudo_export` applies before it writes a frameset, so the sheet
    shows what the export would carry. `worst=True` inverts it: the LOWEST
    mean-visibility written frames, which is where the stale-reused-window
    failure lives (existence ~1.0 with every per-view visibility below 0.5, on
    an empty crop) -- look at that sheet too, not only the flattering one.

    Nothing here is indexed positionally: the cameras are chosen BY NAME out
    of `kp2d.npz`'s own `cameras` array, and the keypoint colours and leg
    chains come from `viz.core.colors` over its `kp_names`. Frames are the
    ABSOLUTE slots `frame_start + t`, read through the same synced reader the
    pass used.
    """
    import cv2
    from jarvis_jax.predict.synced_reader import load_plan, read_window
    from viz.core.colors import PALETTE, keypoint_groups, leg_chains

    meta = json.load(open(os.path.join(bout_dir, "mvq_meta.json")))
    z = np.load(os.path.join(bout_dir, "fly0", "kp2d.npz"), allow_pickle=True)
    kp2d, conf = z["kp2d"], z["conf"]
    npz_cams = [str(c) for c in z["cameras"]]
    kp_names = [str(n) for n in z["kp_names"]]
    session_dir = session_dir or meta["session_dir"]
    cameras = list(cameras) if cameras else npz_cams[:2]
    for c in cameras:
        if c not in npz_cams:
            raise ValueError(f"camera {c!r} is not in {bout_dir}'s kp2d.npz "
                             f"({npz_cams}); cameras are resolved BY NAME")
    cam_idx = [npz_cams.index(c) for c in cameras]      # BY NAME, never positional
    groups = keypoint_groups(kp_names)
    colour = {i: PALETTE[g] for g, idxs in groups.items() for i in idxs}
    # the leg CHAINS (ThxCx -> ... -> TaTip), drawn as polylines: a lone blue
    # dot out in space is unreadable, while a leg that leaves the body along
    # its own chain is obviously a leg -- and a leg landing on the arena floor
    # is obviously wrong
    chains = leg_chains(kp_names)

    exist = np.asarray(meta["per_frame"]["exist"][0], np.float32)
    T = int(meta["n_frames"])
    frame_start = int(meta.get("frame_start", 0))
    # Frames the pass WROTE, ranked by how much the first chosen camera can
    # see. A crop with no visible keypoint says nothing about whether the
    # keypoints sit on the fly (and the export drops such a frameset), so the
    # default sheet skips those and `worst=True` shows only them.
    wrote = np.isfinite(kp2d[:, cam_idx[0], :, 0]).any(-1)
    seen = (conf[:, cam_idx[0]] >= 0.5).mean(-1)
    if worst:
        cand = np.flatnonzero(wrote)
        cand = cand[np.argsort(seen[cand])][:max(n_frames, 1)]
        pick = np.sort(cand)
    else:
        cand = np.flatnonzero(wrote & (seen >= vis_min_frac))
        pick = (cand[np.linspace(0, len(cand) - 1, min(n_frames, len(cand))).astype(int)]
                if cand.size else np.flatnonzero(wrote)[:n_frames])
    if not len(pick):
        pick = np.linspace(0, T - 1, n_frames).astype(int)
    print(f"[sheet] {bout_dir}: {int(wrote.sum())}/{T} frames written, "
          f"{int((wrote & (seen >= vis_min_frac)).sum())} with >= {vis_min_frac:.0%} of "
          f"{cameras[0]}'s keypoints visible; picked "
          f"{'the lowest-visibility' if worst else 'evenly spaced visible'} frames", flush=True)

    plan = load_plan(session_dir)
    cols = max(1, min(int(cols), len(pick)))
    nblk = int(np.ceil(len(pick) / cols))
    sheet = np.full((cell * nblk * len(cameras), cell * cols, 3), 20, np.uint8)
    for i, t in enumerate(pick):
        t = int(t)
        blk, col = divmod(i, cols)
        frames, present = next(iter(read_window(session_dir, cameras, plan,
                                                frame_start + t, 1)))
        for row, (cam, ci) in enumerate(zip(cameras, cam_idx)):
            if not present[row]:
                continue
            img = frames[row][:, :, ::-1].copy()          # RGB -> BGR for cv2
            uv = kp2d[t, ci]
            ok = np.isfinite(uv).all(-1)
            if ok.any():
                cx, cy = np.nanmean(uv[ok], 0)
            else:
                cx, cy = img.shape[1] / 2, img.shape[0] / 2
            half = cell // 2
            x0 = int(np.clip(cx - half, 0, max(img.shape[1] - cell, 0)))
            y0 = int(np.clip(cy - half, 0, max(img.shape[0] - cell, 0)))
            crop = img[y0:y0 + cell, x0:x0 + cell]
            pad = np.full((cell, cell, 3), 20, np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            shown = {}
            for k in range(uv.shape[0]):
                if not ok[k] or conf[t, ci, k] < 0.5:
                    continue
                x, y = int(round(uv[k, 0] - x0)), int(round(uv[k, 1] - y0))
                if 0 <= x < cell and 0 <= y < cell:
                    shown[k] = (x, y)
            for chain in chains.values():
                # only ADJACENT pairs, and only when both are drawn: skipping a
                # missing joint would connect two segments that are not
                # neighbours and draw a leg that does not exist
                for ka, kb in zip(chain[:-1], chain[1:]):
                    if ka in shown and kb in shown:
                        cv2.line(pad, shown[ka], shown[kb], PALETTE["legs"], 1, cv2.LINE_AA)
            for k, (x, y) in shown.items():
                cv2.circle(pad, (x, y), 2, colour.get(k, PALETTE["fit"]), -1)
            cv2.putText(pad, cam, (4, cell - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if row == 0:
                cv2.putText(pad, f"f{frame_start + t} e{exist[t]:.2f} v{seen[t]:.2f}",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
            r = blk * len(cameras) + row
            sheet[r * cell:(r + 1) * cell, col * cell:(col + 1) * cell] = pad
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    if not cv2.imwrite(out_png, sheet):
        raise RuntimeError(f"cv2.imwrite failed for {out_png}")
    print(f"[sheet] {out_png}: {len(pick)} frames x {cameras} from {bout_dir}", flush=True)
    return out_png


# -------------------------------------------------------------------- driver
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", help="recording dir (mp4s + calibration)")
    p.add_argument("--calib-dir", default=None, help="default <session-dir>/calibration")
    p.add_argument("--cameras", default=None,
                   help=f"comma list; default recording.cameras of {DEF_RECORDING_CFG}")
    p.add_argument("--recording-cfg", default=DEF_RECORDING_CFG)
    p.add_argument("--anatomy", default=DEF_ANATOMY_CFG,
                   help="config whose model.KP_NAMES is the WRITTEN keypoint order")
    p.add_argument("--run", help="mvq final/ dir, or the run dir with --step")
    p.add_argument("--step", default=None, help="checkpoint step (int); omit for a final/ dir")
    p.add_argument("--centerdetect", help="CenterDetect ckpt dir (epoch_XXX)")
    p.add_argument("--bout", action="append", default=[], metavar="START:END",
                   help="frame span, end exclusive; repeatable, one bout dir each")
    p.add_argument("--num-animals", type=int, default=1,
                   help="1 only -- this is the single-fly route")
    p.add_argument("--out", help="run root; bouts land in <out>/bouts/bout_<idx:05d>")
    p.add_argument("--link-cameras", action="store_true",
                   help="build <out>/_cams/<Cam>.mp4 symlinks for a recording whose mp4s "
                        "carry a suffix (Clip), and read the frames from there")
    p.add_argument("--fly-sex", default=None, choices=("female", "male"),
                   help="the recording's KNOWN sex (free_running Session11 notes.txt: all "
                        "female); default is the majority of the model's typed reads")
    p.add_argument("--behavior", default="free_running",
                   help="recorded in mvq_meta.json and carried into the export manifest")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--min-score", type=float, default=0.2, help="CenterDetect peak threshold")
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--max-resid-px", type=float, default=25.0)
    p.add_argument("--merge-dist-units", type=float, default=30.0)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--attn-impl", default=None, help='"xla" to run a cuDNN-trained run on CPU')
    p.add_argument("--force", action="store_true", help="re-lift a bout that is already current")
    # --- scan mode
    p.add_argument("--scan", action="append", default=[], metavar="START:END",
                   help="trackability probe of this span instead of lifting anything")
    p.add_argument("--scan-n", type=int, default=20, help="probe frames per --scan span")
    # --- contact-sheet mode
    p.add_argument("--contact-sheet", default=None, metavar="BOUT_DIR",
                   help="render an already-written bout dir (no model needed)")
    p.add_argument("--out-png", default=None, help="--contact-sheet output path")
    p.add_argument("--sheet-cameras", default=None,
                   help="comma list of the two cameras to draw, BY NAME")
    p.add_argument("--sheet-frames", type=int, default=12)
    p.add_argument("--sheet-cols", type=int, default=6,
                   help="frames per row-block (the rows within a block are the cameras)")
    p.add_argument("--sheet-worst", action="store_true",
                   help="show the LOWEST-visibility written frames instead of evenly "
                        "spaced visible ones -- the stale-reused-window failure")
    return p


def main(argv=None):
    from omegaconf import OmegaConf
    a = build_parser().parse_args(argv)

    if a.contact_sheet:
        if not a.out_png:
            raise SystemExit("--contact-sheet needs --out-png")
        cams = ([c.strip() for c in a.sheet_cameras.split(",") if c.strip()]
                if a.sheet_cameras else None)
        return contact_sheet(a.contact_sheet, a.out_png, cameras=cams,
                             n_frames=int(a.sheet_frames), session_dir=a.session_dir,
                             cols=int(a.sheet_cols), worst=bool(a.sheet_worst))

    if int(a.num_animals) != 1:
        raise SystemExit(f"--num-animals must be 1 (this is the single-fly route), "
                         f"got {a.num_animals}")
    if not a.session_dir:
        raise SystemExit("--session-dir is required")
    cameras = ([c.strip() for c in a.cameras.split(",") if c.strip()]
               if a.cameras else default_cameras(a.recording_cfg))
    source_dir = os.path.abspath(a.session_dir)
    session_dir = source_dir
    if a.link_cameras:
        if not a.out:
            raise SystemExit("--link-cameras needs --out (the links live in <out>/_cams)")
        link_cameras(source_dir, os.path.join(a.out, "_cams"), cameras)
        session_dir = os.path.abspath(os.path.join(a.out, "_cams"))
        print(f"[singlefly] reading frames from {session_dir} (symlinks to {source_dir})",
              flush=True)
    calib_dir = a.calib_dir or os.path.join(session_dir, "calibration")

    if a.scan:
        from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
        from jarvis_jax.tracking.coarse_centres import CenterDetector
        if not a.centerdetect:
            raise SystemExit("--scan needs --centerdetect")
        rt = ReprojectionTool(str(calib_dir))
        if list(rt.cameras.keys()) != list(cameras):
            raise SystemExit(f"calibration glob order {list(rt.cameras.keys())} != canonical "
                             f"camera order {cameras}")
        rows = scan_spans(session_dir, cameras, np.asarray(rt.camera_matrices, np.float32),
                          CenterDetector(a.centerdetect, min_score=a.min_score),
                          [parse_span(s) for s in a.scan], n_probe=int(a.scan_n),
                          min_views=int(a.min_views), max_resid_px=float(a.max_resid_px))
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "scan_trackability.json"), "w") as f:
                json.dump({"session_dir": session_dir, "spans": rows}, f, indent=1)
        return rows

    if not (a.bout and a.run and a.centerdetect and a.out):
        raise SystemExit("the pass needs --bout, --run, --centerdetect and --out")
    spans = [parse_span(s) for s in a.bout]
    model_names = [str(n) for n in OmegaConf.load(a.anatomy).model.KP_NAMES]

    from coarse_pass_mvq import SlotReader
    from jarvis_jax.predict.synced_reader import load_plan
    from jarvis_jax.tracking.coarse_centres import CenterDetector
    from jarvis_jax.tracking.lift_mvq import MVQRunner

    t0 = time.time()
    runner = MVQRunner(a.run, step=(int(a.step) if a.step is not None else None),
                       attn_impl=a.attn_impl, calib_dir=calib_dir, cameras=cameras,
                       batch=int(a.batch), identity="sex", containment=False)
    detector = CenterDetector(a.centerdetect, min_score=float(a.min_score))
    print(f"[singlefly] models loaded in {time.time() - t0:.1f}s (mvq step "
          f"{runner.step_label}, K={runner.K}, slots={runner.I}, "
          f"exist_thresh={runner.exist_thresh})\n"
          f"[singlefly] gates {runner.gates_string()}\n"
          f"[singlefly] {len(spans)} span(s): {spans}", flush=True)

    plan = load_plan(session_dir)
    results = []
    for i, (start, end) in enumerate(spans):
        out_dir = os.path.join(a.out, "bouts", f"bout_{i:05d}")
        gates_string = runner.gates_string()
        if not a.force and _bout_is_current(out_dir, gates_string):
            print(f"[singlefly] bout {i} ({start}:{end}): skip (already current)", flush=True)
            continue
        frames = list(range(start, end))
        reader = SlotReader(session_dir, cameras, plan, start_slot=start, stride=1)
        try:
            res = lift_singlefly_bout(
                runner, detector, (reader(f) for f in frames), frames=frames,
                out_dir=out_dir, model_names=model_names,
                min_views=int(a.min_views), max_resid_px=float(a.max_resid_px),
                merge_dist_units=float(a.merge_dist_units), fly_sex=a.fly_sex,
                progress_every=int(a.progress_every),
                meta_extra={"bout": int(i), "session_dir": session_dir,
                            "source_session_dir": source_dir,
                            "frame_start": int(start),
                            "bout_frame_range": [int(start), int(end)],
                            "behavior": a.behavior,
                            "recording_sex": a.fly_sex or "unknown",
                            "centerdetect": os.path.abspath(str(a.centerdetect)),
                            "min_score": float(a.min_score)})
        finally:
            reader.close()
        results.append(res)
    print(f"[singlefly] done: {len(results)} bout(s) written to {a.out} in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return results


if __name__ == "__main__":
    main()
