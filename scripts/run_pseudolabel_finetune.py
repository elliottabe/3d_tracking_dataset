#!/usr/bin/env python3
"""End-to-end: gated silhouette pseudo-labels from pipeline outputs -> finetune v3->v4.

SESSION-GENERIC: courtship spans many separate recordings (Session0: one
recording dir; Session1: ~14 recording dirs), each with its OWN
videos/calibration/predictions/bouts. `cfg.label_sources` is therefore a list
of per-recording dicts (see `resolve_label_source` for the schema) -- NEVER a
single global `cfg.recording` reused across recordings, which would reproject
with the wrong calibration, read the wrong videos, and collide `file_name`s
across recordings.

For each source, this driver resolves its own recording view (calibration
dir, cameras, predictions dir, bouts csv), walks its `run_root`'s
`bouts/bout_*` dirs, and for every (bout, fly) that has completed pipeline
outputs (`outputs.h5`, `kp2d.npz`, `qc_perframe.npz` -- Task 1's per-frame
QC), reprojects the FK'd sites using THAT recording's own calibration, gates
them against the detector 2-D + SAM3 masks (Task 4/5's
`courtship_pseudolabel`), and turns the surviving labels into V3-format COCO
records (Task 3's `build_pseudolabel_dataset`), namespaced by
`rec_tag = basename(session_dir)` so frames from different recordings never
collide under the same file_name.

STREAMING (C1): a full run spans hundreds of bouts x 2 flies x many
recordings, each bout holding full-res `rgb`/`mask` arrays -- accumulating
every bout's records in one Python list before writing would use hundreds of
GB of RAM. Instead, records are built and flushed to disk ONE BOUT AT A TIME
via `build_pseudolabel_dataset.PseudoLabelWriter` (both flies of a bout added
together in one `add_records()` call, so frames they share land in one COCO
image, exactly as a single whole-list write would), then dropped from RAM
before the next bout. `writer.finalize()` writes the COCO json once, after
every source has been processed, then mixed with the curated real V3
annotations to continue-train the v3 ViTPose checkpoint into a new v4
checkpoint (Task 6's `finetune_detector.finetune`).

Incomplete (bout, fly) dirs (pipeline still running / never run) are skipped
(counted and logged, not silent) -- this driver only consumes
already-finished pipeline output, it does not run the pipeline itself (see
`scripts/slurm_bout_array.py` for that, which is resumable and
idempotent). A bout whose video frames fail to read (corrupt/short video) is
also skipped with a warning naming the bout, rather than killing the whole
multi-hour, multi-recording run.

Usage:
    Configure `label_sources` in configs/detector_finetune.yaml (a list of
    per-recording dicts; see the commented example there). A bare CLI list
    override is impractical once entries are dicts -- set it in the config
    file, or via Hydra's per-field `+label_sources.0.session_dir=...` style
    overrides:
        python scripts/run_pseudolabel_finetune.py
"""
import os
import glob

import numpy as np
import hydra
from omegaconf import OmegaConf

import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.tracking.bout_masks import load_bout_masks
from jarvis_jax.tracking.courtship_pseudolabel import (
    reproject_sites, gate_pseudolabels, records_for_bout, GateCfg)
from jarvis_jax.tracking.build_pseudolabel_dataset import PseudoLabelWriter
from jarvis_jax.tracking.finetune_detector import finetune
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_bout import all_cams_frames, open_video_captures


def resolve_label_source(src):
    """Resolve one `label_sources` entry into a fully-resolved recording view.

    `src` is a plain dict (the caller has already done any
    `OmegaConf.to_container(..., resolve=True)` needed) with required keys:
      session_dir  the recording dir (has `Cam*.mp4` + `calibration/`).
      run_root     this recording's pipeline output root (has
                   `bouts/bout_*/fly*/{outputs.h5,kp2d.npz,qc_perframe.npz}`).
    and optional overrides:
      predictions_dir  masks source (`bout_*/sam3_masks.npz`); default = the
                       sole `Predictions_3D_*` dir under session_dir -- if
                       there are 0 or >1 candidates, this field is REQUIRED
                       (the error lists every candidate found).
      bouts_csv        default `session_dir/courtship_bouts_unified_summary.csv`
                       if it exists, else REQUIRED -- bout/mask conventions
                       vary per recording (e.g. Session1 recordings have no
                       unified summary csv).
      calib_dir        default `session_dir/calibration`.
      cameras          default sorted basenames (without `.yaml`) of
                       `calib_dir/Cam*.yaml`.

    Returns a dict with keys: session_dir, run_root, calib_dir, cameras,
    predictions_dir, bouts_csv.
    """
    if not src.get("session_dir"):
        raise ValueError(f"label_sources entry missing required 'session_dir': {src!r}")
    if not src.get("run_root"):
        raise ValueError(f"label_sources entry missing required 'run_root': {src!r}")
    session_dir = str(src["session_dir"])
    run_root = str(src["run_root"])
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(
            f"label_sources entry: session_dir does not exist: {session_dir!r}")

    calib_dir = (str(src["calib_dir"]) if src.get("calib_dir")
                 else os.path.join(session_dir, "calibration"))

    cameras = src.get("cameras")
    if cameras:
        cameras = list(cameras)
    else:
        yaml_paths = sorted(glob.glob(os.path.join(calib_dir, "Cam*.yaml")))
        cameras = [os.path.splitext(os.path.basename(p))[0] for p in yaml_paths]
        if not cameras:
            raise ValueError(
                f"label_sources entry {session_dir!r}: no 'cameras' given and no "
                f"'Cam*.yaml' found under calib_dir={calib_dir!r} to auto-resolve them")

    predictions_dir = src.get("predictions_dir")
    if predictions_dir:
        predictions_dir = str(predictions_dir)
    else:
        candidates = sorted(
            d for d in glob.glob(os.path.join(session_dir, "Predictions_3D_*"))
            if os.path.isdir(d))
        if len(candidates) != 1:
            raise ValueError(
                f"label_sources entry {session_dir!r}: cannot auto-resolve "
                f"predictions_dir -- found {len(candidates)} 'Predictions_3D_*' "
                f"dir(s) {candidates}; set predictions_dir= explicitly for this source")
        predictions_dir = candidates[0]

    bouts_csv = src.get("bouts_csv")
    if bouts_csv:
        bouts_csv = str(bouts_csv)
    else:
        default_csv = os.path.join(session_dir, "courtship_bouts_unified_summary.csv")
        if not os.path.isfile(default_csv):
            raise ValueError(
                f"label_sources entry {session_dir!r}: no 'bouts_csv' given and the "
                f"default {default_csv!r} does not exist (bout/mask conventions vary "
                f"per recording, e.g. Session1); set bouts_csv= explicitly for this source")
        bouts_csv = default_csv

    return dict(session_dir=session_dir, run_root=run_root, calib_dir=calib_dir,
                cameras=cameras, predictions_dir=predictions_dir, bouts_csv=bouts_csv)


def start_frame_for(bouts_csv, session_dir, bout_idx):
    """Absolute start-frame for bout_idx from THIS source's bouts_csv (mirrors
    `run_bout.bout_start_frame`'s logic, parameterized per-source
    instead of reading a single global `cfg.recording`).

    NOTE: reads the session's bouts CSV (the same source D2/SAM3 itself parses
    via `jarvis_jax.predict.sam3_driver.parse_bouts`), not the D2
    manifest.json -- see `run_bout.bout_start_frame`'s docstring for
    why the CSV is the authoritative, append-only source."""
    tag = session_tag_for(session_dir)
    bouts = parse_bouts(bouts_csv, tag, bout_ids=[bout_idx])
    if not bouts:
        raise KeyError(f"bout_idx {bout_idx} not found in {bouts_csv} (fly_id tag={tag})")
    return int(bouts[0]["start"])


def _mask_npz(predictions_dir, bout_idx):
    """Locate this bout's SAM3 masks -- same layout `run_bout.py`
    itself reads from (`<predictions_dir>/bout_<idx:05d>/sam3_masks.npz`)."""
    return os.path.join(str(predictions_dir), f"bout_{bout_idx:05d}", "sam3_masks.npz")


def _bout_dirs(run_root):
    """Sorted `bouts/bout_*` dirs under one recording's pipeline run-root."""
    return sorted(glob.glob(os.path.join(run_root, "bouts", "bout_*")))


def _fly_artifacts(fly_dir):
    """(out_h5, kp2d_npz, qc_pf) for one `fly*` dir if it has every artifact
    this driver needs, else None (pipeline still running / never run for
    this fly -- skipped, not fatal)."""
    out_h5 = os.path.join(fly_dir, "outputs.h5")
    kp2d_npz = os.path.join(fly_dir, "kp2d.npz")
    qc_pf = os.path.join(fly_dir, "qc_perframe.npz")
    if os.path.exists(out_h5) and os.path.exists(kp2d_npz) and os.path.exists(qc_pf):
        return out_h5, kp2d_npz, qc_pf
    return None


def _records_for_fly_dir(resolved, rt, cams, gcfg, rec_tag, fly_dir, out_h5, kp2d_npz, qc_pf):
    """Build gated pseudo-label COCO records for one completed (bout, fly) dir,
    using THIS recording's own reprojection tool / cameras / masks / bouts_csv
    (`resolved`, from `resolve_label_source`) -- never a single global
    `cfg.recording`.

    C1: if reading this bout's video frames fails (`all_cams_frames` raising
    -- e.g. a short/corrupt video), the failure is caught HERE (around only
    the frame-read + record-build tail), logged with the bout/fly named, and
    an empty list is returned so one bad bout can't kill a multi-hour,
    multi-recording run. The T-consistency guard below is NOT covered by this
    catch (it runs before the try) -- a stale/mismatched artifact is a real
    bug and must still fail loudly, not be silently skipped."""
    kp3d_mm = np.asarray(ioh5.load(out_h5)["kp3d_mm"])
    z = np.load(kp2d_npz)
    det, conf = z["kp2d"], z["conf"]
    pf = dict(np.load(qc_pf))

    bout_idx = int(os.path.basename(os.path.dirname(fly_dir)).split("_")[-1])
    fly = int(os.path.basename(fly_dir).replace("fly", ""))

    mask_npz = _mask_npz(resolved["predictions_dir"], bout_idx)
    masks = load_bout_masks(mask_npz, fly)

    # T-consistency guard: masks and outputs.h5's kp3d_mm must cover the exact
    # same bout frame range. A stale outputs.h5 left from a different (e.g.
    # differently-trimmed) run would silently desync gating from masks_dict --
    # fail loudly instead of gating garbage (mirrors run_bout.py's
    # stac_ik.h5-vs-masks stale-artifact guard, lines 307-317). NOT caught by
    # the frame-read try/except below.
    if masks["masks"].shape[0] != kp3d_mm.shape[0]:
        raise RuntimeError(
            f"{fly_dir}: outputs.h5 kp3d_mm T={kp3d_mm.shape[0]} != masks T="
            f"{masks['masks'].shape[0]} ({mask_npz}); stale/mismatched artifact "
            f"-- delete it and rerun this bout/fly.")

    # NOTE (keypoint-order footgun -- BEFORE un-shelving the finetune path):
    #   kp3d_mm (and, after the O->model reorder in run_bout, kp2d.npz)
    #   are in cfg.model.KP_NAMES == XML SITE order. So mesh2d/det/labels here are
    #   all XML-order -> gating below is internally consistent. BUT the pseudo-label
    #   COCO written from `labels` is therefore XML-order, while the REAL red_data
    #   COCO the detector trains on is in the detector's tracking order O (see
    #   configs/detector/vitpose_v3.yaml kp_names). Mixing the two orders in
    #   ConcatV3 would train the detector on contradictory channel semantics.
    #   FIX before enabling: emit pseudo-labels in order O (apply the inverse of
    #   courtship_predict_2d.detector_to_model_perm to `labels` here), OR rebuild
    #   real red_data in XML order. Also: load_bout_masks above is called WITHOUT
    #   expected_cameras/verify_mask_camera_order (unlike run_bout) --
    #   add that guard here too when un-shelving.
    mesh2d = reproject_sites(rt, kp3d_mm)
    labels, _ = gate_pseudolabels(mesh2d, det, conf, pf, masks["valid"], cfg=gcfg)

    start = start_frame_for(resolved["bouts_csv"], resolved["session_dir"], bout_idx)
    caps = open_video_captures(resolved["session_dir"], cams)
    try:
        frames = enumerate(all_cams_frames(caps, start, kp3d_mm.shape[0]))
        return records_for_bout(labels, masks["masks"], frames, cams, rec_tag, start)
    except RuntimeError as e:
        print(f"[pseudolabel] WARNING: bout {bout_idx} fly{fly} ({fly_dir}): "
              f"frame read failed ({e}); skipping this (bout, fly) and "
              f"continuing", flush=True)
        return []
    finally:
        for cap in caps:
            cap.release()


def _process_source(src, gcfg, writer):
    """Resolve one label_sources entry and STREAM its gated pseudo-label
    records into `writer`, one bout at a time -- both flies of a bout are
    gathered and added together in ONE `writer.add_records()` call so frames
    they share (same camera frame, two flies) land in one COCO image with N
    annotations, exactly as a single whole-list write would (C1). No more
    than one bout's rgb/mask arrays are ever held in RAM at once. Uses THIS
    source's own calibration/cameras/masks/bouts (never a single global
    `cfg.recording`) and a `rec_tag` derived from ITS session_dir so
    file_names never collide across recordings.

    Returns a counts dict for this source (I3 visibility): bouts_scanned,
    fly_dirs_scanned, fly_dirs_skipped_incomplete, records_written,
    visible_keypoints."""
    resolved = resolve_label_source(src)
    rt = ReprojectionTool(resolved["calib_dir"])
    cams = resolved["cameras"]
    rec_tag = os.path.basename(os.path.normpath(resolved["session_dir"]))

    counts = dict(bouts_scanned=0, fly_dirs_scanned=0,
                  fly_dirs_skipped_incomplete=0, records_written=0,
                  visible_keypoints=0)
    for bout_dir in _bout_dirs(resolved["run_root"]):
        counts["bouts_scanned"] += 1
        bout_records = []
        for fly_dir in sorted(glob.glob(os.path.join(bout_dir, "fly*"))):
            counts["fly_dirs_scanned"] += 1
            artifacts = _fly_artifacts(fly_dir)
            if artifacts is None:
                counts["fly_dirs_skipped_incomplete"] += 1
                continue
            out_h5, kp2d_npz, qc_pf = artifacts
            bout_records += _records_for_fly_dir(
                resolved, rt, cams, gcfg, rec_tag, fly_dir, out_h5, kp2d_npz, qc_pf)
        if bout_records:
            counts["records_written"] += len(bout_records)
            counts["visible_keypoints"] += sum(
                int((np.asarray(r["keypoints"], float)[:, 2] > 0).sum())
                for r in bout_records)
            writer.add_records(bout_records)     # flushed to disk; dropped from RAM here
    return counts


def _require_label_sources(label_sources):
    """I3: raise a clear error if `label_sources` is empty -- an empty list
    would silently fall through to a real-only retrain that could be
    (mis)reported as a successful v3->v4 silhouette-bootstrap."""
    if not label_sources:
        raise ValueError(
            "cfg.label_sources is empty -- refusing to run a real-only "
            "finetune that could be silently mistaken for a successful "
            "v3->v4 silhouette-bootstrap; configure at least one "
            "label_sources entry in configs/detector_finetune.yaml (see its "
            "comments for the per-entry schema).")


def _require_nonempty_pseudo_dataset(writer):
    """I3: raise a clear error if the pseudo-label dataset `writer` produced
    has zero images or zero annotations -- an upstream bug (e.g. every bout
    skipped as incomplete, every frame gated out) must not silently degrade
    this run to an effectively real-only retrain reported as success."""
    if not writer.images or not writer.annotations:
        raise ValueError(
            f"pseudo-label dataset at {writer.out_root!r} split "
            f"{writer.split!r} has {len(writer.images)} image(s) and "
            f"{len(writer.annotations)} annotation(s) after processing all "
            "label_sources -- refusing to finetune on an effectively empty "
            "pseudo set (check label_sources point at completed pipeline "
            "runs, and that gate thresholds in cfg.gate aren't excluding "
            "everything).")


@hydra.main(version_base=None, config_path="../configs", config_name="detector_finetune")
def main(cfg):
    gcfg = GateCfg(**OmegaConf.to_container(cfg.gate, resolve=True))

    label_sources = list(cfg.label_sources)
    _require_label_sources(label_sources)

    # C1: stream pseudo-labels to disk per-bout via PseudoLabelWriter instead
    # of accumulating every bout's full-res rgb/mask arrays in one giant
    # `all_records` list (hundreds of GB across all bouts x 2 flies x every
    # recording -> OOM on the real run).
    writer = PseudoLabelWriter(cfg.pseudo_root, split="train")
    totals = dict(bouts_scanned=0, fly_dirs_scanned=0,
                  fly_dirs_skipped_incomplete=0, records_written=0,
                  visible_keypoints=0)
    for src in label_sources:                          # each src = a per-recording dict
        src_dict = OmegaConf.to_container(src, resolve=True)
        src_counts = _process_source(src_dict, gcfg, writer)
        print(f"[pseudolabel] source {src_dict.get('session_dir')}: "
              f"bouts_scanned={src_counts['bouts_scanned']} "
              f"fly_dirs_scanned={src_counts['fly_dirs_scanned']} "
              f"fly_dirs_skipped_incomplete={src_counts['fly_dirs_skipped_incomplete']} "
              f"records_written={src_counts['records_written']} "
              f"visible_keypoints={src_counts['visible_keypoints']}", flush=True)
        for k in totals:
            totals[k] += src_counts[k]

    ann_path = writer.finalize()
    print(f"[pseudolabel] TOTAL across {len(label_sources)} source(s): "
          f"bouts_scanned={totals['bouts_scanned']} "
          f"fly_dirs_scanned={totals['fly_dirs_scanned']} "
          f"fly_dirs_skipped_incomplete={totals['fly_dirs_skipped_incomplete']} "
          f"records_written={totals['records_written']} "
          f"visible_keypoints={totals['visible_keypoints']} "
          f"images={len(writer.images)} annotations={len(writer.annotations)} "
          f"-> {ann_path}", flush=True)

    _require_nonempty_pseudo_dataset(writer)

    res = finetune(v3_ckpt=cfg.v3_ckpt, real_root=cfg.real_root,
                   pseudo_root=cfg.pseudo_root, out_dir=cfg.out_dir,
                   val_recordings=list(cfg.val_recordings),
                   **OmegaConf.to_container(cfg.finetune, resolve=True))
    print("FINETUNE RESULT:", res)


if __name__ == "__main__":
    main()
