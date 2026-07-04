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
`bouts/bout_*/fly*` dirs, and for every (bout, fly) that has completed
pipeline outputs (`outputs.h5`, `kp2d.npz`, `qc_perframe.npz` -- Task 1's
per-frame QC), reprojects the FK'd sites using THAT recording's own
calibration, gates them against the detector 2-D + SAM3 masks (Task 4/5's
`courtship_pseudolabel`), and turns the surviving labels into V3-format COCO
records (Task 3's `build_pseudolabel_dataset`), namespaced by
`rec_tag = basename(session_dir)` so frames from different recordings never
collide under the same file_name. Records from every source are accumulated
into ONE list and written as ONE pseudo-label dataset, then mixed with the
curated real V3 annotations to continue-train the v3 ViTPose checkpoint into
a new v4 checkpoint (Task 6's `finetune_detector.finetune`).

Incomplete (bout, fly) dirs (pipeline still running / never run) are skipped
silently -- this driver only consumes already-finished pipeline output, it
does not run the pipeline itself (see `scripts/slurm_courtship_array.py` for
that, which is resumable and idempotent).

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
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks
from jarvis_jax.cse.courtship_pseudolabel import (
    reproject_sites, gate_pseudolabels, records_for_bout, GateCfg)
from jarvis_jax.cse.build_pseudolabel_dataset import write_pseudolabel_coco
from jarvis_jax.cse.finetune_detector import finetune
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_courtship_bout import all_cams_frames, open_video_captures


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
    `run_courtship_bout.bout_start_frame`'s logic, parameterized per-source
    instead of reading a single global `cfg.recording`).

    NOTE: reads the session's bouts CSV (the same source D2/SAM3 itself parses
    via `jarvis_jax.predict.sam3_driver.parse_bouts`), not the D2
    manifest.json -- see `run_courtship_bout.bout_start_frame`'s docstring for
    why the CSV is the authoritative, append-only source."""
    tag = session_tag_for(session_dir)
    bouts = parse_bouts(bouts_csv, tag, bout_ids=[bout_idx])
    if not bouts:
        raise KeyError(f"bout_idx {bout_idx} not found in {bouts_csv} (fly_id tag={tag})")
    return int(bouts[0]["start"])


def _mask_npz(predictions_dir, bout_idx):
    """Locate this bout's SAM3 masks -- same layout `run_courtship_bout.py`
    itself reads from (`<predictions_dir>/bout_<idx:05d>/sam3_masks.npz`)."""
    return os.path.join(str(predictions_dir), f"bout_{bout_idx:05d}", "sam3_masks.npz")


def _fly_dirs(run_root):
    """Sorted `bouts/bout_*/fly*` dirs under one recording's pipeline run-root
    that have every artifact this driver needs (`outputs.h5`, `kp2d.npz`,
    `qc_perframe.npz`); incomplete ones (pipeline still running, or never
    run) are skipped."""
    for fly_dir in sorted(glob.glob(os.path.join(run_root, "bouts", "bout_*", "fly*"))):
        out_h5 = os.path.join(fly_dir, "outputs.h5")
        kp2d_npz = os.path.join(fly_dir, "kp2d.npz")
        qc_pf = os.path.join(fly_dir, "qc_perframe.npz")
        if os.path.exists(out_h5) and os.path.exists(kp2d_npz) and os.path.exists(qc_pf):
            yield fly_dir, out_h5, kp2d_npz, qc_pf


def _records_for_fly_dir(resolved, rt, cams, gcfg, rec_tag, fly_dir, out_h5, kp2d_npz, qc_pf):
    """Build gated pseudo-label COCO records for one completed (bout, fly) dir,
    using THIS recording's own reprojection tool / cameras / masks / bouts_csv
    (`resolved`, from `resolve_label_source`) -- never a single global
    `cfg.recording`."""
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
    # fail loudly instead of gating garbage (mirrors run_courtship_bout.py's
    # stac_ik.h5-vs-masks stale-artifact guard, lines 307-317).
    if masks["masks"].shape[0] != kp3d_mm.shape[0]:
        raise RuntimeError(
            f"{fly_dir}: outputs.h5 kp3d_mm T={kp3d_mm.shape[0]} != masks T="
            f"{masks['masks'].shape[0]} ({mask_npz}); stale/mismatched artifact "
            f"-- delete it and rerun this bout/fly.")

    mesh2d = reproject_sites(rt, kp3d_mm)
    labels, _ = gate_pseudolabels(mesh2d, det, conf, pf, masks["valid"], cfg=gcfg)

    start = start_frame_for(resolved["bouts_csv"], resolved["session_dir"], bout_idx)
    caps = open_video_captures(resolved["session_dir"], cams)
    try:
        frames = enumerate(all_cams_frames(caps, start, kp3d_mm.shape[0]))
        return records_for_bout(labels, masks["masks"], frames, cams, rec_tag, start)
    finally:
        for cap in caps:
            cap.release()


def _records_for_source(src, gcfg):
    """Resolve one label_sources entry and build every (bout, fly)'s gated
    pseudo-label records for it, using ITS OWN calibration/cameras/masks/bouts
    (never the single global `cfg.recording`) and a `rec_tag` derived from ITS
    session_dir so file_names never collide across recordings."""
    resolved = resolve_label_source(src)
    rt = ReprojectionTool(resolved["calib_dir"])
    cams = resolved["cameras"]
    rec_tag = os.path.basename(os.path.normpath(resolved["session_dir"]))

    records = []
    for fly_dir, out_h5, kp2d_npz, qc_pf in _fly_dirs(resolved["run_root"]):
        records += _records_for_fly_dir(
            resolved, rt, cams, gcfg, rec_tag, fly_dir, out_h5, kp2d_npz, qc_pf)
    return records


@hydra.main(version_base=None, config_path="../configs", config_name="detector_finetune")
def main(cfg):
    gcfg = GateCfg(**OmegaConf.to_container(cfg.gate, resolve=True))

    all_records = []
    for src in cfg.label_sources:                          # each src = a per-recording dict
        src_dict = OmegaConf.to_container(src, resolve=True)
        all_records += _records_for_source(src_dict, gcfg)

    write_pseudolabel_coco(cfg.pseudo_root, all_records, split="train")

    res = finetune(v3_ckpt=cfg.v3_ckpt, real_root=cfg.real_root,
                   pseudo_root=cfg.pseudo_root, out_dir=cfg.out_dir,
                   val_recordings=list(cfg.val_recordings),
                   **OmegaConf.to_container(cfg.finetune, resolve=True))
    print("FINETUNE RESULT:", res)


if __name__ == "__main__":
    main()
