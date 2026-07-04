#!/usr/bin/env python3
"""End-to-end: gated silhouette pseudo-labels from pipeline outputs -> finetune v3->v4.

Walks each `cfg.label_sources` run-root's `bouts/bout_*/fly*` dirs, and for
every (bout, fly) that has completed pipeline outputs (`outputs.h5`,
`kp2d.npz`, `qc_perframe.npz` -- Task 1's per-frame QC), reprojects the FK'd
sites, gates them against the detector 2-D + SAM3 masks (Task 4/5's
`courtship_pseudolabel`), and turns the surviving labels into V3-format COCO
records (Task 3's `build_pseudolabel_dataset`). The merged pseudo-label
dataset is then mixed with the curated real V3 annotations and used to
continue-train the v3 ViTPose checkpoint into a new v4 checkpoint (Task 6's
`finetune_detector.finetune`).

Incomplete (bout, fly) dirs (pipeline still running / never run) are skipped
silently -- this driver only consumes already-finished pipeline output, it
does not run the pipeline itself (see `scripts/slurm_courtship_array.py` for
that, which is resumable and idempotent).

Usage:
    python scripts/run_pseudolabel_finetune.py \
        label_sources=[/path/to/run_root1,/path/to/run_root2]
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_courtship_bout import all_cams_frames, open_video_captures, bout_start_frame


def _mask_npz(cfg, src, bout_idx):
    """Locate this bout's SAM3 masks -- same layout `run_courtship_bout.py`
    itself reads from (`<predictions_dir>/bout_<idx:05d>/sam3_masks.npz`)."""
    return os.path.join(str(cfg.recording.predictions_dir), f"bout_{bout_idx:05d}", "sam3_masks.npz")


def _fly_dirs(src):
    """Sorted `bouts/bout_*/fly*` dirs under a label_sources run-root that have
    every artifact this driver needs (`outputs.h5`, `kp2d.npz`,
    `qc_perframe.npz`); incomplete ones (pipeline still running, or never
    run) are skipped."""
    for fly_dir in sorted(glob.glob(os.path.join(src, "bouts", "bout_*", "fly*"))):
        out_h5 = os.path.join(fly_dir, "outputs.h5")
        kp2d_npz = os.path.join(fly_dir, "kp2d.npz")
        qc_pf = os.path.join(fly_dir, "qc_perframe.npz")
        if os.path.exists(out_h5) and os.path.exists(kp2d_npz) and os.path.exists(qc_pf):
            yield fly_dir, out_h5, kp2d_npz, qc_pf


def _records_for_fly_dir(cfg, rt, cams, gcfg, src, fly_dir, out_h5, kp2d_npz, qc_pf):
    """Build gated pseudo-label COCO records for one completed (bout, fly) dir."""
    kp3d_mm = np.asarray(ioh5.load(out_h5)["kp3d_mm"])
    z = np.load(kp2d_npz)
    det, conf = z["kp2d"], z["conf"]
    pf = dict(np.load(qc_pf))

    bout_idx = int(os.path.basename(os.path.dirname(fly_dir)).split("_")[-1])
    fly = int(os.path.basename(fly_dir).replace("fly", ""))

    masks = load_bout_masks(_mask_npz(cfg, src, bout_idx), fly)
    mesh2d = reproject_sites(rt, kp3d_mm)
    labels, _ = gate_pseudolabels(mesh2d, det, conf, pf, masks["valid"], cfg=gcfg)

    start = bout_start_frame(cfg, bout_idx)
    caps = open_video_captures(str(cfg.recording.session_dir), cams)
    try:
        frames = enumerate(all_cams_frames(caps, start, kp3d_mm.shape[0]))
        rec_tag = os.path.basename(str(cfg.recording.session_dir))
        return records_for_bout(labels, masks["masks"], frames, cams, rec_tag, start)
    finally:
        for cap in caps:
            cap.release()


@hydra.main(version_base=None, config_path="../configs", config_name="detector_finetune")
def main(cfg):
    rt = ReprojectionTool(cfg.recording.calib_dir)
    cams = list(cfg.recording.cameras)
    gcfg = GateCfg(**OmegaConf.to_container(cfg.gate, resolve=True))

    all_records = []
    for src in cfg.label_sources:                          # each src = a run-root
        for fly_dir, out_h5, kp2d_npz, qc_pf in _fly_dirs(src):
            all_records += _records_for_fly_dir(
                cfg, rt, cams, gcfg, src, fly_dir, out_h5, kp2d_npz, qc_pf)

    write_pseudolabel_coco(cfg.pseudo_root, all_records, split="train")

    res = finetune(v3_ckpt=cfg.v3_ckpt, real_root=cfg.real_root,
                   pseudo_root=cfg.pseudo_root, out_dir=cfg.out_dir,
                   val_recordings=list(cfg.val_recordings),
                   **OmegaConf.to_container(cfg.finetune, resolve=True))
    print("FINETUNE RESULT:", res)


if __name__ == "__main__":
    main()
