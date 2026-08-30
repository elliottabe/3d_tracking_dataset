"""Build red_data_3d_v5: one correctly-organized root for 3D training.

Two organizational defects in the old layout drive this module:

1. The SPLIT WAS THE DIRECTORY LAYOUT. general_model/<subset>/ held train/ and
   val/ dirs over the SAME recording, so the split was frame-level and leaked
   (~50% of val framesets had a train frameset within +/-3 frames), and
   regenerating it meant moving image files. Here the split is METADATA
   (split.json, Task 5) and the media tree has no train/ or val/ dir at all.
2. Calibrations were duplicated per subset (21 copies of 3 distinct
   calibrations), hiding the fact that calibration identity is a real
   experimental variable.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

from jarvis_jax.data.calib_groups import CAM_GLOB, group_calibrations

import glob


@dataclass
class SourceRec:
    recording: str
    subset: str | None
    ann_paths: list[str]
    calib_dir: str
    image_root: str
    mask_root: str | None = None
    n_framesets: int = 0
    n_flies: int = 0
    behavior: str = "unknown"
    sex: str = "unknown"
    extra: dict = field(default_factory=dict)


def discover_sources(general_model_root: str, v3_root: str) -> dict[str, SourceRec]:
    """Enumerate every labeled recording across BOTH source trees.

    general_model contributes 21 recordings; red_data_unified_V3 contributes 5
    more that general_model lacks (2026_03_22, 2026_04_07, 2026_04_08, and the
    two 2026_06_11 recordings). The union is 26 -- V3's 3D training only ever
    used 17.
    """
    srcs: dict[str, SourceRec] = {}
    for subset in sorted(os.listdir(general_model_root)):
        cp = os.path.join(general_model_root, subset, "calib_params")
        if not os.path.isdir(cp):
            continue
        for rec in sorted(os.listdir(cp)):
            anns = sorted(glob.glob(os.path.join(
                general_model_root, subset, "annotations", "instances_*.json")))
            srcs[rec] = SourceRec(
                recording=rec, subset=subset, ann_paths=anns,
                calib_dir=os.path.join(cp, rec),
                image_root=os.path.join(general_model_root, subset),
                mask_root=None)
    v3_calib = os.path.join(v3_root, "calib_params")
    v3_anns = sorted(glob.glob(os.path.join(v3_root, "annotations", "instances_*.json")))
    for rec in sorted(os.listdir(v3_calib)):
        if rec in srcs:
            # Already covered by general_model, but V3 is where its masks live.
            srcs[rec].mask_root = os.path.join(v3_root, "sam3_masks")
            continue
        srcs[rec] = SourceRec(
            recording=rec, subset=None, ann_paths=v3_anns,
            calib_dir=os.path.join(v3_calib, rec),
            image_root=v3_root,
            mask_root=os.path.join(v3_root, "sam3_masks"))
    return srcs


def build_manifest(sources: dict[str, SourceRec], out_root: str) -> dict:
    """Write manifest.json and the deduplicated calibrations/ tree."""
    os.makedirs(out_root, exist_ok=True)
    groups = group_calibrations({r: s.calib_dir for r, s in sources.items()})

    calib_out = os.path.join(out_root, "calibrations")
    for rec, grp in groups.items():
        dst = os.path.join(calib_out, grp)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(sources[rec].calib_dir, CAM_GLOB))):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    man = {
        "version": "red_data_3d_v5",
        "calib_groups": sorted(set(groups.values())),
        "recordings": {
            rec: {
                "subset": s.subset,
                "calib_group": groups[rec],
                "n_framesets": s.n_framesets,
                "n_flies": s.n_flies,
                "behavior": s.behavior,
                "sex": s.sex,                      # filled by the Task 6 sexing pass
                "has_masks": s.mask_root is not None,
                "split": None,                     # filled by Task 5
            }
            for rec, s in sorted(sources.items())
        },
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def _looks_like_camera_root(path: str) -> bool:
    """True if `path` itself directly holds `Cam*` directories.

    Distinguishes a source root that is already scoped to one recording
    (children are cameras) from a shared multi-recording root such as
    `general_model/<subset>` or `red_data_unified_V3` (children are
    `calib_params/`, `train/`, `val/`, `annotations/` -- never cameras). Only
    the latter needs `rec` joined on; walking the former as if it needed the
    same join would miss the cameras entirely.
    """
    if not os.path.isdir(path):
        return False
    return any(d.startswith("Cam") and os.path.isdir(os.path.join(path, d))
               for d in os.listdir(path))


def link_media(sources: dict[str, SourceRec], out_root: str, *,
               copy: bool = False) -> None:
    """Materialize images/ and masks/ as ONE flat per-recording tree.

    Symlinks by default: the sources are stable on gscratch and copying costs
    4.4 GB for no benefit. `copy=True` if the tree must be self-contained.
    """
    for rec, s in sources.items():
        for kind, src_root in (("images", s.image_root), ("masks", s.mask_root)):
            if src_root is None:
                continue
            for split in ("train", "val", ""):
                nested = os.path.join(src_root, split, rec) if split else os.path.join(src_root, rec)
                bare = os.path.join(src_root, split) if split else src_root
                if os.path.isdir(nested):
                    base = nested
                elif _looks_like_camera_root(bare):
                    base = bare
                else:
                    continue
                for cam in sorted(os.listdir(base)):
                    src_cam = os.path.join(base, cam)
                    if not os.path.isdir(src_cam):
                        continue
                    dst_cam = os.path.join(out_root, kind, rec, cam)
                    os.makedirs(dst_cam, exist_ok=True)
                    for fn in sorted(os.listdir(src_cam)):
                        dst = os.path.join(dst_cam, fn)
                        if os.path.lexists(dst):
                            continue
                        if copy:
                            shutil.copy2(os.path.join(src_cam, fn), dst)
                        else:
                            os.symlink(os.path.join(src_cam, fn), dst)
