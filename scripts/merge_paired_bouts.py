#!/usr/bin/env python3
"""Merge per-fly preprocessed bouts into a paired dual-fly h5.

Takes two h5 files produced by ``preprocess_keypoints_for_ik.py`` — one per
physical fly csv — and co-locates matching bouts so each bout in the output
carries BOTH flies' keypoints over the same frame window, plus per-frame
validity masks and a ``pair_state`` label (0=none, 1=fly0, 2=fly1, 3=both).

No bouts or frames are dropped: bouts where only one fly is tracked well are
kept with their ``bout_pair_class`` tagged as ``fly0_only`` / ``fly1_only`` /
``mixed`` so downstream analyses can cleanly separate paired vs solo frames.

Usage:
    python scripts/merge_paired_bouts.py \\
        --fly0 /data/.../preprocessed_bout_v1_courtship_fly0.h5 \\
        --fly1 /data/.../preprocessed_bout_v1_courtship_fly1.h5 \\
        --out  /data/.../preprocessed_bout_v1_courtship_paired.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import io_dict_to_hdf5 as ioh5  # noqa: E402
from utils.pair_validity import (  # noqa: E402
    PAIR_BOTH,
    PAIR_FLY0_ONLY,
    PAIR_FLY1_ONLY,
    PAIR_NONE,
    PairValidityConfig,
    classify_bout,
    pair_validity_config_from_dict,
)


def _strip_fly_suffix(fid: str) -> str:
    for suf in ("_fly0", "_fly1"):
        if fid.endswith(suf):
            return fid[: -len(suf)]
    return fid


def _index_by_base_id(data: dict) -> dict:
    """Return {base_fly_id: bout_key} for a per-fly h5."""
    info = data.get("info", {})
    fly_ids = list(info.get("fly_ids", []))
    bout_keys = sorted(k for k in data.keys() if k != "info")
    if len(fly_ids) != len(bout_keys):
        raise ValueError(
            f"info/fly_ids length ({len(fly_ids)}) != bouts ({len(bout_keys)})"
        )
    return {_strip_fly_suffix(fid): bk for fid, bk in zip(fly_ids, bout_keys)}


def merge_paired(
    fly0_path: Path,
    fly1_path: Path,
    out_path: Path,
    pv_cfg: PairValidityConfig,
    force: bool = False,
) -> Path:
    if out_path.exists() and not force:
        print(f"✓ {out_path.name} already exists (use --force to overwrite)")
        return out_path

    print(f"Loading fly0: {fly0_path}")
    d0 = ioh5.load(fly0_path, enable_jax=False)
    print(f"Loading fly1: {fly1_path}")
    d1 = ioh5.load(fly1_path, enable_jax=False)

    idx0 = _index_by_base_id(d0)
    idx1 = _index_by_base_id(d1)

    # Preserve fly0 ordering first, then any fly1-only bouts.
    all_base_ids: list[str] = []
    seen: set[str] = set()
    for base in idx0.keys():
        all_base_ids.append(base)
        seen.add(base)
    for base in idx1.keys():
        if base not in seen:
            all_base_ids.append(base)

    combined: dict = {}
    info: dict = {
        "fly_ids": [],
        "clip_lengths": [],
        "bout_pair_class": [],
        "source_fly0_bout_keys": [],
        "source_fly1_bout_keys": [],
    }
    if "kp_names" in d0.get("info", {}):
        info["kp_names"] = list(d0["info"]["kp_names"])
    if "skeleton_edges" in d0.get("info", {}):
        info["skeleton_edges"] = d0["info"]["skeleton_edges"]

    pair_class_counts: dict[str, int] = {
        "paired": 0, "fly0_only": 0, "fly1_only": 0,
        "mixed": 0, "empty": 0,
    }

    for i, base in enumerate(all_base_ids):
        b0 = d0[idx0[base]] if base in idx0 else None
        b1 = d1[idx1[base]] if base in idx1 else None

        kp0 = np.asarray(b0["keypoints"]) if b0 is not None else None
        kp1 = np.asarray(b1["keypoints"]) if b1 is not None else None

        T = (kp0.shape[0] if kp0 is not None else kp1.shape[0])
        if kp0 is not None and kp1 is not None and kp0.shape[0] != kp1.shape[0]:
            print(f"  ⚠ {base}: shape mismatch fly0={kp0.shape[0]} fly1={kp1.shape[0]} — "
                  f"truncating to min")
            T = min(kp0.shape[0], kp1.shape[0])
            kp0 = kp0[:T]
            kp1 = kp1[:T]
            if b0 is not None:
                b0 = {k: (v[:T] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == kp0.shape[0] else v)
                      for k, v in b0.items()}
            if b1 is not None:
                b1 = {k: (v[:T] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == kp1.shape[0] else v)
                      for k, v in b1.items()}

        # Per-fly valid masks. Prefer masks written by preprocessing; fall back
        # to all-finite critical-kp check if absent.
        def _get_valid(b, kp):
            if b is None or kp is None:
                return np.zeros(T, dtype=bool)
            if "valid_fly" in b:
                return np.asarray(b["valid_fly"], dtype=bool)[:T]
            # fallback: any NaN in keypoints → invalid
            return np.all(np.isfinite(kp), axis=(1, 2))

        valid_fly0 = _get_valid(b0, kp0)
        valid_fly1 = _get_valid(b1, kp1)

        pair_state = (
            valid_fly0.astype(np.uint8) | (valid_fly1.astype(np.uint8) << 1)
        )

        n_both = int((pair_state == PAIR_BOTH).sum())
        n0 = int(valid_fly0.sum())
        n1 = int(valid_fly1.sum())
        pair_class = classify_bout(
            n_both, n0, n1, pv_cfg.min_paired_frames, pv_cfg.min_solo_frames
        )
        pair_class_counts[pair_class] = pair_class_counts.get(pair_class, 0) + 1

        # Fill missing fly's keypoints with NaN so downstream indexing is uniform.
        N = None
        if kp0 is not None:
            N = kp0.shape[1]
        elif kp1 is not None:
            N = kp1.shape[1]
        if kp0 is None:
            kp0 = np.full((T, N, 3), np.nan, dtype=np.float32)
        if kp1 is None:
            kp1 = np.full((T, N, 3), np.nan, dtype=np.float32)

        entry: dict = {
            "keypoints_fly0": kp0.astype(np.float32),
            "keypoints_fly1": kp1.astype(np.float32),
            "valid_fly0": valid_fly0,
            "valid_fly1": valid_fly1,
            "valid_both": (pair_state == PAIR_BOTH),
            "pair_state": pair_state,
        }
        if b0 is not None and "orig_keypoints" in b0:
            entry["orig_keypoints_fly0"] = np.asarray(b0["orig_keypoints"])[:T]
        if b1 is not None and "orig_keypoints" in b1:
            entry["orig_keypoints_fly1"] = np.asarray(b1["orig_keypoints"])[:T]
        if b0 is not None and "alignment_info" in b0:
            entry["alignment_info_fly0"] = b0["alignment_info"]
        if b1 is not None and "alignment_info" in b1:
            entry["alignment_info_fly1"] = b1["alignment_info"]

        new_key = f"bout_{i:03d}"
        combined[new_key] = entry
        info["fly_ids"].append(base)
        info["clip_lengths"].append(int(T))
        info["bout_pair_class"].append(pair_class)
        info["source_fly0_bout_keys"].append(idx0.get(base, ""))
        info["source_fly1_bout_keys"].append(idx1.get(base, ""))

    info["pair_validity"] = {
        "enabled": True,
        "critical_kp_patterns": list(pv_cfg.critical_kp_patterns),
        "ground_kp_patterns": list(pv_cfg.ground_kp_patterns),
        "ground_epsilon_mm": float(pv_cfg.ground_epsilon_mm),
        "floor_percentile": float(pv_cfg.floor_percentile),
        "swap_guard_frames": int(pv_cfg.swap_guard_frames),
        "min_paired_frames": int(pv_cfg.min_paired_frames),
        "min_solo_frames": int(pv_cfg.min_solo_frames),
    }
    combined["info"] = info

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ioh5.save(out_path, combined)

    print("\n" + "=" * 60)
    print(f"✓ Wrote {len(all_base_ids)} paired bouts → {out_path}")
    for k, v in pair_class_counts.items():
        print(f"    {k:10s}: {v}")
    return out_path


def _load_pv_cfg_from_h5(path: Path) -> PairValidityConfig:
    try:
        d = ioh5.load(path, enable_jax=False)
        pv = d.get("info", {}).get("pair_validity", None)
        if pv is not None:
            return pair_validity_config_from_dict(pv)
    except Exception:
        pass
    return PairValidityConfig()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fly0", type=Path, required=True,
                    help="Per-fly0 preprocessed bout h5")
    ap.add_argument("--fly1", type=Path, required=True,
                    help="Per-fly1 preprocessed bout h5")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output paired h5 path")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output")
    ap.add_argument("--min-paired-frames", type=int, default=None)
    ap.add_argument("--min-solo-frames", type=int, default=None)
    args = ap.parse_args()

    pv_cfg = _load_pv_cfg_from_h5(args.fly0)
    if args.min_paired_frames is not None:
        pv_cfg.min_paired_frames = args.min_paired_frames
    if args.min_solo_frames is not None:
        pv_cfg.min_solo_frames = args.min_solo_frames

    merge_paired(args.fly0, args.fly1, args.out, pv_cfg, force=args.force)


if __name__ == "__main__":
    main()
