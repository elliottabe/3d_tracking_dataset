"""Leg-motion QC gate for packed reference-clip h5 files (Task 12).

Why this exists
----------------
In the v2_3 run, `fit_offsets` wrote an all-NaN `offsets` array for 7 of 23
directories and the pipeline reported success. Every bout in those
directories then solved with NaN offsets and all 58 leg joints stayed
frozen at exactly 0 across every frame -- a fly sliding along the floor
with rigid legs. 10 further bouts in otherwise-healthy directories froze
individually from scattered NaN keypoints. 123 of 387 clips (32%) were
unusable, and nothing in the existing gates caught it: correct shapes,
correct DOF count, zero NaN, true `clip_lengths`, verified padding, and it
loaded through the consumer's own loader. All of those are blind to a
joint that never moves. It was found only by watching a render.

This module streams a packed reference-clip h5 (shape as produced by
`scripts/export/pack_reference_clips.py`) clip by clip and flags any clip
whose leg joints never move over their TRUE (unpadded) span, plus reports
NaN-poisoned clips as a secondary, previously-hit failure mode.

Usage
-----
    JAX_PLATFORMS=cpu python scripts/qc/audit_reference_clips.py \\
        --h5 /path/to/reference_clips.h5

Exits 1 if any clip has frozen leg joints (unless --no-fail-on-frozen).
CPU-only, no model compile, no MJX -- safe to run on a login node.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

import h5py
import numpy as np

# Column-name prefixes that identify a leg-joint qpos column in the v2_x
# fruitfly model (58 columns total in the real dataset): coxa abduct/twist,
# coxa, trochanter, femur, tibia, tarsus1..4, tarsal_claw -- x2 legs x3
# thoracic segments x2 sides. `tarsus` (no trailing underscore) intentionally
# matches both `tarsus1_...`..`tarsus4_...`.
LEG_PREFIXES = ('coxa_', 'trochanter_', 'femur_', 'tibia_', 'tarsus', 'tarsal_claw_')
ROOT_NAME = 'free'

# Datasets (besides qpos) worth a NaN sweep -- both measured failure modes
# (all-NaN offsets -> frozen legs, and scattered-NaN keypoints) leave traces
# here.
NAN_SWEEP_KEYS = ('qpos', 'qvel', 'xpos', 'xquat', 'kp_data')


def _read_qpos_names(f: h5py.File) -> List[str]:
    """Read `qpos_names`, a group of per-column scalar strings keyed "0".."N-1"."""
    grp = f['qpos_names']
    names = []
    for i in range(len(grp)):
        v = grp[str(i)][()]
        names.append(v.decode() if isinstance(v, bytes) else str(v))
    return names


def _classify_columns(names: List[str]):
    leg_idx = [i for i, n in enumerate(names) if n.startswith(LEG_PREFIXES)]
    root_idx = [i for i, n in enumerate(names) if n == ROOT_NAME]
    other_idx = [i for i in range(len(names)) if i not in set(leg_idx) and i not in set(root_idx)]
    return leg_idx, root_idx, other_idx


def audit_clips(h5_path: str, leg_ptp_threshold: float = 1e-3) -> Dict:
    """Stream a packed reference-clip h5 and flag frozen-leg / NaN clips.

    Reads `qpos` clip-by-clip (never the whole array at once -- the real
    file is ~500 MB) and, for each clip, restricts every computation to
    its TRUE unpadded span `clip_lengths[i]`. Padding-region motion (a
    repeated final frame, or in corrupted files arbitrary junk) is never
    consulted.

    Returns a dict with:
      n_clips: int
      frozen_leg_clips: list[int] -- clip indices whose max leg-joint ptp
          (over the true span) is below `leg_ptp_threshold`.
      frozen_joint_counts: dict[int, int] -- for frozen clips, how many
          individual leg joints are exactly static (ptp < 1e-6).
      per_clip_max_leg_ptp: np.ndarray, shape (n_clips,)
      nan_clips: dict[str, dict] -- per swept dataset key, the clip
          indices (within their true span) that are all-NaN or any-NaN.
          Covers the other failure mode this tool exists to catch.
    """
    with h5py.File(h5_path, 'r') as f:
        names = _read_qpos_names(f)
        leg_idx, root_idx, _other_idx = _classify_columns(names)
        if not leg_idx:
            raise ValueError(
                f"no leg-joint columns matched prefixes {LEG_PREFIXES} in "
                f"qpos_names -- classification is broken or this is not a "
                f"fruitfly qpos layout")

        qpos = f['qpos']
        clip_lengths = np.asarray(f['clip_lengths'][:])
        n_clips = qpos.shape[0]

        frozen_leg_clips: List[int] = []
        frozen_joint_counts: Dict[int, int] = {}
        per_clip_max_leg_ptp = np.zeros(n_clips, dtype=np.float64)

        nan_clips: Dict[str, Dict[str, List[int]]] = {
            k: {'all_nan': [], 'any_nan': []} for k in NAN_SWEEP_KEYS if k in f
        }

        for i in range(n_clips):
            true_len = int(clip_lengths[i])
            clip_qpos = np.asarray(qpos[i, :true_len, :])
            leg_span = clip_qpos[:, leg_idx]

            leg_ptp = leg_span.max(axis=0) - leg_span.min(axis=0)
            max_ptp = float(np.max(leg_ptp)) if leg_ptp.size else 0.0
            per_clip_max_leg_ptp[i] = max_ptp

            if max_ptp < leg_ptp_threshold:
                frozen_leg_clips.append(i)
                n_static = int(np.count_nonzero(leg_ptp < 1e-6))
                frozen_joint_counts[i] = n_static

            for key, bucket in nan_clips.items():
                arr = np.asarray(f[key][i, :true_len, ...])
                if arr.size == 0:
                    continue
                is_nan = np.isnan(arr)
                if is_nan.all():
                    bucket['all_nan'].append(i)
                elif is_nan.any():
                    bucket['any_nan'].append(i)

        return {
            'n_clips': n_clips,
            'frozen_leg_clips': frozen_leg_clips,
            'frozen_joint_counts': frozen_joint_counts,
            'per_clip_max_leg_ptp': per_clip_max_leg_ptp,
            'nan_clips': nan_clips,
        }


def _print_report(result: Dict, max_report: int) -> None:
    n_clips = result['n_clips']
    frozen = result['frozen_leg_clips']
    print(f"audit_reference_clips: {n_clips} clips checked")
    print(f"  frozen-leg clips: {len(frozen)} / {n_clips}")
    if frozen:
        shown = frozen[:max_report]
        more = len(frozen) - len(shown)
        suffix = f" (+{more} more)" if more > 0 else ""
        print(f"    indices: {shown}{suffix}")
        for idx in shown:
            n_static = result['frozen_joint_counts'].get(idx, 0)
            ptp = result['per_clip_max_leg_ptp'][idx]
            print(f"      clip {idx}: max leg ptp={ptp:.3g}, "
                  f"{n_static} leg joint(s) exactly static (ptp<1e-6)")

    for key, bucket in result.get('nan_clips', {}).items():
        all_nan, any_nan = bucket['all_nan'], bucket['any_nan']
        if all_nan or any_nan:
            print(f"  NaN in '{key}': {len(all_nan)} all-NaN clip(s), "
                  f"{len(any_nan)} any-NaN clip(s)")
            if all_nan:
                print(f"    all-NaN indices: {all_nan[:max_report]}")
            if any_nan:
                print(f"    any-NaN indices: {any_nan[:max_report]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit a packed reference-clip h5 for frozen leg joints "
                    "and NaN-poisoned clips (Task 12 QC gate).")
    ap.add_argument('--h5', required=True, help='Path to the packed reference-clip h5.')
    ap.add_argument('--leg-ptp-threshold', type=float, default=1e-3,
                    help='Leg columns with max peak-to-peak below this over a '
                         "clip's true (unpadded) span are considered frozen. "
                         'Default: 1e-3.')
    ap.add_argument('--max-report', type=int, default=20,
                    help='Max offending clip indices to print per category.')
    ap.add_argument('--fail-on-frozen', dest='fail_on_frozen', action='store_true',
                    default=True,
                    help='Exit 1 if any clip is frozen (default).')
    ap.add_argument('--no-fail-on-frozen', dest='fail_on_frozen', action='store_false',
                    help='Still run and report, but always exit 0.')
    args = ap.parse_args(argv)

    result = audit_clips(args.h5, leg_ptp_threshold=args.leg_ptp_threshold)
    _print_report(result, args.max_report)

    if result['frozen_leg_clips'] and args.fail_on_frozen:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
