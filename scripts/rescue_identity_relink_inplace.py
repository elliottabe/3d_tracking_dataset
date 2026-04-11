#!/usr/bin/env python3
"""Offline identity-relink rescue for an existing combined-bouts H5.

The dual-fly preprocessing pipeline used to run ``identity_relink.relink_pair``
once per fly CSV and discard the sibling output. The two per-fly runs could
disagree on whether a frame was swapped, leaving the merged ``_both.h5`` with
fly0 and fly1 entries that point at the same physical animal for the swapped
frames. The user noticed this in courtship bout 33 (and many others).

This script rescues an existing combined ``_both.h5`` *without* re-running
preprocessing or STAC. For every fly0 / fly1 entry pair (consecutive bouts in
the file, by the convention written by ``batch_split_valid_bouts.py``), it:

  1. Runs ``relink_pair`` jointly on the two flies' ``keypoints`` arrays
     (per-bout, so EMA / velocity / body-length state resets between bouts).
  2. For every per-frame array on the bout dict whose leading dim equals the
     bout length, swaps the t-th slice between fly0 and fly1 wherever
     ``swap_state[t]`` is True. This keeps qpos / xpos / kp_data / etc.
     consistent with the corrected keypoints, so the user's existing STAC IK
     output stays usable.
  3. Recomputes ``valid_fly0`` / ``valid_fly1`` / ``valid_both`` via
     ``compute_pair_validity`` using the *real* fly0 + fly1 pair (the
     preprocessing call was bogus — it passed ``(self, self)``).
  4. Writes the rescued data to a new H5 path (the original is preserved).

Usage:
    python scripts/rescue_identity_relink_inplace.py \\
        --in  /path/to/ik_output_combined_v1_courtship_both.h5 \\
        --out /path/to/ik_output_combined_v1_courtship_both_relinked.h5

After rescue, point the notebook ``h5_path`` at the ``_relinked`` file and
re-run the two-fly render cell on bout 33 — the swap should be gone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import io_dict_to_hdf5 as ioh5  # noqa: E402
from utils.identity_relink import RelinkConfig, relink_pair  # noqa: E402
from utils.pair_validity import (  # noqa: E402
    PairValidityConfig,
    compute_pair_validity,
)


def _bout_keys_sorted(d: dict) -> List[str]:
    return sorted(k for k in d.keys() if k != "info")


def _per_frame_keys(bout: dict, T: int) -> List[str]:
    """Return the keys whose value is an ndarray with leading dim == T."""
    out = []
    for k, v in bout.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim == 0:
            continue
        if v.shape[0] != T:
            continue
        out.append(k)
    return out


def _swap_per_frame_arrays(
    bout0: dict,
    bout1: dict,
    swap_state: np.ndarray,
) -> List[str]:
    """For every per-frame array present in *both* bouts, swap rows where
    ``swap_state`` is True. Returns the list of keys that were swapped."""
    T = swap_state.shape[0]
    keys0 = set(_per_frame_keys(bout0, T))
    keys1 = set(_per_frame_keys(bout1, T))
    # Only swap arrays that exist on both sides AND have matching shape.
    common = []
    for k in sorted(keys0 & keys1):
        a = bout0[k]
        b = bout1[k]
        if a.shape != b.shape:
            continue
        common.append(k)

    mask = swap_state.astype(bool)
    if not mask.any():
        return common

    for k in common:
        a = bout0[k]
        b = bout1[k]
        # In-place swap of the masked rows.
        tmp = a[mask].copy()
        a[mask] = b[mask]
        b[mask] = tmp
    return common


def _pair_bouts(
    bout_keys: List[str],
    info: dict,
) -> List[Tuple[str, str]]:
    """Pair fly0/fly1 entries written by batch_split_valid_bouts.py.

    The writer emits one entry per fly per shared source bout in fly0-then-fly1
    order, so the natural pairing is consecutive (k0, k1) with bucket=='both'.
    We additionally validate the pairing using the per-bout ``source_flies``
    info field when available — it must alternate fly0/fly1.
    """
    src = list(info.get("source_flies", []))
    bucket = list(info.get("bucket", []))
    pairs: List[Tuple[str, str]] = []
    if src and len(src) == len(bout_keys):
        # Use source_flies to confirm pairing.
        i = 0
        while i + 1 < len(bout_keys):
            if (
                src[i] == "fly0"
                and src[i + 1] == "fly1"
                and (not bucket or bucket[i] == "both" == bucket[i + 1])
            ):
                pairs.append((bout_keys[i], bout_keys[i + 1]))
                i += 2
            else:
                # Skip orphan / unexpected ordering.
                i += 1
    else:
        # Fallback: assume strict consecutive pairing.
        for i in range(0, len(bout_keys) - 1, 2):
            pairs.append((bout_keys[i], bout_keys[i + 1]))
    return pairs


def _resolve_kp_names(d: dict) -> List[str]:
    """Find the keypoint name list. Different stages store it in different
    spots — accept any of them."""
    info = d.get("info", {}) or {}
    for k in ("kp_names", "site_names_egocentric"):
        if k in info:
            v = info[k]
            if isinstance(v, dict):
                v = [v[kk] for kk in sorted(v.keys(), key=lambda x: int(x))]
            return list(v)
    # Last-ditch: any bout dict that carries kp_names directly.
    for bk in d:
        if bk == "info":
            continue
        if isinstance(d[bk], dict) and "kp_names" in d[bk]:
            return list(d[bk]["kp_names"])
    raise KeyError("Could not locate kp_names in the H5 (looked under info and bouts)")


def _make_pv_cfg(
    d: dict,
    min_pair_separation_mm: float | None = None,
    colocation_centroid_kp: str | None = None,
) -> PairValidityConfig:
    """Build a PairValidityConfig from the info echo if present, else
    default.

    CLI overrides (``min_pair_separation_mm`` / ``colocation_centroid_kp``)
    take precedence over the echoed values so the user can rescue an old
    combined h5 whose ``info['pair_validity']`` was written before the
    identity-collapse detector was introduced.
    """
    pv_info = (d.get("info", {}) or {}).get("pair_validity", {}) or {}
    kwargs = {}
    for k in (
        "ground_epsilon_mm",
        "floor_percentile",
        "swap_guard_frames",
        "min_paired_frames",
        "min_solo_frames",
        "min_pair_separation_mm",
        "colocation_centroid_kp",
    ):
        if k in pv_info:
            kwargs[k] = pv_info[k]
    if "critical_kp_patterns" in pv_info:
        kwargs["critical_kp_patterns"] = tuple(pv_info["critical_kp_patterns"])
    if "ground_kp_patterns" in pv_info:
        kwargs["ground_kp_patterns"] = tuple(pv_info["ground_kp_patterns"])
    if min_pair_separation_mm is not None:
        kwargs["min_pair_separation_mm"] = float(min_pair_separation_mm)
    if colocation_centroid_kp is not None:
        kwargs["colocation_centroid_kp"] = str(colocation_centroid_kp)
    return PairValidityConfig(**kwargs)


def _pick_keypoint_array(bout: dict) -> Tuple[str, np.ndarray]:
    """Return (key_name, array) for the (T, N, 3) keypoint array used by
    relink.

    *World-frame* keypoints are required — egocentric arrays are useless
    here because both flies sit at the origin in their own frame, so the
    relink algorithm can't tell them apart. Equally important: the array
    must be in a frame *shared* between the two flies, so the relink can
    compare inter-fly distances honestly. Order of preference:

      1. ``orig_keypoints``  (per-fly preprocessed h5; pre-Procrustes,
                              SHARED raw camera frame for both flies — best)
      2. ``keypoints``       (per-fly preprocessed h5; post-Procrustes,
                              per-fly scaled — inter-fly distances are
                              technically miscalibrated but usable)
      3. ``kp_data``         (post-STAC h5; same data as keypoints)
      4. ``marker_sites``    (post-STAC h5; same as kp_data, alt spelling)
      5. ``site_xpos``       (post-STAC h5; STAC's fitted site positions)

    ``xpos_egocentric`` is intentionally excluded — it is centered on each
    fly's body and contains no inter-fly position information.
    """
    for k in ("orig_keypoints", "keypoints", "kp_data", "marker_sites",
              "site_xpos"):
        if k in bout and isinstance(bout[k], np.ndarray):
            arr = bout[k]
            if arr.ndim == 3 and arr.shape[-1] == 3:
                return k, arr
            if arr.ndim == 2 and arr.shape[-1] % 3 == 0:
                # flat (T, N*3) → reshape view
                T = arr.shape[0]
                N = arr.shape[1] // 3
                return k, arr.reshape(T, N, 3)
    raise KeyError("No world-frame (T, N, 3) keypoint array found on the bout dict")


def rescue_relink_both_bucket(
    in_path: Path,
    out_path: Path,
    cfg: RelinkConfig,
    min_pair_separation_mm: float | None = None,
    colocation_centroid_kp: str | None = None,
) -> dict:
    """Run joint per-bout relink + real pair_validity on a combined bouts H5.

    See the module docstring for the why and the algorithm. Returns a summary
    dict suitable for printing or saving alongside the rescued H5.

    Args:
        min_pair_separation_mm: CLI override for
            ``PairValidityConfig.min_pair_separation_mm``. When > 0, frames
            where the two flies' centroid keypoints are closer than this
            threshold are marked invalid for BOTH flies — this catches the
            identity-collapse failure mode where the tracker drops two
            tracks onto the same physical animal.
        colocation_centroid_kp: CLI override for the centroid keypoint name
            used by the co-location detector.
    """
    print(f"\nLoading: {in_path}")
    d = ioh5.load(str(in_path), enable_jax=False)

    bout_keys = _bout_keys_sorted(d)
    info = d.get("info", {}) or {}
    print(f"  loaded {len(bout_keys)} bouts; info keys: {sorted(info.keys())}")

    kp_names = _resolve_kp_names(d)
    print(f"  using {len(kp_names)} keypoint names (first few: {kp_names[:5]})")

    pairs = _pair_bouts(bout_keys, info)
    print(f"  identified {len(pairs)} fly0/fly1 pairs")

    pv_cfg = _make_pv_cfg(
        d,
        min_pair_separation_mm=min_pair_separation_mm,
        colocation_centroid_kp=colocation_centroid_kp,
    )
    print(f"  pair_validity cfg: {pv_cfg}")

    n_pairs_with_swap = 0
    n_pairs_skipped = 0
    total_frames = 0
    swapped_frames = 0
    swap_segments = 0
    colocated_frames = 0
    n_pairs_with_colocation = 0
    swapped_keys_per_pair: Dict[str, Sequence[str]] = {}

    for k0, k1 in pairs:
        b0 = d[k0]
        b1 = d[k1]
        try:
            kp_key0, kp0 = _pick_keypoint_array(b0)
            kp_key1, kp1 = _pick_keypoint_array(b1)
        except KeyError as e:
            print(f"  ! skipping pair ({k0}, {k1}): {e}")
            n_pairs_skipped += 1
            continue

        if kp0.shape != kp1.shape:
            T = min(kp0.shape[0], kp1.shape[0])
            if T == 0 or kp0.shape[1:] != kp1.shape[1:]:
                print(f"  ! skipping pair ({k0}, {k1}): shape mismatch "
                      f"{kp0.shape} vs {kp1.shape}")
                n_pairs_skipped += 1
                continue
            kp0 = kp0[:T]
            kp1 = kp1[:T]

        rl0, rl1, log = relink_pair(kp0, kp1, kp_names, cfg)
        swap_state = np.asarray(log["swap_state"], dtype=bool)

        # Write the corrected keypoint array back into the bout dicts.
        b0[kp_key0] = rl0
        b1[kp_key1] = rl1
        b0["swap_state"] = swap_state.copy()
        b1["swap_state"] = swap_state.copy()
        b0["n_swap_segments"] = int(log["n_swap_segments"])
        b1["n_swap_segments"] = int(log["n_swap_segments"])
        b0["fraction_swapped"] = float(log["fraction_swapped"])
        b1["fraction_swapped"] = float(log["fraction_swapped"])

        # Apply the same swap mask to every other per-frame array on the bout
        # dicts (qpos, qvel, xpos, xquat, site_xpos, xpos_egocentric, kp_data,
        # valid_fly{0,1}, etc) so the rescue stays internally consistent.
        # We must skip the keypoint key we just overwrote.
        # Mask out the keypoint key from each side first.
        kp_key_set = {kp_key0, kp_key1}
        # Build temporary dicts that exclude the kp keys, swap, then restore.
        b0_kp_buf = {k: b0.pop(k) for k in list(kp_key_set) if k in b0}
        b1_kp_buf = {k: b1.pop(k) for k in list(kp_key_set) if k in b1}
        try:
            common_keys = _swap_per_frame_arrays(b0, b1, swap_state)
        finally:
            b0.update(b0_kp_buf)
            b1.update(b1_kp_buf)
        swapped_keys_per_pair[f"{k0}|{k1}"] = common_keys

        # Recompute pair_validity with the *real* (fly0, fly1) pair.
        pv_out = compute_pair_validity(
            rl0, rl1, kp_names, cfg=pv_cfg, swap_state=swap_state,
        )
        b0["valid_fly0"] = pv_out["valid_fly0"]
        b0["valid_fly1"] = pv_out["valid_fly1"]
        b0["valid_both"] = pv_out["valid_both"]
        b1["valid_fly0"] = pv_out["valid_fly0"]
        b1["valid_fly1"] = pv_out["valid_fly1"]
        b1["valid_both"] = pv_out["valid_both"]
        # Persist the collapse mask for downstream diagnostics / re-runs so
        # the rescue is idempotent and the notebook can plot it alongside
        # the wing diagnostics without recomputing.
        pair_colocated = pv_out.get("pair_colocated")
        if pair_colocated is not None:
            b0["pair_colocated"] = pair_colocated
            b1["pair_colocated"] = pair_colocated
            n_coloc = int(pair_colocated.sum())
            colocated_frames += n_coloc
            if n_coloc > 0:
                n_pairs_with_colocation += 1

        total_frames += int(swap_state.shape[0])
        swapped_frames += int(swap_state.sum())
        swap_segments += int(log["n_swap_segments"])
        if log["n_swap_segments"] > 0:
            n_pairs_with_swap += 1

    summary = dict(
        in_path=str(in_path),
        out_path=str(out_path),
        n_pairs=len(pairs),
        n_pairs_with_swap=int(n_pairs_with_swap),
        n_pairs_skipped=int(n_pairs_skipped),
        total_frames=int(total_frames),
        swapped_frames=int(swapped_frames),
        fraction_swapped=(swapped_frames / total_frames) if total_frames else 0.0,
        n_swap_segments=int(swap_segments),
        colocated_frames=int(colocated_frames),
        n_pairs_with_colocation=int(n_pairs_with_colocation),
        fraction_colocated=(colocated_frames / total_frames) if total_frames else 0.0,
        min_pair_separation_mm=float(pv_cfg.min_pair_separation_mm),
        colocation_centroid_kp=str(pv_cfg.colocation_centroid_kp),
        relink_cfg=dict(
            swap_ratio=cfg.swap_ratio,
            bl_tube_factor=cfg.bl_tube_factor,
            max_step_bl=cfg.max_step_bl,
            max_step_abs=cfg.max_step_abs,
            nan_resume_frames=cfg.nan_resume_frames,
            velocity_alpha=cfg.velocity_alpha,
            body_length_alpha=cfg.body_length_alpha,
            body_length_weight=cfg.body_length_weight,
        ),
    )

    # Echo the summary into info so downstream consumers can audit.
    if "info" not in d or not isinstance(d["info"], dict):
        d["info"] = {}
    d["info"]["identity_relink_rescue"] = summary

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving rescued H5: {out_path}")
    ioh5.save(str(out_path), d)
    print("  ✓ saved")

    return summary


def _build_relink_cfg(args: argparse.Namespace) -> RelinkConfig:
    return RelinkConfig(
        swap_ratio=args.swap_ratio,
        bl_tube_factor=args.bl_tube_factor,
        max_step_bl=args.max_step_bl,
        max_step_abs=args.max_step_abs,
        nan_resume_frames=args.nan_resume_frames,
        velocity_alpha=args.velocity_alpha,
        body_length_alpha=args.body_length_alpha,
        body_length_weight=args.body_length_weight,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path,
                        help="Path to the existing combined bouts H5.")
    parser.add_argument("--out", dest="out_path", required=True, type=Path,
                        help="Path for the rescued H5 (must differ from --in).")
    parser.add_argument("--swap-ratio", type=float, default=0.7)
    parser.add_argument("--bl-tube-factor", type=float, default=0.25)
    # Body-length-relative step ceiling — unit-agnostic, works for cgs (cm)
    # or mm or any other input scale because the threshold is computed as
    # max(max_step_abs, max_step_bl * mean_body_length).
    parser.add_argument("--max-step-bl", type=float, default=0.5,
                        help="Predicted-step ceiling as a multiple of body length per frame.")
    parser.add_argument("--max-step-abs", type=float, default=0.0,
                        help="Absolute floor on the predicted-step ceiling, "
                             "in the same units as the input keypoints.")
    parser.add_argument("--nan-resume-frames", type=int, default=3)
    parser.add_argument("--velocity-alpha", type=float, default=0.5)
    parser.add_argument("--body-length-alpha", type=float, default=0.05)
    parser.add_argument("--body-length-weight", type=float, default=0.5)
    # Identity-collapse detector. Overrides whatever value is echoed in the
    # input h5's info['pair_validity']. Pass None (omit the flag) to defer
    # to the echoed config.
    parser.add_argument(
        "--min-pair-separation-mm",
        type=float,
        default=None,
        help="Inter-fly centroid distance (mm) below which frames are marked "
             "invalid for both flies. Rescues identity-collapse bouts where "
             "the tracker dropped two tracks onto the same animal. "
             "Overrides info['pair_validity']['min_pair_separation_mm']. "
             "Recommended: 1.0 for Drosophila courtship.",
    )
    parser.add_argument(
        "--colocation-centroid-kp",
        type=str,
        default=None,
        help="Keypoint name used by the co-location detector as each fly's "
             "centroid. Overrides info['pair_validity']['colocation_centroid_kp']. "
             "Default: 'Scutellum'.",
    )
    args = parser.parse_args(argv)

    if args.out_path == args.in_path:
        parser.error("--out must differ from --in (we never overwrite the original)")
    if not args.in_path.exists():
        parser.error(f"input H5 does not exist: {args.in_path}")

    cfg = _build_relink_cfg(args)
    rescue_relink_both_bucket(
        args.in_path,
        args.out_path,
        cfg,
        min_pair_separation_mm=args.min_pair_separation_mm,
        colocation_centroid_kp=args.colocation_centroid_kp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
