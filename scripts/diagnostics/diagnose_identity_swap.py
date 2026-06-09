"""
Identity-swap diagnostic for multi-fly courtship tracking (Track A).

Locates a target bout in a combined IK h5, finds the matching per-session
prediction folder via ``info/fly_ids``, then compares fly0 vs fly1 raw 3D
positions in the original frame coordinates around the jump frames detected
in the combined bout. Confirms whether per-bout jumps come from upstream
JARVIS identity swaps (fly1 momentarily teleports onto fly0's position, or
vice versa) versus genuine keypoint noise or STAC instability.

Default target: bout_090 in
/data2/users/eabe/datasets/Johnson_lab/courtship/Data_analysis/analysis/v1/ik_output_combined_v1_courtship.h5
which corresponds to Session1/2026_04_02_14_54_28_fly1, csv bout_idx=5,
original frames 476249–476743.

Usage:
    python scripts/diagnose_identity_swap.py
    python scripts/diagnose_identity_swap.py --bout 90 --jump-pct 99
    python scripts/diagnose_identity_swap.py --bout 47
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


COMBINED_DEFAULT = (
    "/data2/users/eabe/datasets/Johnson_lab/courtship/Data_analysis/"
    "analysis/v1/ik_output_combined_v1_courtship.h5"
)
COURTSHIP_ROOT = Path("/data2/users/eabe/datasets/Johnson_lab/courtship")


def _decode_group(g):
    """Decode an h5 'group of indexed datasets' into a python list."""
    keys = sorted(g.keys(), key=int)
    out = []
    for k in keys:
        v = g[k][()]
        if isinstance(v, bytes):
            v = v.decode()
        out.append(v)
    return out


def find_session_folder(fly_id: str) -> Path | None:
    """Match a fly_id like 'Session1/2026_04_02_14_54_28_fly1' to its
    Predictions_3D_* folder by greping info.yaml for the timestamp."""
    m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", fly_id)
    if not m:
        return None
    ts = m.group(1)
    for d in sorted(COURTSHIP_ROOT.glob("Predictions_3D_*")):
        info = d / "info.yaml"
        if info.exists() and ts in info.read_text():
            return d
    return None


def find_bout_in_session(session_dir: Path, fly_label: str, kp_data_combined: np.ndarray) -> tuple[int, int]:
    """Find the bout in the per-session ik_output file whose ``kp_data_stac``
    exactly matches the combined kp_data, returning (bout_index, n_frames)."""
    p = session_dir / "postprocessing" / f"ik_output_v1_courtship_{fly_label}.h5"
    with h5py.File(p, "r") as fs:
        bouts = sorted(k for k in fs.keys() if k.startswith("bout"))
        for b in bouts:
            kps = fs[b]["kp_data_stac"][:]
            if kps.shape == kp_data_combined.shape and np.allclose(kps, kp_data_combined, atol=1e-5):
                return int(b.split("_")[1]), kps.shape[0]
    raise RuntimeError(f"No matching bout found in {p}")


def load_csv_window(session_dir: Path, fly_label: str, start: int, n: int) -> pd.DataFrame:
    df = pd.read_csv(session_dir / f"data3D_{fly_label}.csv", header=[0, 1])
    df.columns = ["_".join(c).strip() for c in df.columns]
    return df.iloc[start:start + n].reset_index(drop=True)


def per_frame_velocity(xyz: np.ndarray) -> np.ndarray:
    v = np.zeros(len(xyz))
    v[1:] = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return v


def diagnose(combined_path: str, bout_idx: int, jump_pct: float = 99.0,
             jump_min_factor: float = 5.0) -> dict:
    """Returns a dict summary suitable for printing or programmatic use."""
    bout_key = f"bout_{bout_idx:03d}"
    with h5py.File(combined_path, "r") as fc:
        info = fc["info"]
        fly_ids = _decode_group(info["fly_ids"])
        kp_names = _decode_group(info["kp_names"])
        if bout_key not in fc:
            raise KeyError(f"{bout_key} not in {combined_path}")
        b = fc[bout_key]
        kp_combined = b["kp_data"][:]                  # (T, N*3)
        T = kp_combined.shape[0]
        kp_combined = kp_combined.reshape(T, -1, 3)    # (T, N, 3)
        marker_sites = b["marker_sites"][:]            # (T, N, 3) post-STAC

    fly_id = fly_ids[bout_idx]
    fly_label = "fly1" if fly_id.endswith("_fly1") else "fly0"
    other_label = "fly0" if fly_label == "fly1" else "fly1"

    # ── Detect jumps in combined kp ───────────────────────────────────────────
    centroid = np.nanmean(kp_combined, axis=1)         # (T, 3)
    cdiff = np.zeros(T)
    cdiff[1:] = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    p99 = float(np.nanpercentile(cdiff, jump_pct))
    median = float(np.nanmedian(cdiff[1:]))
    threshold = max(p99, jump_min_factor * median)
    jump_frames = np.where(cdiff > threshold)[0].tolist()

    # STAC residual (how closely STAC fit the bad data)
    stac_res = float(np.nanmean(np.linalg.norm(marker_sites - kp_combined, axis=-1)))

    out = dict(
        bout_key=bout_key, fly_id=fly_id, fly_label=fly_label, T=T,
        centroid_jump_median=median, centroid_jump_p99=p99,
        centroid_jump_threshold=threshold, jump_frames=jump_frames,
        stac_marker_residual=stac_res,
    )

    # ── Locate session folder ─────────────────────────────────────────────────
    session_dir = find_session_folder(fly_id)
    if session_dir is None:
        out["error"] = "session folder not found"
        return out
    out["session_dir"] = str(session_dir)

    # ── Match the bout in the per-session file ────────────────────────────────
    sess_bout_idx, _ = find_bout_in_session(session_dir, fly_label, b["kp_data"][:] if False else kp_combined.reshape(T, -1))
    out["session_bout_idx"] = sess_bout_idx

    # ── Map to original frame range via the bouts summary csv ─────────────────
    summary_csv = session_dir / f"courtship_bouts_{fly_label}_summary.csv"
    df_sum = pd.read_csv(summary_csv)
    # csv bout_idx is 1-indexed, h5 bout_xxx is 0-indexed
    row = df_sum.iloc[sess_bout_idx]
    start_orig = int(row["start_frame"])
    n_csv = int(row["n_frames"])
    out["original_start_frame"] = start_orig
    out["original_n_frames_csv"] = n_csv

    # ── Load both flies' raw 3D in the same window from the data3D csv ────────
    sub_self = load_csv_window(session_dir, fly_label, start_orig, T)
    sub_other = load_csv_window(session_dir, other_label, start_orig, T)

    def trio(df, name):
        return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].values

    self_scut = trio(sub_self, "Scutellum")
    other_scut = trio(sub_other, "Scutellum")
    self_vel = per_frame_velocity(self_scut)
    other_vel = per_frame_velocity(other_scut)

    # body lengths
    def body_len(df):
        a = trio(df, "Antenna_Base"); z = trio(df, "Abd_tip")
        return np.linalg.norm(a - z, axis=1)

    bl_self = body_len(sub_self)
    bl_other = body_len(sub_other)

    out.update(dict(
        self_vel_p99=float(np.nanpercentile(self_vel, 99)),
        self_vel_max=float(np.nanmax(self_vel)),
        other_vel_p99=float(np.nanpercentile(other_vel, 99)),
        other_vel_max=float(np.nanmax(other_vel)),
        body_len_self_mean=float(np.nanmean(bl_self)),
        body_len_self_std=float(np.nanstd(bl_self)),
        body_len_other_mean=float(np.nanmean(bl_other)),
        body_len_other_std=float(np.nanstd(bl_other)),
    ))

    # ── Per-jump-frame swap test ──────────────────────────────────────────────
    # A swap is suspected if the self-fly's position at the jump frame is closer
    # to the other-fly's *recent* position than to its own previous position.
    swap_evidence = []
    for fr in jump_frames:
        if fr < 1 or fr >= T:
            continue
        self_now = self_scut[fr]
        self_prev = self_scut[fr - 1]
        other_prev = other_scut[fr - 1]
        d_self_to_self_prev = float(np.linalg.norm(self_now - self_prev))
        d_self_to_other_prev = float(np.linalg.norm(self_now - other_prev))
        is_swap = d_self_to_other_prev < d_self_to_self_prev
        swap_evidence.append(dict(
            frame_rel=int(fr),
            frame_orig=int(start_orig + fr),
            d_self_to_self_prev=d_self_to_self_prev,
            d_self_to_other_prev=d_self_to_other_prev,
            looks_like_swap=bool(is_swap),
        ))
    out["swap_evidence"] = swap_evidence
    out["n_jumps_looking_like_swaps"] = sum(1 for e in swap_evidence if e["looks_like_swap"])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default=COMBINED_DEFAULT)
    ap.add_argument("--bout", type=int, default=90)
    ap.add_argument("--jump-pct", type=float, default=99.0)
    args = ap.parse_args()

    rep = diagnose(args.combined, args.bout, args.jump_pct)
    print(f"\n=== {rep['bout_key']}  {rep['fly_id']} ===")
    print(f"  T={rep['T']}, STAC mean residual = {rep['stac_marker_residual']:.4f}")
    print(f"  centroid jump   median={rep['centroid_jump_median']:.4f}  "
          f"p99={rep['centroid_jump_p99']:.4f}  threshold={rep['centroid_jump_threshold']:.4f}")
    print(f"  jump frames (relative): {rep['jump_frames']}")
    if "session_dir" in rep:
        print(f"  session: {rep['session_dir']}  bout_h5={rep['session_bout_idx']}  "
              f"orig_start={rep['original_start_frame']}")
        print(f"  vel  self  p99={rep['self_vel_p99']:.3f}  max={rep['self_vel_max']:.3f}")
        print(f"  vel  other p99={rep['other_vel_p99']:.3f}  max={rep['other_vel_max']:.3f}")
        print(f"  body_len self  mean={rep['body_len_self_mean']:.3f}  std={rep['body_len_self_std']:.3f}")
        print(f"  body_len other mean={rep['body_len_other_mean']:.3f}  std={rep['body_len_other_std']:.3f}")
        print(f"  jumps looking like ID swaps: "
              f"{rep['n_jumps_looking_like_swaps']}/{len(rep['swap_evidence'])}")
        for e in rep["swap_evidence"]:
            tag = "SWAP" if e["looks_like_swap"] else "    "
            print(f"    [{tag}] fr_rel={e['frame_rel']:4d} fr_orig={e['frame_orig']:7d}  "
                  f"d(self,self_prev)={e['d_self_to_self_prev']:7.3f}  "
                  f"d(self,other_prev)={e['d_self_to_other_prev']:7.3f}")
    else:
        print("  ERROR:", rep.get("error"))


if __name__ == "__main__":
    main()
