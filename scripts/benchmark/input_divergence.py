"""Benchmark integrity check: did the four A/B variants really share identical
2D (ViTPose kp2d) inputs?

The benchmark freezes kp2d.npz/kp3d.npz/kp3d_filt.npz per bout-fly and
hardlinks them into each variant root (see freeze_inputs.py, run_variant.py)
so the four scale/calibration variants differ ONLY in downstream treatment.
But scripts/run_bout.py's staleness cascade (masks_are_stale()) deletes those
same artifacts for bout-flies whose recording's sync_plan status is
trim/reindex -- which means Stage A (ViTPose) reruns independently, per
variant, for those bout-flies, silently breaking the controlled comparison.

This module compares each variant's kp2d.npz against the frozen copy for
every (run_key, bout, fly) in the manifest and reports which bout-flies
stayed frozen (hardlink/byte-identical) vs. were regenerated, and by how
much the regenerated 2D actually diverged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.benchmark.manifest import entries, load_manifest

VARIANTS = ("trunk_umeyama_segcal", "trunk_umeyama_nosegcal",
            "all_norm_segcal", "all_norm_nosegcal")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _byte_identical(a: Path, b: Path) -> bool:
    """Cheap identity check: same inode (hardlink) first, else content hash."""
    sa, sb = a.stat(), b.stat()
    if sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino:
        return True
    return _sha256(a) == _sha256(b)


def _nan_safe_dist(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Per-(...,2) euclidean distance, NaN-in-both-arrays == 0, NaN-in-one == inf."""
    ref = ref.astype(np.float64)
    other = other.astype(np.float64)
    is_nan_r = np.isnan(ref)
    is_nan_o = np.isnan(other)
    both_nan = is_nan_r & is_nan_o
    only_one_nan = is_nan_r ^ is_nan_o
    diff = ref - other
    diff = np.where(both_nan, 0.0, diff)
    diff = np.where(only_one_nan, np.inf, diff)
    return np.sqrt(np.sum(diff * diff, axis=-1))


def _nan_safe_abs_diff(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    ref = ref.astype(np.float64)
    other = other.astype(np.float64)
    is_nan_r = np.isnan(ref)
    is_nan_o = np.isnan(other)
    both_nan = is_nan_r & is_nan_o
    only_one_nan = is_nan_r ^ is_nan_o
    diff = np.abs(ref - other)
    diff = np.where(both_nan, 0.0, diff)
    diff = np.where(only_one_nan, np.inf, diff)
    return diff


def kp2d_stats(ref: Path, other: Path) -> dict:
    """Compare two kp2d.npz files' "kp2d" (T,C,K,2) and "conf" (T,C,K) arrays.

    NaN-in-the-same-position in both arrays compares equal (no difference).
    Shape mismatches never raise -- they short-circuit with shape_mismatch=True.
    """
    ref_z = np.load(ref)
    other_z = np.load(other)
    ref_kp2d, ref_conf = ref_z["kp2d"], ref_z["conf"]
    other_kp2d, other_conf = other_z["kp2d"], other_z["conf"]

    if ref_kp2d.shape != other_kp2d.shape or ref_conf.shape != other_conf.shape:
        return {"identical": False, "median_px": None, "p99_px": None,
                "max_px": None, "frac_gt_half_px": None, "max_conf_diff": None,
                "shape_mismatch": True}

    dist = _nan_safe_dist(ref_kp2d, other_kp2d)
    conf_diff = _nan_safe_abs_diff(ref_conf, other_conf)

    max_px = float(np.max(dist)) if dist.size else 0.0
    median_px = float(np.median(dist)) if dist.size else 0.0
    p99_px = float(np.percentile(dist, 99)) if dist.size else 0.0
    frac_gt_half_px = float(np.mean(dist > 0.5)) if dist.size else 0.0
    max_conf_diff = float(np.max(conf_diff)) if conf_diff.size else 0.0

    identical = bool(max_px == 0.0 and max_conf_diff == 0.0)
    return {"identical": identical, "median_px": median_px, "p99_px": p99_px,
            "max_px": max_px, "frac_gt_half_px": frac_gt_half_px,
            "max_conf_diff": max_conf_diff, "shape_mismatch": False}


def _kp2d_rel(run_key: str, bout: int, fly: int) -> Path:
    return (Path(run_key) / "bouts" / f"bout_{int(bout):05d}"
             / f"fly{fly}" / "kp2d.npz")


def scan_divergence(manifest: dict, variants_root: Path,
                    frozen_root: Path) -> list[dict]:
    """Compare each variant's kp2d.npz against the frozen copy, per bout-fly."""
    variants_root = Path(variants_root)
    frozen_root = Path(frozen_root)
    rows = []
    for e in entries(manifest):
        for fly in e["flies"]:
            rel = _kp2d_rel(e["run_key"], e["bout"], fly)
            frozen_path = frozen_root / rel

            present, missing = [], []
            frozen_identical, regenerated = [], []
            stats = {}
            for variant in VARIANTS:
                vpath = variants_root / variant / rel
                if not vpath.is_file():
                    missing.append(variant)
                    continue
                present.append(variant)
                if not frozen_path.is_file():
                    # No frozen reference to compare against -- can't be sure
                    # it's frozen-identical, so treat conservatively as
                    # regenerated but without a numeric comparison.
                    regenerated.append(variant)
                    stats[variant] = None
                    continue
                if _byte_identical(frozen_path, vpath):
                    frozen_identical.append(variant)
                else:
                    regenerated.append(variant)
                    stats[variant] = kp2d_stats(frozen_path, vpath)

            worst = None
            worst_vals = [stats[v]["max_px"] for v in regenerated
                          if stats.get(v) is not None
                          and stats[v]["max_px"] is not None]
            if worst_vals:
                worst = max(worst_vals)

            rows.append({
                "run_key": e["run_key"], "bout": e["bout"], "fly": fly,
                "present": present, "missing": missing,
                "frozen_identical": frozen_identical,
                "regenerated": regenerated,
                "worst": worst, "stats": stats,
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    n_bout_flies = len(rows)
    n_clean = sum(1 for r in rows if not r["regenerated"])
    n_divergent = sum(1 for r in rows if r["regenerated"])
    n_incomplete = sum(1 for r in rows if r["missing"])
    worst_vals = [r["worst"] for r in rows if r["worst"] is not None]
    worst_max_px = max(worst_vals) if worst_vals else None
    return {"n_bout_flies": n_bout_flies, "n_clean": n_clean,
            "n_divergent": n_divergent, "n_incomplete": n_incomplete,
            "worst_max_px": worst_max_px}


def _print_table(rows: list[dict]) -> None:
    for r in rows:
        tags = []
        if r["missing"]:
            tags.append("INCOMPLETE")
        if r["regenerated"]:
            tags.append("DIVERGENT")
        if not tags:
            tags.append("CLEAN")
        status = "+".join(tags)
        print(f"{r['run_key']} bout={r['bout']} fly={r['fly']}: {status}")
        if r["missing"]:
            print(f"    missing from: {', '.join(r['missing'])}")
        for variant in r["regenerated"]:
            s = r["stats"].get(variant)
            if s is None:
                print(f"    {variant}: regenerated (no frozen reference to compare)")
            elif s["shape_mismatch"]:
                print(f"    {variant}: regenerated, SHAPE MISMATCH")
            else:
                print(f"    {variant}: regenerated -- median={s['median_px']:.4f}px "
                      f"p99={s['p99_px']:.4f}px max={s['max_px']:.4f}px")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/benchmark/bouts.yaml"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    benchmark_root = Path(manifest["benchmark_root"])
    variants_root = benchmark_root / "variants"
    frozen_root = benchmark_root / "frozen"

    rows = scan_divergence(manifest, variants_root, frozen_root)
    _print_table(rows)
    summary = summarize(rows)
    print()
    print(f"summary: {json.dumps(summary)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows, "summary": summary},
                                       indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
