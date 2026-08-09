#!/usr/bin/env python3
"""Flag courtship bouts where a fly cannot be reconstructed, for exclusion.

A fly can only be triangulated in a frame that at least ``min_views`` cameras
saw. In this rig the arena is a long narrow strip and the cameras do not all
cover its ends, so a fly that parks at one end is genuinely OUTSIDE most
cameras' field of view -- measured on Session0, 5.1% of (fly, camera, frame)
views are out of frame, against only 1.4% that are in frame but unsegmented.
That is geometry, not a segmentation bug, and no amount of mask repair fixes
it. The honest response is to exclude the affected bouts rather than to emit
3D that was never observable.

A bout is KEPT only when EVERY fly clears the threshold: a bout in which one
fly reconstructs perfectly and the other never does is not a usable courtship
bout, since every interaction measure needs both.

Evidence, in order of preference (recorded per bout as `basis`):

  mask  -- ``sam3_masks.npz``'s ``valid (A,C,T)``: SAM3 actually located that
           fly in that camera. Authoritative, and the same signal the
           ``masks.min_views`` gate already uses at triangulation time.
  kp2d  -- per-view median detector confidence from ``kp2d.npz`` (the metric
           behind ``detector.view_conf_thresh``), for recordings whose masks
           were not kept (all of Session1). PERMISSIVE: the detector reports
           high confidence on ~5-8% of crops containing no fly, so this
           over-counts usable views. Treat a kp2d-basis fraction as an UPPER
           BOUND -- measured on Session0 bout 22 fly0 it reads 0.218 where the
           mask basis reads 0.000.

Non-destructive by default: writes a report and does nothing else. ``--apply``
writes an ``EXCLUDED.json`` marker into each failing bout dir, which is
reversible; it never deletes pose data.

Usage:
    python scripts/qc/bout_reconstructable.py --root <processed>/courtship
    python scripts/qc/bout_reconstructable.py --min-frac 0.8 --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"
for p in (str(PROJECT_DIR), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship"
MIN_VIEWS_DEFAULT = 4      # matches configs/pipeline.yaml masks.min_views
MIN_FRAC_DEFAULT = 0.8     # see module docstring / the report's distribution
VIEW_CONF_DEFAULT = 0.6    # matches configs/detector/vitpose_v3.yaml


def coverage_from_valid(valid, min_views=MIN_VIEWS_DEFAULT):
    """valid (C,T) bool for ONE fly -> fraction of frames with >= min_views."""
    valid = np.asarray(valid, bool)
    if valid.ndim != 2:
        raise ValueError(f"coverage_from_valid expects (C,T), got {valid.shape}")
    if valid.shape[1] == 0:
        return 0.0
    return float((valid.sum(axis=0) >= min_views).mean())


def coverage_from_conf(conf, min_views=MIN_VIEWS_DEFAULT,
                       view_conf=VIEW_CONF_DEFAULT):
    """conf (T,C,K) detector confidence for ONE fly -> fraction of frames with
    >= min_views cameras whose MEDIAN keypoint confidence clears `view_conf`."""
    from jarvis_jax.tracking.triangulate import view_median_conf
    conf = np.asarray(conf, np.float32)
    if conf.ndim != 3:
        raise ValueError(f"coverage_from_conf expects (T,C,K), got {conf.shape}")
    if conf.shape[0] == 0:
        return 0.0
    vm = view_median_conf(conf)                       # (T,C)
    return float(((vm >= view_conf).sum(axis=1) >= min_views).mean())


def bout_verdict(fractions, min_frac=MIN_FRAC_DEFAULT):
    """Per-fly coverage fractions -> (keep: bool, reason: str).

    Keep only if EVERY fly clears `min_frac`; a bout is unusable when either
    animal is unobservable, however good the other one is.
    """
    fractions = list(fractions)
    if not fractions:
        return False, "no flies found"
    worst = min(fractions)
    if worst >= min_frac:
        return True, f"all flies >= {min_frac:.2f} (worst {worst:.3f})"
    bad = [i for i, f in enumerate(fractions) if f < min_frac]
    detail = ", ".join(f"fly{i}={fractions[i]:.3f}" for i in bad)
    return False, f"below {min_frac:.2f}: {detail}"


def _mask_valid(rec_dir: Path, bout: int):
    npz = rec_dir / "sam3_masks" / f"bout_{bout:05d}" / "sam3_masks.npz"
    if not npz.is_file():
        return None
    with np.load(npz, allow_pickle=True) as z:
        return np.asarray(z["valid"], bool)            # (A,C,T)


def _kp2d_confs(rec_dir: Path, bout: int):
    """[conf (T,C,K), ...] per fly dir, or None when the bout has no kp2d."""
    bdir = rec_dir / "pose" / "bouts" / f"bout_{bout:05d}"
    if not bdir.is_dir():
        return None
    out = []
    for fly_dir in sorted(bdir.glob("fly*")):
        kp = fly_dir / "kp2d.npz"
        if not kp.is_file():
            continue
        with np.load(kp) as z:
            out.append(np.asarray(z["conf"], np.float32))
    return out or None


def scan_recording(rec_dir: Path, *, min_views=MIN_VIEWS_DEFAULT,
                   min_frac=MIN_FRAC_DEFAULT, view_conf=VIEW_CONF_DEFAULT,
                   basis="auto"):
    """One recording -> list of per-bout dicts."""
    rec_dir = Path(rec_dir)
    bouts_dir = rec_dir / "pose" / "bouts"
    if not bouts_dir.is_dir():
        return []
    rows = []
    for bdir in sorted(bouts_dir.glob("bout_*")):
        try:
            bout = int(bdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        fracs, used = None, None
        if basis in ("auto", "mask"):
            valid = _mask_valid(rec_dir, bout)
            if valid is not None:
                fracs = [coverage_from_valid(valid[a], min_views)
                         for a in range(valid.shape[0])]
                used = "mask"
        if fracs is None and basis in ("auto", "kp2d"):
            confs = _kp2d_confs(rec_dir, bout)
            if confs is not None:
                fracs = [coverage_from_conf(c, min_views, view_conf)
                         for c in confs]
                used = "kp2d"
        if fracs is None:
            rows.append({"recording": rec_dir.name, "bout": bout,
                         "basis": "none", "fractions": [], "keep": False,
                         "reason": "no masks and no kp2d -- cannot assess"})
            continue
        keep, reason = bout_verdict(fracs, min_frac)
        rows.append({"recording": rec_dir.name, "bout": bout, "basis": used,
                     "fractions": [round(f, 4) for f in fracs],
                     "keep": keep, "reason": reason})
    return rows


def scan_root(root: Path, **kw):
    """Walk <root>/<session>/<recording>/ and scan every recording found."""
    root = Path(root)
    rows = []
    for sess in sorted(p for p in root.iterdir() if p.is_dir()):
        for rec in sorted(p for p in sess.iterdir() if p.is_dir()):
            for r in scan_recording(rec, **kw):
                r["session"] = sess.name
                rows.append(r)
    return rows


def write_report(rows, out_json: Path, out_csv: Path | None = None):
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if r["keep"]]
    payload = {
        "n_bouts": len(rows),
        "n_keep": len(kept),
        "n_drop": len(rows) - len(kept),
        "by_basis": {b: sum(1 for r in rows if r["basis"] == b)
                     for b in sorted({r["basis"] for r in rows})},
        "bouts": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    if out_csv:
        out_csv = Path(out_csv)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["session", "recording", "bout", "basis", "keep",
                        "worst_fly_fraction", "fractions", "reason"])
            for r in rows:
                w.writerow([r.get("session", ""), r["recording"], r["bout"],
                            r["basis"], int(r["keep"]),
                            min(r["fractions"]) if r["fractions"] else "",
                            ";".join(str(x) for x in r["fractions"]),
                            r["reason"]])
    return payload


def apply_exclusions(rows, root: Path):
    """Write an EXCLUDED.json marker into each failing bout dir.

    Reversible and non-destructive: no pose data is deleted. Bouts that pass
    have any stale marker REMOVED, so re-running after a threshold change
    cannot leave a bout wrongly excluded.
    """
    root = Path(root)
    n_add = n_clear = 0
    for r in rows:
        bdir = (root / r.get("session", "") / r["recording"] / "pose"
                / "bouts" / f"bout_{int(r['bout']):05d}")
        if not bdir.is_dir():
            continue
        marker = bdir / "EXCLUDED.json"
        if r["keep"]:
            if marker.exists():
                marker.unlink()
                n_clear += 1
        else:
            marker.write_text(json.dumps({
                "reason": r["reason"], "basis": r["basis"],
                "fractions": r["fractions"],
                "tool": "scripts/qc/bout_reconstructable.py",
            }, indent=2))
            n_add += 1
    return n_add, n_clear


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--min-views", type=int, default=MIN_VIEWS_DEFAULT)
    ap.add_argument("--min-frac", type=float, default=MIN_FRAC_DEFAULT)
    ap.add_argument("--view-conf", type=float, default=VIEW_CONF_DEFAULT)
    ap.add_argument("--basis", choices=["auto", "mask", "kp2d"], default="auto")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/qc/bout_reconstructable.json"))
    ap.add_argument("--csv", type=Path,
                    default=Path("docs/qc/bout_reconstructable.csv"))
    ap.add_argument("--apply", action="store_true",
                    help="write EXCLUDED.json markers into failing bout dirs")
    args = ap.parse_args(argv)

    rows = scan_root(args.root, min_views=args.min_views,
                     min_frac=args.min_frac, view_conf=args.view_conf,
                     basis=args.basis)
    payload = write_report(rows, args.out, args.csv)

    print(f"scanned {payload['n_bouts']} bouts under {args.root}")
    print(f"  basis: {payload['by_basis']}")
    print(f"  keep {payload['n_keep']}   drop {payload['n_drop']}"
          f"   (min_views={args.min_views}, min_frac={args.min_frac})")
    drops = [r for r in rows if not r["keep"]]
    if drops:
        print(f"\n{'session':<10}{'recording':<22}{'bout':>5}{'basis':>7}  fractions")
        for r in sorted(drops, key=lambda r: min(r["fractions"]) if r["fractions"] else -1):
            print(f"{r.get('session',''):<10}{r['recording']:<22}{r['bout']:>5}"
                  f"{r['basis']:>7}  {r['fractions']}   {r['reason']}")
    print(f"\nreport: {args.out}")
    if args.csv:
        print(f"csv:    {args.csv}")
    if args.apply:
        n_add, n_clear = apply_exclusions(rows, args.root)
        print(f"applied: wrote {n_add} EXCLUDED.json markers, "
              f"cleared {n_clear} stale ones")
    else:
        print("(no markers written; pass --apply to flag the bout dirs)")
    return payload


if __name__ == "__main__":
    main()
