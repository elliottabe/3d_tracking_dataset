#!/usr/bin/env python3
"""Roll every recording of one session's mvq run up into a single scorecard.

The last link of the session chain built by `scripts/slurm/mvq_session_pipeline.sh`:

    lift[ts] -> precompute[ts] -> ik[ts] -> aggregate[ts]      (per recording)
                                              aggregate[*] -> collect   <- this

`aggregate` (jarvis_jax.tracking.session_qc.aggregate_session_qc) already
reduces one recording's `bouts/bout_*/fly*/qc.json` to `qc/session_qc.json`.
This step reduces the *session*: it reads every recording's `session_qc.json`
plus the per-bout artifacts the aggregate never looks at -- `coverage.json`,
`track_qc.json`, and the lifter's `mvq_meta.json` -- and writes

    <processed>/<Session>/<run-name>_session_summary.json
    <processed>/<Session>/<run-name>_session_summary.md

one row per recording plus a TOTAL row.

What the columns mean (all names, never bare indices -- fly0 = FEMALE,
fly1 = MALE after canonicalization, and `mvq_meta.json: fly_sex` is read rather
than assumed so a future swap cannot silently relabel a column):

    bouts                  bout dirs under <run>/bouts/
    bout-flies solved      qc.json files the aggregate found (2 per finished bout)
    frames                 sum of mvq_meta n_frames over bouts (per fly, not doubled)
    LOO median / p90 px    leave-one-out reprojection, over the per-bout-fly medians
    reproj median px       multi-view reprojection, over the per-bout-fly medians
    IoU median             mesh-vs-mask hard IoU; on the mvq route this is a
                           RELATIVE A/B number, not an absolute gate (see
                           docs/benchmark/2026-09-mvq/pipeline-audit-2026-09-05.md §9)
    female / male missing% frames the lift wrote no instance for, / frames
    containment-dropped %  keypoints the mask-containment gate removed, of the
                           finite keypoints it tested
    bouts w/ unsolvable    bouts where some fly wrote unsolvable.json (no
                           >=30-frame finite run to fit)

Everything is best-effort: a missing or unparseable file becomes a line in
`warnings`, never an exception. A recording whose aggregate has not run yet
still gets a row (from its bout dirs) with the QC columns blank.

Usage:
    python scripts/session_collect.py --session-name Session1 \\
        --processed /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \\
        --run-name pose_mvq_p3b
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
from pathlib import Path

# fly dir -> sex label. Overridden per bout by mvq_meta.json's own `fly_sex`.
DEFAULT_FLY_SEX = {"fly0": "female", "fly1": "male"}


# ---------------------------------------------------------------------------
# small pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def read_json(path, warnings: list, *, required: bool = False):
    """json.load(path) or None, appending a warning instead of raising."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        if required:
            warnings.append(f"missing {path}")
        return None
    except Exception as e:  # noqa: BLE001 -- a corrupt file must not kill the roll-up
        warnings.append(f"unreadable {path}: {type(e).__name__}: {e}")
        return None


def _finite(values):
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):
            out.append(f)
    return out


def percentile(values, q: float):
    """Linear-interpolated percentile (numpy's default method), None if empty.

    Implemented here rather than via numpy so the collect job stays a pure-python
    CPU step -- it is submitted with --gpus=0 and imports nothing heavy."""
    vals = sorted(_finite(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def median(values):
    return percentile(values, 50.0)


def _pct(num, den):
    """num/den as a percentage, None when den is 0/unknown."""
    if not den:
        return None
    return 100.0 * float(num) / float(den)


def bout_index(bout_dir) -> int:
    m = re.match(r"bout_(\d+)$", os.path.basename(str(bout_dir)))
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# per-recording collection
# ---------------------------------------------------------------------------

def collect_bouts(run_root: Path, warnings: list) -> dict:
    """Walk <run_root>/bouts/bout_*/ for the per-bout artifacts."""
    acc = {
        "n_bouts": 0,
        "n_frames_lift": 0,
        "n_missing": {"fly0": 0, "fly1": 0},
        "n_collapsed": {"fly0": 0, "fly1": 0},
        "containment_dropped": 0,
        "containment_tested": 0,
        "sex_head_disagree_frac": {"fly0": None, "fly1": None},
        "identity_source": {},
        "n_mvq_meta": 0,
        "n_coverage": 0,
        "n_track_qc": 0,
        "n_track_merged_bouts": 0,
        "coverage_below_min_flies": 0,
        "bouts_with_unsolvable": [],
        "missing_stac_ik": [],
        "fly_sex": dict(DEFAULT_FLY_SEX),
    }
    # frame-weighted accumulators for the disagree fractions
    dis_num = {"fly0": 0.0, "fly1": 0.0}
    dis_den = {"fly0": 0.0, "fly1": 0.0}

    bouts_dir = run_root / "bouts"
    if not bouts_dir.is_dir():
        warnings.append(f"missing {bouts_dir}")
        return acc

    for bd in sorted(p for p in bouts_dir.glob("bout_*") if p.is_dir()):
        acc["n_bouts"] += 1
        bname = bd.name

        meta = read_json(bd / "mvq_meta.json", warnings)
        if meta is None:
            warnings.append(f"no mvq_meta.json for {bname}")
        else:
            acc["n_mvq_meta"] += 1
            n_fr = int(meta.get("n_frames") or 0)
            acc["n_frames_lift"] += n_fr
            fs = meta.get("fly_sex") or {}
            for fly, sex in fs.items():
                acc["fly_sex"][fly] = sex
            for fly in ("fly0", "fly1"):
                acc["n_missing"][fly] += int((meta.get("n_missing") or {}).get(fly, 0) or 0)
                acc["n_collapsed"][fly] += int((meta.get("n_collapsed") or {}).get(fly, 0) or 0)
                d = (meta.get("sex_head_disagree_frac") or {}).get(fly)
                if d is not None and n_fr:
                    dis_num[fly] += float(d) * n_fr
                    dis_den[fly] += n_fr
            cr = meta.get("containment_report") or {}
            for key in ("n_kp_dropped_other_mask", "n_kp_dropped_spike"):
                acc["containment_dropped"] += sum(
                    int(v or 0) for v in (cr.get(key) or {}).values())
            acc["containment_tested"] += sum(
                int(v or 0) for v in (cr.get("n_kp_finite_before") or {}).values())
            src = meta.get("identity_resolved") or meta.get("identity") or "unknown"
            acc["identity_source"][src] = acc["identity_source"].get(src, 0) + 1

        cov = read_json(bd / "coverage.json", warnings)
        if cov is not None:
            acc["n_coverage"] += 1
            for _fly, blk in (cov.get("per_fly") or {}).items():
                if (blk or {}).get("status") != "ok":
                    acc["coverage_below_min_flies"] += 1

        tq = read_json(bd / "track_qc.json", warnings)
        if tq is not None:
            acc["n_track_qc"] += 1
            if int(tq.get("n_merged") or 0) > 0:
                acc["n_track_merged_bouts"] += 1

        for fd in sorted(p for p in bd.glob("fly*") if p.is_dir()):
            if (fd / "unsolvable.json").exists():
                if bname not in acc["bouts_with_unsolvable"]:
                    acc["bouts_with_unsolvable"].append(bname)
            elif not (fd / "stac_ik.h5").exists():
                acc["missing_stac_ik"].append(f"{bname}/{fd.name}")

    for fly in ("fly0", "fly1"):
        if dis_den[fly]:
            acc["sex_head_disagree_frac"][fly] = dis_num[fly] / dis_den[fly]
    return acc


def collect_recording(rec_dir: Path, run_name: str) -> dict:
    """One recording's row: aggregate QC + per-bout artifacts."""
    warnings: list = []
    run_root = rec_dir / run_name
    row = {
        "timestamp": rec_dir.name,
        "run_root": str(run_root),
        "has_run_root": run_root.is_dir(),
    }

    sq = read_json(run_root / "qc" / "session_qc.json", warnings, required=True)
    rows = (sq or {}).get("rows") or []
    loo = [r.get("loo_px") for r in rows]
    reproj = [r.get("reproj_px") for r in rows]
    iou_hard = [r.get("iou_hard") for r in rows]
    iou_soft = [r.get("iou_soft") for r in rows]
    row["qc"] = {
        "n_bout_flies": int((sq or {}).get("n_bouts_flies") or 0),
        "total_frames_bout_flies": int((sq or {}).get("total_frames") or 0),
        # medians recomputed from `rows` so LOO median and p90 come from the
        # same population; fall back to the aggregate's own scalar when the
        # rows are absent (older session_qc.json).
        "loo_px_median": median(loo) if loo else (sq or {}).get("loo_px_median"),
        "loo_px_p90": percentile(loo, 90.0),
        "reproj_px_median": (median(reproj) if reproj
                             else (sq or {}).get("reproj_px_median")),
        "iou_hard_median": (median(iou_hard) if iou_hard
                            else (sq or {}).get("iou_hard_median")),
        "iou_soft_median": (median(iou_soft) if iou_soft
                            else (sq or {}).get("iou_soft_median")),
    }

    bouts = collect_bouts(run_root, warnings)
    row["bouts"] = bouts
    fly_sex = bouts["fly_sex"]
    female = next((f for f, s in fly_sex.items() if s == "female"), "fly0")
    male = next((f for f, s in fly_sex.items() if s == "male"), "fly1")
    frames = bouts["n_frames_lift"]
    row["derived"] = {
        "female_fly": female,
        "male_fly": male,
        "female_missing_pct": _pct(bouts["n_missing"].get(female, 0), frames),
        "male_missing_pct": _pct(bouts["n_missing"].get(male, 0), frames),
        "containment_dropped_pct": _pct(bouts["containment_dropped"],
                                        bouts["containment_tested"]),
        "n_bouts_with_unsolvable": len(bouts["bouts_with_unsolvable"]),
    }
    row["warnings"] = warnings
    return row


def collect_session(processed: Path, session_name: str, run_name: str) -> dict:
    session_dir = processed / session_name
    recordings = []
    warnings: list = []
    if not session_dir.is_dir():
        warnings.append(f"missing session dir {session_dir}")
    else:
        for rec in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            if not (rec / run_name).is_dir():
                continue
            recordings.append(collect_recording(rec, run_name))
    if not recordings:
        warnings.append(f"no recording has a {run_name}/ run root under {session_dir}")

    total = _total_row(recordings)
    return {
        "session": session_name,
        "run_name": run_name,
        "processed_root": str(processed),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_recordings": len(recordings),
        "recordings": recordings,
        "total": total,
        "missing_stac_ik": [f"{r['timestamp']}/{m}"
                            for r in recordings for m in r["bouts"]["missing_stac_ik"]],
        "bouts_with_unsolvable": [f"{r['timestamp']}/{b}"
                                  for r in recordings
                                  for b in r["bouts"]["bouts_with_unsolvable"]],
        "warnings": warnings,
    }


def _total_row(recordings: list) -> dict:
    """Session totals. Counts are summed; the percentages are recomputed from
    the summed numerators/denominators (a mean of per-recording percentages
    would weight a 3-bout recording like a 30-bout one), and the LOO/reproj/IoU
    numbers are the median over every recording's median -- named as such in
    the table header so nobody reads them as a pooled per-bout median."""
    n_frames = sum(r["bouts"]["n_frames_lift"] for r in recordings)
    miss = {"female": 0, "male": 0}
    for r in recordings:
        d = r["derived"]
        miss["female"] += r["bouts"]["n_missing"].get(d["female_fly"], 0)
        miss["male"] += r["bouts"]["n_missing"].get(d["male_fly"], 0)
    dropped = sum(r["bouts"]["containment_dropped"] for r in recordings)
    tested = sum(r["bouts"]["containment_tested"] for r in recordings)
    return {
        "n_recordings": len(recordings),
        "n_bouts": sum(r["bouts"]["n_bouts"] for r in recordings),
        "n_bout_flies": sum(r["qc"]["n_bout_flies"] for r in recordings),
        "n_frames_lift": n_frames,
        "loo_px_median": median([r["qc"]["loo_px_median"] for r in recordings]),
        "loo_px_p90": percentile([r["qc"]["loo_px_p90"] for r in recordings], 90.0),
        "reproj_px_median": median([r["qc"]["reproj_px_median"] for r in recordings]),
        "iou_hard_median": median([r["qc"]["iou_hard_median"] for r in recordings]),
        "female_missing_pct": _pct(miss["female"], n_frames),
        "male_missing_pct": _pct(miss["male"], n_frames),
        "containment_dropped_pct": _pct(dropped, tested),
        "n_bouts_with_unsolvable": sum(r["derived"]["n_bouts_with_unsolvable"]
                                       for r in recordings),
        "n_missing_stac_ik": sum(len(r["bouts"]["missing_stac_ik"]) for r in recordings),
    }


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def _f(v, digits=2):
    return "-" if v is None else f"{float(v):.{digits}f}"


COLUMNS = ["recording", "bouts", "bout-flies solved", "frames",
           "LOO median px", "LOO p90 px", "reproj median px", "IoU hard median",
           "female missing %", "male missing %", "containment-dropped %",
           "bouts w/ unsolvable"]


def render_markdown(summary: dict) -> str:
    lines = [f"# {summary['session']} / `{summary['run_name']}` session summary",
             "",
             f"Generated {summary['generated']} from "
             f"`{summary['processed_root']}/{summary['session']}/*/{summary['run_name']}/`.",
             f"{summary['n_recordings']} recording(s). "
             "fly0 = FEMALE, fly1 = MALE (read from each bout's `mvq_meta.json: fly_sex`).",
             "",
             "| " + " | ".join(COLUMNS) + " |",
             "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for r in summary["recordings"]:
        q, b, d = r["qc"], r["bouts"], r["derived"]
        lines.append("| " + " | ".join([
            r["timestamp"], str(b["n_bouts"]), str(q["n_bout_flies"]),
            str(b["n_frames_lift"]),
            _f(q["loo_px_median"]), _f(q["loo_px_p90"]), _f(q["reproj_px_median"]),
            _f(q["iou_hard_median"], 3),
            _f(d["female_missing_pct"], 1), _f(d["male_missing_pct"], 1),
            _f(d["containment_dropped_pct"], 2),
            str(d["n_bouts_with_unsolvable"]),
        ]) + " |")
    t = summary["total"]
    lines.append("| " + " | ".join([
        "**TOTAL**", str(t["n_bouts"]), str(t["n_bout_flies"]), str(t["n_frames_lift"]),
        _f(t["loo_px_median"]), _f(t["loo_px_p90"]), _f(t["reproj_px_median"]),
        _f(t["iou_hard_median"], 3),
        _f(t["female_missing_pct"], 1), _f(t["male_missing_pct"], 1),
        _f(t["containment_dropped_pct"], 2), str(t["n_bouts_with_unsolvable"]),
    ]) + " |")
    lines += ["",
              "`frames` is the lifted frame count per fly (sum of each bout's "
              "`mvq_meta.json: n_frames`), NOT the aggregate's `total_frames`, "
              "which counts every bout twice (once per fly). TOTAL's px/IoU cells "
              "are the median over the per-recording medians."]

    miss = summary["missing_stac_ik"]
    lines += ["", f"## Bouts missing `stac_ik.h5` ({len(miss)})", ""]
    lines += ([f"- `{m}`" for m in miss] if miss
              else ["_none — every bout-fly that was not declared unsolvable has an IK fit._"])

    uns = summary["bouts_with_unsolvable"]
    lines += ["", f"## Bout-flies declared unsolvable ({len(uns)})", ""]
    lines += ([f"- `{u}`" for u in uns] if uns else ["_none._"])

    warns = list(summary["warnings"])
    for r in summary["recordings"]:
        warns += [f"{r['timestamp']}: {w}" for w in r["warnings"]]
    lines += ["", f"## Warnings ({len(warns)})", ""]
    lines += ([f"- {w}" for w in warns[:200]] if warns else ["_none._"])
    if len(warns) > 200:
        lines.append(f"- … and {len(warns) - 200} more (see the JSON)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-name", required=True,
                   help="session folder under --processed, e.g. Session1")
    p.add_argument("--processed", required=True,
                   help="processed assay root, e.g. "
                        "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")
    p.add_argument("--run-name", required=True,
                   help="per-recording run root name, e.g. pose_mvq_p3b")
    p.add_argument("--out-dir", default=None,
                   help="where the two summary files land "
                        "(default: <processed>/<session-name>)")
    args = p.parse_args()

    processed = Path(args.processed)
    summary = collect_session(processed, args.session_name, args.run_name)

    out_dir = Path(args.out_dir) if args.out_dir else processed / args.session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.run_name}_session_summary.json"
    out_md = out_dir / f"{args.run_name}_session_summary.md"
    # atomic-ish: write then replace, so a preempted requeue never leaves a
    # half-written summary that looks like a finished one.
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out_json)
    tmp = out_md.with_suffix(".md.tmp")
    tmp.write_text(render_markdown(summary))
    tmp.replace(out_md)

    t = summary["total"]
    print(f"[collect] {summary['session']}/{summary['run_name']}: "
          f"{summary['n_recordings']} recording(s), {t['n_bouts']} bouts, "
          f"{t['n_bout_flies']} bout-flies solved, {t['n_frames_lift']} frames, "
          f"LOO median {_f(t['loo_px_median'])} px, "
          f"{t['n_missing_stac_ik']} bout-flies missing stac_ik.h5, "
          f"{len(summary['warnings'])} session-level warning(s)")
    print(f"[collect] -> {out_json}")
    print(f"[collect] -> {out_md}")


if __name__ == "__main__":
    main()
