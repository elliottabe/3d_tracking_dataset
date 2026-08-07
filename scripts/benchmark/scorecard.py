"""Cohort scorecard: aggregate per-bout metrics, gap-to-free-running ratios,
and the apart/close partner-proximity split (spec: Track 0)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COHORTS = ("free_running", "courtship_male", "courtship_female")


def cohort_of(entry: dict, fly: int, male_fly: int | None) -> str:
    if entry["assay"] == "free_running":
        return "free_running"
    if male_fly is None:
        raise ValueError(f"courtship entry {entry.get('run_key')} needs sex.json "
                         f"male_fly to resolve the cohort")
    return "courtship_male" if fly == male_fly else "courtship_female"


def split_series(series: np.ndarray, proximity: np.ndarray | None,
                 threshold: float) -> dict:
    series = np.asarray(series, dtype=np.float64)
    out = {"all": float(np.nanmedian(series)) if series.size else None,
           "apart": None, "close": None}
    if proximity is None or not series.size:
        return out
    prox = np.asarray(proximity, dtype=np.float64)
    prox = prox[prox.shape[0] - series.shape[0]:]      # align (T-2,) vs (T,)
    for name, mask in (("apart", prox >= threshold), ("close", prox < threshold)):
        if mask.any():
            with np.errstate(all="ignore"):
                v = np.nanmedian(series[mask])
            out[name] = None if np.isnan(v) else float(v)
    return out


def build_scorecard(per_bout: list[dict]) -> dict:
    cohorts: dict[str, dict] = {}
    n_bouts: dict[str, int] = {}
    for c in COHORTS:
        items = [b for b in per_bout if b["cohort"] == c]
        if not items:
            continue
        n_bouts[c] = len(items)
        keys = sorted({k for b in items for k in b["scalars"]})
        agg = {k: float(np.nanmedian([b["scalars"][k] for b in items
                                      if k in b["scalars"]])) for k in keys}
        split_keys = sorted({k for b in items for k in b.get("splits", {})})
        for sk in split_keys:
            for part in ("apart", "close"):
                vals = [b["splits"][sk][part] for b in items
                        if b.get("splits", {}).get(sk, {}).get(part) is not None]
                if vals:
                    agg[f"{sk}_{part}"] = float(np.nanmedian(vals))
        cohorts[c] = agg
    gap_ratios: dict[str, dict] = {}
    base = cohorts.get("free_running", {})
    for c in ("courtship_male", "courtship_female"):
        if c not in cohorts:
            continue
        gap_ratios[c] = {k: cohorts[c][k] / base[k]
                         for k in cohorts[c] if base.get(k) not in (None, 0)}
    return {"cohorts": cohorts, "gap_ratios": gap_ratios, "n_bouts": n_bouts,
            "per_bout": per_bout}


def scorecard_markdown(scorecard: dict) -> str:
    lines = ["# Benchmark scorecard", ""]
    for c, agg in scorecard["cohorts"].items():
        lines += [f"## {c} (n={scorecard['n_bouts'][c]})", "",
                  "| metric | median |", "|---|---|"]
        lines += [f"| {k} | {v:.4g} |" for k, v in sorted(agg.items())]
        lines.append("")
    if scorecard["gap_ratios"]:
        lines += ["## Gap ratios (courtship / free_running, 1.0 = parity)", "",
                  "| metric | " + " | ".join(scorecard["gap_ratios"]) + " |",
                  "|---|" + "---|" * len(scorecard["gap_ratios"])]
        keys = sorted({k for g in scorecard["gap_ratios"].values() for k in g})
        for k in keys:
            row = [f"{scorecard['gap_ratios'][c].get(k, float('nan')):.3g}"
                   for c in scorecard["gap_ratios"]]
            lines.append(f"| {k} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_scorecard(path: Path, scorecard: dict) -> None:
    path = Path(path)
    slim = dict(scorecard)
    slim["per_bout"] = [{k: v for k, v in b.items() if k != "series"}
                        for b in scorecard.get("per_bout", [])]
    path.write_text(json.dumps(_jsonable(slim), indent=2, sort_keys=True))
    path.with_suffix(".md").write_text(scorecard_markdown(scorecard))
