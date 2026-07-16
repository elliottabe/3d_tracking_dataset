"""Aggregate per-bout/fly QC (qc.qc_report output) into a session dashboard."""
from __future__ import annotations
import json, os, re
import numpy as np


def _parse_name(p):
    m = re.search(r"bout_(\d+)_fly(\d)", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def aggregate_session_qc(bout_qc_paths, out_json, *, plot_dir=None) -> dict:
    rows = []
    for p in bout_qc_paths:
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        b, f = _parse_name(p)
        rows.append(dict(bout=b, fly=f,
                         iou_hard=d.get("silhouette_iou", {}).get("hard_median", float("nan")),
                         iou_soft=d.get("silhouette_iou", {}).get("soft_median", float("nan")),
                         reproj_px=d.get("per_camera_reproj_px", {}).get("median", float("nan")),
                         loo_px=d.get("loo_reproj_px", {}).get("median", float("nan")),
                         n_frames=d.get("n_frames", 0)))
    def _med(k):
        v = [r[k] for r in rows if r[k] == r[k]]      # drop nan
        return float(np.median(v)) if v else float("nan")
    summ = dict(n_bouts_flies=len(rows),
                iou_hard_median=_med("iou_hard"), iou_soft_median=_med("iou_soft"),
                reproj_px_median=_med("reproj_px"), loo_px_median=_med("loo_px"),
                total_frames=int(sum(r["n_frames"] for r in rows)), rows=rows)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(summ, open(out_json, "w"), indent=2)
    if plot_dir and rows:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(plot_dir, exist_ok=True)
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["bout"], rows[i]["fly"]))
        labels = [f"{rows[i]['bout']}.{rows[i]['fly']}" for i in order]
        for key, fname in [("iou_hard", "iou_by_bout.png"), ("reproj_px", "reproj_by_bout.png")]:
            fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.3), 4))
            ax.bar(range(len(rows)), [rows[i][key] for i in order])
            ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_ylabel(key); ax.set_title(f"{key} by bout.fly")
            fig.tight_layout(); fig.savefig(os.path.join(plot_dir, fname), dpi=110); plt.close(fig)
    return summ
