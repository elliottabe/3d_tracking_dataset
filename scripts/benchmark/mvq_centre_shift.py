#!/usr/bin/env python
"""Centre-shift robustness spike for the mvq lifter (P4a spec §6).

The mvq lifter was trained with window centres jittered by only 0.3mm
(`jitter_units=3.0`, `MM_PER_UNIT=0.1`) around the host fly's own labelled 3D
bbox midpoint. The next phase (mask-free front end) instead places windows
from a COARSE track interpolated to 800fps, whose centre can be off from the
true fly centre by ~0.6mm. This script measures how the UNPROMPTED typed-slot
policy (`policy_instance(..., prompted=False, has_mask=False)` -- the
inference-time choice, no ground truth) degrades as the window centre is
displaced by a known amount on the val split.

EXPECTATION (write before running): the unprompted typed-slot policy error
stays within 10% of its unshifted value and the miss fraction under 2% up to
a 1mm (10 unit) centre shift; if it climbs steeply before 1mm the
interpolated coarse centres of the mask-free pass are not safe and the §6
retrain with 1mm jitter is required.

Shift is applied via `V12WindowDataset(..., center_shift_units=...)`
(`jarvis_jax/data/v12_windows.py`): eval-mode only, a fixed random in-plane
direction and an EXACT magnitude per window, seeded on `(seed, i, 99)` so
every shift setting only differs from the others in that one magnitude and
each run reproduces exactly.

Writes `figures/2026-09-mvq/p4_maskfree/centre_shift.{png,json}`.
"""
import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)
from jarvis_jax.data.v12_windows import V12WindowDataset, window_batches
from jarvis_jax.models.mvq.checkpoint import load_mvq_model
from jarvis_jax.models.mvq.policy import policy_instance
from jarvis_jax.train.train_mvq import MM_PER_UNIT, normalize_crops

SHIFTS_MM = [0.0, 0.5, 1.0, 2.0, 3.0]


def run(a):
    """Run the centre-shift spike for parsed args `a` (the same fields
    `build_arg_parser()` below defines: `run`, `step`, `attn_impl`, `root`,
    `out`, `batch`). Writes `centre_shift.{png,json}` under `a.out` (same as
    always) and returns the JSON dict (`{"rows": [...], ...}`) directly, so
    a caller (`scripts/benchmark/mvq_v2_acceptance.py`'s "centre shift"
    acceptance row) can read `rows` without re-parsing the written JSON."""
    step = int(a.step) if (a.step is not None and a.step != "latest") else a.step
    model, meta = load_mvq_model(a.run, step=step, attn_impl=a.attn_impl)
    run_name = os.path.basename(os.path.dirname(a.run.rstrip("/")))
    step_label = a.step if a.step is not None else f"final(total_steps={meta['train']['total_steps']})"

    rows = []
    for mm in SHIFTS_MM:
        ds = V12WindowDataset(a.root, "val", T=1, train=False, center_shift_units=mm / MM_PER_UNIT)
        errs, miss, n = [], 0, 0
        for b in window_batches(ds, a.batch, shuffle=False, drop_last=False, num_workers=8):
            B = b["crops"].shape[0]
            out = model(normalize_crops(jnp.asarray(b["crops"])), jnp.asarray(b["cam_valid"]), jnp.asarray(b["M"]),
                        jnp.asarray(b["t_local"]), jnp.asarray(b["prompt_mask"]), prompt_on=jnp.zeros((B,), bool))
            xyz = np.asarray(out["xyz"]); ex = 1 / (1 + np.exp(-np.asarray(out["exist_logit"])))
            for bi in range(B):
                n += 1
                inst = policy_instance(ex[bi], xyz[bi], prompted=False, has_mask=False)
                if inst is None:
                    miss += 1; continue
                gt, has = b["kp3d_local"][bi, 0, 0], b["has3d"][bi, 0, 0]
                if has.any():
                    errs.append(float(np.linalg.norm(xyz[bi, inst, 0][has] - gt[has], axis=-1).mean()))
        rows.append(dict(shift_mm=mm, mpjpe_policy_mm=float(np.mean(errs)) * MM_PER_UNIT if errs else float("nan"),
                         miss_frac=miss / max(n, 1), n=n))
        print(rows[-1], flush=True)

    os.makedirs(a.out, exist_ok=True)
    base_err = rows[0]["mpjpe_policy_mm"]
    err_line = 1.1 * base_err
    miss_line = 0.02
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    shifts = [r["shift_mm"] for r in rows]
    errs_mm = [r["mpjpe_policy_mm"] for r in rows]
    misses = [r["miss_frac"] for r in rows]
    ax1.plot(shifts, errs_mm, "o-", color="tab:blue", label="policy MPJPE (mm)")
    ax1.axhline(err_line, linestyle="--", color="gray", label=f"1.1x unshifted ({err_line:.3f}mm)")
    ax1.set_xlabel("centre shift (mm)"); ax1.set_ylabel("policy MPJPE (mm)")
    ax1.set_title("unprompted policy error vs centre shift"); ax1.legend(fontsize=7)
    ax2.plot(shifts, misses, "o-", color="tab:red", label="miss fraction")
    ax2.axhline(miss_line, linestyle="--", color="gray", label="0.02 decision line")
    ax2.set_xlabel("centre shift (mm)"); ax2.set_ylabel("miss fraction")
    ax2.set_title("unprompted policy miss rate vs centre shift"); ax2.legend(fontsize=7)
    fig.suptitle(f"mvq {run_name} step={step_label} -- centre-shift spike (val, unprompted policy)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = os.path.join(a.out, "centre_shift.png")
    fig.savefig(png_path, dpi=130); plt.close(fig)
    print("wrote", png_path)

    result = {"run": a.run, "step": a.step, "run_name": run_name,
             "err_decision_line_mm": err_line, "miss_decision_line": miss_line,
             "rows": rows}
    json_path = os.path.join(a.out, "centre_shift.json")
    json.dump(result, open(json_path, "w"), indent=1)
    print("wrote", json_path)
    return result


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a final/ dir (default), or (with --step) the RUN dir (parent of final/ and ckpt/)")
    ap.add_argument("--step", default=None, help="load ckpt/<step> (or 'latest') instead of final/")
    ap.add_argument("--attn_impl", default=None, help="override the run's own attn_impl (e.g. 'xla' on CPU)")
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", default="figures/2026-09-mvq/p4_maskfree")
    ap.add_argument("--batch", type=int, default=16)
    return ap


def main():
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
