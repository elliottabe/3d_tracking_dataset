#!/usr/bin/env python
"""Post-training temperature calibration for the mvq existence and per-view
visibility heads (design spec `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-
design.md` §5 "existence calibration"; §4 "Calibration after training").

WHY. The existence head reports ~0.45 sigmoid on present flies in earlier
runs even though `MVQRunner`/every downstream reader (`slot_read`,
`pick_typed_pair`, `pick_mask_pair`, `coarse_track`) gates on the fixed
`exist_thresh=0.5`: the raw head is over-confident in one direction and
under-confident in another, so 0.5 does not mean "as likely present as
absent". Temperature scaling (Guo et al. 2017) rescales the LOGIT by one
learned scalar per head so `sigmoid(logit / T)` matches the empirical
frequency, without touching argmax/ranking (dividing every logit by the
same positive T is monotone) -- this is exactly why `MVQRunner.infer` can
apply it and `read_typed`/`pick_typed_pair`/`policy_instance` keep picking
the SAME slot they always did (see `tests/test_mvq_calibrate.py`'s
`test_runner_applies_the_stored_temperature`).

EXPECTATION (write before running, CLAUDE.md): the pre-scaling existence
curve departs from the diagonal in BOTH directions at once, at different
confidence levels -- read the SIGN off the per-bin table/figure rather than
assuming one global direction (a mid-2026 note on this same head reported
~0.45 sigmoid on PRESENT flies, i.e. UNDER-confidence in the region that
covers present flies -- but low-confidence bins, which mostly cover ABSENT
flies, can just as well sit BELOW the diagonal, i.e. over-confident, at the
very same run). After fitting one temperature per head, the WELL-POPULATED
bins of both curves should move measurably closer to the diagonal and the
ECE should drop; the +-0.05 band on `max_gap_min_n` (bins with n >= min_n
only) is the spec's PASS/FAIL line for a checkpoint meant to ship (Task 7's
acceptance suite) -- NOT the raw `max_gap`, which unlike the bin-size-
weighted ECE can be pinned entirely to a single near-empty bin (n=1) that
one val fresh sample dominates; the raw number is still reported (and
plotted) for reference. A single scalar temperature also cannot fix a curve
that is under-confident in one region and roughly correct or over-confident
in another (its own shape is non-monotonic in a way one multiplicative
constant cannot undo) -- expect SOME bins to get closer and others to get
further from the diagonal after scaling; read the per-bin table, not just
the two summary numbers, before concluding calibration helped or hurt. A
curve that is ALREADY on the diagonal (T ~ 1, max_gap/max_gap_min_n small
before scaling too) means there is nothing to fix -- record that as the
finding, do not force a temperature away from 1 just because the script
ran.

WHAT THIS SCRIPT DOES. One UNPROMPTED forward pass over the val split
(`V12WindowDataset(root, "val", T=1, train=False)`, the same forward
`train_mvq.evaluate` uses: `normalize_crops`, `prompt_on` all-False),
collecting:
  * exist_logit (N,I) against the label-driven `slot_target` from
    `assign_slots` (a slot HOLDS a labelled fly, `assign_slots`' own
    host-first/typed-by-sex rule -- no prediction enters the target), scored
    only on slots `slot_ignore` does not mark ignorable (an unlabelled fly
    of known/unknown sex could legitimately occupy that slot with no
    contradicting label) -- exactly `train_mvq.evaluate`'s per-slot
    existence bookkeeping.
  * vis_logit (N,F,T,C,K), gathered per labelled fly by its assigned slot
    (`losses_mvq._gather_inst`, the SAME gather `mvq_loss`'s term 4 uses),
    against `vis2d`, masked by `cam_valid` (that view was actually shown to
    the model) AND `fly_valid & (assign >= 0)` (a labelled fly that got a
    slot at all -- `losses_mvq.mvq_loss`'s own `fv_eff`; a fly whose typed
    slot collided and was dropped to -1 has no assigned readout to score).

`fit_temperature` finds T by bisection on d(NLL)/dT (no scipy, 1e-4
tolerance, clamped to [0.25, 10]) and `reliability` bins probs into 10
equal-width [0,1) bins (last bin closed on 1.0), reporting per-bin
edges/acc/conf/n plus `max_gap` (over POPULATED bins only), `max_gap_min_n`
(over bins with `n >= min_n`, default 20 -- None if none qualify; THIS is
what the acceptance suite gates on) and `ece` (bin-size-weighted mean gap,
over populated bins). The script then reads `<run>/final/
mvq_run.json` (or, with `--stage-out`, a separate copy -- production runs
must never be overwritten by a checkpoint run this script is only proving
end-to-end), adds a `"calibration"` block, atomic-writes it back, and saves
a two-panel reliability figure (existence, visibility; before vs after;
diagonal +- 0.05 band).

Run (GPU node, one visible device):
    module load cuda/12.9.1
    export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
    unset LD_LIBRARY_PATH JAX_PLATFORMS
    export PYTHONPATH=third_party/jarvis_jax:. HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
    CUDA_VISIBLE_DEVICES=<n> python scripts/benchmark/mvq_calibrate.py \\
        --run <checkpoint>/final --out figures/2026-09-mvq/v2_train \\
        --figure-name calibration_p3b.png \\
        --stage-out OutFiles/v2_calib_check/<name>/mvq_run.json

(`--figure-name` defaults to `calibration.png`, the name a real v2-final run
should use; a proof run against an existing production checkpoint -- e.g.
P3b before v2 exists -- should pick a distinct name so it never collides
with a later real run's figure.)
"""
from __future__ import annotations

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
from jarvis_jax.train.losses_mvq import _gather_inst
from jarvis_jax.train.matching import assign_slots, slot_ignore
from jarvis_jax.train.train_mvq import normalize_crops
from jarvis_jax.tracking.resume import atomic_save_json

DEFAULT_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"


# --------------------------------------------------------------------------
# fit_temperature / reliability -- pure numpy, no jax/GPU dependence, so
# `tests/test_mvq_calibrate.py` exercises them with no checkpoint at all.
# --------------------------------------------------------------------------
def fit_temperature(logits, targets, mask, *, lo=0.25, hi=10.0, tol=1e-4, max_iter=100):
    """Scalar T minimising the BCE of `sigmoid(logits / T)` against `targets`
    on the entries where `mask` is True, by bisection on d(NLL)/dT.

    For one entry, NLL(t) = -[y log s(t) + (1-y) log(1-s(t))] with
    s(t) = sigmoid(z/t); d NLL/dt = (s(t)-y) * d(z/t)/dt = -(z/t^2)(s(t)-y).
    Summed over the masked entries this is `d(t) = -(1/t^2) * sum((s-y)*z)`;
    `d` is smooth and changes sign at most once on (0, inf) for this family
    (a single logistic-regression-style rescaling), so plain bisection on
    `d` finds the root. If `d` does NOT change sign over `[lo, hi]` the NLL
    is monotone on the whole clamp range and the minimiser is whichever
    endpoint `d`'s sign points at (never silently returned as "the bisection
    midpoint" -- that would report a fake root). An empty mask (nothing to
    calibrate against) returns T=1 -- the "already calibrated" answer, not
    a made-up number.
    """
    logits = np.asarray(logits, np.float64).reshape(-1)
    targets = np.asarray(targets, bool).reshape(-1)
    mask = np.asarray(mask, bool).reshape(-1)
    z = logits[mask]
    y = targets[mask].astype(np.float64)
    if z.size == 0:
        return 1.0

    def d(t):
        s = 1.0 / (1.0 + np.exp(-z / t))
        return float(np.sum((s - y) * z) * (-1.0 / (t * t)))

    d_lo, d_hi = d(lo), d(hi)
    if d_lo == 0.0:
        return float(lo)
    if d_hi == 0.0:
        return float(hi)
    if np.sign(d_lo) == np.sign(d_hi):
        # NLL monotone over the whole clamp range -- the unconstrained
        # minimiser lies outside [lo, hi]; report the nearer clamp edge.
        return float(lo) if d_lo > 0 else float(hi)
    a, b, fa = lo, hi, d_lo
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = d(m)
        if abs(fm) < 1e-12 or (b - a) / 2.0 < tol:
            return float(m)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return float(0.5 * (a + b))


def reliability(probs, targets, mask, bins=10, min_n=20):
    """10 (default) equal-width bins over [0,1]; returns
    `{"edges", "acc", "conf", "n", "max_gap", "max_gap_min_n", "ece"}`.

    `edges` (bins+1,) the bin boundaries; `acc`/`conf`/`n` (bins,) the
    empirical frequency, mean predicted prob, and count per bin (NaN
    acc/conf where `n == 0` -- an empty bin has no "reliability" to report).
    `max_gap` is `max(|acc-conf|)` over POPULATED bins only (an empty bin
    must not silently read as a perfect bin); `ece` is the bin-size-weighted
    mean of the same gaps (also over populated bins only).

    `max_gap` and `ece` both have a blind spot a real val split hits: a bin
    with only 1-2 samples can sit anywhere in [0,1] by pure chance, so one
    val fresh sample can single-handedly set `max_gap` (ECE is somewhat
    protected -- it is WEIGHTED by `n`, so a tiny bin barely moves it -- but
    is then too lenient to catch a real, broad miscalibration spread evenly
    over well-populated bins). `max_gap_min_n` is `max(|acc-conf|)` over only
    the bins with `n >= min_n` (None if no bin qualifies) -- the acceptance
    metric a single sparse bin cannot dominate, and the one
    `scripts/benchmark/mvq_v2_acceptance.py::check_calibration` gates on.
    """
    probs = np.asarray(probs, np.float64).reshape(-1)
    targets = np.asarray(targets, bool).reshape(-1)
    mask = np.asarray(mask, bool).reshape(-1)
    p = probs[mask]
    y = targets[mask].astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    acc = np.full(bins, np.nan)
    conf = np.full(bins, np.nan)
    n = np.zeros(bins, np.int64)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        n[b] = int(sel.sum())
        if n[b] > 0:
            acc[b] = float(y[sel].mean())
            conf[b] = float(p[sel].mean())
    populated = n > 0
    gaps = np.abs(acc[populated] - conf[populated]) if populated.any() else np.zeros(0)
    max_gap = float(gaps.max()) if gaps.size else float("nan")
    ece = (float(np.sum(gaps * n[populated]) / n[populated].sum())
           if populated.any() else float("nan"))
    big = n >= min_n
    gaps_big = np.abs(acc[big] - conf[big]) if big.any() else np.zeros(0)
    max_gap_min_n = float(gaps_big.max()) if gaps_big.size else None
    return {"edges": edges.tolist(), "acc": acc.tolist(), "conf": conf.tolist(),
            "n": n.tolist(), "max_gap": max_gap, "max_gap_min_n": max_gap_min_n, "ece": ece}


# --------------------------------------------------------------------------
# forward pass over the val split
# --------------------------------------------------------------------------
def collect_val_logits(run, *, step=None, attn_impl=None, root=DEFAULT_ROOT, batch=16):
    """One UNPROMPTED pass over `V12WindowDataset(root, "val", T=1)`.

    Returns a dict: `exist_logit`/`exist_target`/`exist_mask` all (N,I);
    `vis_logit`/`vis_target`/`vis_mask` all (N,F,T,C,K); `n_val` = N (the
    number of val framesets actually seen -- `window_batches(drop_last=
    False)` never drops or pads, so this is `len(ds)` exactly, 153 on the
    v12 val split at the time of writing).
    """
    model, meta = load_mvq_model(run, step=step, attn_impl=attn_impl)
    ds = V12WindowDataset(root, "val", T=1, train=False)
    exist_logits, exist_targets, exist_masks = [], [], []
    vis_logits, vis_targets, vis_masks = [], [], []
    n_val = 0
    for b in window_batches(ds, batch, shuffle=False, drop_last=False, num_workers=8):
        B0 = int(b["crops"].shape[0])
        n_val += B0
        on = np.zeros(B0, bool)
        out = model(normalize_crops(jnp.asarray(b["crops"])), jnp.asarray(b["cam_valid"]),
                    jnp.asarray(b["M"]), jnp.asarray(b["t_local"]), jnp.asarray(b["prompt_mask"]),
                    prompt_on=jnp.asarray(on))
        exist_logit = np.asarray(out["exist_logit"])                          # (B0,I)
        I = exist_logit.shape[1]
        has_f = b["has3d"].astype(np.float32)
        cen = ((b["kp3d_local"] * has_f[..., None]).sum((2, 3))
               / np.maximum(has_f.sum((2, 3)), 1.0)[..., None])
        dist = np.linalg.norm(cen, axis=-1)
        assign, slot_t = assign_slots(jnp.asarray(b["fly_sex"]), jnp.asarray(b["fly_valid"]),
                                      jnp.asarray(on), jnp.asarray(dist), I)
        assign, slot_t = np.asarray(assign), np.asarray(slot_t)
        # a slot HOLDING a labelled fly is never ignored, even if an
        # unlabelled animal's sex would otherwise excuse it (train_mvq.
        # evaluate's own `ignore & ~slot_t`).
        ignore = np.asarray(slot_ignore(jnp.asarray(b["unlabelled_sex"]), I)) & ~slot_t
        exist_logits.append(exist_logit)
        exist_targets.append(slot_t)
        exist_masks.append(~ignore)

        vis_logit = np.asarray(_gather_inst(out["vis_logit"], jnp.asarray(assign)))   # (B0,F,T,C,K)
        fv_eff = np.asarray(b["fly_valid"]) & (assign >= 0)                     # losses_mvq.mvq_loss's fv_eff
        vis_mask = (np.asarray(b["cam_valid"])[:, None, :, :, None]
                   & fv_eff[:, :, None, None, None])
        vis_mask = np.broadcast_to(vis_mask, vis_logit.shape)
        vis_logits.append(vis_logit)
        vis_targets.append(np.asarray(b["vis2d"]))
        vis_masks.append(vis_mask)
    return {"exist_logit": np.concatenate(exist_logits, axis=0),
            "exist_target": np.concatenate(exist_targets, axis=0),
            "exist_mask": np.concatenate(exist_masks, axis=0),
            "vis_logit": np.concatenate(vis_logits, axis=0),
            "vis_target": np.concatenate(vis_targets, axis=0),
            "vis_mask": np.concatenate(vis_masks, axis=0),
            "n_val": n_val}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def calibrate(collected):
    """`collect_val_logits`'s dict -> `(t_exist, t_vis, before, after)` where
    `before`/`after` are `{"exist": reliability(...), "vis": reliability(...)}`
    at T=1 and at the fitted temperatures respectively."""
    t_exist = fit_temperature(collected["exist_logit"], collected["exist_target"],
                              collected["exist_mask"])
    t_vis = fit_temperature(collected["vis_logit"], collected["vis_target"],
                            collected["vis_mask"])
    before = {"exist": reliability(_sigmoid(collected["exist_logit"]), collected["exist_target"],
                                   collected["exist_mask"]),
             "vis": reliability(_sigmoid(collected["vis_logit"]), collected["vis_target"],
                                collected["vis_mask"])}
    after = {"exist": reliability(_sigmoid(collected["exist_logit"] / t_exist), collected["exist_target"],
                                  collected["exist_mask"]),
            "vis": reliability(_sigmoid(collected["vis_logit"] / t_vis), collected["vis_target"],
                               collected["vis_mask"])}
    return t_exist, t_vis, before, after


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _plot_panel(ax, before, after, title):
    """Standard reliability diagram: x = mean predicted prob (conf) per bin,
    y = empirical frequency (acc) per bin, one curve before scaling and one
    after -- a point ABOVE the diagonal is under-confident there, BELOW is
    over-confident; the +-0.05 band is the spec's acceptance threshold."""
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="diagonal")
    xs = np.linspace(0, 1, 100)
    ax.fill_between(xs, xs - 0.05, xs + 0.05, color="gray", alpha=0.15, label="+-0.05 band")
    ok_b = np.asarray(before["n"]) > 0
    ok_a = np.asarray(after["n"]) > 0
    conf_b, acc_b = np.asarray(before["conf"])[ok_b], np.asarray(before["acc"])[ok_b]
    conf_a, acc_a = np.asarray(after["conf"])[ok_a], np.asarray(after["acc"])[ok_a]
    ax.plot(conf_b, acc_b, "o-", color="tab:red", label=f"before (max_gap={before['max_gap']:.3f})")
    ax.plot(conf_a, acc_a, "o-", color="tab:blue", label=f"after (max_gap={after['max_gap']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability (conf)"); ax.set_ylabel("empirical frequency (acc)")
    ax.set_title(title); ax.legend(fontsize=6, loc="upper left")


def save_figure(png_path, before, after, t_exist, t_vis, run_name, n_val):
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    _plot_panel(ax1, before["exist"], after["exist"], f"existence (T={t_exist:.3f})")
    _plot_panel(ax2, before["vis"], after["vis"], f"per-view visibility (T={t_vis:.3f})")
    fig.suptitle(f"mvq {run_name} -- val reliability (n={n_val} framesets, unprompted)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


# --------------------------------------------------------------------------
# mvq_run.json read/merge/write
# --------------------------------------------------------------------------
def write_calibration(meta_path, t_exist, t_vis, before, after, n_val, *, out_path=None):
    """Reads `meta_path`, adds/replaces its `"calibration"` block, and
    atomic-writes the result to `out_path` (default: back to `meta_path`
    itself). NEVER pass a production run's `final/mvq_run.json` as
    `out_path` unless that run is meant to start being read calibrated --
    a checkpoint being merely PROVEN end-to-end (spec Task 6 step 4, no v2
    run yet) must go to a staging copy instead."""
    with open(meta_path) as f:
        meta = json.load(f)
    meta["calibration"] = {
        "exist_temperature": float(t_exist),
        "vis_temperature": float(t_vis),
        "reliability_exist": after["exist"],
        "reliability_vis": after["vis"],
        "reliability_exist_before": before["exist"],
        "reliability_vis_before": before["vis"],
        "n_val": int(n_val),
    }
    dest = out_path or meta_path
    atomic_save_json(dest, meta)
    return dest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def run(a):
    step = int(a.step) if (a.step is not None and a.step != "latest") else a.step
    collected = collect_val_logits(a.run, step=step, attn_impl=a.attn_impl, root=a.root,
                                   batch=a.batch)
    t_exist, t_vis, before, after = calibrate(collected)
    run_name = os.path.basename(os.path.dirname(a.run.rstrip("/"))) or a.run

    os.makedirs(a.out, exist_ok=True)
    fig_name = a.figure_name or "calibration.png"
    png_path = save_figure(os.path.join(a.out, fig_name),
                           before, after, t_exist, t_vis, run_name, collected["n_val"])
    print("wrote", png_path)

    meta_path = a.run if a.run.endswith("mvq_run.json") else os.path.join(a.run, "mvq_run.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"{meta_path} does not exist -- --run must be a final/ dir "
                                f"(or the mvq_run.json file itself)")
    dest = write_calibration(meta_path, t_exist, t_vis, before, after, collected["n_val"],
                             out_path=a.stage_out)
    print("wrote", dest)

    result = {"run": a.run, "run_name": run_name, "n_val": collected["n_val"],
             "exist_temperature": t_exist, "vis_temperature": t_vis,
             "before": {"exist_max_gap": before["exist"]["max_gap"],
                       "exist_max_gap_min_n": before["exist"]["max_gap_min_n"],
                       "exist_ece": before["exist"]["ece"],
                       "vis_max_gap": before["vis"]["max_gap"],
                       "vis_max_gap_min_n": before["vis"]["max_gap_min_n"],
                       "vis_ece": before["vis"]["ece"]},
             "after": {"exist_max_gap": after["exist"]["max_gap"],
                      "exist_max_gap_min_n": after["exist"]["max_gap_min_n"],
                      "exist_ece": after["exist"]["ece"],
                      "vis_max_gap": after["vis"]["max_gap"],
                      "vis_max_gap_min_n": after["vis"]["max_gap_min_n"],
                      "vis_ece": after["vis"]["ece"]},
             "mvq_run_json": dest, "figure": png_path}
    json_path = os.path.join(a.out, fig_name.replace(".png", ".json"))
    with open(json_path, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", json_path)
    print(json.dumps(result, indent=1))
    return result


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="a final/ dir (default), or (with --step) the RUN dir")
    ap.add_argument("--step", default=None, help="load ckpt/<step> (or 'latest') instead of final/")
    ap.add_argument("--attn_impl", default=None, help="override the run's own attn_impl (e.g. 'xla' on CPU)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="figures/2026-09-mvq/v2_train",
                    help="figure (+ result json) directory")
    ap.add_argument("--figure-name", default=None,
                    help="figure filename under --out (default calibration_p3b.png)")
    ap.add_argument("--stage-out", default=None,
                    help="write the updated mvq_run.json HERE instead of in place -- use this for "
                        "any checkpoint that is not the production run this calibration is meant "
                        "for (spec Task 6 step 4: prove the script on P3b, never touch its own "
                        "final/mvq_run.json)")
    return ap


def main():
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
