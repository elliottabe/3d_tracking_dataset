"""Pslow / Pfast classification method-diagram figure.

Renders a single multi-panel figure walking through the classification
pipeline implemented in :mod:`utils.song_analysis` (per-pulse features)
and :mod:`utils.pulse_types` (PCA + GMM clustering):

    A. Tip-Z trace with detected pulse peaks.
    B. Per-pulse alignment cascade (raw → detrend → z-score → sign-align).
    C. Symmetry index s = (a · flip(b)) / (||a|| ||b||).
    D. Pooled PCA scores + 2-component GMM ellipses.
    E. Per-cluster mean symmetry — the rule that names Pslow vs Pfast.
    F. Mean Pslow / Pfast centroid waveforms.

The function refits the same PCA + GMM that
:func:`utils.pulse_type_cache.get_pulse_type_labels` fits, so the
clusters in panels D-F match the labels stored in the cache (both use
``random_state=0``).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from utils.pulse_types import (
    PulseTypeConfig,
    PulseTypeModel,
    fit_pulse_type_model,
)


PSLOW_COLOR = "#EF7B45"
PFAST_COLOR = "#3D5A80"


def _gather_pooled(results: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror of :func:`pulse_type_cache._gather_waveforms`."""
    waves, syms = [], []
    for r in results:
        for side in ("L", "R"):
            pf = (r.get("song0", {}).get("sides", {})
                   .get(side, {}).get("pulse_features"))
            if not pf:
                continue
            w = np.asarray(pf.get("waveforms", np.zeros((0, 0))))
            s = np.asarray(pf.get("symmetry", np.zeros(0)))
            if w.ndim != 2 or w.shape[0] == 0 or s.shape[0] != w.shape[0]:
                continue
            waves.append(w)
            syms.append(s)
    if not waves:
        return np.zeros((0, 0)), np.zeros(0)
    Wmax = max(w.shape[1] for w in waves)
    padded = []
    for w in waves:
        if w.shape[1] == Wmax:
            padded.append(w)
        else:
            pad = np.full((w.shape[0], Wmax - w.shape[1]), np.nan)
            padded.append(np.concatenate([w, pad], axis=1))
    W = np.concatenate(padded, axis=0)
    S = np.concatenate(syms, axis=0)
    finite = np.isfinite(W).all(axis=1) & np.isfinite(S)
    return W[finite], S[finite]


def _pick_example_bout(
    results: Sequence[dict],
    pair_idx: Optional[int] = None,
    min_pulses: int = 8,
):
    """Return ``(result, dom_side, z_trace, peak_frames, pulse_features)``."""
    fallback = None
    for r in results:
        if pair_idx is not None and r.get("pair_idx") != pair_idx:
            continue
        ms = r.get("song0") or {}
        dom = ms.get("dominant_wing")
        if not dom:
            continue
        side = ms.get("sides", {}).get(dom, {})
        pf = side.get("pulse_features") or {}
        peaks = np.asarray(pf.get("peak_frames", []), dtype=int)
        wd = ms.get("wing_data", {})
        tip_key = f"Wing{dom}_V13"
        z = np.asarray(wd.get(tip_key, {}).get("z", np.array([])), dtype=float)
        if peaks.size == 0 or z.size == 0:
            continue
        if peaks.size >= min_pulses:
            return r, dom, z, peaks, pf
        if fallback is None:
            fallback = (r, dom, z, peaks, pf)
        if pair_idx is not None:
            return fallback
    return fallback


def _panel_a_trace(ax, z, peaks, fs, half_window_ms=120.0):
    if peaks.size == 0:
        ax.text(0.5, 0.5, "no pulses", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return
    p_center = int(peaks[len(peaks) // 2])
    half = int(half_window_ms * 1e-3 * fs)
    lo = max(0, p_center - half)
    hi = min(len(z), p_center + half + 1)
    seg = z[lo:hi]
    t = (np.arange(lo, hi) - p_center) / fs * 1000.0
    ax.plot(t, seg, color="0.2", lw=1.0)
    in_view = peaks[(peaks >= lo) & (peaks < hi)]
    for p in in_view:
        ax.axvline((p - p_center) / fs * 1000.0,
                   color="#E76F51", lw=1.0, ls="--", alpha=0.85)
    ax.axvline(0, color="#E76F51", lw=1.4, ls="-", alpha=0.9)
    ax.set_xlabel("time relative to center pulse (ms)")
    ax.set_ylabel("wing tip z (world)")
    ax.set_title(f"A. detected pulses ({in_view.size}/{peaks.size} in view)")
    ax.grid(alpha=0.3)


def _panel_b_cascade(ax, z, peaks, fs, half_w_ms=12.5):
    half_w = int(round(half_w_ms * 1e-3 * fs))
    W = 2 * half_w + 1
    inside = (peaks - half_w >= 0) & (peaks + half_w + 1 <= len(z))
    cands = peaks[inside]
    if cands.size == 0:
        ax.text(0.5, 0.5, "no full-width window",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    p = int(cands[len(cands) // 2])
    raw = z[p - half_w:p + half_w + 1].astype(float)
    t_ms = (np.arange(W) - half_w) / fs * 1000.0

    x = np.arange(W, dtype=float)
    xc = x - x.mean()
    slope = ((raw - raw.mean()) * xc).sum() / max((xc * xc).sum(), 1e-12)
    intercept = raw.mean() - slope * x.mean()
    detr = raw - (slope * x + intercept)

    zsc = detr / max(detr.std(), 1e-9)
    aligned = zsc * (-1.0 if zsc[half_w] < 0 else 1.0)

    rows = [
        (raw - raw.mean(), "raw (mean-removed)", "0.4"),
        (detr,             "detrended",          "0.3"),
        (zsc,              "z-scored",           PFAST_COLOR),
        (aligned,          "sign-aligned",       PSLOW_COLOR),
    ]
    offsets = np.linspace(0.0, -3.5 * (len(rows) - 1), len(rows))
    for (s, label, color), off in zip(rows, offsets):
        ax.plot(t_ms, s + off, color=color, lw=1.3)
        ax.text(t_ms[0] - 1.0, off, label, va="center", ha="right",
                fontsize=8, color=color)
    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("time (ms)")
    ax.set_yticks([])
    ax.set_title("B. per-pulse alignment cascade")
    return aligned, t_ms


def _panel_c_symmetry(ax, aligned, t_ms):
    W = len(aligned)
    half = W // 2
    a = aligned[:half]
    b = aligned[W - half:]
    b_flip = b[::-1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    s = float((a * b_flip).sum() / denom) if denom > 1e-12 else 0.0

    t_a = t_ms[:half]
    ax.plot(t_a, a, color=PFAST_COLOR, lw=1.6, label="first half  a(t)")
    ax.plot(t_a, b_flip, color=PSLOW_COLOR, lw=1.6, ls="--",
            label="reversed second half  b(-t)")
    ax.fill_between(t_a, a, b_flip, color="0.7", alpha=0.25)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("aligned amplitude")
    ax.set_title(f"C. symmetry index  s = (a · b_flip) / (|a||b|) = {s:+.3f}")
    ax.legend(loc="upper right", fontsize=8)


def _panel_d_pca_gmm(ax, scores, hard, model, label_map, max_points=4000):
    color_for_label = {"Pslow": PSLOW_COLOR, "Pfast": PFAST_COLOR}
    pt_colors = np.array(
        [color_for_label[label_map[int(k)]] for k in hard]
    )
    n = scores.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, max_points, replace=False)
    else:
        sel = np.arange(n)
    ax.scatter(scores[sel, 0], scores[sel, 1],
               c=pt_colors[sel], s=4, alpha=0.35, linewidths=0)

    means = model.gmm.means_
    covs = model.gmm.covariances_
    for k in range(model.gmm.n_components):
        mu = means[k][:2]
        cov = covs[k][:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * 2.0 * np.sqrt(eigvals)
        c = color_for_label[label_map[int(k)]]
        ax.add_patch(Ellipse(xy=mu, width=width, height=height,
                             angle=angle, facecolor="none",
                             edgecolor=c, lw=1.6))
        ax.scatter([mu[0]], [mu[1]], marker="x", s=70, c=c,
                   linewidths=2.0)
        ax.annotate(label_map[int(k)], xy=mu, xytext=(8, 8),
                    textcoords="offset points", color=c,
                    fontsize=10, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"D. pooled PCA scores + 2-comp GMM (N={n})"
    )


def _panel_e_symmetry_bars(ax, model, label_map):
    syms = model.component_symmetry
    K = len(syms)
    color_for_label = {"Pslow": PSLOW_COLOR, "Pfast": PFAST_COLOR}
    colors = [color_for_label[label_map[k]] for k in range(K)]
    labels = [label_map[k] for k in range(K)]
    xs = np.arange(K)
    ax.bar(xs, syms, color=colors, edgecolor="k", linewidth=0.5)
    for x, val in zip(xs, syms):
        ax.text(x, val, f"{val:+.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean symmetry index")
    ax.set_title("E. cluster → type by mean symmetry\n(higher = more symmetric = Pslow)")


def _panel_f_centroids(ax, pulse_type_results, fs):
    pooled = pulse_type_results.get("pooled_waveforms", {}) or {}
    centroids = pulse_type_results.get("centroids", {}) or {}
    counts = pulse_type_results.get("counts", {}) or {}
    color_for_label = {"Pslow": PSLOW_COLOR, "Pfast": PFAST_COLOR}
    Ws = [np.asarray(v).shape[1] for v in pooled.values()
          if np.asarray(v).ndim == 2 and np.asarray(v).size]
    W = max(Ws) if Ws else 0
    if W == 0:
        ax.text(0.5, 0.5, "no centroid data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    t_ms = (np.arange(W) - W / 2.0) / fs * 1000.0
    for name in ("Pslow", "Pfast"):
        mu = np.asarray(centroids.get(name, np.zeros(0)))
        pool = np.asarray(pooled.get(name, np.zeros((0, 0))))
        if mu.size == 0:
            continue
        c = color_for_label[name]
        if pool.shape[0] > 1:
            std = pool.std(axis=0)
            ax.fill_between(t_ms[:mu.size], mu - std, mu + std,
                            color=c, alpha=0.20, lw=0)
        ax.plot(t_ms[:mu.size], mu, color=c, lw=1.8,
                label=f"{name} (n={counts.get(name, 0)})")
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("z-scored amplitude")
    ax.set_title("F. canonical Pslow / Pfast waveforms")
    ax.legend(loc="upper right", fontsize=8)


def make_method_figure(
    results: Sequence[dict],
    pulse_type_results: dict,
    *,
    fs: float = 800.0,
    example_pair_idx: Optional[int] = None,
    cfg: Optional[PulseTypeConfig] = None,
):
    """Render the 2x3 Pslow/Pfast classification method figure.

    Parameters
    ----------
    results : sequence of per-pair result dicts (must contain
        ``song0.sides[L|R].pulse_features.{waveforms,symmetry,peak_frames}``
        and ``song0.wing_data``).
    pulse_type_results : output of
        :func:`utils.pulse_type_cache.get_pulse_type_labels`.
    fs : sampling rate of the wing-tip trace.
    example_pair_idx : pair index to draw panels A-C from. If ``None``,
        the first bout with at least 8 detected pulses is used.
    cfg : :class:`PulseTypeConfig` for the refit. Defaults match the
        cache settings (``random_state=0``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    pooled_w, pooled_s = _gather_pooled(results)
    if cfg is None:
        cfg = PulseTypeConfig()
    if pooled_w.shape[0] < cfg.n_components:
        raise ValueError(
            f"too few pooled pulses ({pooled_w.shape[0]}) to refit the GMM"
        )
    model: PulseTypeModel = fit_pulse_type_model(pooled_w, pooled_s, cfg)
    scores = model.pca.transform(pooled_w)
    hard = model.gmm.predict(scores)

    ex = _pick_example_bout(results, pair_idx=example_pair_idx)
    if ex is None:
        raise RuntimeError("no example bout with detected pulses")
    r_, dom_, z_, peaks_, pf_ = ex

    fig, axes = plt.subplots(
        2, 3, figsize=(13.5, 7.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.32, "left": 0.10},
    )
    _panel_a_trace(axes[0, 0], z_, peaks_, fs)
    cascade = _panel_b_cascade(axes[0, 1], z_, peaks_, fs)
    if cascade is not None:
        aligned, t_ms_w = cascade
        _panel_c_symmetry(axes[0, 2], aligned, t_ms_w)
    _panel_d_pca_gmm(axes[1, 0], scores, hard, model, model.label_map)
    _panel_e_symmetry_bars(axes[1, 1], model, model.label_map)
    _panel_f_centroids(axes[1, 2], pulse_type_results, fs)

    pidx = r_.get("pair_idx", "?")
    fig.suptitle(
        f"Pslow / Pfast classification stages   "
        f"(example: pair {pidx}, dom={dom_};   "
        f"pooled pulses N={pooled_w.shape[0]})",
        fontsize=11, fontweight="bold", y=0.99,
    )
    return fig
