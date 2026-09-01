#!/usr/bin/env python3
"""IK-optimized global body scale: pick the data->model scale that minimizes
the REAL STAC marker residual, instead of a geometric heuristic.

Motivation (measured, not hypothetical). Three geometric scale estimators
disagree by +/-12% on the same fly (Session0 2025_10_20_13_20_04, fly1/male):

    current pipeline (single-bout trunk Umeyama):     0.010994
    robust recording-level trunk Umeyama:              0.013242
    pose-invariant rigid-segment lengths (23 joint-to-joint
      distances -- the cleanest size evidence, cannot change with pose):
                                                        0.011789

For the female the rigid-segment estimate is barely usable at all (IQR
0.0071-0.0125). Meanwhile STAC (``stac_mjx.compute_stac``) has NO scale
parameter of its own: it only alternates POSE and OFFSET optimization
(``offset_optimization`` regularizes just a masked subset of the 50 marker
offsets via ``m_reg_coef``). So when the fixed input scale is wrong, the
solver launders the size error into the marker offsets instead -- the known
"markers float off the mesh" artifact (see ``scripts/viz/compare_stac_fits.py``).

This module does NOT touch the frozen jaxls/stac solver. Instead it treats
scale as a single scalar hyperparameter and picks it by a 1-D search over
log(scale), scoring each candidate by actually running the existing IK
(``jarvis_jax.tracking.stac.fit_offsets_once`` + ``ik_only_bout``) and reading
back the real marker residual.

CRITICAL: at every candidate scale, marker offsets are REFIT FROM SCRATCH
(``fit_offsets_once``), never reused from a different candidate's fit. If
offsets were shared across candidates, the 50 marker offsets would simply
re-absorb whatever size error the "wrong" scale introduces -- the very
degeneracy (offsets laundering scale) that this search exists to break --
and the scale->residual curve would go flat by construction, telling you
nothing about identifiability.

CLI:
    python scripts/fit_global_scale.py --run-root <recording>/pose --fly 1 \\
        [--bouts 1,2,3] [--lo 0.008 --hi 0.016] [--n-frames 150] \\
        [--method golden|grid --n-grid 7] [--anatomy configs/anatomy/v1.yaml] \\
        [--out scale_ik.json] [--dry-run]

Prints the full scale->residual curve every time (not just the argmin): a
flat curve is a real, useful result -- it means scale is weakly identified
from markers alone -- not a failure of the search.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scripts.estimate_recording_scale import bout_kp3d_paths, _bout_idx_from_path
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from estimate_recording_scale import bout_kp3d_paths, _bout_idx_from_path

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1-D scale search primitives -- pure, no GPU, unit-tested directly
# ---------------------------------------------------------------------------

def candidate_scales(lo: float, hi: float, n: int) -> List[float]:
    """``n`` log-spaced candidate scales in ``[lo, hi]``, both endpoints
    included. Scale is a ratio (data-units -> model-units), so a log grid is
    the natural spacing -- linear spacing would over-sample the top of the
    range and under-sample the bottom. Deterministic (no randomness)."""
    if n < 1:
        raise ValueError(f"candidate_scales: n must be >= 1, got {n}")
    if lo <= 0 or hi <= 0:
        raise ValueError(f"candidate_scales: lo/hi must be > 0, got lo={lo} hi={hi}")
    if hi < lo:
        raise ValueError(f"candidate_scales: hi ({hi}) must be >= lo ({lo})")
    if n == 1:
        return [float(lo)]
    log_lo, log_hi = math.log(lo), math.log(hi)
    step = (log_hi - log_lo) / (n - 1)
    return [float(math.exp(log_lo + i * step)) for i in range(n)]


_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0    # ~0.618, golden ratio conjugate
_INV_PHI2 = 1.0 - _INV_PHI                 # ~0.382


def golden_section(f: Callable[[float], float], lo: float, hi: float, *,
                    tol_rel: float = 0.01, max_evals: int = 12
                    ) -> Tuple[float, dict]:
    """Minimize scalar ``f`` over ``[lo, hi]`` by golden-section search on
    ``log(scale))``. ``f`` is called with values in the ORIGINAL (non-log)
    scale -- the log transform is purely an internal bracketing device so
    the search samples proportionally across the range, matching
    ``candidate_scales``'s log spacing.

    Pure: ``f`` may be any callable (real IK scoring, a synthetic parabola
    for tests, ...), so this needs no GPU and no jarvis_jax import.

    Stops when the bracket has shrunk to within ``tol_rel`` (interpreted as
    a width in log-space, i.e. approximately a relative width in original
    scale for small ``tol_rel``) or ``max_evals`` function calls have been
    made, whichever comes first -- so it always terminates, even for a
    perfectly flat ``f`` (bracket shrinkage is geometric and independent of
    the function's values).

    Returns ``(best_x, info)`` where ``info = {"evals": {x: f(x), ...},
    "n_evals": int}`` -- the full evaluation trace, not just the winner, so
    callers can see the objective's shape (see module docstring: a flat
    curve is a real result).
    """
    if lo <= 0 or hi <= 0:
        raise ValueError(f"golden_section: lo/hi must be > 0, got lo={lo} hi={hi}")
    if hi <= lo:
        raise ValueError(f"golden_section: hi ({hi}) must be > lo ({lo})")

    evals: Dict[float, float] = {}

    def _score(log_x: float) -> float:
        x = math.exp(log_x)
        val = evals.get(x)
        if val is None:
            val = float(f(x))
            evals[x] = val
        return val

    a, b = math.log(lo), math.log(hi)
    c = a + _INV_PHI2 * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = _score(c), _score(d)
    n_evals = 2

    while n_evals < max_evals and (b - a) > tol_rel:
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + _INV_PHI2 * (b - a)
            fc = _score(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = _score(d)
        n_evals += 1

    best_log_x = c if fc <= fd else d
    best_x = math.exp(best_log_x)
    return best_x, {"evals": dict(evals), "n_evals": n_evals}


# ---------------------------------------------------------------------------
# Real IK scoring (GPU) -- everything above this line is unit-testable
# without touching it.
# ---------------------------------------------------------------------------

def score_scale(cfg, kp3d, kp_names, scale: float, *, offsets_path: str,
                 frames: Optional[Sequence[int]] = None) -> dict:
    """Run the existing IK for ONE candidate ``scale`` over (optionally) a
    frame subsample, and score it by the REAL median STAC marker residual.

    IMPORTANT: marker offsets are refit FROM SCRATCH at this scale
    (``fit_offsets_once``) rather than reusing offsets fit at a different
    candidate scale. STAC has no scale parameter of its own; reusing offsets
    across candidates would let the 50 marker offsets absorb whatever size
    error a wrong scale introduces, flattening the scale->residual objective
    by construction -- exactly the "markers float off the mesh" degeneracy
    (offsets laundering a scale error) that this whole search exists to
    break. Every candidate must get its own honest offset fit.

    Writes into a fresh ``tempfile.TemporaryDirectory`` (removed on return),
    never into the user's processed data.

    Returns ``{"scale": scale, "marker_px": <median |kp_data - marker_sites|
    over all finite marker-frames>, "n_frames": n}``.
    """
    import h5py
    from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout

    kp3d = np.asarray(kp3d, dtype=np.float64)
    if frames is not None:
        kp3d = kp3d[np.asarray(frames)]
    kp_names = list(kp_names)
    n_frames = int(kp3d.shape[0])

    with tempfile.TemporaryDirectory(prefix="fit_global_scale_") as tmp:
        save_path = Path(tmp)
        fit_offsets_once(cfg, kp3d, kp_names, offsets_path=offsets_path,
                          save_path=save_path, scale=scale)
        h5_path = ik_only_bout(cfg, kp3d, kp_names, offsets_path=offsets_path,
                                out_h5="ik_only.h5", save_path=save_path,
                                scale=scale)
        with h5py.File(h5_path, "r") as f:
            kp_data = np.asarray(f["kp_data"]).reshape(n_frames, -1, 3)
            marker_sites = np.asarray(f["marker_sites"])

    resid = np.linalg.norm(kp_data - marker_sites, axis=-1)
    marker_px = float(np.nanmedian(resid))
    return {"scale": float(scale), "marker_px": marker_px, "n_frames": n_frames}


def _pool_frames(bout_dirs: Sequence[Path], n_frames: int) -> Tuple[np.ndarray, int]:
    """Load ``kp3d_filt.npz`` (falling back to ``kp3d.npz``) from every dir
    in ``bout_dirs``, drop frames with any non-finite marker, concatenate in
    the given order, then deterministically pick up to ``n_frames`` evenly
    spaced frames from the pool (``np.linspace`` indices, duplicates
    collapsed for small pools)."""
    arrays = []
    for d in bout_dirs:
        d = Path(d)
        filt = d / "kp3d_filt.npz"
        raw = d / "kp3d.npz"
        path = filt if filt.exists() else raw
        if not path.exists():
            raise FileNotFoundError(f"_pool_frames: no kp3d(_filt).npz under {d}")
        with np.load(path) as z:
            kp3d = np.asarray(z["kp3d"], dtype=np.float64)
        valid = np.all(np.isfinite(kp3d), axis=(1, 2))
        if valid.any():
            arrays.append(kp3d[valid])
    if not arrays:
        raise ValueError(f"_pool_frames: no usable (all-finite) frames across {bout_dirs}")
    pooled = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]

    n = max(1, min(int(n_frames), pooled.shape[0]))
    idx = np.unique(np.linspace(0, pooled.shape[0] - 1, n).round().astype(int))
    return pooled[idx], int(idx.size)


def fit_global_scale(bout_dirs: Sequence[Path], cfg, *, lo: float, hi: float,
                      n_frames: int = 150, method: str = "golden",
                      n_grid: int = 7, offsets_path: str = "offsets_scale_search.h5",
                      score_fn: Optional[Callable] = None) -> dict:
    """Per fly: pool a frame subsample across ``bout_dirs``, search for the
    single global scale minimizing the real STAC marker residual.

    ``score_fn`` defaults to ``score_scale`` and is the seam this is
    unit-tested through (see ``tests/test_fit_global_scale.py``): pass a
    monkeypatched ``score_fn(cfg, kp3d, kp_names, scale, offsets_path=...)
    -> {"marker_px": ...}`` to test the search logic without a GPU.

    Returns ``{"scale": best, "curve": {scale: marker_px, ...},
    "method": method, "n_frames": n}``.
    """
    if score_fn is None:
        score_fn = score_scale
    if method not in ("golden", "grid"):
        raise ValueError(f"fit_global_scale: unknown method {method!r} (want 'golden'|'grid')")

    kp_names = list(cfg.model.KP_NAMES)
    kp3d, n_used = _pool_frames(bout_dirs, n_frames)

    curve: Dict[float, float] = {}

    def objective(scale: float) -> float:
        result = score_fn(cfg, kp3d, kp_names, scale, offsets_path=offsets_path)
        marker_px = float(result["marker_px"])
        curve[float(scale)] = marker_px
        return marker_px

    if method == "golden":
        best_scale, _search_info = golden_section(objective, lo, hi)
    else:  # "grid"
        for s in candidate_scales(lo, hi, n_grid):
            objective(s)
        best_scale = min(curve, key=curve.get)

    return {
        "scale": float(best_scale),
        "curve": dict(sorted(curve.items())),
        "method": method,
        "n_frames": n_used,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_bout_dirs(run_root: Path, fly: int, bouts: Optional[List[int]]) -> List[Path]:
    """Fly-bout directories under ``<run_root>/bouts/bout_*/fly<fly>``,
    restricted to ``bouts`` (bout indices) when given, else every bout found.
    Reuses ``estimate_recording_scale.bout_kp3d_paths`` (already handles the
    kp3d_filt.npz/kp3d.npz fallback and sorted bout-index parsing) rather
    than reimplementing that resolution."""
    paths = bout_kp3d_paths(run_root, fly)
    if bouts is not None:
        wanted = set(bouts)
        paths = [p for p in paths if _bout_idx_from_path(p) in wanted]
    return [p.parent for p in paths]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="IK-optimized global body scale: choose scale by "
                     "minimizing the real STAC marker residual.")
    ap.add_argument("--run-root", required=True, help="<recording>/pose dir")
    ap.add_argument("--fly", type=int, required=True)
    ap.add_argument("--bouts", default=None,
                     help="comma-separated bout indices (default: all bouts "
                          "under --run-root for --fly)")
    ap.add_argument("--lo", type=float, default=0.008)
    ap.add_argument("--hi", type=float, default=0.016)
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--method", default="golden", choices=["golden", "grid"])
    ap.add_argument("--n-grid", type=int, default=7)
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml",
                     help="anatomy config for KP_NAMES + mjcf path "
                          "(selects the hydra anatomy=<stem> override)")
    ap.add_argument("--out", default=None, help="default <run-root>/scale_ik.json")
    ap.add_argument("--dry-run", action="store_true", help="print, don't write")
    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    bouts = [int(b) for b in args.bouts.split(",")] if args.bouts else None
    bout_dirs = _resolve_bout_dirs(run_root, args.fly, bouts)
    if not bout_dirs:
        raise SystemExit(
            f"fit_global_scale: no bout dirs with kp3d(_filt).npz found under "
            f"{run_root} for fly{args.fly} bouts={bouts}")

    cfg = _load_pipeline_cfg(args.anatomy)

    result = fit_global_scale(
        bout_dirs, cfg, lo=args.lo, hi=args.hi, n_frames=args.n_frames,
        method=args.method, n_grid=args.n_grid)

    print(f"[fit_global_scale] run_root={run_root} fly={args.fly} "
          f"bouts={sorted(_bout_idx_from_path_safe(d) for d in bout_dirs)} "
          f"method={result['method']} n_frames={result['n_frames']}")
    print("  scale -> marker_px:")
    for s in sorted(result["curve"]):
        marker = " <== best" if math.isclose(s, result["scale"]) else ""
        print(f"    {s:.6f} -> {result['curve'][s]:.6f}{marker}")
    vals = list(result["curve"].values())
    spread_pct = (max(vals) - min(vals)) / min(vals) * 100 if min(vals) > 0 else float("nan")
    print(f"  best scale = {result['scale']:.6f}  (curve spread {spread_pct:.1f}% -- "
          f"a small spread means scale is WEAKLY identified from markers alone)")

    out_path = Path(args.out) if args.out else run_root / "scale_ik.json"
    if args.dry_run:
        print(f"[dry-run] would write {out_path}")
        return 0

    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, str(out_path))
    print(f"wrote {out_path}")
    return 0


def _bout_idx_from_path_safe(bout_fly_dir: Path) -> int:
    """``_bout_idx_from_path`` expects a *file* path two levels under the
    bout dir (``bout_dir/flyN/kp3d.npz``); the CLI only has the bout-fly
    DIR, so hand it a synthetic child path purely to reuse the same regex
    parsing (avoids a second copy of the ``bout_(\\d+)`` pattern here)."""
    return _bout_idx_from_path(Path(bout_fly_dir) / "kp3d.npz")


def _load_pipeline_cfg(anatomy: str, overrides: Optional[Sequence[str]] = None):
    """Compose the SAME stac-capable config ``scripts/run_bout.py``'s
    ``@hydra.main`` normally gets from ``configs/pipeline.yaml``, standalone.

    This CLI addresses a run by ``--run-root``/``--fly`` rather than a hydra
    ``recording`` group, but ``fit_offsets_once``/``ik_only_bout`` (via
    ``stac_mjx.run_stac``) need the FULL cfg tree (``cfg.stac``,
    ``cfg.model``, ``cfg.ik``, ...) regardless -- including branches
    this script never reads -- because ``stac_mjx.io.save_data_to_h5`` does
    a full ``OmegaConf`` resolve of the whole config before writing every
    h5 (see ``configs/pipeline.yaml``'s ``dataset``/``preprocessing`` block
    comments for the same constraint). ``anatomy`` is a path like
    ``configs/anatomy/v1.yaml``; only its stem is used, as the hydra
    ``anatomy=<stem>`` override -- every other group (paths, stac, recording,
    detector, silhouette, outputs) is whatever ``configs/pipeline.yaml``
    defaults to, since none of those recording-specific fields are read by
    the functions this script calls (kp3d comes from ``--run-root`` directly,
    not from ``cfg.recording``).
    """
    try:
        import utils.path_utils  # noqa: F401  (registers OmegaConf resolvers)
    except ImportError:
        repo = str(_REPO_ROOT)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import utils.path_utils  # noqa: F401
    import hydra

    anatomy_name = Path(anatomy).stem
    ov = [f"anatomy={anatomy_name}"] + list(overrides or []) + [
        "hydra/job_logging=disabled", "hydra/hydra_logging=disabled"]
    with hydra.initialize_config_dir(config_dir=str(_REPO_ROOT / "configs"),
                                      version_base=None):
        cfg = hydra.compose(config_name="pipeline", overrides=ov)
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
