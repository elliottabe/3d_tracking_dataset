"""Trunk-marker Procrustes body-size scale for the courtship STAC stage.

Ported (adapted) from `scripts/preprocess_keypoints_for_ik.py`
(`_umeyama_scale_per_frame`, ~lines 363-415, and the trunk path of
`compute_shared_scale`, ~lines 418-504). Keep this module in sync if the
algorithm there changes.

The courtship inference pipeline feeds RAW triangulated 3D keypoints straight
into STAC, which are ~78x the MuJoCo model's rest-pose scale -- without a
body-size scale, STAC's jaxls LM-batch Jacobian is catastrophically
ill-conditioned and the solve stalls. This module estimates ONE scale
(multiply keypoints by it to match the model) from rigid trunk markers, using
each frame's rotation/translation-invariant centered spread so the estimate
is independent of heading and of which legs are tracked.
"""
from __future__ import annotations

from typing import List, Optional

import mujoco
import numpy as np

DEFAULT_TRUNK_KEYPOINTS = ['Scutellum', 'WingL_base', 'WingR_base', 'Abd_A4', 'Abd_tip']


def _umeyama_scale_per_frame(P: np.ndarray, ref_centered: np.ndarray,
                             robust: str = 'huber', n_iter: int = 3) -> np.ndarray:
    """Per-frame Umeyama least-squares similarity scale (data -> model rest).

    For each frame, the optimal similarity scale mapping the trunk markers ``P``
    onto the fixed model rest trunk ``ref_centered`` is
    ``s = trace(D·S) / Σ‖p_centered‖²`` where ``U D Vᵀ = svd(refᵀ·data)`` and
    ``S`` is the reflection-correction sign matrix (Umeyama 1991). This scale is
    provably ≤ the norm-ratio ``‖ref‖/‖data‖`` (von Neumann trace inequality), so
    it never over-scales when the shapes don't perfectly match.

    Args:
        P: (F, n, 3) trunk-marker positions, one set per valid frame.
        ref_centered: (n, 3) model rest trunk positions, already centered.
        robust: 'huber' runs IRLS down-weighting outlier markers; 'none' does a
            single unweighted fit.
        n_iter: IRLS iterations when robust='huber'.

    Returns:
        (F,) per-frame scale (data -> model).
    """
    P = np.asarray(P, dtype=np.float64)
    Y = ref_centered[None].astype(np.float64)           # (1, n, 3)
    F, n, _ = P.shape
    w = np.ones((F, n), dtype=np.float64)
    c = np.ones(F, dtype=np.float64)
    iters = n_iter if robust == 'huber' else 1
    for _ in range(iters):
        wsum = w.sum(axis=1)[:, None, None]             # (F, 1, 1)
        wn = w[..., None]                               # (F, n, 1)
        Xbar = (wn * P).sum(axis=1, keepdims=True) / np.maximum(wsum, 1e-12)
        Ybar = (wn * Y).sum(axis=1, keepdims=True) / np.maximum(wsum, 1e-12)
        Xc = P - Xbar                                   # (F, n, 3)
        Yc = Y - Ybar                                   # (F, n, 3)
        H = np.einsum('fn,fni,fnj->fij', w, Yc, Xc)     # (F, 3, 3) = refᵀ·data
        U, Dv, Vt = np.linalg.svd(H)
        dsign = np.sign(np.linalg.det(H))
        dsign[dsign == 0] = 1.0
        trDS = Dv[:, 0] + Dv[:, 1] + dsign * Dv[:, 2]
        denom = (w * (Xc ** 2).sum(-1)).sum(axis=1)     # (F,)
        c = trDS / np.maximum(denom, 1e-12)
        if robust != 'huber':
            break
        Smid = np.ones((F, 3)); Smid[:, 2] = dsign
        R = np.einsum('fij,fj,fjk->fik', U, Smid, Vt)   # (F, 3, 3) data -> model
        Yhat = c[:, None, None] * np.einsum('fij,fnj->fni', R, Xc)
        resid = np.linalg.norm(Yhat - Yc, axis=-1)      # (F, n)
        rr = resid.reshape(-1)
        med = np.median(rr)
        sigma = 1.4826 * np.median(np.abs(rr - med)) + 1e-12
        delta = 1.345 * sigma                           # Huber threshold
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-12))
    return c


def compute_trunk_scale(kp3d: np.ndarray, kp_names: List[str], model_xml: str, *,
                        trunk_names: Optional[List[str]] = None, estimator: str = 'umeyama',
                        robust_stat: str = 'median', robust: str = 'none') -> float:
    """One body-size scale (multiply keypoints by it to match the model) from rigid
    trunk markers. kp3d (T,K,3), kp_names length K, model_xml path to the MuJoCo XML.
    Mirrors preprocess_keypoints_for_ik.compute_shared_scale (trunk path).
    Returns float. Raises ValueError if <3 trunk markers are present in BOTH kp_names
    and the model's tracking[...] sites (the caller relies on a real scale; a silent
    1.0 would re-introduce the stall)."""
    names = list(trunk_names) if trunk_names else list(DEFAULT_TRUNK_KEYPOINTS)

    mj = mujoco.MjModel.from_xml_path(str(model_xml))
    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)
    tracking_site_idx = {}
    for i in range(mj.nsite):
        site_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if site_name and site_name.startswith('tracking[') and site_name.endswith(']'):
            tracking_site_idx[site_name[len('tracking['):-1]] = i

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    present = [n for n in names if n in name_to_idx and n in tracking_site_idx]
    if len(present) < 3:
        raise ValueError(
            f"compute_trunk_scale needs >=3 trunk markers present in both kp_names and "
            f"the model's tracking[...] sites; requested {names}, found {present}")

    # Reference (model rest pose) trunk positions, centered.
    ref = np.array([d.site_xpos[tracking_site_idx[n]] for n in present])
    ref_centered = ref - ref.mean(axis=0)
    ref_spread = float(np.sqrt((ref_centered ** 2).sum()))

    # Data trunk points over frames where all trunk markers are finite
    # (rotation/translation invariant once centered per frame).
    tidx = [name_to_idx[n] for n in present]
    P = np.asarray(kp3d, dtype=np.float64)[:, tidx, :]
    valid = np.all(np.isfinite(P), axis=(1, 2))
    P = P[valid]
    if P.shape[0] == 0:
        raise ValueError("compute_trunk_scale: no frames with all trunk markers finite")

    # Reference per-frame norm-ratio spread (also used as the legacy estimate and
    # the umeyama fallback).
    spreads = np.sqrt(((P - P.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2)))
    data_spread = float(np.median(spreads) if robust_stat == 'median' else np.mean(spreads))
    # Degenerate trunk (all markers collapsed to a point) -> scale would be inf/nan,
    # silently re-introducing the ill-conditioned STAC stall. Fail loudly instead.
    if data_spread <= 1e-10:
        raise ValueError(
            f"compute_trunk_scale: degenerate trunk marker spread ({data_spread:.3e}); "
            f"cannot compute a body-size scale")
    norm_ratio_scale = ref_spread / data_spread

    if estimator == 'norm_ratio':
        return float(norm_ratio_scale)

    # Umeyama least-squares similarity scale per frame, optionally Huber-IRLS
    # reweighted to resist a mistracked trunk marker. Never over-scales.
    c = _umeyama_scale_per_frame(P, ref_centered, robust=robust)
    c = c[np.isfinite(c) & (c > 0)]
    if c.size == 0:
        return float(norm_ratio_scale)
    return float(np.median(c) if robust_stat == 'median' else np.mean(c))
