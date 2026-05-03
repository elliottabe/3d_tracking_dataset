"""Pslow / Pfast pulse-type clustering for *D. melanogaster* courtship song.

Implements the second half of the Clemens et al. 2018 pipeline (after the
per-pulse features computed in :mod:`utils.song_analysis`): pool all
z-scored, aligned pulse waveforms across pairs, reduce with PCA, fit a
two-component Gaussian mixture, and assign each cluster to ``'Pslow'`` or
``'Pfast'`` by comparing per-cluster mean symmetry index (Clemens finding:
the symmetric mode is the slow mode).

Also exposes a Mahalanobis-distance template classifier
(:func:`qda_pulse_classifier`) — the closer analogue of what Clemens
reports — so bouts with too few pulses for their own GMM can still be
labelled against the globally-fit centroids.

No dependency on the notebook or on the analysis pipeline beyond the
``(N, W)`` pulse-waveform array and the per-pulse symmetry vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# -----------------------------------------------------------------------------
# Configuration + model dataclass
# -----------------------------------------------------------------------------


@dataclass
class PulseTypeConfig:
    """Hyperparameters for the Pslow/Pfast clusterer."""

    # PCA dimensionality fed into the GMM. Clemens used 8 on 10 kHz / 251-
    # sample waveforms; 6 is adequate for 20-sample 800 Hz windows.
    n_pca: int = 6
    # Number of mixture components. Always 2 (Pslow / Pfast).
    n_components: int = 2
    # GMM covariance structure. "full" is fine for 2 components in 6-D.
    covariance_type: str = "full"
    # RNG seed for the PCA/GMM fit (fully reproducible labels).
    random_state: int = 0
    # Cluster → type assignment rule. "symmetry" picks the cluster with the
    # higher mean symmetry index as Pslow (Clemens: symmetric = slow).
    label_by: str = "symmetry"


@dataclass
class PulseTypeModel:
    """Fitted clusterer plus the artifacts callers need to plot and
    classify new pulses."""

    pca: PCA
    gmm: GaussianMixture
    # {cluster_id_int: 'Pslow'|'Pfast'}
    label_map: Dict[int, str]
    # (n_components, W) mean waveform per type, in input (waveform) space.
    centroid_waveforms: np.ndarray
    # Per-component mean symmetry index (used to verify the label_map).
    component_symmetry: np.ndarray


# -----------------------------------------------------------------------------
# Fit / classify
# -----------------------------------------------------------------------------


def _as_float(waveforms: np.ndarray) -> np.ndarray:
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2:
        raise ValueError(
            f"waveforms must be (N, W); got shape {waves.shape}"
        )
    return waves


def fit_pulse_type_model(
    waveforms: np.ndarray,
    symmetry: np.ndarray,
    cfg: Optional[PulseTypeConfig] = None,
) -> PulseTypeModel:
    """Fit PCA → 2-component GMM on pooled z-scored pulse waveforms.

    Parameters
    ----------
    waveforms : (N, W) float
        Per-pulse waveforms as returned by
        :func:`utils.song_analysis.extract_pulse_waveforms` (already
        z-scored and sign-aligned).
    symmetry : (N,) float
        Per-pulse symmetry index from
        :func:`utils.song_analysis.compute_pulse_symmetry`. Used only
        for cluster→type assignment, not for fitting.
    cfg : PulseTypeConfig or None

    Returns
    -------
    PulseTypeModel
        Fitted PCA + GMM plus the cluster→type map and the mean
        waveform per type.
    """
    if cfg is None:
        cfg = PulseTypeConfig()
    waves = _as_float(waveforms)
    sym = np.asarray(symmetry, dtype=float)
    if sym.shape[0] != waves.shape[0]:
        raise ValueError(
            f"symmetry length {sym.shape[0]} != waveforms.shape[0] "
            f"{waves.shape[0]}"
        )
    if waves.shape[0] < cfg.n_components:
        raise ValueError(
            f"Need at least {cfg.n_components} pulses to fit a "
            f"{cfg.n_components}-component GMM; got {waves.shape[0]}."
        )

    n_pca = min(cfg.n_pca, waves.shape[1], waves.shape[0])
    pca = PCA(n_components=n_pca, random_state=cfg.random_state)
    X = pca.fit_transform(waves)

    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        random_state=cfg.random_state,
    )
    gmm.fit(X)
    hard = gmm.predict(X)

    # Per-component artefacts: mean symmetry and mean waveform.
    comp_sym = np.zeros(cfg.n_components, dtype=float)
    centroid_waves = np.zeros((cfg.n_components, waves.shape[1]), dtype=float)
    for k in range(cfg.n_components):
        sel = hard == k
        if sel.any():
            comp_sym[k] = float(np.nanmean(sym[sel]))
            centroid_waves[k] = waves[sel].mean(axis=0)
        else:
            comp_sym[k] = np.nan

    # Cluster → type assignment.
    if cfg.label_by != "symmetry":
        raise ValueError(f"Unknown label_by: {cfg.label_by}")
    pslow_k = int(np.nanargmax(comp_sym))
    label_map = {
        k: ("Pslow" if k == pslow_k else "Pfast")
        for k in range(cfg.n_components)
    }

    return PulseTypeModel(
        pca=pca,
        gmm=gmm,
        label_map=label_map,
        centroid_waveforms=centroid_waves,
        component_symmetry=comp_sym,
    )


def classify_pulses(
    waveforms: np.ndarray, model: PulseTypeModel
) -> np.ndarray:
    """Return (N,) array of ``'Pslow'`` / ``'Pfast'`` labels via the GMM.

    Hard-assigns each pulse to its most-likely component, then maps
    cluster IDs to type names via ``model.label_map``.
    """
    waves = _as_float(waveforms)
    if waves.shape[0] == 0:
        return np.zeros(0, dtype=object)
    X = model.pca.transform(waves)
    hard = model.gmm.predict(X)
    return np.array([model.label_map[int(k)] for k in hard], dtype=object)


def qda_pulse_classifier(
    waveforms: np.ndarray, model: PulseTypeModel
) -> np.ndarray:
    """Template / QDA fallback classifier (Clemens 2018 methods).

    Per pulse, compute Mahalanobis distance in PCA space to each GMM
    component using its own covariance, then pick the argmin. Equivalent
    to QDA with the fitted GMM's parameters — useful when you want a
    template-based label (the Clemens-reported method) instead of the
    full GMM posterior used by :func:`classify_pulses`.
    """
    waves = _as_float(waveforms)
    if waves.shape[0] == 0:
        return np.zeros(0, dtype=object)
    X = model.pca.transform(waves)
    K = model.gmm.n_components
    dists = np.empty((X.shape[0], K), dtype=float)
    for k in range(K):
        mu = model.gmm.means_[k]
        cov = model.gmm.covariances_[k]
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        diff = X - mu
        dists[:, k] = np.einsum("ni,ij,nj->n", diff, inv, diff)
    hard = dists.argmin(axis=1)
    return np.array([model.label_map[int(k)] for k in hard], dtype=object)
