"""Per-bout male/female identification for paired Drosophila courtship.

In Drosophila, courtship song is produced almost exclusively by the male, so
the simplest robust rule is "the fly with more song = male". This module
implements that rule with a body-length tiebreaker for the rare bout where
both flies are silent (male courtship can include quiet orienting phases).

Input is two ``analyze_fly_song`` outputs plus the raw keypoint arrays for
both flies in a bout. No h5 or config coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class SexIdConfig:
    #: Minimum song_fraction gap required for the song-based rule to be
    #: considered confident. If both flies sing with similar fractions
    #: (e.g. 0.10 vs 0.11) we fall back to body length.
    min_song_gap: float = 0.02
    #: Below this absolute song fraction on BOTH flies, the bout is
    #: effectively silent — fall back to body length.
    silence_threshold: float = 0.01
    #: Keypoints whose Euclidean distance we use as body length.
    body_length_kp_from: str = "Scutellum"
    body_length_kp_to: str = "Abd_tip"


def _median_body_length(
    kp: np.ndarray, kp_names: Sequence[str], cfg: SexIdConfig
) -> float:
    """Median |Scutellum − Abd_tip| over all frames with finite data."""
    kp = np.asarray(kp)
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    names = list(kp_names)
    i_from = names.index(cfg.body_length_kp_from)
    i_to = names.index(cfg.body_length_kp_to)
    d = np.linalg.norm(kp[:, i_from, :] - kp[:, i_to, :], axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")


def identify_male_female(
    fly0_result: Dict,
    fly1_result: Dict,
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[SexIdConfig] = None,
) -> Dict:
    """Return the male/female assignment for one bout.

    Parameters
    ----------
    fly0_result, fly1_result : dict
        Outputs of ``song_analysis.analyze_fly_song`` (must contain a
        ``summary.song_fraction`` entry).
    fly0_kp, fly1_kp : array
        ``kp_data`` (or any (T, N, 3) keypoint array) used for the
        body-length tiebreaker. Should be in the SAME units and frame of
        reference for both flies so body lengths are comparable.
    kp_names : Sequence[str]
        Names along the N axis.
    cfg : SexIdConfig, optional

    Returns
    -------
    dict with keys:
        ``male_id``          — 'fly0' or 'fly1'
        ``female_id``        — the other one
        ``criterion``        — 'song_fraction' or 'body_length'
        ``confidence``       — [0, 1], larger is more confident
        ``song_fraction_male`` / ``song_fraction_female``
        ``body_length_male`` / ``body_length_female`` — data-unit lengths
        ``disagree``         — True if the body-length criterion would
                               have flipped the song-based assignment
                               (sanity-check field for plots / tables).
    """
    if cfg is None:
        cfg = SexIdConfig()

    sf0 = float(fly0_result["summary"]["song_fraction"])
    sf1 = float(fly1_result["summary"]["song_fraction"])

    bl0 = _median_body_length(fly0_kp, kp_names, cfg)
    bl1 = _median_body_length(fly1_kp, kp_names, cfg)

    gap = sf0 - sf1
    both_silent = (sf0 < cfg.silence_threshold) and (sf1 < cfg.silence_threshold)
    ambiguous = abs(gap) < cfg.min_song_gap

    if both_silent or ambiguous:
        # Fall back to body-length: male is the smaller fly in Drosophila.
        if np.isfinite(bl0) and np.isfinite(bl1) and bl0 != bl1:
            male = "fly0" if bl0 < bl1 else "fly1"
            # Confidence = normalized size gap, capped at 1.
            size_gap = abs(bl0 - bl1) / max(bl0, bl1)
            confidence = float(min(1.0, size_gap * 5.0))
            criterion = "body_length"
        else:
            # Degenerate: no useful signal. Default to fly0 = male with
            # zero confidence so the caller can filter these out.
            male = "fly0"
            confidence = 0.0
            criterion = "body_length"
    else:
        male = "fly0" if gap > 0 else "fly1"
        # Confidence = normalized song-fraction gap, squashed.
        confidence = float(min(1.0, abs(gap) / max(1e-6, max(sf0, sf1)) ))
        criterion = "song_fraction"

    female = "fly1" if male == "fly0" else "fly0"
    sf_male = sf0 if male == "fly0" else sf1
    sf_female = sf1 if male == "fly0" else sf0
    bl_male = bl0 if male == "fly0" else bl1
    bl_female = bl1 if male == "fly0" else bl0

    # Sanity check: would the body-length rule have flipped this?
    disagree = False
    if criterion == "song_fraction" and np.isfinite(bl0) and np.isfinite(bl1):
        body_male = "fly0" if bl0 < bl1 else "fly1"
        disagree = (body_male != male)

    return {
        "male_id": male,
        "female_id": female,
        "criterion": criterion,
        "confidence": confidence,
        "song_fraction_male": sf_male,
        "song_fraction_female": sf_female,
        "body_length_male": bl_male,
        "body_length_female": bl_female,
        "disagree": bool(disagree),
    }
