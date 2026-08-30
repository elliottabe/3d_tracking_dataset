"""Which of two flies in one frameset is the male?

WHY PAIRWISE. In the absolute labels, sex is perfectly confounded with
recording: all 4 male-labelled recordings are 100% male, all 4 female ones 100%
female. A per-fly classifier can score perfectly by learning lighting or one
individual's quirks, and nothing in an in-recording validation would show it.
The two-fly courtship recordings remove the confound completely -- both animals
appear in the same frame, same camera, same instant.

WHY ANTISYMMETRIC. Every feature is f(a) - f(b) and the model carries NO
intercept, so P(a is male) == 1 - P(b is male) identically. A model that could
call both flies in a courtship pair male is not modelling the problem.

CUES. Females are larger with a longer, more pointed abdomen; males are smaller
with a blunter, darker tip. Overall SIZE is a real cue here, so features are
deliberately NOT scale-normalised. Note the recorded gotcha: mask AREA is
backwards during courtship because the male extends a wing during song, so his
silhouette is LARGER -- that is why these features come from 3-D keypoint
geometry, not from silhouette extent.
"""
from __future__ import annotations

import re

import numpy as np

_FRAME_RE = re.compile(r"Frame_(\d+)")

# (name_a, name_b) segment lengths used as scalar descriptors.
_SEGMENTS = [
    ("Scutellum", "Abd_tip"),     # body length
    ("Scutellum", "Abd_A4"),      # thorax->mid-abdomen
    ("Abd_A4", "Abd_tip"),        # abdomen taper section
    ("WingL_base", "WingL_V12"),  # wing length
    ("EyeL", "EyeR"),             # head width
    ("T1L_FeTi", "T1L_TiTa"),     # tibia
]


def _scalars(kp3d: np.ndarray, names: list[str]) -> np.ndarray:
    out = []
    for a, b in _SEGMENTS:
        if a in names and b in names:
            d = kp3d[names.index(a)] - kp3d[names.index(b)]
            v = float(np.linalg.norm(d))
            out.append(v if np.isfinite(v) else 0.0)
        else:
            out.append(0.0)
    finite = kp3d[np.isfinite(kp3d).all(axis=-1)]
    extent = (float(np.linalg.norm(finite.max(0) - finite.min(0)))
              if finite.shape[0] >= 2 else 0.0)
    out.append(extent)
    body = out[0]
    # abdomen fraction of body: shape cue that survives a size difference
    out.append(out[2] / body if body > 1e-9 else 0.0)
    return np.asarray(out, np.float64)


def pair_features(kp3d_a, kp3d_b, keypoint_names) -> np.ndarray:
    """Antisymmetric descriptor of the ORDERED pair (a, b)."""
    a = _scalars(np.asarray(kp3d_a, np.float64), list(keypoint_names))
    b = _scalars(np.asarray(kp3d_b, np.float64), list(keypoint_names))
    f = a - b
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def fit_pairwise(X, y, groups) -> dict:
    """Logistic regression, NO intercept, leave-one-group-out CV.

    `groups` MUST be clip ids, never frameset ids: adjacent framesets inside a
    clip are near-duplicates, and grouping by frameset would leak exactly the
    way this project's dataset split used to.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    groups = np.asarray(groups)

    accs, uniq = [], sorted(set(groups.tolist()))
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        m = LogisticRegression(fit_intercept=False, max_iter=2000)
        m.fit(X[tr], y[tr])
        accs.append(float(m.score(X[te], y[te])))

    final = LogisticRegression(fit_intercept=False, max_iter=2000)
    final.fit(X, y)
    return {"coef": final.coef_[0].copy(), "intercept": 0.0,
            "fold_accuracy": accs, "cv_groups": uniq,
            "cv_mean": float(np.mean(accs)) if accs else float("nan"),
            "_model": final}


def predict_male_slot(model, kp3d_a, kp3d_b, keypoint_names):
    """-> (slot, probability). slot 0 means `kp3d_a` is the male."""
    f = pair_features(kp3d_a, kp3d_b, keypoint_names)
    z = float(np.dot(model["coef"], f))       # no intercept => exactly antisymmetric
    p_b_male = 1.0 / (1.0 + np.exp(-z))
    return (1, p_b_male) if p_b_male >= 0.5 else (0, 1.0 - p_b_male)


def clip_index(merged: dict, *, gap: int = 100) -> dict[str, list[str]]:
    """Group frameset keys into contiguous clips (frame gaps > `gap` split).

    Annotation order is NOT spatial (ann0 < ann1 in only 53% of framesets) but
    IS a stable track identity within a clip -- measured: on 113 near-adjacent
    frameset pairs, slot 0 stayed with the nearer fly 100% of the time. So one
    'which slot is the male' decision labels a whole clip.
    """
    per_rec: dict[str, list[tuple[int, str]]] = {}
    for key, v in merged["framesets"].items():
        m = _FRAME_RE.search(key)
        if m is None:
            continue
        per_rec.setdefault(v["recording"], []).append((int(m.group(1)), key))
    clips: dict[str, list[str]] = {}
    for rec, items in per_rec.items():
        items.sort()
        idx, cur, prev = 0, [], None
        for fr, key in items:
            if prev is not None and fr - prev > gap:
                clips[f"{rec}#clip{idx:03d}"] = cur
                idx += 1
                cur = []
            cur.append(key)
            prev = fr
        if cur:
            clips[f"{rec}#clip{idx:03d}"] = cur
    return clips
