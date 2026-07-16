"""1-Euro filtering of per-camera 2D keypoints, applied pre-triangulation.

ViTPose's continuous soft-argmax decode reveals high-frequency 2D shimmer that
EfficientTrack's quantized argmax hid. The 1-Euro filter removes it at the
source -- per camera, per keypoint, along time -- adapting its cutoff to speed
(strong smoothing when still, low lag when moving fast) so it does not smear
fast motion. Low-confidence frames split the signal into independent segments
so the filter never smooths across a tracking dropout, and named keypoints
(default: wings) are passed through untouched to preserve song kinematics.

Default-off: wired as a gated Stage-A step in scripts/run_bout.py.
"""
import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro_1d(x, *, dt=1.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """1-D 1-Euro filter over x (T,). NaNs reset filter state (segment split)
    and stay NaN in the output; finite samples are always finite out."""
    x = np.asarray(x, np.float64)
    T = x.shape[0]
    out = np.full(T, np.nan)
    x_prev = dx_prev = None
    for t in range(T):
        if not np.isfinite(x[t]):
            x_prev = dx_prev = None
            continue
        if x_prev is None:
            out[t] = x[t]; x_prev = x[t]; dx_prev = 0.0
            continue
        dx = (x[t] - x_prev) / dt
        a_d = _alpha(d_cutoff, dt)
        edx = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(edx)
        a = _alpha(cutoff, dt)
        xhat = a * x[t] + (1.0 - a) * x_prev
        out[t] = xhat; x_prev = xhat; dx_prev = edx
    return out


def filter_kp2d_oneeuro(kp2d, conf, kp_names, *, conf_thresh=0.3,
                        min_cutoff=1.0, beta=0.0, d_cutoff=1.0,
                        preserve_raw_patterns=("Wing",)):
    """kp2d (T,C,K,2), conf (T,C,K) -> filtered kp2d (T,C,K,2).

    Per (camera, keypoint): filter x and y over time; frames with
    conf < conf_thresh are treated as gaps (filter resets, raw value kept).
    Keypoints whose name contains any preserve_raw_patterns entry are left raw.
    Coverage invariant: never introduces a NaN where the input was finite."""
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf)
    T, C, K, _ = kp2d.shape
    out = kp2d.copy()
    keep_raw = {i for i, n in enumerate(kp_names)
                if any(p in n for p in preserve_raw_patterns)}
    for c in range(C):
        for k in range(K):
            if k in keep_raw:
                continue
            low = conf[:, c, k] < conf_thresh
            for d in range(2):
                v = kp2d[:, c, k, d].copy()
                v[low] = np.nan
                f = one_euro_1d(v, dt=1.0, min_cutoff=min_cutoff,
                                beta=beta, d_cutoff=d_cutoff)
                good = np.isfinite(f)
                out[good, c, k, d] = f[good]       # raw kept where filter=NaN (gaps)
    return out
