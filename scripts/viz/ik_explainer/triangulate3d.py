#!/usr/bin/env python3
"""Triangulate the explainer clip's 2D into 3D (mm) and temporally filter it.

UNITS: the DLTs map mm -> px, so triangulating our own 2D yields mm directly.
No 0.1 factor appears here; that conversion exists only in
clip_io.load_shipped_kp3d_mm for the baseline comparison.

Filtering is a PREREQUISITE, not polish: jarvis_jax/tracking/filter.py records
that raw triangulated distal keypoints (esp. *_TaTip) jump many mm frame to
frame and STAC then bends the leg to chase the outlier. Act 4's whole claim is
that tarsal tips track the keypoints.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io   # noqa: E402

# Matches configs/detector/vitpose_v3.yaml.
CONF_THRESH = 0.3
VIEW_CONF_THRESH = 0.6


def _default_filter_cfg() -> dict:
    """Load the `filtering:` block from configs/pipeline.yaml.

    `filter_keypoints` (utils/keypoint_filter.py) reads its cfg with dotted
    attribute access on sub-blocks it doesn't explicitly enable (e.g.
    `filter_cfg.interpolation`, gated by a `.get(..., True)` default) -- it
    expects the full nested schema of an OmegaConf DictConfig, the same shape
    production's `cfg.filtering` (scripts/run_bout.py) already has. A minimal
    `{"enabled": True, "preserve_raw_patterns": [...]}` dict passes the
    top-level `.get(...)` guards but raises `ConfigAttributeError: Missing
    key interpolation` the moment that sub-block is accessed, so load the
    real block rather than typing a partial one by hand.
    """
    import yaml
    with open(_REPO / "configs" / "pipeline.yaml") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg["filtering"])


FILTER_CFG = _default_filter_cfg()


def triangulate(kp2d, conf, cam_mats, *, conf_thresh=CONF_THRESH,
                view_conf_thresh=VIEW_CONF_THRESH):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3) mm, conf3d)."""
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    kp2d = np.nan_to_num(np.asarray(kp2d, np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)   # NaN px would poison the SVD
    return triangulate_keypoints(kp2d, np.asarray(conf, np.float32),
                                 np.asarray(cam_mats, np.float32),
                                 conf_thresh=conf_thresh,
                                 view_conf_thresh=view_conf_thresh)


def filter_kp3d(kp3d, conf3d, kp_names, filter_cfg=None):
    """Temporally filter (T,K,3) mm keypoints ahead of STAC.

    `filter_keypoints` (utils/keypoint_filter.py) reads its cfg with BOTH
    `.get(...)` and dotted attribute access (e.g. `filter_cfg.interpolation`),
    i.e. it expects an OmegaConf DictConfig, not a plain dict -- confirmed by
    tests/test_courtship_kp_filter.py using `OmegaConf.create`. A plain dict
    passes the `.get(...)` guards but raises AttributeError the moment a
    sub-block (interpolation defaults to enabled) is accessed by attribute, so
    wrap here rather than silently handing it a plain dict.
    """
    from omegaconf import OmegaConf
    from jarvis_jax.tracking.filter import filter_bout_kp3d
    cfg = OmegaConf.create(dict(filter_cfg or FILTER_CFG))
    return filter_bout_kp3d(kp3d, conf3d, list(kp_names), cfg)


def run(clip: str = clip_io.CLIP_DEFAULT):
    d = clip_io.out_dirs(clip)
    cam_mats, _cams = clip_io.load_dlt(str(Path(clip) / "calibration"))
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp_names = [str(n) for n in z["kp_names"]]

    kp3d, conf3d = triangulate(z["kp2d"], z["conf"], cam_mats)
    np.savez_compressed(d["predictions"] / "03_kp3d.npz", kp3d=kp3d,
                        conf3d=conf3d, kp_names=np.array(kp_names))
    print(f"wrote 03_kp3d.npz {kp3d.shape} mm  finite={np.isfinite(kp3d).all(-1).mean():.3f}")

    filt = filter_kp3d(kp3d, conf3d, kp_names)
    np.savez_compressed(d["predictions"] / "04_kp3d_filt.npz", kp3d=filt,
                        conf3d=conf3d, kp_names=np.array(kp_names))
    print(f"wrote 04_kp3d_filt.npz {filt.shape} mm")
    return kp3d, filt


def qc_vs_baseline(clip: str = clip_io.CLIP_DEFAULT):
    """Compare our 3D against the shipped csv, and measure the filter's effect.

    EXPECTATION if ordering and units are right: per-keypoint median
    disagreement is SMALL and roughly uniform across keypoints. Given 1 px of
    2D error costs 9.3 um, a well-behaved run should sit in the tens of um to
    ~0.1 mm. A few hundred um on distal tarsi is plausible (different detector
    run); a UNIFORM offset of ~mm, or one keypoint wildly off, is a unit or
    ordering error.
    FALSIFICATION: disagreement clustered by keypoint GROUP (all head markers
    off by the same large amount) => the detector->model permutation is wrong.

    Filter expectation: *_TaTip max acceleration drops substantially
    (filter.py's own precedent: 12.5 -> 0.9 mm). No drop => filter not applied.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = clip_io.out_dirs(clip)
    ours = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)
    filt = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)
    kp_names = [str(n) for n in ours["kp_names"]]

    base_xyz, _bc, base_names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(clip))
    # baseline is DETECTOR order; ours is MODEL order.
    idx = clip_io.detector_to_model_index()
    base = base_xyz[:, idx]                       # -> MODEL order
    n = min(len(base), len(ours["kp3d"]))
    diff = np.linalg.norm(ours["kp3d"][:n] - base[:n], axis=-1)      # (n,K)
    med = np.nanmedian(diff, axis=0)

    def max_accel(a, name_sub="TaTip"):
        cols = [i for i, nm in enumerate(kp_names) if name_sub in nm]
        v = np.diff(a[:, cols], axis=0)
        acc = np.linalg.norm(np.diff(v, axis=0), axis=-1)
        return float(np.nanmax(acc))

    raw_acc = max_accel(ours["kp3d"])
    filt_acc = max_accel(filt["kp3d"])

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    ax[0].bar(range(len(med)), med)
    ax[0].set_xticks(range(len(med)))
    ax[0].set_xticklabels(kp_names, rotation=90, fontsize=6)
    ax[0].set_ylabel("median |ours - shipped| (mm)")
    ax[0].set_title(f"re-predicted vs shipped baseline, n={n} frames "
                    f"(1 px of 2D error = 9.3 um in 3D)")
    ax[1].bar(["raw", "filtered"], [raw_acc, filt_acc], color=["grey", "green"])
    ax[1].set_ylabel("max *_TaTip acceleration (mm/frame^2)")
    ax[1].set_title(f"filter effect: {raw_acc:.2f} -> {filt_acc:.2f} mm")
    out = d["qc"] / "03_kp3d_vs_shipped.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    import json
    (d["qc"] / "qc.json").write_text(json.dumps(
        {"median_diff_mm": dict(zip(kp_names, med.round(5).tolist())),
         "tatip_max_accel_raw_mm": raw_acc,
         "tatip_max_accel_filtered_mm": filt_acc,
         "n_frames_compared": int(n)}, indent=2))
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--qc", action="store_true",
                    help="also run the baseline + filter QC comparison")
    a = ap.parse_args()
    run(a.clip)
    if a.qc:
        qc_vs_baseline(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
