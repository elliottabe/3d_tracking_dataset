"""Per-bout tracking-quality metrics for the benchmark suite.

All heavy inputs are the pipeline's existing per-bout artifacts (see plan
Global Constraints for schemas). Functions here are pure numpy where possible;
mujoco/jarvis_jax are imported lazily inside the functions that need them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEG_QPOS_PREFIXES = ('coxa_', 'trochanter_', 'femur_', 'tibia_', 'tarsus')
TRUNK_KEYPOINTS = {'Scutellum', 'WingL_base', 'WingR_base', 'Abd_A4', 'Abd_tip',
                   'Abd_A1', 'Abd_A2', 'Abd_A3'}


def _iou(qc, field):
    """qc.json's mesh-vs-mask IoU, under either key name (renamed 2026-09-01)."""
    blk = qc.get("mesh_mask_iou") or qc.get("silhouette_iou") or {}
    return blk.get(field)


def kp_group(name: str) -> str:
    if name in TRUNK_KEYPOINTS:
        return 'trunk'
    if name[:2] in ('T1', 'T2', 'T3'):
        return 'leg'
    if name.startswith('Wing'):
        return 'wing'
    return 'other'


# --- temporal stability -----------------------------------------------------

def jitter_series(qpos: np.ndarray, names_qpos: list[str]) -> np.ndarray:
    """(T-2,) per-frame median |second difference| over LEG qpos columns.

    Rad/frame^2 acceleration proxy: high-frequency wobble scores high, smooth
    fast motion scores low. Sampling-rate free (both assays share the rig fps).
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names_qpos]
    cols = [i for i, n in enumerate(names) if n.startswith(LEG_QPOS_PREFIXES)]
    if not cols:
        raise ValueError("no leg qpos columns found")
    d2 = np.diff(np.asarray(qpos, dtype=np.float64)[:, cols], n=2, axis=0)
    return np.nanmedian(np.abs(d2), axis=1)


def kp3d_spike_rate(kp3d: np.ndarray, kp_names: list[str],
                    thr_bl: float = 0.04) -> float:
    """Fraction of (frame, keypoint) samples whose 3D acceleration exceeds
    thr_bl body lengths.

    kp3d (T,K,3) raw triangulated keypoints in ANY consistent unit -- the
    threshold is normalised by the median Antenna_Base<->Abd_tip distance
    because the frozen benchmark tree triangulates in ~0.1 mm units while
    clip-format calibrations give mm. Acceleration = magnitude of the second
    temporal difference. thr_bl=0.04 reproduces the 0.1 mm threshold on a
    2.4 mm fly. This is the TAIL metric the median-style scores above cannot
    see: the 2026-08-14 jax-vs-jarvis A/B
    (docs/benchmark/2026-08-14-jax-vs-jarvis-stability/notes.md) measured
    identical median jitter but a 5x spike-count gap between plain DLT and
    fused lifting -- single-view leg swaps surface here and nowhere else.
    NaN samples (occlusion gaps) never count as spikes.
    """
    a = np.asarray(kp3d, dtype=np.float64)
    if a.shape[0] < 3:
        return 0.0
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in kp_names]
    i_ant, i_abd = names.index('Antenna_Base'), names.index('Abd_tip')
    with np.errstate(invalid='ignore'):
        body = float(np.nanmedian(
            np.linalg.norm(a[:, i_ant] - a[:, i_abd], axis=-1)))
    if not np.isfinite(body) or body <= 0:
        raise ValueError("cannot estimate body length for spike threshold")
    d2 = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=-1)   # (T-2,K)
    with np.errstate(invalid='ignore'):
        return float(np.mean(d2 > thr_bl * body))


# --- joint limits -----------------------------------------------------------

def joint_bounds(mj_model) -> tuple[np.ndarray, np.ndarray]:
    """(nq,) lower/upper qpos bounds; +/-inf for free/ball/unlimited joints."""
    import mujoco
    lb = np.full(mj_model.nq, -np.inf)
    ub = np.full(mj_model.nq, np.inf)
    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        adr = int(mj_model.jnt_qposadr[j])
        if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        lo, hi = mj_model.jnt_range[j]
        if lo == 0.0 and hi == 0.0:          # MuJoCo "unlimited" sentinel
            continue
        lb[adr], ub[adr] = float(lo), float(hi)
    return lb, ub


def joint_limit_violation_rate(qpos: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                               tol: float = 1e-4) -> float:
    """Fraction of (frame, bounded-dof) samples outside [lb-tol, ub+tol]."""
    qpos = np.asarray(qpos, dtype=np.float64)
    bounded = np.isfinite(lb) | np.isfinite(ub)
    if not bounded.any():
        return 0.0
    q = qpos[:, bounded]
    viol = (q < (lb[bounded] - tol)) | (q > (ub[bounded] + tol))
    return float(viol.mean())


# --- reprojection -----------------------------------------------------------

def _project(cam_mats: np.ndarray, X: np.ndarray) -> np.ndarray:
    """cam_mats (C,4,3) [uv_h = [X 1] @ P], X (T,K,3) -> (T,C,K,2) pixels."""
    Xh = np.concatenate([X, np.ones((*X.shape[:-1], 1))], axis=-1)   # (T,K,4)
    uvh = np.einsum('tkf,cfe->tcke', Xh, cam_mats)                    # (T,C,K,3)
    return uvh[..., :2] / np.where(np.abs(uvh[..., 2:3]) < 1e-12, np.nan,
                                   uvh[..., 2:3])


def reproj_series_by_group(kp3d_mm: np.ndarray, kp2d: np.ndarray,
                           conf: np.ndarray, cam_mats: np.ndarray,
                           kp_names: list[str], conf_thr: float = 0.3
                           ) -> dict[str, np.ndarray]:
    """Per-frame median reprojection error (px) per keypoint group.

    Projects the FITTED 3D sites into every camera and compares against the
    observed 2D detections, masking low-confidence detections. Frames where a
    group has no confident observation are NaN.
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in kp_names]
    err = np.linalg.norm(_project(cam_mats, kp3d_mm) - kp2d, axis=-1)  # (T,C,K)
    err = np.where(conf >= conf_thr, err, np.nan)
    out: dict[str, np.ndarray] = {}
    for grp in ('leg', 'wing', 'trunk'):
        cols = [i for i, n in enumerate(names) if kp_group(n) == grp]
        if not cols:
            continue
        with np.errstate(all='ignore'):
            out[grp] = np.nanmedian(err[:, :, cols].reshape(err.shape[0], -1),
                                    axis=1)
    return out


# --- multi-animal proximity ---------------------------------------------------

def proximity_bl(kp3d_self: np.ndarray, kp3d_partner: np.ndarray) -> np.ndarray:
    """(T,) inter-fly centroid distance in unit-free "body lengths".

    Body length proxy = median per-frame RMS keypoint spread of the self fly,
    so the number is invariant to the raw triangulation units.
    """
    a = np.asarray(kp3d_self, dtype=np.float64)
    b = np.asarray(kp3d_partner, dtype=np.float64)
    with np.errstate(all='ignore'):
        ca = np.nanmedian(a, axis=1)
        cb = np.nanmedian(b, axis=1)
        spread = np.sqrt(np.nansum((a - ca[:, None, :]) ** 2, axis=(1, 2))
                         / max(a.shape[1], 1))
        body = float(np.nanmedian(spread))
    return np.linalg.norm(ca - cb, axis=-1) / max(body, 1e-12)


# --- bout-level driver --------------------------------------------------------

def compute_bout_metrics(bout_dir: Path, partner_dir: Path | None = None,
                         calib_dir: Path | None = None) -> dict:
    """Compute all metrics for one bout-fly dir from its on-disk artifacts."""
    import h5py
    bout_dir = Path(bout_dir)
    with h5py.File(bout_dir / 'outputs.h5', 'r') as f:
        qpos = f['qpos'][:]
        kp3d_mm = f['kp3d_mm'][:]
        kp_names = [n.decode() for n in f['kp_names'][:]]
    with h5py.File(bout_dir / 'stac_ik.h5', 'r') as f:
        names_qpos = [n.decode() for n in f['names_qpos'][:]]
    qc = json.loads((bout_dir / 'qc.json').read_text())
    pf = np.load(bout_dir / 'qc_perframe.npz')

    series = {
        'jitter': jitter_series(qpos, names_qpos),
        'soft_iou': pf['soft_iou'],
        'hard_iou': pf['hard_iou'],
        'reproj_px': pf['reproj_px'],
    }
    scalars = {
        'reproj_px_median': qc['per_camera_reproj_px']['median'],
        'loo_px_median': qc['loo_reproj_px']['median'],
        # qc.json key renamed silhouette_iou -> mesh_mask_iou (2026-09-01, the
        # metric is mesh-vs-mask overlap and never was the deleted polish).
        # Fall back so scorecards over qc.json written before the rename still
        # read -- the frozen 13-bout baseline predates it.
        'soft_iou_median': _iou(qc, 'soft_median'),
        'hard_iou_median': _iou(qc, 'hard_median'),
    }
    # Raw-triangulation spike rate (pre-filter, pre-scale): the tail metric
    # median jitter/IoU cannot see -- see kp3d_spike_rate. Optional so bouts
    # frozen before this metric existed still score.
    kp3d_raw_path = bout_dir / 'kp3d.npz'
    if kp3d_raw_path.exists():
        with np.load(kp3d_raw_path) as z:
            scalars['kp3d_spike_rate'] = kp3d_spike_rate(z['kp3d'], kp_names)

    if calib_dir is not None:
        from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
        rt = ReprojectionTool(str(calib_dir))
        kp2d_f = np.load(bout_dir / 'kp2d.npz')
        by_group = reproj_series_by_group(
            kp3d_mm, kp2d_f['kp2d'], kp2d_f['conf'],
            np.asarray(rt.camera_matrices), kp_names)
        for grp, s in by_group.items():
            series[f'reproj_px_{grp}'] = s
            scalars[f'reproj_px_{grp}_median'] = float(np.nanmedian(s))

    kp3d_src = bout_dir / ('kp3d_filt.npz'
                           if (bout_dir / 'kp3d_filt.npz').exists() else 'kp3d.npz')
    if partner_dir is not None:
        p_src = Path(partner_dir) / ('kp3d_filt.npz'
                                     if (Path(partner_dir) / 'kp3d_filt.npz').exists()
                                     else 'kp3d.npz')
        series['proximity_bl'] = proximity_bl(
            np.load(kp3d_src)['kp3d'], np.load(p_src)['kp3d'])

    scalars['jitter_median'] = float(np.nanmedian(series['jitter']))
    return {'series': {k: np.asarray(v) for k, v in series.items()},
            'scalars': scalars, 'n_frames': int(qpos.shape[0])}
