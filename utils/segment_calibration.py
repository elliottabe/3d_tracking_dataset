"""Subject-specific per-segment model calibration.

Measures each body segment's length directly from the fly's joint keypoints
(median inter-keypoint distance) and produces per-segment scale factors that
morph the generic body model to *this* fly's proportions. A single global
isotropic scale + STAC marker offsets cannot fix proportion mismatches (e.g.
legs ~15% longer per segment, head ~17% further out); scaling the model's
segments does, so IK joint angles, marker residual, and the visual mesh all
improve together.

The fly skeleton has a keypoint at essentially every joint, so each model
segment's length is the distance between the two keypoints that bound it. A
segment is described by:
  - ``geom_body``:   the body whose geom *is* this segment (scaled visually).
  - ``length_body``: the body whose ``pos`` encodes the segment length (its
    offset from the proximal joint); scaling it lengthens the segment.
For a leg chain ``A->B``, ``geom_body = body(A)`` and ``length_body = body(B)``.
The head is a self-segment (``geom_body == length_body == head``).

This module produces the per-segment scale list; ``stac_mjx.rescale.rescale_per_segment``
applies it to the MJCF.
"""

from typing import Dict, List, Optional

import numpy as np

# Per-leg kinematic chain of keypoints (proximal -> distal).
_LEG_CHAIN = ['ThxCx', 'Tro', 'FeTi', 'TiTa', 'TaT1', 'TaT3', 'TaTip']
_LEGS = ['T1L', 'T1R', 'T2L', 'T2R', 'T3L', 'T3R']


def build_segment_map(keypoint_model_pairs: Dict[str, str]) -> List[Dict]:
    """Build the list of calibratable segments from KEYPOINT_MODEL_PAIRS.

    Args:
        keypoint_model_pairs: mapping keypoint name -> model body name.

    Returns:
        List of segment dicts with keys: name, prox_kp, dist_kp(s), geom_body,
        length_body, mirror (side-independent key for L/R symmetry), self_segment.
    """
    kmp = dict(keypoint_model_pairs)
    segs: List[Dict] = []

    # Legs: consecutive keypoints along each leg chain.
    for leg in _LEGS:
        num, side = leg[:2], leg[2]            # 'T1', 'L'
        kps = [f'{leg}_{seg}' for seg in _LEG_CHAIN]
        for i in range(len(kps) - 1):
            a, b = kps[i], kps[i + 1]
            if a in kmp and b in kmp:
                segs.append({
                    'name': f'{leg}_{_LEG_CHAIN[i]}',
                    'prox_kp': a, 'dist_kp': b,
                    'geom_body': kmp[a], 'length_body': kmp[b],
                    'mirror': f'{num}_{_LEG_CHAIN[i]}',   # T1_Tro mirrors L/R
                    'self_segment': False,
                })

    # Abdomen chain: Scutellum -> Abd_A4 -> Abd_tip (length on abdomen bodies).
    for prox, dist in [('Scutellum', 'Abd_A4'), ('Abd_A4', 'Abd_tip')]:
        if prox in kmp and dist in kmp:
            segs.append({
                'name': f'abd_{dist}', 'prox_kp': prox, 'dist_kp': dist,
                'geom_body': kmp[dist], 'length_body': kmp[dist],
                'mirror': None, 'self_segment': True,
            })

    # Head: self-segment. Length = Scutellum -> mean(eye/antenna). geom+length = head body.
    head_dists = [k for k in ('EyeL', 'EyeR', 'Antenna_Base') if k in kmp]
    if 'Scutellum' in kmp and head_dists:
        segs.append({
            'name': 'head', 'prox_kp': 'Scutellum', 'dist_kp': head_dists,
            'geom_body': kmp[head_dists[0]], 'length_body': kmp[head_dists[0]],
            'mirror': None, 'self_segment': True,
        })
    return segs


def _median_dist(kp: np.ndarray, names: List[str], a: str, b) -> Optional[float]:
    """Median (over frames) distance between keypoint a and b (b may be a list -> centroid)."""
    if a not in names:
        return None
    pa = kp[:, names.index(a), :]
    if isinstance(b, (list, tuple)):
        idxs = [names.index(x) for x in b if x in names]
        if not idxs:
            return None
        pb = np.nanmean(kp[:, idxs, :], axis=1)
    else:
        if b not in names:
            return None
        pb = kp[:, names.index(b), :]
    dd = np.linalg.norm(pa - pb, axis=-1)
    finite = dd[np.isfinite(dd)]
    return float(np.median(finite)) if finite.size else None


def estimate_segment_scales(keypoints: np.ndarray,
                            kp_names: List[str],
                            segment_map: List[Dict],
                            model_ref_lengths: Dict[str, float],
                            symmetry: bool = True,
                            clamp: tuple = (0.7, 1.4)) -> List[Dict]:
    """Compute per-segment scale = data_length / model_length.

    Args:
        keypoints: globally-scaled keypoints, (n_frames, n_kp, 3), model units.
        kp_names: keypoint names matching axis 1.
        segment_map: from build_segment_map().
        model_ref_lengths: segment name -> model rest-pose length (same units).
        symmetry: average L/R mirrored segments (fills an amputated/occluded side
            from its intact partner).
        clamp: (lo, hi) bounds; scales outside are clamped (reject tracking blowups).

    Returns:
        List of {name, geom_body, length_body, self_segment, scale} for segments
        with a usable scale (others default to 1.0 = unchanged).
    """
    kp = np.asarray(keypoints)
    raw: Dict[str, Optional[float]] = {}
    for seg in segment_map:
        dlen = _median_dist(kp, kp_names, seg['prox_kp'], seg['dist_kp'])
        mlen = model_ref_lengths.get(seg['name'])
        raw[seg['name']] = (dlen / mlen) if (dlen and mlen and mlen > 1e-9) else None

    # Symmetry: fill/average L/R by the side-independent mirror key.
    if symmetry:
        groups: Dict[str, List[str]] = {}
        for seg in segment_map:
            if seg['mirror']:
                groups.setdefault(seg['mirror'], []).append(seg['name'])
        for mirror, members in groups.items():
            vals = [raw[n] for n in members if raw[n] is not None]
            if vals:
                avg = float(np.mean(vals))
                for n in members:
                    raw[n] = avg            # mirror fills missing side (e.g. amputated T1L)

    out: List[Dict] = []
    for seg in segment_map:
        s = raw[seg['name']]
        if s is None:
            continue
        s = float(np.clip(s, clamp[0], clamp[1]))
        out.append({'name': seg['name'], 'geom_body': seg['geom_body'],
                    'length_body': seg['length_body'],
                    'self_segment': seg['self_segment'], 'scale': s})
    return out


def read_segment_scales(h5_path) -> Optional[List[Dict]]:
    """Read per-segment calibration scales from a preprocessed/STAC output h5.

    Looks up the ``info/segment_scales`` group written by preprocessing.

    Args:
        h5_path: path to a preprocessed (or STAC output) h5.

    Returns:
        List of ``{geom_body, length_body, scale}`` ready for
        ``rescale_per_segment`` / ``load_calibrated_spec``, or None if absent.
    """
    import h5py
    with h5py.File(str(h5_path), 'r') as f:
        g = f.get('info/segment_scales')
        if g is None:
            return None

        def _v(node):
            x = node[()]
            return x.decode() if isinstance(x, bytes) else x

        out = [{'geom_body': str(_v(g[k]['geom_body'])),
                'length_body': str(_v(g[k]['length_body'])),
                'scale': float(_v(g[k]['scale']))} for k in g.keys()]
    return out or None


def load_calibrated_spec(xml_path, segment_scales=None, global_scale: float = 1.0):
    """Load the base body model and reproduce an individual fly's calibrated model.

    Single entry point for the subject-specific model: an optional global
    isotropic scale (``dm_scale_spec``) followed by the per-segment morph
    (``rescale_per_segment``). Use with ``read_segment_scales`` to rebuild a
    fly's exact model anywhere (analysis, notebooks, re-rendering) from the
    scales stored during preprocessing.

    Args:
        xml_path: base MJCF path (e.g. fruitfly_v1_free.xml).
        segment_scales: list of ``{geom_body, length_body, scale}`` (from
            ``read_segment_scales``) or a dict keyed by segment name; None -> no morph.
        global_scale: optional global isotropic scale (default 1.0 = none).

    Returns:
        ``mujoco.MjSpec`` (morphed, not yet compiled). Call ``.compile()`` for a model.

    Example:
        scales = read_segment_scales(preprocessed_h5)
        model = load_calibrated_spec(base_xml, scales).compile()
    """
    import sys as _sys
    from pathlib import Path as _P
    import mujoco
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "stac-mjx"))
    from stac_mjx import rescale

    spec = mujoco.MjSpec.from_file(str(xml_path))
    if global_scale and abs(global_scale - 1.0) > 1e-9:
        spec = rescale.dm_scale_spec(spec, global_scale)
    if segment_scales:
        segs = (list(segment_scales.values())
                if hasattr(segment_scales, 'values') else list(segment_scales))
        rescale.rescale_per_segment(spec, segs)
    return spec


def load_calibrated_model(xml_path, segment_scales=None, global_scale: float = 1.0):
    """Compiled-model convenience wrapper around ``load_calibrated_spec``.

    Returns a ``mujoco.MjModel`` morphed to the individual fly.
    """
    return load_calibrated_spec(xml_path, segment_scales, global_scale).compile()


def model_reference_lengths(mj_model, segment_map: List[Dict]) -> Dict[str, float]:
    """Model rest-pose length for each segment, from tracking-site positions.

    Args:
        mj_model: compiled base MuJoCo model (mj_forward already valid at rest).
        segment_map: from build_segment_map().

    Returns:
        segment name -> rest-pose length (model units).
    """
    import mujoco
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)

    def site(kp):
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, f'tracking[{kp}]')
        return data.site_xpos[i].copy() if i >= 0 else None

    out: Dict[str, float] = {}
    for seg in segment_map:
        pa = site(seg['prox_kp'])
        dist = seg['dist_kp']
        if isinstance(dist, (list, tuple)):
            pts = [site(x) for x in dist]
            pts = [p for p in pts if p is not None]
            pb = np.mean(pts, axis=0) if pts else None
        else:
            pb = site(dist)
        if pa is not None and pb is not None:
            out[seg['name']] = float(np.linalg.norm(pa - pb))
    return out
