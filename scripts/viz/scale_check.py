"""Visual scale-verification tool: does a candidate body scale actually fit
the animal?

The pipeline's per-recording body scale used to be fit from ONE arbitrary
bout (see the scale-from-first-bout defect); ``scripts/estimate_recording_scale.py``
replaces that with a robust per-fly estimate. Numbers alone (span ratios) are
easy to misread; this module renders the MuJoCo model with the observed 3D
keypoints overlaid under several candidate scales side by side so a human can
see which one actually sits on the body.

By default the model is rendered at its REST pose, in which case only the
TRUNK row of the picture is trustworthy: legs are folded/extended
differently in the rest pose than in any real frame, so a rest-pose leg
span ratio conflates POSE with SIZE (a leg can look "too short" purely
because the model's rest leg angle differs from the real fly's leg angle in
that frame, independent of scale). Trunk sites are rigid relative to the
thorax root, so the trunk row's size comparison is pose-invariant and is not
subject to this confound.

Passing ``qpos`` drives the model to the SAME fitted pose as the observed
frame (see ``render_scale_check``'s ``qpos`` argument) before both reading
the model's tracking-site positions and rendering the mesh, so the LEG (and
wing) rows become a fair size comparison too -- the whole animal, not just
the trunk. Caveat: that fitted ``qpos`` was itself solved (by STAC/IK) at
whatever scale was used to produce it, so posed-mode ratios are not fully
independent of the CURRENT scale used upstream. The model's own geometry
(bone lengths, joint limits) is fixed by the XML regardless of which scale
solved the pose, so this is still a fair comparison of the OBSERVED
keypoints' size against the model -- just note the fitted pose itself
carries a little of that scale's fingerprint.

For each (scale, frame) the observed keypoints are:
  1. multiplied by the candidate scale, then
  2. rigidly aligned (rotation + translation ONLY, via Kabsch -- no scaling)
     onto the model's ``tracking[...]`` sites (rest pose, or the fitted pose
     when ``qpos`` is given), using only the TRUNK and LEG markers.

Wing markers (``WingL_V12``/``WingL_V13``/``WingR_V12``/``WingR_V13``, per
``scripts.benchmark.metrics.kp_group``) are excluded from the alignment fit
and from the size reference: the model rests with wings extended out to the
side, but a real fly at rest folds its wings over its back, so the model's
wing geometry is not a valid alignment or size target for the observed wing
markers. They are still rendered (in their own colour) and their span ratio
is still reported, purely as a diagnostic -- a mismatch there is EXPECTED
(folded vs. extended) and is not evidence of a bad scale.

Because a rigid transform never changes the size of a point cloud, a WRONG
scale shows up directly as spheres sitting outside (too small a scale) or
inside (too large a scale) the mesh, even after alignment removes any
rotation/translation ambiguity.

Model sites are resolved via ``scripts.estimate_recording_scale``'s
``_tracking_site_idx`` (reused by import, not reimplemented) -- raw
``mj_name2id`` lookups on ``tracking[<name>]`` strings return -1 and
``site_xpos[-1]`` silently yields the wrong site.

CLI:
    python scripts/viz/scale_check.py \\
        --bout-dir <run_root>/bouts/bout_00014/fly1 \\
        --scales 0.010994,0.013242,0.012890 \\
        --labels current,trunk_umeyama,all_norm \\
        --qpos-source stac \\
        [--frames 0,300,600] [--anatomy configs/anatomy/v1.yaml] \\
        --out docs/benchmark/scale-check/S0_bout14_fly1.png

``--qpos-source`` selects which fitted pose (if any) drives the model:
``stac`` (default) reads ``<bout-dir>/stac_ik.h5``'s ``qpos`` dataset,
``refined`` reads ``<bout-dir>/qpos_refined.npz``'s ``qpos`` array, ``none``
keeps the rest pose. A missing file falls back to rest pose with a warning
rather than crashing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.estimate_recording_scale import (  # noqa: E402
    _load_anatomy_cfg,
    _tracking_site_idx,
)
from scripts.benchmark.metrics import kp_group  # noqa: E402

TRUNK_RGBA = np.array([0.20, 0.90, 0.20, 1.0], dtype=np.float32)
LEG_RGBA = np.array([0.20, 0.55, 1.00, 1.0], dtype=np.float32)
WING_RGBA = np.array([1.00, 0.55, 0.00, 1.0], dtype=np.float32)
MODEL_RGBA = np.array([1.00, 1.00, 1.00, 0.95], dtype=np.float32)
_GROUP_RGBA = {"trunk": TRUNK_RGBA, "leg": LEG_RGBA, "wing": WING_RGBA}
_ALIGN_GROUPS = ("trunk", "leg")
_RATIO_GROUPS = ("trunk", "leg", "wing")


# ---------------------------------------------------------------------------
# rigid (rotation + translation only) alignment
# ---------------------------------------------------------------------------

def _rigid_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation ``R`` + translation ``t`` minimizing
    ``||(R @ src.T).T + t - dst||^2``. NO scaling -- size is the quantity
    under test elsewhere in this module and must never be absorbed here.

    ``src``/``dst`` are (N, 3), N >= 1. With N == 1 or N == 2 the rotation is
    underdetermined about one/two axes; the SVD solution below still returns
    a valid (if not fully constrained) rotation for those degenerate cases.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    Sc = src - mu_s
    Dc = dst - mu_d
    H = Sc.T @ Dc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _align_frame(scaled: np.ndarray, name_to_idx: Dict[str, int],
                 align_names: List[str], ref_by_name: Dict[str, np.ndarray]
                 ) -> np.ndarray:
    """Rigidly align ``scaled`` (K, 3) onto ``ref_by_name`` using only the
    finite ``align_names`` markers. Falls back to translation-only (1-2
    finite markers) or identity (0 finite markers) rather than raising --
    NaN rows in ``scaled`` propagate to NaN in the output, they are simply
    not drawn."""
    idxs = [name_to_idx[n] for n in align_names]
    pts = scaled[idxs]
    ref = np.array([ref_by_name[n] for n in align_names])
    finite = np.all(np.isfinite(pts), axis=1)
    src = pts[finite]
    dst = ref[finite]
    if src.shape[0] >= 3:
        R, t = _rigid_align(src, dst)
    elif src.shape[0] >= 1:
        R = np.eye(3)
        t = dst.mean(axis=0) - src.mean(axis=0)
    else:
        R = np.eye(3)
        t = np.zeros(3)
    return (R @ scaled.T).T + t


# ---------------------------------------------------------------------------
# span-ratio diagnostics
# ---------------------------------------------------------------------------

def _group_span_ratio(scaled: np.ndarray, name_to_idx: Dict[str, int],
                      names: List[str], ref_by_name: Dict[str, np.ndarray]
                      ) -> Optional[float]:
    """observed span / model span for one keypoint group, one frame.

    Span is the centered RMS spread (``sqrt(sum(centered**2))``), which is
    invariant to rotation/translation -- so this is identical whether
    computed on the raw scaled points or on their rigidly aligned versions.
    Needs >= 2 finite markers; returns None otherwise (caller skips it,
    doesn't fail)."""
    present = [n for n in names if n in name_to_idx]
    if not present:
        return None
    idxs = [name_to_idx[n] for n in present]
    pts = scaled[idxs]
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    used = [n for n, f in zip(present, finite) if f]
    if len(pts) < 2:
        return None
    ref = np.array([ref_by_name[n] for n in used])
    obs_span = float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum()))
    ref_span = float(np.sqrt(((ref - ref.mean(axis=0)) ** 2).sum()))
    if ref_span <= 1e-12:
        return None
    return obs_span / ref_span


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _add_sphere(scene, pos: np.ndarray, rgba: np.ndarray, size: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), rgba.astype(np.float32))
    scene.ngeom += 1


def _render_panel(renderer, d, cam, aligned: np.ndarray,
                  name_to_idx: Dict[str, int], groups: Dict[str, str],
                  ref_by_name: Dict[str, np.ndarray], marker: float) -> np.ndarray:
    """Render the model exactly as ``d`` currently holds it (rest pose, or
    whatever pose the caller last forwarded it to) plus the overlay spheres.
    ``ref_by_name`` are the model's own tracking-site positions in that same
    pose (white spheres), for a direct visual size comparison."""
    renderer.update_scene(d, camera=cam)
    scn = renderer.scene
    for pos in ref_by_name.values():
        _add_sphere(scn, pos, MODEL_RGBA, marker * 0.7)
    for name, idx in name_to_idx.items():
        grp = groups.get(name)
        if grp not in _GROUP_RGBA:
            continue
        p = aligned[idx]
        if not np.all(np.isfinite(p)):
            continue
        _add_sphere(scn, p, _GROUP_RGBA[grp], marker)
    return renderer.render()


def _label_strip(text: str, height: int, width: int = 190) -> Optional[np.ndarray]:
    """A left-margin strip with a readable row label, or None (draw nothing
    rather than crash) if PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    img = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    line_h = 14
    y0 = max(4, height // 2 - line_h * len(lines) // 2)
    for i, line in enumerate(lines):
        draw.text((8, y0 + i * line_h), line, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def _model_sites_and_groups(model_xml: str, kp_names: List[str]
                            ) -> Tuple["mujoco.MjModel", Dict[str, int], List[str], Dict[str, str]]:
    site_idx = _tracking_site_idx(model_xml)
    mj = site_idx["__mj__"]
    present = [n for n in kp_names if n in site_idx]
    groups = {n: kp_group(n) for n in present}
    return mj, site_idx, present, groups


def _posed_tracking_sites(mj, d, site_idx: Dict[str, int], present: List[str],
                          qpos_frame: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Model ``tracking[...]`` site positions with the model driven to
    ``qpos_frame`` (or its rest pose, ``mj.qpos0``, when ``qpos_frame`` is
    None). Mutates ``d`` in place -- also leaves it in the state the caller
    wants to render immediately afterward, so both the overlay reference and
    the rendered mesh are guaranteed to be the same pose."""
    d.qpos[:] = qpos_frame if qpos_frame is not None else mj.qpos0
    mujoco.mj_forward(mj, d)
    return {n: d.site_xpos[site_idx[n]].copy() for n in present}


def render_scale_check(kp3d: np.ndarray, kp_names: Sequence[str], model_xml: str,
                       scales: Sequence[float], frames: Sequence[int], out_png,
                       labels: Optional[Sequence[str]] = None,
                       size: Tuple[int, int] = (420, 420),
                       marker: Optional[float] = None,
                       qpos: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
    """Render a scale x frame grid PNG and return per-scale span ratios.

    Args:
        kp3d: (T, K, 3) raw triangulated keypoints (NOT scaled).
        kp_names: length-K keypoint names.
        model_xml: MuJoCo fly model XML path.
        scales: candidate body scales; one grid ROW each.
        frames: frame indices into ``kp3d``; one grid COLUMN each.
        out_png: output PNG path.
        labels: optional per-scale text labels (parallel to ``scales``).
        size: (width, height) of each rendered panel, before row labels.
        marker: sphere radius; default scales with the model extent.
        qpos: optional (T, nq) fitted joint configuration, same frame axis
            as ``kp3d``. When given, EACH rendered frame drives the model to
            ``qpos[frame]`` (``d.qpos[:] = qpos[frame]`` then
            ``mujoco.mj_forward``) before both reading the tracking-site
            reference positions and rendering the mesh, so leg/wing rows
            become a fair size comparison instead of conflating pose with
            size (see module docstring for the caveat this implies). When
            None (default), behaviour is unchanged: rest pose throughout.

    Returns:
        List (parallel to ``scales``) of ``{"trunk": r, "leg": r, "wing": r}``
        span ratios (observed / model, median over ``frames``). A group
        missing >= 2 finite markers in every rendered frame gets ``nan``.
        With ``qpos`` given, "model" here means the FITTED pose per frame,
        not rest -- trunk ratios should barely move (trunk is rigid relative
        to the thorax root); leg/wing ratios are the whole point of posed
        mode and should be read as pose-fair.
    """
    kp3d = np.asarray(kp3d, dtype=np.float64)
    kp_names = list(kp_names)
    T, K, _ = kp3d.shape
    if K != len(kp_names):
        raise ValueError(f"kp3d has K={K} but {len(kp_names)} kp_names given")

    mj, site_idx, present, groups = _model_sites_and_groups(model_xml, kp_names)
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    align_names = [n for n in present if groups.get(n) in _ALIGN_GROUPS]
    ratio_names = {g: [n for n in present if groups.get(n) == g] for g in _RATIO_GROUPS}

    if qpos is not None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape[0] != T:
            raise ValueError(f"qpos has {qpos.shape[0]} frames but kp3d has T={T}")
        if qpos.shape[1] != mj.nq:
            raise ValueError(f"qpos has nq={qpos.shape[1]} but model nq={mj.nq}")

    d = mujoco.MjData(mj)
    mujoco.mj_forward(mj, d)

    # cam.lookat is re-centered per FRAME below (on that frame's own posed
    # tracking-site mean) rather than fixed once here: a fitted qpos's free
    # joint can translate the whole animal far from the origin (measured on
    # real stac_ik.h5 qpos: root xyz ranging ~0.17-0.7 m against a model
    # extent of ~0.65 m), so a camera fixed at the REST-pose center/extent
    # would crop the animal out of frame entirely for a posed render.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj, cam)
    cam.azimuth, cam.elevation = 90.0, -25.0
    cam.distance = mj.stat.extent * 1.3

    width, height = size
    marker_r = marker if marker is not None else mj.stat.extent * 0.012

    ratios_by_scale: List[Dict[str, float]] = []
    grid_rows: List[np.ndarray] = []

    with mujoco.Renderer(mj, height=height, width=width) as renderer:
        for si, scale in enumerate(scales):
            group_vals: Dict[str, List[float]] = {g: [] for g in _RATIO_GROUPS}
            panel_imgs = []
            for frame in frames:
                if frame < 0 or frame >= T:
                    raise IndexError(f"frame {frame} out of range for kp3d with T={T}")
                qpos_frame = qpos[frame] if qpos is not None else None
                ref_by_name = _posed_tracking_sites(mj, d, site_idx, present, qpos_frame)
                cam.lookat[:] = np.mean(list(ref_by_name.values()), axis=0)
                raw = kp3d[frame]
                scaled = raw * scale
                for g in _RATIO_GROUPS:
                    r = _group_span_ratio(scaled, name_to_idx, ratio_names[g], ref_by_name)
                    if r is not None:
                        group_vals[g].append(r)
                aligned = _align_frame(scaled, name_to_idx, align_names, ref_by_name)
                panel_imgs.append(_render_panel(renderer, d, cam, aligned, name_to_idx,
                                                groups, ref_by_name, marker_r))
            ratios_by_scale.append({
                g: (float(np.median(vals)) if vals else float("nan"))
                for g, vals in group_vals.items()
            })
            row = np.concatenate(panel_imgs, axis=1)
            label_text = f"scale {scale:.6f}"
            if labels is not None and si < len(labels) and labels[si]:
                label_text += f"\n{labels[si]}"
            strip = _label_strip(label_text, row.shape[0])
            if strip is not None:
                row = np.concatenate([strip, row], axis=1)
            grid_rows.append(row)

    grid = np.concatenate(grid_rows, axis=0)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, grid)
    return ratios_by_scale


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_kp3d(fly_dir: Path) -> np.ndarray:
    for name in ("kp3d_filt.npz", "kp3d.npz"):
        p = fly_dir / name
        if p.exists():
            with np.load(p) as z:
                return np.asarray(z["kp3d"], dtype=np.float64)
    raise FileNotFoundError(f"no kp3d_filt.npz or kp3d.npz found in {fly_dir}")


def _resolve_qpos(bout_dir: Path, source: str) -> Tuple[Optional[np.ndarray], str]:
    """Load the (T, nq) fitted qpos named by ``--qpos-source``.

    Returns ``(qpos_or_None, description)`` -- ``description`` is a short,
    human-readable string ("rest", "stac", "refined", or "rest
    (<file> missing)") meant to be folded into each row's label so the
    picture says which pose it's showing. A missing file prints a warning
    and falls back to rest pose (``None``) rather than raising.
    """
    if source == "none":
        return None, "rest"
    if source == "stac":
        p = bout_dir / "stac_ik.h5"
        if not p.exists():
            print(f"WARNING: {p} not found; falling back to rest pose", file=sys.stderr)
            return None, "rest (stac_ik.h5 missing)"
        import h5py
        with h5py.File(p, "r") as f:
            return np.asarray(f["qpos"], dtype=np.float64), "stac"
    if source == "refined":
        p = bout_dir / "qpos_refined.npz"
        if not p.exists():
            print(f"WARNING: {p} not found; falling back to rest pose", file=sys.stderr)
            return None, "rest (qpos_refined.npz missing)"
        with np.load(p) as z:
            return np.asarray(z["qpos"], dtype=np.float64), "refined"
    raise ValueError(f"unknown --qpos-source {source!r}")


def _pick_default_frames(kp3d: np.ndarray, kp_names: List[str], model_xml: str,
                         n: int = 3) -> List[int]:
    """``n`` evenly spaced frame indices with full trunk+leg marker coverage,
    falling back to evenly spaced frames over the whole recording (with a
    warning) if no frame has full coverage."""
    site_idx = _tracking_site_idx(model_xml)
    name_to_idx = {name: i for i, name in enumerate(kp_names)}
    align_names = [name for name in kp_names
                   if name in site_idx and kp_group(name) in _ALIGN_GROUPS]
    idxs = [name_to_idx[name] for name in align_names]
    T = kp3d.shape[0]
    coverage = np.all(np.isfinite(kp3d[:, idxs, :]), axis=(1, 2)) if idxs else np.zeros(T, bool)
    valid = np.flatnonzero(coverage)
    if valid.size == 0:
        print("WARNING: no frame has full trunk+leg marker coverage; "
              "picking evenly spaced frames over the whole recording anyway",
              file=sys.stderr)
        valid = np.arange(T)
    if valid.size <= n:
        return sorted(valid.tolist())
    pick_pos = np.linspace(0, valid.size - 1, n).round().astype(int)
    return sorted(set(int(valid[p]) for p in pick_pos))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bout-dir", required=True, type=Path,
                    help="<run_root>/bouts/bout_XXXXX/flyN")
    ap.add_argument("--scales", required=True,
                    help="comma-separated candidate body scales")
    ap.add_argument("--labels", default=None,
                    help="comma-separated labels, parallel to --scales")
    ap.add_argument("--frames", default=None,
                    help="comma-separated frame indices; default: 3 evenly "
                         "spaced frames with full marker coverage")
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml")
    ap.add_argument("--qpos-source", choices=["none", "stac", "refined"], default="stac",
                    help="drive the model to a fitted pose per frame instead "
                         "of rest ('stac': stac_ik.h5, 'refined': "
                         "qpos_refined.npz, 'none': rest pose)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    cfg = _load_anatomy_cfg(args.anatomy)
    kp_names = list(cfg.model.KP_NAMES)
    model_xml = str(cfg.mjcf_path)

    kp3d = _load_kp3d(args.bout_dir)
    scales = [float(x) for x in args.scales.split(",")]
    labels = [x.strip() for x in args.labels.split(",")] if args.labels else None
    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = _pick_default_frames(kp3d, kp_names, model_xml)

    qpos, pose_desc = _resolve_qpos(args.bout_dir, args.qpos_source)
    if labels is None:
        labels = [f"pose={pose_desc}" for _ in scales]
    else:
        labels = [f"{lbl} (pose={pose_desc})" for lbl in labels]

    print(f"[scale_check] bout_dir={args.bout_dir} T={kp3d.shape[0]} frames={frames} "
          f"pose={pose_desc}")

    ratios = render_scale_check(kp3d, kp_names, model_xml, scales, frames, args.out,
                                labels=labels, qpos=qpos)

    header = f"{'label':<32}{'scale':>12}{'trunk':>10}{'leg':>10}{'wing':>10}"
    print(header)
    for i, s in enumerate(scales):
        lbl = labels[i] if labels else ""
        r = ratios[i]
        print(f"{lbl:<32}{s:>12.6f}{r['trunk']:>10.3f}{r['leg']:>10.3f}{r['wing']:>10.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
