"""Panel functions for the consolidated courtship figure (Figure 6).

Each ``panel_*`` function takes a target ``Axes`` (or list of axes) plus the
data it needs. Functions are self-contained so any panel can be re-rendered
into a different axes without rebuilding the full figure (the "swappable"
requirement). ``assemble_figure`` wires up the canonical layout and returns a
dict of axes that the caller hands to the panel functions.

Layout (183 mm x 130 mm, 4 rows):

    Row 1: 6 cropped video frames
    Row 2: 6 MuJoCo render frames
    Row 3: [wing-z trace stacked over scutellum-z trace] | [singing vs walking
           z-height boxplot]   widths 2:1
    Row 4: [pulse-bout classification] | [pulse vs sine totals]
           | [L/R wing dominance]      widths 1:1:1
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Style + constants
# -----------------------------------------------------------------------------

SONG_COLORS: Dict[str, str] = {
    'pulse':  '#d62728',
    'sine':   '#1f77b4',
    'waggle': '#9467bd',
    'quiet':  '#bdbdbd',
}

# Pale fills for axvspan shading.
_SEG_FILL: Dict[str, str] = {
    'pulse':  '#E76F5133',
    'sine':   '#2A9D8F33',
    'waggle': '#E9C46A33',
    'quiet':  '#cccccc11',
}
_SEG_EDGE: Dict[str, str] = {
    'pulse':  '#E76F51',
    'sine':   '#2A9D8F',
    'waggle': '#E9C46A',
    'quiet':  '#999999',
}

WING_COLORS: Dict[str, str] = {
    'WingL_V13': '#A855F7',  # purple
    'WingR_V13': '#38BDF8',  # blue
}

PULSE_TYPE_COLORS: Dict[str, str] = {
    'Pslow': '#fb8c00',  # orange
    'Pfast': '#1565c0',  # deep blue
}

DOMINANCE_COLORS: Dict[str, str] = {
    'L': '#A855F7',
    'R': '#38BDF8',
}


def apply_paper_style(min_font_pt: float = 6.0) -> None:
    """Set rcParams to paper-figure defaults with all fonts >= min_font_pt."""
    base = max(7.0, min_font_pt)
    tick = max(6.0, min_font_pt)
    mpl.rcParams.update({
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
        'font.family':     'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size':       base,
        'axes.labelsize':  base,
        'axes.titlesize':  base,
        'xtick.labelsize': tick,
        'ytick.labelsize': tick,
        'legend.fontsize': tick,
        'axes.spines.right': False,
        'axes.spines.top':   False,
        'axes.linewidth':  0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'figure.dpi':     150,
        'savefig.dpi':    300,
        'savefig.bbox':   'standard',
    })


# -----------------------------------------------------------------------------
# Layout assembly
# -----------------------------------------------------------------------------

def assemble_figure(
    fig_width_mm: float = 183.0,
    fig_height_mm: float = 130.0,
    n_frames_strip: int = 6,
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Build the 4-row consolidated layout. Returns (fig, axes_dict).

    Layout (top to bottom):
        Row 1 [labeled top-down frame] | [wing V13 z + scutellum z stacked]
        Row 2  6 video frames with all keypoints overlayed
        Row 3 [sine in-phase] | [joint angle density] | [Pslow/Pfast wave]
              | [scutellum z courtship vs free walking]
        Row 4  6 MuJoCo render frames

    axes_dict keys:
        'kp_label'    : axes (Row 1 left)
        'wing'        : axes (Row 1 right top)
        'scut'        : axes (Row 1 right bottom, sharex with wing)
        'video'       : list of N axes (Row 2)
        'sine_phase'  : axes (Row 3 col 0)
        'angle_2d'    : axes (Row 3 col 1)
        'pulse_class' : axes (Row 3 col 2)
        'zheight'     : axes (Row 3 col 3)
        'render'      : list of N axes (Row 4)
    """
    apply_paper_style()
    fig = plt.figure(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    sub_row1, sub_video, sub_row3, sub_render = fig.subfigures(
        4, 1, height_ratios=[1.4, 1.0, 1.6, 1.0], hspace=0.05,
    )

    # Row 1: labeled keypoint frame (left) + wing/scut traces (right)
    sf1_left, sf1_right = sub_row1.subfigures(
        1, 2, width_ratios=[1, 5], wspace=0.05,
    )
    ax_kp_label = sf1_left.subplots(1, 1)
    ax_kp_label.set_xticks([]); ax_kp_label.set_yticks([])
    for spine in ax_kp_label.spines.values():
        spine.set_visible(False)
    sf1_left.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)

    ax_wing, ax_scut = sf1_right.subplots(
        2, 1, sharex=True, height_ratios=[2, 1],
    )
    sf1_right.subplots_adjust(
        left=0.06, right=0.99, top=0.94, bottom=0.18, hspace=0.10,
    )

    # Row 2: (n_frames_strip - 1) video frames + exemplar sine in-phase trace
    sf2_left, sf2_right = sub_video.subfigures(
        1, 2, width_ratios=[5.0, 1.4], wspace=0.04,
    )
    n_video = max(1, int(n_frames_strip) - 1)
    ax_video = list(sf2_left.subplots(1, n_video))
    if n_video == 1:
        ax_video = [ax_video[0]] if not isinstance(ax_video, list) else ax_video
    for ax in ax_video:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sf2_left.subplots_adjust(
        wspace=0.04, left=0.03, right=0.99, top=0.92, bottom=0.04,
    )
    ax_sine_phase = sf2_right.subplots(1, 1)
    sf2_right.subplots_adjust(
        left=0.16, right=0.97, top=0.92, bottom=0.22,
    )

    # Row 3: 4 mechanism panels (col 0 = polar, cols 1-3 = cartesian)
    sf3_polar, sf3_rest = sub_row3.subfigures(
        1, 2, width_ratios=[1.0, 3.0], wspace=0.05,
    )
    ax_wing_phase_polar = sf3_polar.subplots(
        1, 1, subplot_kw={'projection': 'polar'},
    )
    sf3_polar.subplots_adjust(
        left=0.18, right=0.88, top=0.82, bottom=0.22,
    )
    ax_angle_2d, ax_pulse, ax_zheight = sf3_rest.subplots(1, 3)
    sf3_rest.subplots_adjust(
        left=0.05, right=0.99, top=0.92, bottom=0.20, wspace=0.45,
    )

    # Row 4: render strip
    ax_render = list(sub_render.subplots(1, n_frames_strip))
    for ax in ax_render:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sub_render.subplots_adjust(
        wspace=0.04, left=0.03, right=0.99, top=0.96, bottom=0.04,
    )

    return fig, {
        'kp_label':         ax_kp_label,
        'wing':             ax_wing,
        'scut':             ax_scut,
        'video':            ax_video,
        'sine_phase':       ax_sine_phase,
        'wing_phase_polar': ax_wing_phase_polar,
        'angle_2d':         ax_angle_2d,
        'pulse_class':      ax_pulse,
        'zheight':          ax_zheight,
        'render':           ax_render,
    }


# -----------------------------------------------------------------------------
# Row 1: video frame strip
# -----------------------------------------------------------------------------

def _dlt_load(csv_path: str | Path) -> np.ndarray:
    """Load 11 DLT coefficients (one per line) from a ``*_dlt.csv`` file."""
    coeffs = np.loadtxt(str(csv_path)).astype(float).reshape(-1)
    if coeffs.size != 11:
        raise ValueError(
            f'expected 11 DLT coefficients in {csv_path}, got {coeffs.size}'
        )
    return coeffs


def _dlt_project(coeffs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project 3D world points to 2D pixel coords with the standard 11-param DLT.

    ``xyz`` is broadcast over leading axes; trailing dim must be 3. Returns an
    array with the same leading shape and trailing dim 2 (u, v in pixels).
    """
    L = np.asarray(coeffs, dtype=float).reshape(11)
    pts = np.asarray(xyz, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError(f'xyz last dim must be 3; got {pts.shape}')
    X = pts[..., 0]
    Y = pts[..., 1]
    Z = pts[..., 2]
    denom = L[8] * X + L[9] * Y + L[10] * Z + 1.0
    u = (L[0] * X + L[1] * Y + L[2] * Z + L[3]) / denom
    v = (L[4] * X + L[5] * Y + L[6] * Z + L[7]) / denom
    return np.stack([u, v], axis=-1)


def _open_video(mp4_path: str | Path):
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open video: {mp4_path}')
    return cap


def _read_frame(cap, fidx: int, roi: Optional[Tuple[int, int, int, int]] = None):
    """Seek and read frame ``fidx``; return cropped RGB ndarray or ``None``."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
    return frame


def _center_to_roi(
    u: float, v: float, w: int, h: int, frame_w: int, frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Build a (w, h) crop centered on (u, v) and clamp it inside the frame.

    Returns ``(x, y, w, h)`` with the same ``(w, h)``; ``x, y`` are shifted to
    keep the crop fully inside ``[0, frame_w) × [0, frame_h)`` when ``(u, v)``
    is near an edge. Returns ``None`` if ``(u, v)`` is non-finite.
    """
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    w = int(w); h = int(h)
    x = int(round(float(u) - w / 2.0))
    y = int(round(float(v) - h / 2.0))
    x = max(0, min(x, int(frame_w) - w))
    y = max(0, min(y, int(frame_h) - h))
    return (x, y, w, h)


def _center_xyz_to_roi(
    center_xyz: np.ndarray,
    dlt_coeffs: np.ndarray,
    crop_wh: Tuple[int, int],
    frame_w: int,
    frame_h: int,
    kp_scale: float = 1.0,
) -> Optional[Tuple[int, int, int, int]]:
    """Project a single 3D point through DLT and build a centered crop ROI."""
    pt = np.asarray(center_xyz, dtype=float) * float(kp_scale)
    uv = _dlt_project(dlt_coeffs, pt)
    return _center_to_roi(uv[0], uv[1], crop_wh[0], crop_wh[1], frame_w, frame_h)


def panel_video_strip(
    axes: Sequence[plt.Axes],
    mp4_path: str | Path,
    frame_indices: Sequence[int],
    roi: Optional[Tuple[int, int, int, int]] = None,
    titles: Optional[Sequence[str]] = None,
    fs: float = 800.0,
    video_frame_offset: int = 0,
    center_xyz: Optional[np.ndarray] = None,
    crop_wh: Optional[Tuple[int, int]] = None,
    dlt_coeffs: Optional[np.ndarray] = None,
    kp_scale: float = 1.0,
) -> None:
    """Show a horizontal sequence of cropped video frames.

    Parameters
    ----------
    axes : list of N axes
    mp4_path : path to a Cam<serial>.mp4 file
    frame_indices : N integer frame indices (bout-relative when
        ``video_frame_offset`` is used; otherwise session-absolute)
    roi : (x, y, w, h) crop in pixels (origin top-left); None = no crop
    titles : N strings to label each frame; default = "t = ... ms"
    fs : sample rate (Hz) used only to label time-titles when titles is None
    video_frame_offset : added to each ``fidx`` before seeking the mp4
    center_xyz, crop_wh, dlt_coeffs, kp_scale : optional per-frame ROI. When
        ``center_xyz`` (T, 3), ``crop_wh=(w, h)``, and ``dlt_coeffs`` are all
        given, a dynamic ROI centered on ``center_xyz[fidx]`` is computed and
        used in place of ``roi``.
    """
    import cv2
    if titles is None:
        titles = [f't = {int(round(f / fs * 1000))} ms' for f in frame_indices]
    if len(titles) != len(axes) or len(frame_indices) != len(axes):
        raise ValueError('axes, frame_indices, titles must all be the same length')

    dynamic_roi = (
        center_xyz is not None and crop_wh is not None and dlt_coeffs is not None
    )
    center_arr = np.asarray(center_xyz) if dynamic_roi else None

    cap = _open_video(mp4_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        for ax, fidx, title in zip(axes, frame_indices, titles):
            if dynamic_roi:
                roi_t = _center_xyz_to_roi(
                    center_arr[int(fidx)], dlt_coeffs, crop_wh,
                    frame_w, frame_h, kp_scale=kp_scale,
                )
                if roi_t is None:
                    roi_t = roi
            else:
                roi_t = roi
            frame = _read_frame(cap, int(fidx) + int(video_frame_offset), roi_t)
            if frame is None:
                ax.text(0.5, 0.5, f'frame\n{fidx}\nN/A',
                        ha='center', va='center', transform=ax.transAxes)
                continue
            ax.imshow(frame)
            ax.set_title(title, pad=2)
    finally:
        cap.release()


def panel_kp_label_frame(
    ax: plt.Axes,
    mp4_path: str | Path,
    frame_index: int,
    kp_xyz: np.ndarray,
    kp_names: Sequence[str],
    dlt_coeffs: np.ndarray,
    label_kps: Sequence[str] = ('Scutellum', 'WingL_V13', 'WingR_V13'),
    roi: Optional[Tuple[int, int, int, int]] = None,
    label_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
    kp_color: str = '#ffd400',
    kp_scale: float = 1.0,
    video_frame_offset: int = 0,
    text_kwargs: Optional[Dict] = None,
    scatter_kwargs: Optional[Dict] = None,
    center_xyz: Optional[np.ndarray] = None,
    crop_wh: Optional[Tuple[int, int]] = None,
) -> None:
    """Single video frame with selected keypoints drawn + labeled.

    Projects ``kp_xyz[frame_index, idx, :]`` to pixels via DLT, applies the ROI
    offset, and draws a dot + text for each entry of ``label_kps`` (defaults
    to Scutellum + the two V13 wing tips).

    When ``center_xyz`` (shape ``(3,)`` or ``(T, 3)``) and ``crop_wh=(w, h)``
    are both given, the ROI is computed dynamically by projecting the center
    through DLT and centering a ``(w, h)`` crop on the resulting pixel,
    clamped to the raw video bounds.
    """
    import cv2
    sk = {'s': 14, 'edgecolors': 'k', 'linewidths': 0.4,
          'zorder': 3, **(scatter_kwargs or {})}
    tk = {'fontsize': 6, 'color': kp_color,
          'path_effects': [], **(text_kwargs or {})}
    label_offsets = label_offsets or {}

    cap = _open_video(mp4_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_t = roi
    if center_xyz is not None and crop_wh is not None:
        c_arr = np.asarray(center_xyz)
        c_pt = c_arr[int(frame_index)] if c_arr.ndim == 2 else c_arr
        roi_dyn = _center_xyz_to_roi(
            c_pt, dlt_coeffs, crop_wh, frame_w, frame_h, kp_scale=kp_scale,
        )
        if roi_dyn is not None:
            roi_t = roi_dyn
    try:
        frame = _read_frame(cap, int(frame_index) + int(video_frame_offset), roi_t)
    finally:
        cap.release()
    if frame is None:
        ax.text(0.5, 0.5, f'frame {frame_index} N/A',
                ha='center', va='center', transform=ax.transAxes)
        return
    ax.imshow(frame)

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    kp_arr = np.asarray(kp_xyz)
    if kp_arr.ndim == 2 and kp_arr.shape[-1] != 3:
        kp_arr = kp_arr.reshape(kp_arr.shape[0], -1, 3)
    pts3 = kp_arr[int(frame_index)] * float(kp_scale)
    uv = _dlt_project(dlt_coeffs, pts3)
    if roi_t is not None:
        uv = uv - np.array([roi_t[0], roi_t[1]], dtype=float)

    xs, ys = [], []
    for name in label_kps:
        if name not in name_to_idx:
            continue
        u, v = uv[name_to_idx[name]]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        xs.append(u); ys.append(v)
        dx, dy = label_offsets.get(name, (8, -4))
        ax.annotate(name, xy=(u, v), xytext=(u + dx, v + dy), **tk)
    if xs:
        ax.scatter(xs, ys, c=kp_color, **sk)


def panel_video_strip_with_kp(
    axes: Sequence[plt.Axes],
    mp4_path: str | Path,
    frame_indices: Sequence[int],
    kp_xyz_per_frame: np.ndarray,
    kp_names: Sequence[str],
    dlt_coeffs: np.ndarray,
    overlay_kps: Optional[Sequence[str]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    titles: Optional[Sequence[str]] = None,
    fs: float = 800.0,
    kp_color: str = '#ffd400',
    kp_scale: float = 1.0,
    video_frame_offset: int = 0,
    scatter_kwargs: Optional[Dict] = None,
    center_xyz: Optional[np.ndarray] = None,
    crop_wh: Optional[Tuple[int, int]] = None,
) -> None:
    """Same as :func:`panel_video_strip` but draws projected keypoint dots
    on every frame.

    ``frame_indices`` are bout-relative — used as-is to index
    ``kp_xyz_per_frame``. The video seek uses ``video_frame_offset + fidx`` so
    that callers can point ``mp4_path`` at the full session video while
    keypoints / segments stay in bout-local coordinates. Set
    ``video_frame_offset`` to the bout's session-absolute start frame.

    ``overlay_kps`` defaults to all entries of ``kp_names``. Points whose
    projected coordinates fall outside the cropped frame are silently clipped.

    When ``center_xyz`` (shape ``(T, 3)`` in the same world frame as
    ``kp_xyz_per_frame``) and ``crop_wh=(w, h)`` are both given, a per-frame
    ROI centered on ``center_xyz[fidx]`` is computed via DLT and used instead
    of ``roi``. The ROI is clamped to the raw video bounds.
    """
    import cv2
    sk = {'s': 4, 'edgecolors': 'none', 'linewidths': 0,
          'zorder': 3, **(scatter_kwargs or {})}
    if titles is None:
        titles = [f't = {int(round(f / fs * 1000))} ms' for f in frame_indices]
    if len(titles) != len(axes) or len(frame_indices) != len(axes):
        raise ValueError('axes, frame_indices, titles must all be the same length')

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    if overlay_kps is None:
        kp_idx = np.arange(len(kp_names))
    else:
        kp_idx = np.array([name_to_idx[n] for n in overlay_kps if n in name_to_idx],
                          dtype=int)

    kp_arr = np.asarray(kp_xyz_per_frame)
    if kp_arr.ndim == 2 and kp_arr.shape[-1] != 3:
        kp_arr = kp_arr.reshape(kp_arr.shape[0], -1, 3)

    dynamic_roi = center_xyz is not None and crop_wh is not None
    center_arr = np.asarray(center_xyz) if dynamic_roi else None

    cap = _open_video(mp4_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        for ax, fidx, title in zip(axes, frame_indices, titles):
            if dynamic_roi:
                roi_t = _center_xyz_to_roi(
                    center_arr[int(fidx)], dlt_coeffs, crop_wh,
                    frame_w, frame_h, kp_scale=kp_scale,
                )
                if roi_t is None:
                    roi_t = roi
            else:
                roi_t = roi
            frame = _read_frame(cap, int(fidx) + int(video_frame_offset), roi_t)
            if frame is None:
                ax.text(0.5, 0.5, f'frame\n{fidx}\nN/A',
                        ha='center', va='center', transform=ax.transAxes)
                continue
            ax.imshow(frame)
            ax.set_title(title, pad=2)
            pts3 = kp_arr[int(fidx)][kp_idx] * float(kp_scale)
            uv = _dlt_project(dlt_coeffs, pts3)
            if roi_t is not None:
                uv = uv - np.array([roi_t[0], roi_t[1]], dtype=float)
            H, W = frame.shape[:2]
            m = (np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
                 & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < H))
            if m.any():
                ax.scatter(uv[m, 0], uv[m, 1], c=kp_color, **sk)
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# Row 2: MuJoCo render strip
# -----------------------------------------------------------------------------

def panel_render_strip(
    axes: Sequence[plt.Axes],
    qpos_array: np.ndarray,
    frame_indices: Sequence[int],
    viz=None,
    mj_model=None,
    camera: str = 'track1',
    height_px: int = 256,
    width_px: int = 256,
    crop: Optional[Tuple[int, int, int, int]] = None,
    titles: Optional[Sequence[str]] = None,
) -> None:
    """Render N frames from a qpos array into the given axes.

    Prefers ``viz`` (a ``FlyVisualizer`` instance from
    ``fly_neuromech.visualizer.visualizer``) when provided — this picks up
    all custom render settings (lighting, skybox, body coloring, camera
    presets) from whatever settings JSON the caller loaded. Falls back to
    a vanilla ``mujoco.Renderer(mj_model)`` when ``viz`` is None.

    Parameters
    ----------
    axes : list of N axes
    qpos_array : (T, nq) joint angles for the bout
    frame_indices : N integer indices into ``qpos_array``
    viz : FlyVisualizer instance (preferred); if given, ``mj_model`` and
        ``camera`` are read from ``viz`` unless the caller passes
        ``camera`` explicitly.
    mj_model : mujoco.MjModel — used only when ``viz`` is None.
    camera : named camera (``'track1'``, ``'hero'``, ...)
    height_px, width_px : render resolution in pixels (pre-crop)
    crop : (x, y, w, h) in render-pixel coordinates applied after render;
        lets you zoom into the fly without re-rendering.
    titles : optional per-frame titles
    """
    if titles is not None and len(titles) != len(axes):
        raise ValueError('titles must match axes length when provided')
    if viz is None and mj_model is None:
        raise ValueError('pass either `viz` or `mj_model`')

    def _crop(img):
        if crop is None:
            return img
        x, y, w, h = crop
        return img[y:y + h, x:x + w]

    if viz is not None:
        for i, (ax, fidx) in enumerate(zip(axes, frame_indices)):
            pixels = viz.render_frame(
                qpos_array[int(fidx)],
                camera=camera, height=height_px, width=width_px,
            )
            ax.imshow(_crop(pixels))
            if titles is not None:
                ax.set_title(titles[i], pad=2)
        return

    import mujoco
    mj_data = mujoco.MjData(mj_model)
    with mujoco.Renderer(mj_model, height=height_px, width=width_px) as renderer:
        for i, (ax, fidx) in enumerate(zip(axes, frame_indices)):
            mj_data.qpos[:] = qpos_array[int(fidx)]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            pixels = renderer.render()
            ax.imshow(_crop(pixels))
            if titles is not None:
                ax.set_title(titles[i], pad=2)


# -----------------------------------------------------------------------------
# Two-fly courtship scene (male + female + floor) for row 2 renders
# -----------------------------------------------------------------------------

def _courtship_pair_anatomy(cameras: Optional[Sequence[str]] = None):
    """AnatomyConfig with every fly category duplicated per ``_fly0``/``_fly1``.

    Categories use ``body_substring=[name, flyN], all=true`` so each rule
    matches only the geoms on its intended fly even though both flies share
    the same unsuffixed body names before MjSpec ``attach_body`` renames
    them with the per-fly suffix.
    """
    from mujoco_visualizer.config import AnatomyConfig, CategoryRule

    pairs = [
        # (category_name_root, body_substrings_before_suffix, all)
        ('thorax',        ['thorax'],        False),
        ('head',          ['head'],          False),
        ('abdomen',       ['abdomen'],       False),
        ('antenna_left',  ['antenna', 'left'],  True),
        ('antenna_right', ['antenna', 'right'], True),
        ('haltere_left',  ['haltere', 'left'],  True),
        ('haltere_right', ['haltere', 'right'], True),
        ('proboscis',     ['proboscis', 'rostrum', 'labrum', 'labell',
                           'haustellum'], False),
        ('wing_left',     ['wing_left'],  False),
        ('wing_right',    ['wing_right'], False),
        ('T1_left',       ['t1', 'left'],  True),
        ('T1_right',      ['t1', 'right'], True),
        ('T2_left',       ['t2', 'left'],  True),
        ('T2_right',      ['t2', 'right'], True),
        ('T3_left',       ['t3', 'left'],  True),
        ('T3_right',      ['t3', 'right'], True),
    ]

    # eye_red must precede `head` so the `head_red` eye geom is bucketed there.
    # Use geom_substring for the fly-tag constraint (geom names carry the
    # ``_fly0`` / ``_fly1`` suffix too), so each rule ANDs the body match
    # with the geom-name tag match. This avoids the edge case where a rule
    # with ``all=False`` over multiple body substrings would match on the
    # tag alone and sweep in unrelated bodies (e.g. proboscis rule grabbing
    # wing geoms because both contain "fly0").
    rules: List[CategoryRule] = []
    for tag in ('fly0', 'fly1'):
        rules.append(CategoryRule(
            name=f'eye_red_{tag}',
            geom_substring=['red', tag], all=True,
        ))
        for cat, subs, require_all in pairs:
            rules.append(CategoryRule(
                name=f'{cat}_{tag}',
                body_substring=list(subs),
                geom_substring=[tag],
                all=require_all,
            ))

    return AnatomyConfig(
        cameras=list(cameras or []),
        categories=rules,
    )


def _remap_settings_keys(settings: dict, suffix: str) -> dict:
    """Return a copy of ``settings`` with color keys suffixed (``_fly0`` etc).

    Only the ``colors`` block is remapped; flags / camera / lighting / floor /
    skybox apply globally and are left untouched.
    """
    import copy as _copy
    out = _copy.deepcopy(settings)
    if isinstance(out.get('colors'), dict):
        out['colors'] = {f'{k}{suffix}': v for k, v in out['colors'].items()}
    return out


def build_courtship_pair_visualizer(
    flybody_xml: str,
    floor_xml: str,
    settings_fly0: Optional[str] = None,
    settings_fly1: Optional[str] = None,
    root_body: str = 'thorax',
    spawn_pos=(0.0, 0.0, -0.005),
):
    """Compose a floor + two fly bodies (``_fly0`` male, ``_fly1`` female).

    Returns a :class:`mujoco_visualizer.Visualizer` whose ``self.model.nq``
    equals ``fly_nq * 2`` and whose category buckets are per-fly, so the two
    fly-specific settings JSONs color each fly independently.

    Parameters
    ----------
    flybody_xml : str
        Path to the single-fly MuJoCo XML (e.g. ``fruitfly_v1_free.xml``).
    floor_xml : str
        Path to a worldbody XML containing a ``geom name="floor"`` plane.
    settings_fly0, settings_fly1 : str or None
        Settings file names (preset names or absolute paths) applied to each
        fly's geoms. Uses the mujoco_visualizer settings resolver, so bare
        names like ``'Earthy_V1_courtship_fly0'`` work. ``None`` skips.
    root_body : str
        Body to attach from the fly spec (default ``'thorax'``).
    spawn_pos : (x, y, z)
        Spawn offset for both flies (applied via the floor frame).
    """
    import json
    import mujoco
    from mujoco_visualizer import Visualizer
    from mujoco_visualizer.render_settings import _resolve_settings_path

    fly_spec_0 = mujoco.MjSpec.from_file(flybody_xml)
    fly_spec_1 = mujoco.MjSpec.from_file(flybody_xml)
    floor_spec = mujoco.MjSpec.from_file(floor_xml)
    spawn_frame = floor_spec.worldbody.add_frame(
        pos=list(spawn_pos), quat=[1, 0, 0, 0],
    )
    spawn_frame.attach_body(fly_spec_0.body(root_body), '', '_fly0')
    spawn_frame.attach_body(fly_spec_1.body(root_body), '', '_fly1')

    anatomy = _courtship_pair_anatomy()
    viz = Visualizer(spec=floor_spec, anatomy=anatomy)

    def _load(name: str, suffix: str) -> None:
        try:
            path = _resolve_settings_path(name)
        except FileNotFoundError:
            path = Path(name)
        with open(path) as f:
            raw = json.load(f)
        viz.load_settings(_remap_settings_keys(raw, suffix))

    if settings_fly0 is not None:
        _load(settings_fly0, '_fly0')
    if settings_fly1 is not None:
        _load(settings_fly1, '_fly1')
    return viz


# -----------------------------------------------------------------------------
# Row 3 left top: wing V13 z traces (L + R) with song shading
# -----------------------------------------------------------------------------

def _hex_to_fill(hex_rgb: str, alpha: float = 0.20) -> str:
    """Append an alpha byte to a ``#rrggbb`` hex color; pass through ``#rrggbbaa``."""
    s = str(hex_rgb)
    if s.startswith('#') and len(s) == 7:
        return f'{s}{int(round(alpha * 255)):02x}'
    return s


def _colored_text_legend(
    ax: plt.Axes,
    handles_labels: Optional[Tuple[list, list]] = None,
    **legend_kwargs,
):
    """Draw a legend whose text entries are colored to match their series
    (no line/box marker). Replaces the standard handle+label legend.

    ``handles_labels`` defaults to ``ax.get_legend_handles_labels()``. Any
    extra ``legend_kwargs`` are forwarded to ``ax.legend`` and override the
    marker-hiding defaults if the caller wants to tweak placement/spacing.
    """
    if handles_labels is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = handles_labels
    if not handles:
        return None

    defaults = {
        'frameon': False,
        'handlelength': 0,
        'handletextpad': 0,
        'borderpad': 0.2,
        'labelspacing': 0.25,
    }
    defaults.update(legend_kwargs)
    leg = ax.legend(handles, labels, **defaults)

    colors = []
    for h in handles:
        c = None
        for getter in ('get_color', 'get_facecolor', 'get_edgecolor'):
            if hasattr(h, getter):
                try:
                    c = getattr(h, getter)()
                except Exception:
                    c = None
                if c is not None:
                    break
        colors.append(c if c is not None else 'k')

    for txt, color in zip(leg.get_texts(), colors):
        txt.set_color(color)
    for h in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
        try:
            h.set_visible(False)
        except Exception:
            pass
    return leg


def _merge_segments_by_type(
    segs_list: Sequence[Iterable[dict]],
) -> List[dict]:
    """Union overlapping/adjacent same-type intervals across multiple segment
    lists (e.g. L + R sides) so downstream shading paints each region once
    and there's no double-shading overlap where sides agree.
    """
    by_type: Dict[str, List[Tuple[int, int]]] = {}
    for segs in segs_list:
        for s in segs or []:
            by_type.setdefault(s['type'], []).append(
                (int(s['start']), int(s['end']))
            )
    merged: List[dict] = []
    for t, ivs in by_type.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append({'type': t, 'start': cur_s, 'end': cur_e})
                cur_s, cur_e = s, e
        merged.append({'type': t, 'start': cur_s, 'end': cur_e})
    merged.sort(key=lambda d: d['start'])
    return merged


def _resolve_pulse_subtype(
    seg: dict,
    peak_frames: Optional[np.ndarray],
    subtype_labels: Optional[np.ndarray],
) -> Optional[str]:
    """Majority Pslow/Pfast label among pulses whose peaks fall in ``seg``.

    Returns ``None`` if the segment isn't a pulse segment, or the pulse-type
    inputs are missing / have no peaks inside the segment.
    """
    if seg.get('type') != 'pulse':
        return None
    if peak_frames is None or subtype_labels is None:
        return None
    pf = np.asarray(peak_frames)
    sl = np.asarray(subtype_labels)
    if pf.size == 0 or sl.size == 0 or pf.size != sl.size:
        return None
    s, e = int(seg['start']), int(seg['end'])
    m = (pf >= s) & (pf <= e)
    if not m.any():
        return None
    inside = sl[m]
    n_slow = int((inside == 'Pslow').sum())
    n_fast = int((inside == 'Pfast').sum())
    if n_slow == 0 and n_fast == 0:
        return None
    return 'Pslow' if n_slow >= n_fast else 'Pfast'


def _shade_segments(
    ax: plt.Axes,
    segments: Iterable[dict],
    fs: float,
    frame_range: Optional[Tuple[int, int]] = None,
    seen_labels: Optional[set] = None,
    skip: Tuple[str, ...] = ('quiet',),
    fills: Optional[Dict[str, str]] = None,
    edges: Optional[Dict[str, str]] = None,
    pulse_peak_frames: Optional[np.ndarray] = None,
    pulse_subtype_labels: Optional[np.ndarray] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    fill_alpha: float = 0.20,
) -> None:
    """Paint axvspans for each non-quiet song segment.

    When ``pulse_peak_frames`` + ``pulse_subtype_labels`` are supplied, pulse
    segments are colored per the dominant pulse sub-type (Pslow/Pfast). Caller
    may override fills / edges via ``fills`` / ``edges`` dicts (shallow-merged
    over module defaults), and pulse-subtype colors via ``pulse_type_colors``.
    """
    if seen_labels is None:
        seen_labels = set()
    _fills = {**_SEG_FILL, **(fills or {})}
    _edges = {**_SEG_EDGE, **(edges or {})}
    _pt = {**PULSE_TYPE_COLORS, **(pulse_type_colors or {})}

    for seg in segments:
        stype = seg['type']
        if stype in skip:
            continue
        s, e = int(seg['start']), int(seg['end'])
        if frame_range is not None:
            lo, hi = frame_range
            if e <= lo or s >= hi:
                continue
            s, e = max(s, lo), min(e, hi)
        s_ms = s / fs * 1000.0
        e_ms = e / fs * 1000.0

        key = stype
        fill = _fills.get(stype, '#cccccc33')
        edge = _edges.get(stype)
        sub = _resolve_pulse_subtype(seg, pulse_peak_frames, pulse_subtype_labels)
        if sub is not None:
            key = sub
            fill = _hex_to_fill(_pt[sub], fill_alpha)
            edge = _pt[sub]

        label = key.capitalize() if key not in seen_labels else None
        seen_labels.add(key)
        ax.axvspan(s_ms, e_ms, facecolor=fill, edgecolor=edge,
                   linewidth=0, zorder=0, label=label)


def panel_wing_z_traces(
    ax: plt.Axes,
    t_ms: np.ndarray,
    wingL_z: np.ndarray,
    wingR_z: np.ndarray,
    segments_L: Iterable[dict],
    segments_R: Iterable[dict],
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    pulse_type_side: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    wing_colors: Optional[Dict[str, str]] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    pulse_vline_kwargs: Optional[Dict] = None,
) -> None:
    """Plot left/right wing-V13 z-position with song shading + per-pulse markers.

    Segment shading is drawn once over the union of L+R segments (so L/R
    overlap does not double-shade). Each individual pulse event is marked
    with a vertical line colored by its own Pslow / Pfast label, so every
    pulse is individually identifiable.

    Parameters
    ----------
    pulse_type_side : optional {'L': {'peak_frames': ndarray, 'labels': ndarray},
        'R': {...}}. When provided, vertical lines are drawn at each
        ``peak_frame`` colored per the corresponding Pslow/Pfast label.
    wing_colors, pulse_type_colors : shallow-merge overrides for module defaults.
    line_kwargs : extra kwargs passed to ``ax.plot`` for both wing traces.
    pulse_vline_kwargs : extra kwargs passed to ``ax.axvline`` for pulse markers.
    legend_kwargs : extra kwargs forwarded to ``ax.legend``.
    """
    wc = {**WING_COLORS, **(wing_colors or {})}
    pt = {**PULSE_TYPE_COLORS, **(pulse_type_colors or {})}
    lk = {'lw': 0.7, **(line_kwargs or {})}
    vk = {'lw': 0.5, 'alpha': 0.8, 'zorder': 1, **(pulse_vline_kwargs or {})}
    lg = {'loc': 'upper right', 'ncols': 2,
          'columnspacing': 0.6, **(legend_kwargs or {})}

    seen: set = set()
    merged = _merge_segments_by_type([segments_L, segments_R])
    _shade_segments(ax, merged, fs, frame_range=frame_range, seen_labels=seen)

    peaks_all: List[np.ndarray] = []
    labs_all: List[np.ndarray] = []
    if pulse_type_side:
        for side in ('L', 'R'):
            d = pulse_type_side.get(side) or {}
            pf = np.asarray(d.get('peak_frames', []))
            lb = np.asarray(d.get('labels', []))
            if pf.size and lb.size == pf.size:
                peaks_all.append(pf)
                labs_all.append(lb)
    if peaks_all:
        pf_all = np.concatenate(peaks_all)
        lb_all = np.concatenate(labs_all).astype(str)
        order = np.argsort(pf_all)
        pf_all, lb_all = pf_all[order], lb_all[order]
        if pf_all.size >= 2:
            keep = np.concatenate(([True], np.diff(pf_all) > 1))
            pf_all, lb_all = pf_all[keep], lb_all[keep]
        if frame_range is not None:
            lo, hi = frame_range
            m = (pf_all >= lo) & (pf_all < hi)
            pf_all, lb_all = pf_all[m], lb_all[m]
        for f, lab in zip(pf_all, lb_all):
            tm = float(f) / fs * 1000.0
            color = pt.get(lab, 'k')
            label = lab if lab not in seen else None
            seen.add(lab)
            ax.axvline(tm, color=color, label=label, **vk)

    ax.plot(t_ms, wingL_z, color=wc['WingL_V13'], label='Wing L V13', **lk)
    ax.plot(t_ms, wingR_z, color=wc['WingR_V13'], label='Wing R V13', **lk)
    ax.set_ylabel('Wing V13\nz (mm)')
    _colored_text_legend(ax, **lg)


# -----------------------------------------------------------------------------
# Row 3 left bottom: scutellum z trace
# -----------------------------------------------------------------------------

def panel_scutellum_z_trace(
    ax: plt.Axes,
    t_ms: np.ndarray,
    scutellum_z: np.ndarray,
    segments: Optional[Iterable[dict]] = None,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    pulse_peak_frames: Optional[np.ndarray] = None,
    pulse_subtype_labels: Optional[np.ndarray] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    line_color: str = 'k',
    line_kwargs: Optional[Dict] = None,
) -> None:
    """Plot scutellum (body) z-position over the same time interval.

    When ``pulse_peak_frames`` + ``pulse_subtype_labels`` are provided, pulse
    segments are shaded by dominant Pslow/Pfast type.
    """
    lk = {'lw': 0.7, **(line_kwargs or {})}
    if segments is not None:
        _shade_segments(ax, segments, fs, frame_range=frame_range,
                        pulse_peak_frames=pulse_peak_frames,
                        pulse_subtype_labels=pulse_subtype_labels,
                        pulse_type_colors=pulse_type_colors)
    ax.plot(t_ms, scutellum_z, color=line_color, **lk)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Scutellum\nz (mm)')


# -----------------------------------------------------------------------------
# Row 3 right: singing vs free-walking z-height
# -----------------------------------------------------------------------------

def panel_z_height_singing_vs_walking(
    ax: plt.Axes,
    pulse_z: np.ndarray,
    sine_z: np.ndarray,
    walking_z: np.ndarray,
    kind: str = 'box',
    colors: Optional[Sequence[str]] = None,
    alpha: float = 0.55,
    box_kwargs: Optional[Dict] = None,
    violin_kwargs: Optional[Dict] = None,
    show_points: bool = True,
    point_kwargs: Optional[Dict] = None,
    jitter_width: float = 0.15,
    rng_seed: int = 0,
    title: str = 'z height by state',
) -> None:
    """Compare scutellum z-height during pulse, sine, and free walking.

    ``colors`` overrides the default (pulse/sine/walking) triplet. When
    ``show_points`` is True (default), raw samples are jittered and scattered
    on top of each box/violin.
    """
    p = np.asarray(pulse_z,   dtype=float); p = p[np.isfinite(p)]
    s = np.asarray(sine_z,    dtype=float); s = s[np.isfinite(s)]
    w = np.asarray(walking_z, dtype=float); w = w[np.isfinite(w)]
    data = [p, s, w]
    labels = [
        f'pulse\n(n={p.size})',
        f'sine\n(n={s.size})',
        f'free walk\n(n={w.size})',
    ]
    cols = list(colors) if colors else [
        SONG_COLORS['pulse'], SONG_COLORS['sine'], '#888888',
    ]
    positions = [0, 1, 2]

    if show_points:
        pk = {'s': 6, 'linewidths': 0.3, 'edgecolors': 'k',
              'alpha': 0.7, 'zorder': 1, **(point_kwargs or {})}
        rng = np.random.default_rng(rng_seed)
        for pos, vals, c in zip(positions, data, cols):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
            ax.scatter(pos + jitter, vals, c=c, **pk)

    if kind == 'violin':
        vk = {'widths': 0.7, 'showmeans': True, 'showextrema': False,
              **(violin_kwargs or {})}
        parts = ax.violinplot(data, positions=positions, **vk)
        for pc, c in zip(parts['bodies'], cols):
            pc.set_facecolor(c); pc.set_alpha(alpha)
            pc.set_edgecolor('k'); pc.set_linewidth(0.4)
            pc.set_zorder(2)
        for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            lc = parts.get(key)
            if lc is not None:
                lc.set_color('k'); lc.set_linewidth(0.9); lc.set_zorder(3)
    else:
        bk = {'widths': 0.55, 'patch_artist': True, 'showfliers': False,
              'medianprops': dict(color='k', lw=0.8),
              'whiskerprops': dict(lw=0.6),
              'capprops':     dict(lw=0.6),
              'boxprops':     dict(lw=0.6),
              **(box_kwargs or {})}
        bp = ax.boxplot(data, positions=positions, **bk)
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c); patch.set_alpha(alpha)
            patch.set_zorder(2)
        for key in ('medians', 'whiskers', 'caps'):
            for line in bp.get(key, []):
                line.set_zorder(3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scutellum z (mm)')
    ax.set_title(title, pad=2)


# -----------------------------------------------------------------------------
# Sine in-phase: extended vs folded wing z-traces over a sine bout
# -----------------------------------------------------------------------------

WING_PHASE_COLORS: Dict[str, str] = {
    'extended': '#1f77b4',
    'folded':   '#d62728',
}


def panel_sine_wing_inphase(
    ax: plt.Axes,
    t_ms: np.ndarray,
    wing_extended_z: np.ndarray,
    wing_folded_z: np.ndarray,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    sine_segments: Optional[Iterable[dict]] = None,
    colors: Optional[Dict[str, str]] = None,
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = 'Sine song: extended + folded wing in phase',
) -> None:
    """Overlaid extended- vs folded-wing V13 z traces over a slice.

    Caller resolves which wing is extended at each frame (typically by
    comparing per-frame extension angles) and passes the two resulting traces
    here.
    """
    cc = {**WING_PHASE_COLORS, **(colors or {})}
    lk = {'lw': 0.8, **(line_kwargs or {})}
    lg = {'loc': 'upper right', **(legend_kwargs or {})}

    if sine_segments is not None:
        sine_only = [s for s in sine_segments if s.get('type') == 'sine']
        _shade_segments(ax, sine_only, fs, frame_range=frame_range)

    ax.plot(t_ms, wing_extended_z, color=cc['extended'],
            label='Extended', **lk)
    ax.plot(t_ms, wing_folded_z, color=cc['folded'],
            label='Folded', **lk)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Wing V13 z (mm)')
    ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)


# -----------------------------------------------------------------------------
# Wing-angle distributions (histogram + KDE) for pulse vs sine
# -----------------------------------------------------------------------------

def _gaussian_kde_1d(
    x: np.ndarray, grid: np.ndarray, bw: Optional[float] = None
) -> np.ndarray:
    """Lightweight 1-D Gaussian KDE (Silverman bandwidth by default)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(grid, dtype=float)
    if bw is None:
        sd = float(np.std(x, ddof=1))
        if sd <= 0:
            sd = float(np.std(x))
        if sd <= 0:
            return np.zeros_like(grid, dtype=float)
        # Silverman's rule of thumb for univariate Gaussian KDE.
        bw = 1.06 * sd * n ** (-1.0 / 5.0)
    if bw <= 0:
        return np.zeros_like(grid, dtype=float)
    # Vectorised Gaussian sum; chunk to bound memory if needed.
    diff = (grid[:, None] - x[None, :]) / bw
    K = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    return K.sum(axis=1) / (n * bw)


def panel_joint_angle_density(
    ax: plt.Axes,
    ext_pulse: np.ndarray,
    ext_sine: np.ndarray,
    bins: int = 40,
    range_deg: Tuple[float, float] = (0.0, 180.0),
    colors: Optional[Dict[str, str]] = None,
    hist_alpha: float = 0.25,
    kde_lw: float = 1.4,
    show_hist: bool = True,
    show_kde: bool = True,
    bw: Optional[float] = None,
    title: str = 'Wing angle: pulse vs sine',
    legend_kwargs: Optional[Dict] = None,
) -> None:
    """Overlaid 1-D histogram + KDE of the extended-wing angle by song state.

    Two distributions are drawn on a single axes:

    * Pulse  (solid, ``SONG_COLORS['pulse']``)
    * Sine   (solid, ``SONG_COLORS['sine']``)

    Histograms (probability-density normalised) are drawn translucent and
    KDE curves (Silverman bandwidth) are overlaid on top.
    """
    sc = {'pulse': SONG_COLORS['pulse'], 'sine': SONG_COLORS['sine'],
          **(colors or {})}
    edges = np.linspace(range_deg[0], range_deg[1], int(bins) + 1)
    grid = np.linspace(range_deg[0], range_deg[1], 400)

    series = (
        ('Pulse', ext_pulse, sc['pulse'], '-'),
        ('Sine',  ext_sine,  sc['sine'],  '-'),
    )

    handles, labels = [], []
    for label, x, color, ls in series:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            continue
        if show_hist:
            ax.hist(x, bins=edges, density=True, color=color,
                    alpha=hist_alpha, histtype='stepfilled',
                    edgecolor='none')
        if show_kde:
            y = _gaussian_kde_1d(x, grid, bw=bw)
            (line,) = ax.plot(grid, y, color=color, lw=kde_lw,
                              linestyle=ls, label=label)
            handles.append(line); labels.append(label)
        else:
            handles.append(plt.Line2D([], [], color=color, lw=kde_lw,
                                      linestyle=ls))
            labels.append(label)

    ax.set_xlim(*range_deg)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wing angle (deg)')
    ax.set_ylabel('Density')
    ax.set_title(title, pad=2)
    lg = {'loc': 'upper right', **(legend_kwargs or {})}
    if handles:
        _colored_text_legend(ax, handles_labels=(handles, labels), **lg)


# -----------------------------------------------------------------------------
# Row 3 col 0: L vs R wing phase difference during sine song
# -----------------------------------------------------------------------------

def panel_wing_phase_polar(
    ax: plt.Axes,
    phase_rad: np.ndarray,
    bins: int = 36,
    color: Optional[str] = None,
    density: bool = True,
    mean_vector: bool = True,
    title: str = 'L–R wing phase (sine)',
    bar_kwargs: Optional[Dict] = None,
    mean_kwargs: Optional[Dict] = None,
) -> None:
    """Polar histogram of L-vs-R wing phase difference (radians).

    Parameters
    ----------
    ax : matplotlib polar axes (must be created with ``projection='polar'``).
    phase_rad : 1-D array of phase differences (radians, any wrapping).
    bins : number of equal-width angular bins over [-pi, pi].
    density : if True, normalise so the histogram integrates to 1 over 2*pi
        (probability density per radian).
    mean_vector : if True, draw a line from origin to the mean resultant
        vector R = mean(exp(i*phase)); arrow direction = arg(R), length |R|.
    """
    if not getattr(ax, 'name', '') == 'polar':
        raise ValueError('panel_wing_phase_polar requires a polar axes')

    color = color or SONG_COLORS['sine']
    bk = {'edgecolor': 'white', 'linewidth': 0.4, 'alpha': 0.85,
          **(bar_kwargs or {})}
    mk = {'color': '#222222', 'linewidth': 1.4, **(mean_kwargs or {})}

    x = np.asarray(phase_rad, dtype=float)
    x = x[np.isfinite(x)]
    x = np.angle(np.exp(1j * x))                  # wrap to [-pi, pi]

    if x.size == 0:
        ax.set_title(title, pad=2)
        return

    edges = np.linspace(-np.pi, np.pi, int(bins) + 1)
    counts, _ = np.histogram(x, bins=edges)
    width = 2 * np.pi / bins
    centers = edges[:-1] + width / 2.0
    if density and counts.sum() > 0:
        heights = counts / (counts.sum() * width)
    else:
        heights = counts.astype(float)

    bar_width = 0.65 * width
    ax.bar(centers, heights, width=bar_width, color=color,
           bottom=0.0, **bk)

    rmax = float(heights.max()) if heights.size else 1.0
    if rmax <= 0.0:
        rmax = 1.0
    ax.set_ylim(0.0, rmax * 1.08)

    if mean_vector and x.size > 1:
        R = np.mean(np.exp(1j * x))
        r_len = float(np.abs(R))
        r_ang = float(np.angle(R))
        ax.plot([r_ang], [rmax * 1.04],
                marker='o', markersize=4.5,
                markerfacecolor=mk.get('color', '#222222'),
                markeredgecolor='white', markeredgewidth=0.6,
                linestyle='none', zorder=5, clip_on=False)
        # ax.text(0.5, 0.5, f'|R|={r_len:.2f}',
        #         transform=ax.transAxes, ha='center', va='center',
        #         fontsize=6, color='#222222')

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetalim(-np.pi, np.pi)
    ax.set_thetagrids(
        [0, 90, 180, -90],
        labels=['0', 'π/2', '±π', '-π/2'],
        fontsize=6,
    )
    ax.set_rgrids(
        np.linspace(rmax / 3.0, rmax, 3),
        labels=[''] * 3,
    )
    ax.tick_params(axis='x', pad=-2)
    ax.grid(True, color='#bbbbbb', linewidth=0.4, alpha=0.8)
    ax.set_facecolor('#f4f4f4')
    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(0.6)
    ax.set_title(title, pad=2, fontsize=7)


# -----------------------------------------------------------------------------
# Row 4 col 0: pulse classification (Pslow vs Pfast)
# -----------------------------------------------------------------------------

def panel_pulse_classification(
    ax: plt.Axes,
    pulse_type_results: Dict[str, object],
    show_std: bool = True,
    show_examples: bool = False,
    max_examples: int = 40,
    colors: Optional[Dict[str, str]] = None,
    mean_kwargs: Optional[Dict] = None,
    std_alpha: float = 0.20,
    example_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = 'Pslow vs Pfast waveform',
) -> None:
    """Plot mean Pslow and Pfast pulse waveforms.

    Parameters
    ----------
    pulse_type_results : dict returned by
        :func:`utils.pulse_type_cache.get_pulse_type_labels`
        (keys: ``centroids``, ``counts``, ``pooled_waveforms``, ``fs``).
    show_std : if True, shade +/- 1 std around each mean.
    show_examples : if True, draw up to ``max_examples`` individual
        waveforms per class as thin semi-transparent lines.
    """
    centroids = pulse_type_results.get('centroids', {}) or {}
    counts = pulse_type_results.get('counts', {}) or {}
    pooled = pulse_type_results.get('pooled_waveforms', {}) or {}
    fs = float(pulse_type_results.get('fs', 800.0))

    pt = {**PULSE_TYPE_COLORS, **(colors or {})}
    mk = {'lw': 1.2, **(mean_kwargs or {})}
    ek = {'lw': 0.3, 'alpha': 0.15, **(example_kwargs or {})}
    lg = {'loc': 'upper right', 'borderaxespad': 0.2,
          **(legend_kwargs or {})}

    slow = np.asarray(centroids.get('Pslow', np.zeros(0)))
    fast = np.asarray(centroids.get('Pfast', np.zeros(0)))

    if slow.size == 0 and fast.size == 0:
        ax.text(0.5, 0.5, 'no pulses', ha='center', va='center',
                transform=ax.transAxes)
        return

    W = max(slow.size, fast.size)
    t_ms = (np.arange(W) - W / 2.0) / fs * 1000.0  # centered on pulse peak

    for name, mean_wf in (('Pslow', slow), ('Pfast', fast)):
        if mean_wf.size == 0:
            continue
        color = pt[name]
        n = counts.get(name, 0)

        if show_examples:
            pool = np.asarray(pooled.get(name, np.zeros((0, 0))))
            if pool.size > 0:
                if pool.shape[0] > max_examples:
                    rng = np.random.default_rng(0)
                    sel = rng.choice(pool.shape[0], max_examples, replace=False)
                    pool = pool[sel]
                ax.plot(t_ms[:pool.shape[1]], pool.T,
                        color=color, zorder=1, **ek)

        if show_std:
            pool = np.asarray(pooled.get(name, np.zeros((0, 0))))
            if pool.shape[0] > 1:
                std = pool.std(axis=0)
                ax.fill_between(t_ms[:mean_wf.size],
                                mean_wf - std, mean_wf + std,
                                color=color, alpha=std_alpha,
                                linewidth=0, zorder=2)

        ax.plot(t_ms[:mean_wf.size], mean_wf, color=color,
                label=f'{name} (n={n})', zorder=3, **mk)

    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('wing z (z-score)')
    ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)


# -----------------------------------------------------------------------------
# Row 4 col 1: total pulse vs sine across pairs
# -----------------------------------------------------------------------------

def panel_pulse_sine_totals(
    ax: plt.Axes,
    df,
    results: Sequence[dict],
    pulse_type_results: Dict[str, object],
    fs: float = 800.0,
    show_seconds: bool = True,
    colors: Optional[Dict[str, str]] = None,
    bar_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = 'sine / Pslow / Pfast per pair',
) -> None:
    """Per-pair Sine / Pslow / Pfast composition.

    For each pair, splits the pulse fraction between Pslow and Pfast using
    the dominant-side pulse-type label counts from ``pulse_type_results``
    (``utils.pulse_type_cache.get_pulse_type_labels``). Bars are sorted by
    total song (sine + pslow + pfast).
    """
    needed = ('frac_pulse', 'frac_sine', 'T', 'pair_idx')
    for c in needed:
        if c not in df.columns:
            raise KeyError(f'df missing column {c!r}')

    labels = (pulse_type_results or {}).get('labels', {}) or {}

    # Map pair_idx -> (key0, dominant_side) via results
    pair_dom: Dict[int, str] = {}
    for r in results:
        pidx = int(r.get('pair_idx', -1))
        dw = str(r.get('song0', {}).get('dominant_wing', 'L')).upper()
        pair_dom[pidx] = 'L' if dw.startswith('L') else 'R'

    sub = df[list(needed)].copy().reset_index(drop=True)
    frac_pslow = np.zeros(len(sub))
    frac_pfast = np.zeros(len(sub))
    frac_pulse = sub['frac_pulse'].values
    for i, pidx in enumerate(sub['pair_idx'].astype(int).values):
        side = pair_dom.get(pidx, 'L')
        side_labels = labels.get(pidx, {}).get(side, None)
        if side_labels is None or len(side_labels) == 0:
            continue
        lab = np.asarray(side_labels)
        n_slow = int((lab == 'Pslow').sum())
        n_fast = int((lab == 'Pfast').sum())
        denom = n_slow + n_fast
        if denom == 0:
            continue
        frac_pslow[i] = frac_pulse[i] * (n_slow / denom)
        frac_pfast[i] = frac_pulse[i] * (n_fast / denom)

    frac_sine = sub['frac_sine'].values
    total_song = frac_sine + frac_pslow + frac_pfast
    order = np.argsort(total_song)
    frac_sine = frac_sine[order]
    frac_pslow = frac_pslow[order]
    frac_pfast = frac_pfast[order]
    xs = np.arange(len(sub))

    col = {'sine':  SONG_COLORS['sine'],
           'Pslow': PULSE_TYPE_COLORS['Pslow'],
           'Pfast': PULSE_TYPE_COLORS['Pfast'],
           **(colors or {})}
    bk = {'width': 1.0, 'linewidth': 0, **(bar_kwargs or {})}
    lg = {'loc': 'upper left', 'borderaxespad': 0.2, 'ncols': 3,
          'columnspacing': 0.6, **(legend_kwargs or {})}

    ax.bar(xs, frac_sine,  color=col['sine'],  label='sine',  **bk)
    ax.bar(xs, frac_pslow, bottom=frac_sine,                  color=col['Pslow'], label='Pslow', **bk)
    ax.bar(xs, frac_pfast, bottom=frac_sine + frac_pslow,     color=col['Pfast'], label='Pfast', **bk)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f'pair (n={len(sub)})')
    ax.set_ylabel('fraction of bout')
    ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)

    if show_seconds:
        T_sorted = sub['T'].values[order]
        total_sorted = total_song[order]
        med_total_s = float(np.nanmedian((T_sorted / fs) * total_sorted))
        ax.text(0.98, 0.02,
                f'median song time/bout: {med_total_s:.2f} s',
                ha='right', va='bottom', transform=ax.transAxes,
                fontsize=mpl.rcParams['xtick.labelsize'])


# -----------------------------------------------------------------------------
# Row 4 col 2: L/R wing dominance
# -----------------------------------------------------------------------------

def panel_lr_dominance(
    ax: plt.Axes,
    results: Sequence[dict],
    colors: Optional[Dict[str, str]] = None,
    n_bins: int = 21,
    hist_kwargs: Optional[Dict] = None,
    midline_kwargs: Optional[Dict] = None,
) -> None:
    """L/R wing dominance: per-bout L pulse-fraction histogram.

    Title annotates the population L vs R counts.
    """
    dcol = {**DOMINANCE_COLORS, **(colors or {})}
    hk = {'edgecolor': 'white', 'linewidth': 0.4, **(hist_kwargs or {})}
    mk = {'color': 'k', 'lw': 0.5, 'ls': '--', **(midline_kwargs or {})}

    dom = [str(r['song0'].get('dominant_wing', 'L')).upper()[0] for r in results]
    counts = Counter(dom)
    n_L = counts.get('L', 0)
    n_R = counts.get('R', 0)
    n_total = n_L + n_R

    pulse_frac_L = []
    for r in results:
        sides = r['song0'].get('sides', {})
        L_lab = np.asarray(sides.get('L', {}).get('frame_labels', np.array([])))
        R_lab = np.asarray(sides.get('R', {}).get('frame_labels', np.array([])))
        n_pulse_L = int((L_lab == 'pulse').sum()) if L_lab.size else 0
        n_pulse_R = int((R_lab == 'pulse').sum()) if R_lab.size else 0
        denom = n_pulse_L + n_pulse_R
        if denom == 0:
            continue
        pulse_frac_L.append(n_pulse_L / denom)
    pulse_frac_L = np.asarray(pulse_frac_L)

    if pulse_frac_L.size == 0:
        ax.text(0.5, 0.5, 'no pulses', ha='center', va='center',
                transform=ax.transAxes)
        return

    bins = np.linspace(0, 1, n_bins)
    ax.hist(pulse_frac_L, bins=bins, color=dcol['L'], **hk)
    ax.axvline(0.5, **mk)
    ax.set_xlim(0, 1)
    ax.set_xlabel('L pulse fraction (per pair)')
    ax.set_ylabel('# pairs')
    ax.set_title(
        f'L/R dominance: {n_L} L  /  {n_R} R  (n={n_total})', pad=2,
    )
