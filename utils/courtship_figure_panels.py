"""Panel functions for the consolidated courtship figure (Figure 6).

Each ``panel_*`` function takes a target ``Axes`` (or list of axes) plus the
data it needs. Functions are self-contained so any panel can be re-rendered
into a different axes without rebuilding the full figure (the "swappable"
requirement). ``assemble_figure`` wires up the canonical layout and returns a
dict of axes that the caller hands to the panel functions.

Layout (183 mm x 130 mm, 4 rows):

    Row 1: 6 cropped video frames
    Row 2: 6 MuJoCo render frames
    Row 3: [wing-z trace stacked over scutellum-z trace] | [singing vs running
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
import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


# -----------------------------------------------------------------------------
# Style + constants
# -----------------------------------------------------------------------------

SONG_COLORS: Dict[str, str] = {
    'pulse':  '#F97316',  # warm orange  (pulse song → F, H)
    'sine':   '#14B8A6',  # teal         (sine song  → F, H)
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
        'svg.fonttype':    'none',
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size':       base,
        'axes.labelsize':  base,
        'axes.titlesize':  base,
        'xtick.labelsize': tick,
        'ytick.labelsize': tick,
        'legend.fontsize': tick,
        'axes.spines.right': False,
        'axes.spines.top':   False,
        'axes.linewidth':  1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
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
    n_render_strip: int = 4,
    n_video_frames: Optional[int] = None,
) -> Tuple[plt.Figure, Dict[str, object]]:
    """Build the 4-row consolidated layout. Returns (fig, axes_dict).

    Layout (top to bottom):
        Row 1  video frames with KP overlay | sine in-phase trace
        Row 2  wing V13 z + scutellum z stacked (full width)
        Row 3 [polar L-R phase] | [joint angle density] | [Pslow/Pfast]
              | [scutellum z courtship vs free running]
        Row 4  [N MuJoCo render frames] | [pitch align] | [per-bout violin]

    axes_dict keys:
        'video'        : list of N axes (Row 1 left)
        'sine_phase'   : axes (Row 1 right)
        'wing'         : axes (Row 2 top)
        'scut'         : axes (Row 2 bottom, sharex with wing)
        'wing_phase_polar' : polar axes (Row 3 col 0)
        'angle_2d'     : axes (Row 3 col 1)
        'pulse_class'  : axes (Row 3 col 2)
        'zheight'      : axes (Row 3 col 3)
        'render'       : list of N axes (Row 4 left)
        'pitch'        : axes (Row 4 middle)
        'align_violin' : axes (Row 4 right)
    """
    apply_paper_style()
    fig = plt.figure(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    sub_video, sub_traces, sub_row3, sub_render = fig.subfigures(
        4, 1, height_ratios=[1.0, 1.15, 1.6, 1.0], hspace=0.10,
    )

    # Row 1: video frames (left) + exemplar sine in-phase trace (right). When
    # ``n_video_frames`` is set, the sine panel expands into the reclaimed
    # width so per-frame width stays roughly constant.
    orig_n_video = max(1, int(n_frames_strip) - 1)
    n_video = max(1, int(n_video_frames if n_video_frames is not None
                         else orig_n_video))
    sf1_left_ratio = 5.0 * (n_video / orig_n_video)
    sf1_right_ratio = 1.4 + (5.0 - sf1_left_ratio)
    sf1_left, sf1_right = sub_video.subfigures(
        1, 2, width_ratios=[sf1_left_ratio, sf1_right_ratio], wspace=0.02,
    )
    ax_video = list(sf1_left.subplots(1, n_video))
    if n_video == 1:
        ax_video = [ax_video[0]] if not isinstance(ax_video, list) else ax_video
    for ax in ax_video:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sf1_left.subplots_adjust(
        wspace=0.06, left=0.05, right=0.99, top=0.80, bottom=0.06,
    )
    ax_sine_phase = sf1_right.subplots(1, 1)
    sf1_right.subplots_adjust(
        left=0.12, right=0.98, top=0.80, bottom=0.24,
    )

    # Row 2: wing V13 z + scutellum z stacked across the full figure width.
    ax_wing, ax_scut = sub_traces.subplots(
        2, 1, sharex=True, height_ratios=[1.4, 1.0],
    )
    ax_wing.tick_params(bottom=False, labelbottom=False)
    ax_wing.spines['bottom'].set_visible(False)
    sub_traces.subplots_adjust(
        left=0.07, right=0.97, top=0.88, bottom=0.28, hspace=0.16,
    )

    # Row 3: 4 mechanism panels (col 0 = polar, cols 1-3 = cartesian)
    sf3_polar, sf3_rest = sub_row3.subfigures(
        1, 2, width_ratios=[1.0, 3.0], wspace=0.06,
    )
    ax_wing_phase_polar = sf3_polar.subplots(
        1, 1, subplot_kw={'projection': 'polar'},
    )
    sf3_polar.subplots_adjust(
        left=0.20, right=0.88, top=0.82, bottom=0.22,
    )
    ax_angle_2d, ax_pulse, ax_zheight = sf3_rest.subplots(1, 3)
    sf3_rest.subplots_adjust(
        left=0.06, right=0.97, top=0.90, bottom=0.22, wspace=0.48,
    )

    # Row 4: render strip (left) + pitch-align trace (middle) + violin (right)
    sub_render_left, sub_render_mid, sub_render_right = sub_render.subfigures(
        1, 3, width_ratios=[int(n_render_strip), 2.0, 1.2], wspace=0.08,
    )
    ax_render = list(sub_render_left.subplots(1, int(n_render_strip)))
    if int(n_render_strip) == 1:
        ax_render = [ax_render[0]] if not isinstance(ax_render, list) else ax_render
    for ax in ax_render:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sub_render_left.subplots_adjust(
        wspace=0.06, left=0.05, right=0.98, top=0.94, bottom=0.06,
    )
    ax_pitch = sub_render_mid.subplots(1, 1)
    sub_render_mid.subplots_adjust(
        left=0.18, right=0.96, top=0.90, bottom=0.26,
    )
    ax_align_violin = sub_render_right.subplots(1, 1)
    sub_render_right.subplots_adjust(
        left=0.28, right=0.92, top=0.90, bottom=0.26,
    )

    return fig, {
        'video':            ax_video,
        'sine_phase':       ax_sine_phase,
        'wing':             ax_wing,
        'scut':             ax_scut,
        'wing_phase_polar': ax_wing_phase_polar,
        'angle_2d':         ax_angle_2d,
        'pulse_class':      ax_pulse,
        'zheight':          ax_zheight,
        'render':           ax_render,
        'pitch':            ax_pitch,
        'align_violin':     ax_align_violin,
    }


DEFAULT_PANEL_LETTERS: Sequence[Tuple[str, str]] = (
    ('video',            'A'),
    ('sine_phase',       'B'),
    ('wing',             'C'),
    # 'scut' shares Panel C with 'wing' (stacked row-2 traces)
    ('wing_phase_polar', 'D'),
    ('angle_2d',         'E'),
    ('pulse_class',      'F'),
    ('zheight',          'G'),
    ('render',           'H'),
    ('pitch',            'I'),
    ('align_violin',     'J'),
)


DEFAULT_PANEL_POSITIONS: Dict[str, Tuple[float, float]] = {
    # Per-key (x, y) in axes fractions for the panel-letter anchor.
    # Panels with y-tick labels use more negative x so the letter clears them
    # and sits at the panel's upper-left (everything else to the right).
    'video':            (-0.02, 1.08),
    'sine_phase':       (-0.28, 1.04),
    'wing':             (-0.02, 1.08),
    'wing_phase_polar': (-0.22, 1.20),
    'angle_2d':         (-0.40, 1.06),
    'pulse_class':      (-0.40, 1.06),
    'zheight':          (-0.45, 1.06),
    'render':           (-0.02, 1.08),
    'pitch':            (-0.40, 1.06),
    'align_violin':     (-0.60, 1.06),
}


# Letters for row-leading panels (A, C, D, H) all land at the same figure-x
# so they line up vertically down the left edge of the figure. y still comes
# from DEFAULT_PANEL_POSITIONS (axes fraction).
DEFAULT_PANEL_FIG_X: Dict[str, float] = {
    'video':            0.010,
    'wing':             0.010,
    'wing_phase_polar': 0.010,
    'render':           0.010,
}


def _root_figure(ax) -> plt.Figure:
    """Return the top-level Figure for an axes, running up nested SubFigures."""
    f = ax.figure
    while getattr(f, 'figure', None) is not None and f.figure is not f:
        f = f.figure
    return f


def add_panel_letters(
    axd: Dict[str, object],
    letters: Sequence[Tuple[str, str]] = DEFAULT_PANEL_LETTERS,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: float = 10.0,
    fontweight: str = 'bold',
    titles: Optional[Dict[str, str]] = None,
    title_fontsize: Optional[float] = None,
    title_fontweight: str = 'normal',
    title_sep: float = 4.0,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    fig_x: Optional[Dict[str, float]] = None,
) -> None:
    """Stamp bold panel letters (A, B, C, ...) in the top-left of each panel.

    ``letters`` is a sequence of ``(axd_key, letter)`` pairs. For list-valued
    entries (``'video'``, ``'render'``), the letter lands on the first axes.
    Coordinates are in axes fractions (``transAxes``); negative ``x`` nudges
    the letter outside the axes spine so it clears y-tick labels.

    When ``titles`` maps panel keys to short descriptive strings, each title
    is rendered in regular weight on the same baseline just right of the
    letter (``HPacker`` auto-spaces so the gap is constant regardless of
    letter-width). Per-key position overrides come from ``positions`` (falls
    back to ``DEFAULT_PANEL_POSITIONS``, then to the ``x, y`` arguments).
    """
    title_fs = title_fontsize if title_fontsize is not None else fontsize * 0.85
    pos = {**DEFAULT_PANEL_POSITIONS, **(positions or {})}
    fx_map = {**DEFAULT_PANEL_FIG_X, **(fig_x or {})}

    for key, ch in letters:
        ax = axd.get(key)
        if ax is None:
            continue
        if isinstance(ax, (list, tuple)):
            if not ax:
                continue
            ax = ax[0]
        px, py = pos.get(key, (x, y))
        # Row-leading panels anchor x in figure coords (aligning A/B/D/H down
        # the left edge) while keeping y in axes fraction.
        if key in fx_map:
            root_fig = _root_figure(ax)
            anchor_x = fx_map[key]
            transform = mtransforms.blended_transform_factory(
                root_fig.transFigure, ax.transAxes,
            )
        else:
            anchor_x = px
            transform = ax.transAxes
        title = titles.get(key) if titles else None
        if title:
            letter_ta = offsetbox.TextArea(
                ch, textprops=dict(fontsize=fontsize, fontweight=fontweight),
            )
            title_ta = offsetbox.TextArea(
                title,
                textprops=dict(fontsize=title_fs, fontweight=title_fontweight),
            )
            box = offsetbox.HPacker(
                children=[letter_ta, title_ta],
                align='baseline', pad=0, sep=title_sep,
            )
            aob = offsetbox.AnchoredOffsetbox(
                loc='lower left', child=box, pad=0, borderpad=0,
                frameon=False,
                bbox_to_anchor=(anchor_x, py), bbox_transform=transform,
            )
            aob.set_clip_on(False)
            ax.add_artist(aob)
        else:
            ax.text(anchor_x, py, ch, transform=transform,
                    fontsize=fontsize, fontweight=fontweight,
                    ha='left', va='bottom', clip_on=False)


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
        titles = [f'{int(round(f / fs * 1000))} ms' for f in frame_indices]
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
            _tb = offsetbox.AnchoredText(
                title, loc='lower left', pad=0.0, borderpad=0.0,
                frameon=True, prop=dict(fontsize=7, color='black'),
            )
            _tb.patch.set_boxstyle('square,pad=0.25')
            _tb.patch.set_facecolor('white')
            _tb.patch.set_edgecolor('none')
            _tb.patch.set_alpha(0.9)
            _tb.set_zorder(5)
            ax.add_artist(_tb)
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
    kp_xyz_fly1: Optional[np.ndarray] = None,
    kp_color_fly1: str = '#3a7bff',
    label_kp_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Single video frame with selected keypoints drawn + labeled.

    Projects ``kp_xyz[frame_index, idx, :]`` to pixels via DLT, applies the ROI
    offset, and draws a dot + text for each entry of ``label_kps`` (defaults
    to Scutellum + the two V13 wing tips).

    When ``center_xyz`` (shape ``(3,)`` or ``(T, 3)``) and ``crop_wh=(w, h)``
    are both given, the ROI is computed dynamically by projecting the center
    through DLT and centering a ``(w, h)`` crop on the resulting pixel,
    clamped to the raw video bounds.

    When ``kp_xyz_fly1`` is given it is drawn in ``kp_color_fly1`` on top of
    the fly0 dots (same ``label_kps``), and fly0 gets its own labeled dots.

    When ``label_kp_colors`` is given (``{kp_name: color}``) those per-keypoint
    colors override ``kp_color`` for the fly0 dots + text (useful for matching
    the wing-trace panel colors — purple WingL, blue WingR, black Scutellum).
    """
    import cv2
    sk = {'s': 14, 'edgecolors': 'white', 'linewidths': 0.8,
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

    def _project_and_label(kp_source, color, annotate, per_kp_colors=None):
        arr = np.asarray(kp_source)
        if arr.ndim == 2 and arr.shape[-1] != 3:
            arr = arr.reshape(arr.shape[0], -1, 3)
        pts = arr[int(frame_index)] * float(kp_scale)
        uv_ = _dlt_project(dlt_coeffs, pts)
        if roi_t is not None:
            uv_ = uv_ - np.array([roi_t[0], roi_t[1]], dtype=float)
        xs, ys, cs = [], [], []
        for name in label_kps:
            if name not in name_to_idx:
                continue
            u, v = uv_[name_to_idx[name]]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            c_kp = (per_kp_colors or {}).get(name, color)
            xs.append(u); ys.append(v); cs.append(c_kp)
            if annotate:
                dx, dy = label_offsets.get(name, (8, -4))
                tk_local = {**tk, 'color': c_kp}
                ax.annotate(name, xy=(u, v), xytext=(u + dx, v + dy), **tk_local)
        if xs:
            ax.scatter(xs, ys, c=cs, **sk)

    _project_and_label(kp_xyz, kp_color, annotate=True,
                       per_kp_colors=label_kp_colors)
    if kp_xyz_fly1 is not None:
        _project_and_label(kp_xyz_fly1, kp_color_fly1, annotate=False)


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
    kp_xyz_fly1_per_frame: Optional[np.ndarray] = None,
    kp_color_fly1: str = '#3a7bff',
    masks_per_fly: Optional[Sequence[np.ndarray]] = None,
    mask_colors: Optional[Sequence[str]] = None,
    mask_alpha: float = 0.35,
) -> None:
    """Same as :func:`panel_video_strip` but draws projected keypoint dots
    on every frame.

    When ``masks_per_fly`` is given it is a sequence of ``(len(frame_indices),
    H_full, W_full)`` bool arrays (one per fly, same order as
    ``[kp_xyz_per_frame, kp_xyz_fly1_per_frame]``). Each mask is cropped to the
    current frame ROI and blended onto the frame with ``mask_colors[i]`` at
    ``mask_alpha``. ``mask_colors`` defaults to ``[kp_color, kp_color_fly1]``.

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
        titles = [f'{int(round(f / fs * 1000))} ms' for f in frame_indices]
    if len(titles) != len(axes) or len(frame_indices) != len(axes):
        raise ValueError('axes, frame_indices, titles must all be the same length')

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    if overlay_kps is None:
        kp_idx = np.arange(len(kp_names))
    else:
        kp_idx = np.array([name_to_idx[n] for n in overlay_kps if n in name_to_idx],
                          dtype=int)

    def _as_3d(a):
        a = np.asarray(a)
        if a.ndim == 2 and a.shape[-1] != 3:
            a = a.reshape(a.shape[0], -1, 3)
        return a

    kp_arr = _as_3d(kp_xyz_per_frame)
    kp_arr1 = _as_3d(kp_xyz_fly1_per_frame) if kp_xyz_fly1_per_frame is not None else None

    mask_seq: List[np.ndarray] = list(masks_per_fly or [])
    if mask_seq:
        mc = list(mask_colors or [kp_color, kp_color_fly1])[: len(mask_seq)]
        if len(mc) < len(mask_seq):
            mc = mc + [kp_color] * (len(mask_seq) - len(mc))
        mask_rgba = [
            np.array(
                [int(c.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
                + [float(mask_alpha)],
                dtype=float,
            )
            for c in mc
        ]

    dynamic_roi = center_xyz is not None and crop_wh is not None
    center_arr = np.asarray(center_xyz) if dynamic_roi else None

    cap = _open_video(mp4_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        for i_ax, (ax, fidx, title) in enumerate(zip(axes, frame_indices, titles)):
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
            _tb = offsetbox.AnchoredText(
                title, loc='lower left', pad=0.0, borderpad=0.0,
                frameon=True, prop=dict(fontsize=7, color='black'),
            )
            _tb.patch.set_boxstyle('square,pad=0.25')
            _tb.patch.set_facecolor('white')
            _tb.patch.set_edgecolor('none')
            _tb.patch.set_alpha(0.9)
            _tb.set_zorder(5)
            ax.add_artist(_tb)
            H, W = frame.shape[:2]

            for fi, mask_stack in enumerate(mask_seq):
                m_full = np.asarray(mask_stack[i_ax], dtype=bool)
                if roi_t is not None:
                    x0, y0, w_roi, h_roi = roi_t
                    m_crop = m_full[y0:y0 + h_roi, x0:x0 + w_roi]
                else:
                    m_crop = m_full
                if m_crop.shape[0] != H or m_crop.shape[1] != W:
                    continue
                overlay = np.zeros((H, W, 4), dtype=float)
                overlay[m_crop] = mask_rgba[fi]
                ax.imshow(overlay, interpolation='nearest', zorder=2)

            def _scatter(arr, color):
                pts3 = arr[int(fidx)][kp_idx] * float(kp_scale)
                uv = _dlt_project(dlt_coeffs, pts3)
                if roi_t is not None:
                    uv = uv - np.array([roi_t[0], roi_t[1]], dtype=float)
                m = (np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
                     & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                     & (uv[:, 1] >= 0) & (uv[:, 1] < H))
                if m.any():
                    ax.scatter(uv[m, 0], uv[m, 1], c=color, **sk)

            _scatter(kp_arr, kp_color)
            if kp_arr1 is not None:
                _scatter(kp_arr1, kp_color_fly1)
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# Row 2: MuJoCo render strip
# -----------------------------------------------------------------------------

def floor_align_qpos_pair(
    model,
    qpos_pair: np.ndarray,
    floor_z: Optional[float] = None,
    fly_nq: Optional[int] = None,
    fly_suffixes: Sequence[str] = ('_fly0', '_fly1'),
    floor_geom_name: str = 'floor',
    floor_z_offset: float = 0.0,
) -> np.ndarray:
    """Return a copy of ``qpos_pair`` with each fly's free-joint root-z shifted
    so its lowest-standing geom surface touches ``floor_z`` in every frame.

    ``floor_z_offset`` (default 0) is added to the per-frame shift, so a
    positive value pushes the fly further down into the floor (useful when
    the claw-tip touches visually read as floating at grazing camera angles).

    ``qpos_pair`` is laid out as ``[fly0_qpos | fly1_qpos]`` with each fly
    starting in a 7-dof free joint (xyz + quat), so root-z lives at index
    ``fly_nq * k + 2`` for fly ``k``. ``fly_nq`` defaults to ``model.nq // 2``.

    Geoms are assigned to a fly by a case-insensitive substring match on their
    parent body name against ``fly_suffixes`` (e.g. ``'_fly0'``). The lowest
    surface per fly is ``min(geom_xpos[:, 2] - geom_rbound[:])`` — i.e. the
    bottom of each geom's bounding sphere — which works for mesh feet.

    When ``floor_z`` is ``None`` it is auto-detected from the z of the geom
    named ``floor_geom_name`` (plane geoms store z in ``geom_pos``).
    """
    import mujoco

    qpos_pair = np.asarray(qpos_pair, dtype=float).copy()
    if qpos_pair.ndim != 2:
        raise ValueError(f'qpos_pair must be (T, nq); got {qpos_pair.shape}')
    T, nq = qpos_pair.shape
    if nq != model.nq:
        raise ValueError(f'qpos_pair nq={nq} != model.nq={model.nq}')
    fly_nq = int(fly_nq or (model.nq // len(fly_suffixes)))
    if len(fly_suffixes) * fly_nq != model.nq:
        raise ValueError(f'{len(fly_suffixes)}*{fly_nq} != model.nq={model.nq}')

    if floor_z is None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        floor_z = float(model.geom_pos[gid, 2]) if gid >= 0 else 0.0

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ''
                  for b in range(model.nbody)]
    geom_fly = np.full(model.ngeom, -1, dtype=int)
    for g in range(model.ngeom):
        bname = body_names[model.geom_bodyid[g]].lower()
        for fi, suf in enumerate(fly_suffixes):
            if suf.lower() in bname:
                geom_fly[g] = fi
                break

    # Local AABB corner offsets (8 corners of a unit box in ±1 space). The
    # true world-frame bottom of each geom comes from transforming these by
    # its rotation. Tighter than geom_rbound, so feet actually touch the floor.
    corner_signs = np.array([
        [+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1],
        [-1, +1, +1], [-1, +1, -1], [-1, -1, +1], [-1, -1, -1],
    ], dtype=float)
    local_aabb = np.asarray(model.geom_aabb, dtype=float)  # (ngeom, 6)
    rbound = np.asarray(model.geom_rbound, dtype=float)
    data = mujoco.MjData(model)
    for t in range(T):
        data.qpos[:] = qpos_pair[t]
        mujoco.mj_forward(model, data)
        for fi in range(len(fly_suffixes)):
            mask = np.nonzero(geom_fly == fi)[0]
            if mask.size == 0:
                continue
            bottoms = np.full(mask.size, np.inf, dtype=float)
            for k, g in enumerate(mask):
                c = local_aabb[g, :3]
                s = local_aabb[g, 3:]
                if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s))):
                    # Fall back to bounding sphere for geoms with no AABB.
                    bottoms[k] = data.geom_xpos[g, 2] - rbound[g]
                    continue
                corners_local = c + corner_signs * s            # (8, 3)
                R = data.geom_xmat[g].reshape(3, 3)
                corners_world_z = corners_local @ R.T[:, 2] + data.geom_xpos[g, 2]
                bottoms[k] = float(corners_world_z.min())
            bottoms = bottoms[np.isfinite(bottoms)]
            if bottoms.size == 0:
                continue
            shift = float(floor_z - bottoms.min()) - float(floor_z_offset)
            qpos_pair[t, fi * fly_nq + 2] += shift
    return qpos_pair


def estimate_rig_pose_from_qpos(
    data,
    bout_keys: Optional[Iterable[str]] = None,
    fly_nq: int = 93,
    two_flies_per_bout: bool = True,
    exclude_keys: Sequence[str] = ('info',),
) -> Dict[str, object]:
    """Estimate the rig's xy center and yaw from all fly root-xy samples.

    Fly root xy (qpos indices 0,1 per fly) lives in the camera-calibration
    world frame, but the rig mesh is anchored at the MuJoCo world origin.
    Aggregating every fly's xy across all bouts in ``data`` gives both the
    rig's xy center (sample centroid) and its long-axis orientation
    (principal axis of the centered covariance).

    ``data`` maps bout_key → dict with a ``'qpos'`` array of shape
    ``(T, nq)``. When ``two_flies_per_bout`` is True and ``nq >= 2*fly_nq``,
    fly1's xy is taken from columns ``fly_nq:fly_nq+2`` as well.

    Returns a dict with:
      ``center_xy`` — (cx, cy) of the rig center in the calibration frame
      ``yaw_rad``   — dominant-eigenvector angle, sign-fixed so ``v_x >= 0``
      ``major_len`` — full extent (max-min) along the major axis
      ``minor_len`` — full extent along the minor axis
      ``n_points`` — total number of finite xy samples used

    The rig center is the midpoint of the PCA-aligned bounding box of the
    fly xy samples, not the raw centroid: flies typically spend unequal time
    on different sides of a narrow chamber, so the mean drifts off-center.
    The midpoint is unbiased as long as their range is symmetric about the
    true rig center — a much weaker assumption.
    """
    keys = (list(bout_keys) if bout_keys is not None
            else [k for k in data.keys() if k not in exclude_keys])
    xy_chunks: List[np.ndarray] = []
    for k in keys:
        entry = data.get(k)
        if entry is None or 'qpos' not in entry:
            continue
        qpos = np.asarray(entry['qpos'], dtype=float)
        if qpos.ndim != 2 or qpos.shape[1] < 2:
            continue
        xy_chunks.append(qpos[:, 0:2])
        if two_flies_per_bout and qpos.shape[1] >= 2 * fly_nq:
            xy_chunks.append(qpos[:, fly_nq:fly_nq + 2])
    if not xy_chunks:
        raise ValueError('no xy samples collected from data')
    xy = np.concatenate(xy_chunks, axis=0)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 2:
        raise ValueError(f'not enough finite xy samples: {xy.shape[0]}')
    # PCA on mean-centered data → principal axis direction.
    mean_xy = xy.mean(axis=0)
    cov = np.cov(xy - mean_xy, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    v = evecs[:, 0]
    if v[0] < 0:
        v = -v
    yaw = float(np.arctan2(v[1], v[0]))
    # Rotate samples into the PCA-aligned frame and take the midpoint of the
    # axis-aligned bounding box. This recovers the rig center robustly even
    # when flies spend unequal time on each side of the chamber.
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_inv = np.array([[c, -s], [s, c]])
    xy_local = (xy - mean_xy) @ R_inv.T
    lo = xy_local.min(axis=0)
    hi = xy_local.max(axis=0)
    mid_local = 0.5 * (lo + hi)
    # Map midpoint back into the calibration frame.
    c2, s2 = np.cos(yaw), np.sin(yaw)
    R = np.array([[c2, -s2], [s2, c2]])
    center = mean_xy + R @ mid_local
    return {
        'center_xy': (float(center[0]), float(center[1])),
        'yaw_rad': yaw,
        'major_len': float(hi[0] - lo[0]),
        'minor_len': float(hi[1] - lo[1]),
        'n_points': int(xy.shape[0]),
    }


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
    track_midpoint: bool = False,
    fly_nq: Optional[int] = None,
    cam_distance: float = 0.03,
    cam_azimuth: float = 90.0,
    cam_elevation: float = -20.0,
    cam_lookat_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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

    def _midpoint_cam(q_row):
        import mujoco
        nq_total = q_row.shape[0]
        n_per = int(fly_nq or (nq_total // 2))
        p0 = q_row[0:3]
        p1 = q_row[n_per:n_per + 3]
        mid = 0.5 * (p0 + p1) + np.asarray(cam_lookat_offset, dtype=float)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = mid
        cam.distance = float(cam_distance)
        cam.azimuth = float(cam_azimuth)
        cam.elevation = float(cam_elevation)
        return cam

    if viz is not None:
        for i, (ax, fidx) in enumerate(zip(axes, frame_indices)):
            q_row = qpos_array[int(fidx)]
            cam_arg = _midpoint_cam(q_row) if track_midpoint else camera
            pixels = viz.render_frame(
                q_row,
                camera=cam_arg, height=height_px, width=width_px,
            )
            ax.imshow(_crop(pixels))
            if titles is not None:
                ax.set_title(titles[i], pad=2)
        return

    import mujoco
    mj_data = mujoco.MjData(mj_model)
    with mujoco.Renderer(mj_model, height=height_px, width=width_px) as renderer:
        for i, (ax, fidx) in enumerate(zip(axes, frame_indices)):
            q_row = qpos_array[int(fidx)]
            mj_data.qpos[:] = q_row
            mujoco.mj_forward(mj_model, mj_data)
            cam_arg = _midpoint_cam(q_row) if track_midpoint else camera
            renderer.update_scene(mj_data, camera=cam_arg)
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


def _find_worldbody_geom(floor_spec, rig_geom_name: str):
    for g in floor_spec.worldbody.geoms:
        if g.name == rig_geom_name:
            return g
    names = [g.name for g in floor_spec.worldbody.geoms]
    raise ValueError(
        f'rig geom {rig_geom_name!r} not found in floor spec worldbody; '
        f'available geoms: {names}'
    )


def _apply_rig_pose_override(
    floor_spec,
    floor_xml: str,
    rig_geom_name: str,
    rig_pos: Optional[Sequence[float]],
    rig_quat: Optional[Sequence[float]],
) -> None:
    """Mutate the spec so the rig mesh RENDERS at the requested world pose.

    MuJoCo's mesh compiler re-centers mesh vertices to the mesh COM and aligns
    principal inertial axes to the body frame, so the compiled ``geom_pos`` /
    ``geom_quat`` differ from the spec values. This helper probes that
    mesh-induced shift by compiling a fresh copy of ``floor_xml``, then solves
    for the spec-level ``pos``/``quat`` that produce the desired compiled pose.

    ``rig_pos`` may be length-2 (preserving the XML default's rendered z) or
    length-3. ``rig_quat`` is MuJoCo's scalar-first ``(w, x, y, z)``.
    """
    import mujoco

    target = _find_worldbody_geom(floor_spec, rig_geom_name)

    probe_spec = mujoco.MjSpec.from_file(floor_xml)
    probe_model = probe_spec.compile()
    pgid = mujoco.mj_name2id(
        probe_model, mujoco.mjtObj.mjOBJ_GEOM, rig_geom_name,
    )
    probe_target = _find_worldbody_geom(probe_spec, rig_geom_name)
    spec_pos0 = np.asarray(probe_target.pos, dtype=float)
    spec_quat0 = np.asarray(probe_target.quat, dtype=float)
    comp_pos0 = np.asarray(probe_model.geom_pos[pgid], dtype=float)
    comp_quat0 = np.asarray(probe_model.geom_quat[pgid], dtype=float)
    if not np.allclose(spec_quat0, [1.0, 0.0, 0.0, 0.0], atol=1e-8):
        # Math below assumes the xml-declared geom quat is identity; this is
        # the case for floor.xml. Generalizing would require unrotating the
        # auto-center offset by spec_quat0 first.
        raise NotImplementedError(
            f'rig geom {rig_geom_name!r} has non-identity xml quat '
            f'{spec_quat0.tolist()}; mesh auto-center compensation not '
            f'implemented for that case'
        )
    mesh_auto_pos = comp_pos0 - spec_pos0      # in body frame (= world here)
    mesh_auto_quat = comp_quat0                 # identity ⊗ auto_quat

    desired_compiled_pos = comp_pos0.copy()
    desired_compiled_quat = comp_quat0.copy()
    if rig_pos is not None:
        p = [float(x) for x in rig_pos]
        if len(p) == 2:
            desired_compiled_pos[0] = p[0]
            desired_compiled_pos[1] = p[1]
        elif len(p) == 3:
            desired_compiled_pos = np.asarray(p, dtype=float)
        else:
            raise ValueError(f'rig_pos must have length 2 or 3; got {len(p)}')
    if rig_quat is not None:
        q = [float(x) for x in rig_quat]
        if len(q) != 4:
            raise ValueError(
                f'rig_quat must have length 4 (w, x, y, z); got {len(q)}'
            )
        desired_compiled_quat = np.asarray(q, dtype=float)
        desired_compiled_quat /= np.linalg.norm(desired_compiled_quat)

    spec_quat = _quat_mul(desired_compiled_quat, _quat_conj(mesh_auto_quat))
    spec_pos = desired_compiled_pos - _quat_rotate(spec_quat, mesh_auto_pos)

    target.pos = spec_pos.tolist()
    target.quat = spec_quat.tolist()


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply scalar-first quaternion ``q`` to 3-vector ``v``."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def build_courtship_pair_visualizer(
    flybody_xml: str,
    floor_xml: str,
    settings_fly0: Optional[str] = None,
    settings_fly1: Optional[str] = None,
    root_body: str = 'thorax',
    spawn_pos=(0.0, 0.0, -0.005),
    rig_geom_name: str = 'Happy_house',
    rig_pos: Optional[Sequence[float]] = None,
    rig_quat: Optional[Sequence[float]] = None,
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
    rig_geom_name : str
        Name of the rig/arena geom in ``floor_xml`` whose pose may be
        overridden (e.g. ``'Happy_house'``). Only consulted when
        ``rig_pos`` or ``rig_quat`` is given.
    rig_pos : sequence of float or None
        Optional override for the rig geom's world xyz. Length-2 keeps the
        XML's original z; length-3 sets xyz fully.
    rig_quat : sequence of float or None
        Optional override for the rig geom's quaternion in MuJoCo's
        scalar-first ``(w, x, y, z)`` order.
    """
    import json
    import mujoco
    from mujoco_visualizer import Visualizer
    from mujoco_visualizer.render_settings import _resolve_settings_path

    fly_spec_0 = mujoco.MjSpec.from_file(flybody_xml)
    fly_spec_1 = mujoco.MjSpec.from_file(flybody_xml)
    floor_spec = mujoco.MjSpec.from_file(floor_xml)
    if rig_pos is not None or rig_quat is not None:
        _apply_rig_pose_override(
            floor_spec, floor_xml, rig_geom_name, rig_pos, rig_quat,
        )
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
        # axvspan Patches use a very pale fill but a saturated edgecolor —
        # prefer edgecolor so the legend text reads at full strength.
        for getter in ('get_color', 'get_edgecolor', 'get_facecolor'):
            if hasattr(h, getter):
                try:
                    c = getattr(h, getter)()
                except Exception:
                    c = None
                if c is not None:
                    break
        colors.append(c if c is not None else 'k')

    # Strip alpha so labels render at full opacity even when derived from
    # semi-transparent shading (facecolor 0x33) or invisible edgelines.
    solid_colors = []
    for c in colors:
        rgba = mpl.colors.to_rgba(c)
        if rgba[3] <= 0.0:
            solid_colors.append('k')
        else:
            solid_colors.append((rgba[0], rgba[1], rgba[2], 1.0))
    for txt, color in zip(leg.get_texts(), solid_colors):
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
    time_unit: str = 'ms',
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
    _ms_scale = 1e-3 if time_unit == 's' else 1.0

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
        s_ms = s / fs * 1000.0 * _ms_scale
        e_ms = e / fs * 1000.0 * _ms_scale

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
    min_segment_ms: float = 0.0,
    time_unit: str = 'ms',
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
    vk = {'lw': 1.0, 'alpha': 0.9, 'zorder': 4, **(pulse_vline_kwargs or {})}
    lg = {'loc': 'lower left', 'bbox_to_anchor': (0.38, 0.02),
          'ncols': 1, 'columnspacing': 0.6, **(legend_kwargs or {})}

    _ms_scale = 1e-3 if time_unit == 's' else 1.0

    seen: set = set()
    merged = _merge_segments_by_type([segments_L, segments_R])
    if min_segment_ms > 0:
        min_frames = max(1, int(round(float(min_segment_ms) * fs / 1000.0)))
        merged = [s for s in merged
                  if (int(s['end']) - int(s['start'])) >= min_frames]
    _shade_segments(ax, merged, fs, frame_range=frame_range, seen_labels=seen,
                    time_unit=time_unit)

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
            tm = float(f) / fs * 1000.0 * _ms_scale
            color = pt.get(lab, 'k')
            label = lab if lab not in seen else None
            seen.add(lab)
            ax.axvline(tm, color=color, label=label, **vk)

    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.plot(t_plot, wingL_z, color=wc['WingL_V13'], label='Wing L V13', **lk)
    ax.plot(t_plot, wingR_z, color=wc['WingR_V13'], label='Wing R V13', **lk)
    ax.set_ylabel('Wing tip\nz (mm)')
    if t_plot.size:
        ax.set_xlim(0.0, float(t_plot[-1]))

    # Keep only song-type (Sine/Pulse) and wing-trace labels; Pslow/Pfast
    # per-pulse markers are visible as vertical lines but not in the legend.
    _keep = {'Sine', 'Pulse', 'Wing L V13', 'Wing R V13'}
    _h, _l = ax.get_legend_handles_labels()
    _pairs = [(h, l) for h, l in zip(_h, _l) if l in _keep]
    if _pairs:
        _hs, _ls = zip(*_pairs)
        _colored_text_legend(ax, handles_labels=(list(_hs), list(_ls)), **lg)
    else:
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
    time_unit: str = 'ms',
) -> None:
    """Plot scutellum (body) z-position over the same time interval.

    When ``pulse_peak_frames`` + ``pulse_subtype_labels`` are provided, pulse
    segments are shaded by dominant Pslow/Pfast type.
    """
    lk = {'lw': 0.7, **(line_kwargs or {})}
    _ms_scale = 1e-3 if time_unit == 's' else 1.0
    if segments is not None:
        _shade_segments(ax, segments, fs, frame_range=frame_range,
                        pulse_peak_frames=pulse_peak_frames,
                        pulse_subtype_labels=pulse_subtype_labels,
                        pulse_type_colors=pulse_type_colors,
                        time_unit=time_unit)
    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.plot(t_plot, scutellum_z, color=line_color, **lk)
    ax.set_xlabel('Time (s)' if time_unit == 's' else 'Time (ms)')
    ax.set_ylabel('Scutellum\nz (mm)')


# -----------------------------------------------------------------------------
# Row 4 right: male body pitch vs. target pitch
# -----------------------------------------------------------------------------

def body_pitch_deg_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body-axis elevation in world frame from a MuJoCo quaternion.

    ``quat_wxyz`` is ``(..., 4)`` in ``[w, x, y, z]`` order (MuJoCo free-joint
    convention). The fly model's local ``+X`` is anterior, so the world-frame
    forward vector's z component is ``2*(qx*qz - qw*qy)``; its arcsin is the
    pitch-up angle of the thorax (positive = nose up).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    sin_p = 2.0 * (q[..., 1] * q[..., 3] - q[..., 0] * q[..., 2])
    return np.degrees(np.arcsin(np.clip(sin_p, -1.0, 1.0)))


def panel_male_pitch(
    ax: plt.Axes,
    t_ms: np.ndarray,
    male_pitch_deg: np.ndarray,
    target_pitch_deg: np.ndarray,
    segments: Optional[Iterable[dict]] = None,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    min_segment_ms: float = 10.0,
    male_color: str = '#d62728',
    target_color: str = '#1f77b4',
    zero_line_color: str = '#bdbdbd',
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = '',
    time_unit: str = 'ms',
) -> None:
    """Plot male thorax pitch (red) and target pitch to the female (blue).

    ``male_pitch_deg`` is the thorax body-axis elevation (positive = nose up)
    and ``target_pitch_deg`` is the elevation of the male-scutellum → female-COM
    vector. Both are degrees; where the two traces overlap, the male is aimed
    at the female. Song segments are shaded behind the traces via the same
    `_shade_segments` helper as Panel B; ``min_segment_ms`` filters ultra-short
    detections.
    """
    lk = {'lw': 0.8, **(line_kwargs or {})}
    lg = {'loc': 'upper left', 'ncols': 2,
          'columnspacing': 0.6, **(legend_kwargs or {})}
    _ms_scale = 1e-3 if time_unit == 's' else 1.0

    if segments is not None:
        segs = list(segments)
        if min_segment_ms > 0:
            min_frames = max(1, int(round(float(min_segment_ms) * fs / 1000.0)))
            segs = [s for s in segs
                    if (int(s['end']) - int(s['start'])) >= min_frames]
        _shade_segments(ax, segs, fs, frame_range=frame_range,
                        time_unit=time_unit)

    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.axhline(0.0, color=zero_line_color, lw=0.6, zorder=1)
    ax.plot(t_plot, male_pitch_deg, color=male_color, linestyle='-',
            label='Male', zorder=3, **lk)
    ax.plot(t_plot, target_pitch_deg, color=target_color, linestyle='-',
            label='Female', zorder=2, **lk)
    ax.set_xlabel('Time (s)' if time_unit == 's' else 'Time (ms)')
    ax.set_ylabel('Pitch (°)')
    if t_plot.size:
        ax.set_xlim(0.0, float(t_plot[-1]))
    if title:
        ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)


def panel_pitch_alignment_violin(
    ax: plt.Axes,
    per_bout_values: Sequence[float],
    exemplar_idx: Optional[int] = None,
    violin_color: str = '#c0c0c0',
    dot_color: str = '#555555',
    exemplar_color: str = '#d62728',
    jitter_width: float = 0.12,
    rng_seed: int = 0,
    title: str = '',
) -> None:
    """Violin + per-bout dots of median |pitch alignment| across bouts.

    Each dot is one bout's median absolute alignment (degrees); the
    violin shows the distribution across all bouts. ``exemplar_idx``
    highlights that bout in ``exemplar_color``.
    """
    vals = np.asarray(per_bout_values, dtype=float)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    if finite.size >= 2:
        parts = ax.violinplot(
            finite, positions=[0], widths=0.7, showextrema=False,
            showmedians=False,
        )
        for body in parts['bodies']:
            body.set_facecolor(violin_color)
            body.set_edgecolor('none')
            body.set_alpha(0.55)
    rng = np.random.default_rng(rng_seed)
    jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
    colors = [
        exemplar_color if (exemplar_idx is not None and i == int(exemplar_idx))
        else dot_color
        for i in range(vals.size)
    ]
    sizes = [
        20.0 if (exemplar_idx is not None and i == int(exemplar_idx))
        else 10.0
        for i in range(vals.size)
    ]
    zorders = [
        4 if (exemplar_idx is not None and i == int(exemplar_idx))
        else 3
        for i in range(vals.size)
    ]
    for i in range(vals.size):
        if not finite_mask[i]:
            continue
        ax.scatter(
            jitter[i], vals[i], s=sizes[i], c=colors[i],
            edgecolors='k', linewidths=0.3, zorder=zorders[i],
        )
    if finite.size:
        med = float(np.median(finite))
        ax.hlines(med, -0.35, 0.35, color='k', lw=0.8, zorder=5)
    ax.set_xticks([0])
    ax.set_xticklabels([f'n={int(finite.size)}'])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel('|Pitch align| (°)')
    if title:
        ax.set_title(title, pad=2)


# -----------------------------------------------------------------------------
# Row 3 right: singing vs free-running z-height
# -----------------------------------------------------------------------------

def panel_z_height_singing_vs_running(
    ax: plt.Axes,
    pulse_z: np.ndarray,
    sine_z: np.ndarray,
    running_z: np.ndarray,
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
    """Compare scutellum z-height during pulse, sine, and free running.

    ``colors`` overrides the default (pulse/sine/running) triplet. When
    ``show_points`` is True (default), raw samples are jittered and scattered
    on top of each box/violin.
    """
    p = np.asarray(pulse_z,   dtype=float); p = p[np.isfinite(p)]
    s = np.asarray(sine_z,    dtype=float); s = s[np.isfinite(s)]
    w = np.asarray(running_z, dtype=float); w = w[np.isfinite(w)]
    all_data = [p, s, w]
    all_labels = [
        f'pulse\n(n={p.size})',
        f'sine\n(n={s.size})',
        f'free running\n(n={w.size})',
    ]
    all_cols = list(colors) if colors else [
        SONG_COLORS['pulse'], SONG_COLORS['sine'], '#888888',
    ]
    all_positions = [0, 1, 2]

    # matplotlib's violinplot/boxplot raise on zero-size arrays. Drop empty
    # groups so the panel still renders if one source returns no samples.
    keep = [vals.size > 0 for vals in all_data]
    data      = [v for v, k in zip(all_data,      keep) if k]
    labels    = [l for l, k in zip(all_labels,    keep) if k]
    cols      = [c for c, k in zip(all_cols,      keep) if k]
    positions = [p_ for p_, k in zip(all_positions, keep) if k]

    if show_points:
        pk = {'s': 6, 'linewidths': 0.3, 'edgecolors': 'k',
              'alpha': 0.7, 'zorder': 1, **(point_kwargs or {})}
        rng = np.random.default_rng(rng_seed)
        for pos, vals, c in zip(positions, data, cols):
            jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
            ax.scatter(pos + jitter, vals, c=c, **pk)

    if not data:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels)
        ax.set_ylabel('Scutellum z (mm)')
        ax.set_title(title, pad=2)
        return

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

    # Always show all three x positions for layout stability, even if one
    # group is empty (its label still reflects n=0).
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel('Scutellum z (mm)')
    ax.set_title(title, pad=2)


# -----------------------------------------------------------------------------
# Sine in-phase: extended vs folded wing z-traces over a sine bout
# -----------------------------------------------------------------------------

WING_PHASE_COLORS: Dict[str, str] = {
    'extended': WING_COLORS['WingR_V13'],  # blue   (matches Wing R)
    'folded':   WING_COLORS['WingL_V13'],  # purple (matches Wing L)
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
    lg = {'loc': 'upper center', **(legend_kwargs or {})}

    ax.plot(t_ms, wing_extended_z, color=cc['extended'],
            label='Extending', **lk)
    ax.plot(t_ms, wing_folded_z, color=cc['folded'],
            label='Folding', **lk)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Wing tip z (mm)')
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
    center_stat: str = 'mean',
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
    mean_vector : if True, draw a marker on the rim at the center direction.
    center_stat : ``'mean'`` (default) uses the circular mean arg(mean(e^{iθ}));
        ``'median'`` uses the circular median (angle minimising the sum of
        circular distances Σ π − |π − |θᵢ − α||).
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
        stat = str(center_stat).lower()
        if stat == 'median':
            # Circular median: angle α minimising Σ (π − |π − |xᵢ − α||).
            diffs = np.abs(x[:, None] - x[None, :])
            dist = np.pi - np.abs(np.pi - diffs)
            r_ang = float(x[int(np.argmin(dist.sum(axis=1)))])
        elif stat == 'mean':
            r_ang = float(np.angle(np.mean(np.exp(1j * x))))
        else:
            raise ValueError(
                f"center_stat must be 'mean' or 'median', got {center_stat!r}")
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
