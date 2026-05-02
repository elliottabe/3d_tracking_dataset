"""Walking + Courtship Bout Curation GUI
========================================
Extended version of walking_bout_curation_gui.py.

New features:
  • Walking AND courtship mode (radio selector)
  • Courtship: shows wing tip Y/Z traces alongside leg tips, and auto-detects
    per-fly data3D_fly0/fly1 CSV format
  • Scrollable fly list in sidebar with per-fly "completed" checkbox and
    modification count
  • Per-fly modification tracking across fly switches
  • Hideable Gantt timeline
  • Erase current bout / Combine with next bout (auto-save)
  • Skeleton animation in two stacked rows (bigger views)

Run with:
    streamlit run walking_courtship_curation_gui.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
FPS            = 800
SCALE          = 10.0
CONTEXT_FRAMES = 250
VIEW_HALF      = 2000

LEG_TIPS = [
    "T1L_TaTip", "T1R_TaTip",
    "T2L_TaTip", "T2R_TaTip",
    "T3L_TaTip", "T3R_TaTip",
]
LEG_COLORS = {
    "T1L_TaTip": "#74c476",
    "T1R_TaTip": "#1b7837",
    "T2L_TaTip": "#762a83",
    "T2R_TaTip": "#e08214",
    "T3L_TaTip": "#2166ac",
    "T3R_TaTip": "#d73027",
}
WING_TIPS = ["WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]
WING_COLORS = {
    "WingL_V12": "#9ecae1",
    "WingL_V13": "#3182bd",
    "WingR_V12": "#fdae6b",
    "WingR_V13": "#e6550d",
}
COLOR_DEFAULT         = "#4878cf"
COLOR_MODIFIED        = "#f0a500"
COLOR_SELECTED_BORDER = "#222"
CONF_THRESHOLD        = 0.8

_DEFAULT_INDEX = (
    "/home/user/src/JARVIS-HybridNet/projects/merge_courtship_V3/"
    "predictions/predictions3D/predictions_index.csv"
)

# ── Skeleton topology ──────────────────────────────────────────────────────────
_LEG_CHAINS = [
    ["T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip"],
    ["T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"],
    ["T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip"],
    ["T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip"],
    ["T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip"],
    ["T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip"],
]
_BODY_CHAINS = [
    ["EyeL", "Antenna_Base", "EyeR"],
    ["Antenna_Base", "Scutellum"],
    ["Scutellum", "Abd_A4", "Abd_tip"],
]
_COXA_KPS = {
    f"T{n}{s}_ThxCx": LEG_COLORS.get(f"T{n}{s}_TaTip", "#999")
    for n in "123" for s in "LR"
}


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ── Data I/O ───────────────────────────────────────────────────────────────────

def load_bouts_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_bouts_csv(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _curated_csv(bouts_csv: str) -> Path:
    """Return the working copy path (original is never modified)."""
    p = Path(bouts_csv)
    return p.parent / (p.stem + ".curated" + p.suffix)


def load_3d_data(csv_path: str):
    """Load data3D[_flyN].csv given its full path. Returns (df, kp_names)."""
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df = df.iloc[:-1].reset_index(drop=True)
    seen, kp_names = set(), []
    for col in df.columns:
        base = col.split(".")[0]
        if base not in seen:
            seen.add(base)
            kp_names.append(base)
    return df, kp_names


def _build_kp_pos(df: pd.DataFrame, kp_names: list) -> dict:
    """Pre-extract all keypoints → {kp: np.array(n, 4)} [x/SCALE, y/SCALE, z/SCALE, conf]."""
    kp_pos = {}
    for kp in kp_names:
        cols = [c for c in df.columns if c.split(".")[0] == kp]
        if len(cols) < 3:
            continue
        xyz  = df[cols[:3]].to_numpy(dtype=float) / SCALE
        conf = (df[cols[3]].to_numpy(dtype=float)[:, None]
                if len(cols) >= 4 else np.full((len(xyz), 1), np.nan))
        kp_pos[kp] = np.concatenate([xyz, conf], axis=1)
    return kp_pos


# ── Fly entry builder ──────────────────────────────────────────────────────────

def build_fly_entries(index_path: str, mode: str) -> list:
    """Return flat list of fly_entry dicts from predictions_index.csv.

    Each dict: {key, display_id, bouts_csv, data3d_csv, folder}
    Entries whose bouts_csv does not exist are silently skipped.
    """
    prefix = "walking_bouts" if mode == "Walking" else "courtship_bouts"
    entries = []
    try:
        index_df = pd.read_csv(index_path)
    except Exception:
        return entries

    for _, row in index_df.iterrows():
        folder = Path(str(row["prediction_folder"]))
        fly_id = str(row["fly_id"])

        fly0_data = folder / "data3D_fly0.csv"
        fly1_data = folder / "data3D_fly1.csv"

        if fly0_data.exists() and fly1_data.exists():
            # Two-fly format: separate CSV per fly
            for suffix, d3d in [("_fly0", fly0_data), ("_fly1", fly1_data)]:
                bouts_csv = folder / f"{prefix}{suffix}_summary.csv"
                if not bouts_csv.exists() or bouts_csv.stat().st_size < 2:
                    continue
                entries.append({
                    "key":        str(bouts_csv),
                    "display_id": f"{fly_id} [{suffix[1:]}]",
                    "bouts_csv":  str(bouts_csv),
                    "data3d_csv": str(d3d),
                    "folder":     str(folder),
                })
        else:
            # Single-fly format
            bouts_csv = folder / f"{prefix}_summary.csv"
            if not bouts_csv.exists() or bouts_csv.stat().st_size < 2:
                continue
            d3d = folder / "data3D.csv"
            entries.append({
                "key":        str(bouts_csv),
                "display_id": fly_id,
                "bouts_csv":  str(bouts_csv),
                "data3d_csv": str(d3d),
                "folder":     str(folder),
            })

    return entries


# ── Session state ──────────────────────────────────────────────────────────────

def _init_ss():
    ss = st.session_state
    ss.setdefault("flies",            {})
    ss.setdefault("completed_flies",  set())
    ss.setdefault("current_fly_key",  None)
    ss.setdefault("current_bout_idx", 0)
    ss.setdefault("_mode",            "Walking")
    ss.setdefault("_index_path",      "")


def _fly(key: str) -> dict:
    """Return per-fly sub-dict, creating an empty shell if absent."""
    if key not in st.session_state["flies"]:
        st.session_state["flies"][key] = {
            "bouts_df":  None,
            "modified":  set(),
            "kp_pos":    None,
            "kp_loaded": False,
        }
    return st.session_state["flies"][key]


def _modified_sidecar(bouts_csv: str) -> Path:
    return Path(bouts_csv).with_suffix(".modified.json")


def _load_modified_sidecar(bouts_csv: str, bouts_df: pd.DataFrame) -> set:
    """Return set of list-positions that are recorded as modified on disk."""
    mf = _modified_sidecar(bouts_csv)
    if not mf.exists():
        return set()
    try:
        saved_ids = set(json.loads(mf.read_text()))
        return {i for i, row in bouts_df.iterrows()
                if int(row["bout_idx"]) in saved_ids}
    except Exception:
        return set()


def _save_modified_sidecar(bouts_csv: str, bouts_df: pd.DataFrame, modified: set):
    """Persist modified list-positions → bout_idx values to sidecar JSON."""
    bout_ids = [int(bouts_df.iloc[i]["bout_idx"])
                for i in sorted(modified) if i < len(bouts_df)]
    _modified_sidecar(bouts_csv).write_text(json.dumps(bout_ids))


def _ensure_loaded(entry: dict):
    """Lazy-load bouts_df and kp_pos for a fly entry."""
    import shutil
    f       = _fly(entry["key"])
    curated = _curated_csv(entry["bouts_csv"])
    if f["bouts_df"] is None:
        if not curated.exists():
            shutil.copy2(entry["bouts_csv"], curated)
        df = load_bouts_csv(str(curated))
        f["bouts_df"] = df
        f["modified"] = _load_modified_sidecar(str(curated), df)
    if not f["kp_loaded"]:
        with st.spinner(f"Loading keypoints for {entry['display_id']}…"):
            df_raw, kp_names = load_3d_data(entry["data3d_csv"])
            f["kp_pos"]    = _build_kp_pos(df_raw, kp_names)
            f["kp_loaded"] = True


def _mark_modified(key: str, idx: int, bouts_csv: str = None):
    f = st.session_state["flies"][key]
    f["modified"].add(idx)
    if bouts_csv:
        _save_modified_sidecar(bouts_csv, f["bouts_df"], f["modified"])


def _clear_bout_state(from_idx: int):
    """Delete stale per-bout slider/view keys at index >= from_idx."""
    prefixes = ("sl_start_", "sl_end_", "vc_", "nav_", "ni_start_", "ni_end_", "bounds_")
    to_del = []
    for k in list(st.session_state.keys()):
        for pfx in prefixes:
            if k.startswith(pfx):
                try:
                    num = int(k[len(pfx):].split("_")[0])
                    if num >= from_idx:
                        to_del.append(k)
                except ValueError:
                    pass
    for k in to_del:
        st.session_state.pop(k, None)
    st.session_state.pop("bout_selectbox", None)
    st.session_state.pop("anim_bout_key", None)


# ── Bout mutations ─────────────────────────────────────────────────────────────

def _do_erase(fly_key: str, idx: int, bouts_csv: str):
    """Remove row at idx, renumber bouts, auto-save to curated copy."""
    f       = _fly(fly_key)
    curated = str(_curated_csv(bouts_csv))
    df = f["bouts_df"].copy()
    df = df.drop(index=idx).reset_index(drop=True)
    df["bout_idx"] = range(1, len(df) + 1)
    f["bouts_df"]  = df
    f["modified"] = {i - 1 if i > idx else i
                     for i in f["modified"] if i != idx}
    save_bouts_csv(curated, df)
    _save_modified_sidecar(curated, df, f["modified"])
    _clear_bout_state(0)
    st.session_state["current_bout_idx"] = min(idx, max(0, len(df) - 1))
    st.toast("Bout erased and saved.", icon="🗑️")


def _do_combine(fly_key: str, idx: int, bouts_csv: str):
    """Merge bout idx with idx+1, auto-save to curated copy."""
    f       = _fly(fly_key)
    curated = str(_curated_csv(bouts_csv))
    df = f["bouts_df"].copy()
    if idx >= len(df) - 1:
        return
    new_start = int(min(df.at[idx, "start_frame"], df.at[idx + 1, "start_frame"]))
    new_end   = int(max(df.at[idx, "end_frame"],   df.at[idx + 1, "end_frame"]))
    df.at[idx, "start_frame"] = new_start
    df.at[idx, "end_frame"]   = new_end
    df.at[idx, "n_frames"]    = new_end - new_start + 1
    df.at[idx, "duration_s"]  = (new_end - new_start + 1) / FPS
    df = df.drop(index=idx + 1).reset_index(drop=True)
    df["bout_idx"] = range(1, len(df) + 1)
    f["bouts_df"]  = df
    f["modified"] = {i - 1 if i > idx + 1 else i
                     for i in f["modified"] if i != idx + 1}
    f["modified"].add(idx)
    save_bouts_csv(curated, df)
    _save_modified_sidecar(curated, df, f["modified"])
    _clear_bout_state(0)
    st.toast("Bouts combined and saved.", icon="🔗")


# ── Figure builders ────────────────────────────────────────────────────────────

def make_timeline_fig(bouts_df, modified_rows, selected_idx=None):
    """Gantt-style bar chart — one bar per bout."""
    fig = go.Figure()
    for i, row in enumerate(bouts_df.itertuples()):
        is_sel = i == selected_idx
        color  = COLOR_MODIFIED if i in modified_rows else COLOR_DEFAULT
        dur    = row.end_frame - row.start_frame
        hover  = (
            f"Bout {row.bout_idx}<br>"
            f"Start: {row.start_frame}<br>"
            f"End: {row.end_frame}<br>"
            f"Duration: {dur} frames ({row.duration_s:.2f} s)<br>"
            f"Speed: {row.mean_speed_mm_s:.1f} mm/s"
            f"{'<br><i>MODIFIED</i>' if i in modified_rows else ''}"
        )
        fig.add_trace(go.Bar(
            x=[dur], y=[i], base=[row.start_frame],
            orientation="h",
            marker_color=color,
            marker_line_color=COLOR_SELECTED_BORDER if is_sel else color,
            marker_line_width=3 if is_sel else 0,
            opacity=1.0 if is_sel else 0.75,
            hovertext=hover, hoverinfo="text", showlegend=False,
        ))

    mids = [(r.start_frame + r.end_frame) / 2 for r in bouts_df.itertuples()]
    fig.add_trace(go.Scatter(
        x=mids, y=list(range(len(bouts_df))), mode="markers",
        marker=dict(size=24, opacity=0.01, color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))

    n = len(bouts_df)
    fig.update_layout(
        height=max(200, min(30 * n + 80, 500)),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis_title="Frame", yaxis_title="Bout",
        barmode="overlay", plot_bgcolor="#f8f8f8",
        title="Click a bout · Blue = unmodified · Yellow = modified",
        title_font_size=12,
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n)),
        ticktext=[str(i) for i in range(n)], autorange="reversed",
    )
    return fig


def make_trace_fig(kp_pos, bout_start, bout_end, fps=FPS,
                   view_start=None, view_end=None, show_wings=False):
    """Trace figure: leg Y/Z + confidence (walking); adds wing Y/Z panels (courtship).

    In courtship mode wings get their own rows with Y-axis range locked to the
    data range *within the current bout*, so small oscillations are always visible.
    """
    n_frames = max(len(arr) for arr in kp_pos.values()) if kp_pos else 1
    if view_start is None:
        view_start = max(0, bout_start - CONTEXT_FRAMES)
    if view_end is None:
        view_end = min(n_frames - 1, bout_end + CONTEXT_FRAMES)

    plot_start, plot_end = int(view_start), int(view_end)
    frames = np.arange(plot_start, plot_end + 1)

    available_wings = [w for w in WING_TIPS if w in kp_pos] if show_wings else []

    # ── Wing bout-range (locked to bout frames only) ──────────────────────────
    wing_y_range = wing_z_range = None
    if available_wings:
        bout_f = np.arange(bout_start, min(bout_end + 1, n_frames))
        wy_all, wz_all = [], []
        for w in available_wings:
            bf = bout_f[bout_f < len(kp_pos[w])]
            if len(bf):
                wy_all.append(kp_pos[w][bf, 1])
                wz_all.append(kp_pos[w][bf, 2])
        if wy_all:
            pad = 0.3
            wy = np.concatenate(wy_all)
            wz = np.concatenate(wz_all)
            wing_y_range = [float(np.nanmin(wy)) - pad, float(np.nanmax(wy)) + pad]
            wing_z_range = [float(np.nanmin(wz)) - pad, float(np.nanmax(wz)) + pad]

    # ── Layout: 3 rows (walking) or 5 rows (courtship) ───────────────────────
    if available_wings:
        n_rows      = 5
        row_heights = [0.22, 0.22, 0.20, 0.20, 0.16]
        titles      = ("Y — leg tips (mm)", "Z — leg tips (mm)",
                       "Y — wing tips (mm, bout range)", "Z — wing tips (mm, bout range)",
                       "Mean confidence")
        conf_row    = 5
        conf_yref   = "y5"
    else:
        n_rows      = 3
        row_heights = [0.38, 0.38, 0.24]
        titles      = ("Y — leg tips (mm)", "Z — leg tips (mm)", "Mean confidence")
        conf_row    = 3
        conf_yref   = "y3"

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.06,
        row_heights=row_heights,
    )

    # ── Leg tips ──────────────────────────────────────────────────────────────
    for leg in [l for l in LEG_TIPS if l in kp_pos]:
        arr     = kp_pos[leg]
        clipped = frames[frames < len(arr)]
        color   = LEG_COLORS[leg]
        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 1], mode="lines", name=leg,
            line=dict(color=color, width=1.5), legendgroup=leg,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 2], mode="lines", name=leg,
            line=dict(color=color, width=1.5),
            legendgroup=leg, showlegend=False,
        ), row=2, col=1)

    # ── Wing tips — own rows with bout-locked range ───────────────────────────
    for wing in available_wings:
        arr     = kp_pos[wing]
        clipped = frames[frames < len(arr)]
        color   = WING_COLORS[wing]
        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 1], mode="lines", name=wing,
            line=dict(color=color, width=1.5), legendgroup=wing,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 2], mode="lines", name=wing,
            line=dict(color=color, width=1.5),
            legendgroup=wing, showlegend=False,
        ), row=4, col=1)

    # ── Mean confidence ───────────────────────────────────────────────────────
    conf_arrays = []
    for kp, arr in kp_pos.items():
        if arr.shape[1] >= 4:
            clipped_c = frames[frames < len(arr)]
            conf_arrays.append(arr[clipped_c, 3])
    if conf_arrays:
        min_len   = min(len(a) for a in conf_arrays)
        clipped_f = frames[frames < n_frames][:min_len]
        mean_conf = np.mean([a[:min_len] for a in conf_arrays], axis=0)
        fig.add_trace(go.Scatter(
            x=clipped_f, y=mean_conf, mode="lines", name="mean conf",
            line=dict(color="#555", width=1.2),
            fill="tozeroy", fillcolor="rgba(100,100,100,0.08)", showlegend=False,
        ), row=conf_row, col=1)

    dur_s = (bout_end - bout_start) / fps
    shapes = [
        dict(type="rect", xref="x", yref="paper",
             x0=bout_start, x1=bout_end, y0=0, y1=1,
             fillcolor="rgba(44,160,44,0.10)", layer="below", line=dict(width=0)),
        dict(type="line", xref="x", yref="paper",
             x0=bout_start, x1=bout_start, y0=0, y1=1,
             line=dict(color="#2ca02c", dash="dash", width=2)),
        dict(type="line", xref="x", yref="paper",
             x0=bout_end, x1=bout_end, y0=0, y1=1,
             line=dict(color="#d62728", dash="dash", width=2)),
        dict(type="line", xref="paper", yref=conf_yref,
             x0=0, x1=1, y0=CONF_THRESHOLD, y1=CONF_THRESHOLD,
             line=dict(color="red", dash="dot", width=1)),
    ]
    fig.add_annotation(
        x=1, y=CONF_THRESHOLD, xref="paper", yref=conf_yref,
        text=f"thr {CONF_THRESHOLD}", showarrow=False,
        font=dict(size=9, color="red"), xanchor="right",
    )
    fig.update_layout(
        shapes=shapes,
        height=950 if available_wings else 700,
        margin=dict(l=55, r=20, t=50, b=40),
        hovermode="x", template="none",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=(f"Start {bout_start} → End {bout_end} "
                  f"({bout_end - bout_start + 1} frames, {dur_s:.2f} s)"),
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(range=[plot_start, plot_end], autorange=False)
    fig.update_xaxes(title_text="Frame", row=conf_row, col=1)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=1, range=[-1, 6])
    fig.update_yaxes(title_text="Z (mm)", row=2, col=1, range=[-1, 4])
    if available_wings:
        fig.update_yaxes(title_text="Wing Y (mm)", row=3, col=1, range=[-1, 6])
        fig.update_yaxes(title_text="Wing Z (mm)", row=4, col=1, range=[-1, 4])
    fig.update_yaxes(title_text="Confidence", row=conf_row, col=1, range=[0, 1.05])
    return fig


def _skeleton_proj(kp_pos, fi, available, coord_a, coord_b):
    as_, bs_ = [], []
    for chain in _LEG_CHAINS + _BODY_CHAINS:
        for kp in chain:
            if kp in available and fi < len(kp_pos[kp]):
                as_.append(float(kp_pos[kp][fi, coord_a]))
                bs_.append(float(kp_pos[kp][fi, coord_b]))
            else:
                as_.append(None)
                bs_.append(None)
        as_.append(None)
        bs_.append(None)
    return as_, bs_


def make_animation_fig(kp_pos, bout_start, bout_end, fps=FPS, step=8):
    """Stacked animated skeleton: row 1 = XY top-down, row 2 = XZ side view."""
    n_frames   = max(len(arr) for arr in kp_pos.values()) if kp_pos else 1
    plot_start = max(0, bout_start - CONTEXT_FRAMES)
    plot_end   = min(n_frames - 1, bout_end + CONTEXT_FRAMES)
    indices    = list(range(plot_start, plot_end + 1, step))

    available = set(kp_pos)
    kp_list   = sorted(available)
    pt_colors = [
        LEG_COLORS.get(kp, WING_COLORS.get(kp, _COXA_KPS.get(kp, "#aaa")))
        for kp in kp_list
    ]
    pt_sizes = [
        8 if kp in LEG_COLORS else (6 if kp in WING_COLORS else 5)
        for kp in kp_list
    ]

    x_range = [0, 22]
    y_range = [-1, 6]
    z_range = [0, 5]

    def frame_data(fi):
        xs = [float(kp_pos[kp][fi, 0]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        ys = [float(kp_pos[kp][fi, 1]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        zs = [float(kp_pos[kp][fi, 2]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        sk_xy_x, sk_xy_y = _skeleton_proj(kp_pos, fi, available, 0, 1)
        sk_xz_x, sk_xz_z = _skeleton_proj(kp_pos, fi, available, 0, 2)
        in_bout = bout_start <= fi <= bout_end
        border  = "rgba(44,160,44,0.9)" if in_bout else "rgba(160,160,160,0.6)"
        mk = dict(color=pt_colors, size=pt_sizes, line=dict(color=border, width=1.5))
        # Order must match fig.add_trace order: [sk_xy, pts_xy, sk_xz, pts_xz]
        return [
            go.Scatter(x=sk_xy_x, y=sk_xy_y, mode="lines",
                       line=dict(color="#888", width=1.5), hoverinfo="skip", showlegend=False),
            go.Scatter(x=xs, y=ys, mode="markers", marker=mk,
                       text=kp_list, hoverinfo="text", showlegend=False),
            go.Scatter(x=sk_xz_x, y=sk_xz_z, mode="lines",
                       line=dict(color="#888", width=1.5), hoverinfo="skip", showlegend=False),
            go.Scatter(x=xs, y=zs, mode="markers", marker=mk,
                       text=kp_list, hoverinfo="text", showlegend=False),
        ]

    # Two stacked rows — each gets full width
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Top-down (XY)", "Side view (XZ)"],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.10,
    )
    init = frame_data(indices[0])
    fig.add_trace(init[0], row=1, col=1)
    fig.add_trace(init[1], row=1, col=1)
    fig.add_trace(init[2], row=2, col=1)
    fig.add_trace(init[3], row=2, col=1)

    anim_frames = [
        go.Frame(
            data=frame_data(fi),
            layout=go.Layout(title=dict(
                text=(f"Frame {fi}  ·  step={step}  ·  "
                      f"{'IN BOUT' if bout_start <= fi <= bout_end else 'context'}"),
                font=dict(size=11),
            )),
            name=str(fi),
        )
        for fi in indices
    ]
    fig.frames = anim_frames

    sliders = [{
        "steps": [
            {"args": [[f.name], {"frame": {"duration": 50, "redraw": True},
                                  "transition": {"duration": 0}}],
             "method": "animate", "label": ""}
            for f in anim_frames
        ],
        "active": 0,
        "x": 0.0, "len": 1.0, "y": -0.04,
        "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
        "transition": {"duration": 0},
    }]

    fig.update_layout(
        height=920,
        margin=dict(l=50, r=20, t=60, b=120),
        plot_bgcolor="#f0f0f0",
        showlegend=False,
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.5, "y": -0.08, "xanchor": "center", "yanchor": "top",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 50, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        sliders=sliders,
        title=dict(
            text=(f"Frame {indices[0]}  ·  step={step}  ·  "
                  f"{fps // step} Hz  ·  green border = in bout"),
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(title_text="X (mm)", range=x_range, row=1, col=1)
    fig.update_yaxes(title_text="Y (mm)", range=y_range, row=1, col=1)
    fig.update_xaxes(title_text="X (mm)", range=x_range, row=2, col=1)
    fig.update_yaxes(title_text="Z (mm)", range=z_range, row=2, col=1)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(fly_entries: list) -> dict | None:
    """Render sidebar and return the active fly_entry, or None."""
    ss = st.session_state

    with st.sidebar:
        st.header("Bout Curation")

        # ── Mode ──────────────────────────────────────────────────────────────
        mode = st.radio("Mode", ["Walking", "Courtship"],
                        index=0 if ss["_mode"] == "Walking" else 1,
                        key="mode_radio", horizontal=True)
        if mode != ss["_mode"]:
            ss["_mode"]            = mode
            ss["current_fly_key"]  = None
            ss["current_bout_idx"] = 0
            _clear_bout_state(0)
            _rerun()

        # ── Index path ────────────────────────────────────────────────────────
        index_path = st.text_input(
            "predictions_index.csv",
            value=ss["_index_path"] or _DEFAULT_INDEX,
            key="index_path_input",
        )
        if index_path != ss["_index_path"]:
            ss["_index_path"]      = index_path
            ss["current_fly_key"]  = None
            ss["current_bout_idx"] = 0
            _clear_bout_state(0)
            _rerun()

        if not Path(index_path).exists():
            st.error("File not found.")
            return None

        if not fly_entries:
            st.warning(
                f"No {mode.lower()} bout CSVs found.\n\n"
                "Run `batch_bout_detection.py` first to generate summary files."
            )
            return None

        # Auto-select first fly when none is selected
        entry_keys = [e["key"] for e in fly_entries]
        if ss["current_fly_key"] not in entry_keys:
            ss["current_fly_key"]  = entry_keys[0]
            ss["current_bout_idx"] = 0

        # ── Global summary ────────────────────────────────────────────────────
        n_done      = sum(1 for e in fly_entries if e["key"] in ss["completed_flies"])
        n_mod_flies = sum(
            1 for e in fly_entries
            if ss["flies"].get(e["key"], {}).get("modified")
        )
        st.caption(
            f"{len(fly_entries)} flies · "
            f"{n_done} completed · "
            f"{n_mod_flies} with edits"
        )
        st.divider()
        st.markdown("**✓** done · **✏** edited · **▶** current")

        # ── Scrollable fly list ───────────────────────────────────────────────
        list_height = min(60 * len(fly_entries) + 20, 500)
        with st.container(height=list_height):
            for entry in fly_entries:
                key       = entry["key"]
                key_hash  = str(hash(key) & 0xFFFFFF)
                fly_data  = ss["flies"].get(key, {})
                n_mods    = len(fly_data.get("modified", set()))
                is_done   = key in ss["completed_flies"]
                is_active = key == ss["current_fly_key"]

                col_chk, col_btn = st.columns([0.12, 0.88])

                with col_chk:
                    new_done = st.checkbox(
                        "", value=is_done,
                        key=f"done_{key_hash}",
                        label_visibility="collapsed",
                        help="Mark as completed",
                    )
                    if new_done != is_done:
                        if new_done:
                            ss["completed_flies"].add(key)
                        else:
                            ss["completed_flies"].discard(key)

                with col_btn:
                    prefix = "▶ " if is_active else ("✓ " if is_done else "  ")
                    suffix = f" ✏({n_mods})" if n_mods > 0 else ""
                    label  = f"{prefix}{entry['display_id']}{suffix}"
                    if st.button(label, key=f"sel_{key_hash}",
                                 use_container_width=True):
                        if key != ss["current_fly_key"]:
                            ss["current_fly_key"]  = key
                            ss["current_bout_idx"] = 0
                            _clear_bout_state(0)
                            _rerun()

        # ── Reload button ─────────────────────────────────────────────────────
        active_entry = next(
            (e for e in fly_entries if e["key"] == ss["current_fly_key"]), None
        )
        if active_entry:
            st.divider()
            if st.button("Reload from disk", use_container_width=True):
                k       = active_entry["key"]
                curated = str(_curated_csv(active_entry["bouts_csv"]))
                df = load_bouts_csv(curated)
                ss["flies"][k]["bouts_df"] = df
                ss["flies"][k]["modified"] = _load_modified_sidecar(curated, df)
                ss["current_bout_idx"]     = 0
                _clear_bout_state(0)
                _rerun()

        return active_entry


# ── Bout detail (main panel) ───────────────────────────────────────────────────

def render_bout_detail(entry: dict, mode: str):
    ss       = st.session_state
    fly_key  = entry["key"]
    f        = _fly(fly_key)
    bouts_df = f["bouts_df"]
    kp_pos   = f["kp_pos"]
    modified = f["modified"]
    n_frames = max(len(v) for v in kp_pos.values()) if kp_pos else 0
    n_bouts  = len(bouts_df)

    if n_bouts == 0:
        st.info("No bouts in this summary CSV.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.subheader(f"{entry['display_id']}  —  {mode}")
    st.caption(f"{n_bouts} bouts · {len(modified)} modified")

    # ── Hideable timeline ─────────────────────────────────────────────────────
    show_tl      = st.checkbox("Show timeline", value=True, key="show_timeline")
    selected_idx = max(0, min(ss.get("current_bout_idx", 0), n_bouts - 1))

    if show_tl:
        tl_fig = make_timeline_fig(bouts_df, modified, selected_idx)
        tl_ev  = st.plotly_chart(
            tl_fig, width="stretch",
            key="timeline_chart", on_select="rerun", selection_mode="points",
        )
        _pts = getattr(getattr(tl_ev, "selection", None), "points", None) or []
        if _pts:
            pt    = _pts[0]
            curve = pt.get("curve_number", pt.get("curveNumber"))
            pidx  = pt.get("point_index",  pt.get("pointIndex"))
            if curve == n_bouts and pidx is not None:
                new_sel = pidx
            elif curve is not None and 0 <= curve < n_bouts:
                new_sel = curve
            else:
                new_sel = None
            if new_sel is not None and new_sel != selected_idx:
                ss["current_bout_idx"] = new_sel
                ss["bout_selectbox"]   = new_sel
                ss.pop(f"vc_{new_sel}", None)
                _rerun()

    # ── Bout selectbox ────────────────────────────────────────────────────────
    ss.setdefault("bout_selectbox", selected_idx)

    sel = st.selectbox(
        "Select bout",
        range(n_bouts),
        format_func=lambda i: (
            f"Bout {int(bouts_df.iloc[i]['bout_idx'])}  "
            f"[{int(bouts_df.iloc[i]['start_frame'])}–{int(bouts_df.iloc[i]['end_frame'])}]"
            f"  {bouts_df.iloc[i]['duration_s']:.2f} s"
            f"{'  ✏' if i in modified else ''}"
        ),
        key="bout_selectbox",
    )
    ss["current_bout_idx"] = sel
    selected_idx = sel
    bout_row = bouts_df.iloc[selected_idx]

    st.divider()
    mod_label = "  ✏ **Modified**" if selected_idx in modified else ""
    st.subheader(f"Bout {int(bout_row['bout_idx'])}{mod_label}")

    # ── Navigation + Erase / Combine ─────────────────────────────────────────
    def _go_prev():
        i = ss["current_bout_idx"] - 1
        ss["current_bout_idx"] = i
        ss["bout_selectbox"]   = i

    def _go_next():
        i = ss["current_bout_idx"] + 1
        ss["current_bout_idx"] = i
        ss["bout_selectbox"]   = i

    c_prev, c_next, c_er, c_comb, _ = st.columns([1, 1, 1, 1.6, 2.5])
    with c_prev:
        st.button("← Prev", disabled=(selected_idx == 0), on_click=_go_prev)
    with c_next:
        st.button("Next →", disabled=(selected_idx >= n_bouts - 1), on_click=_go_next)
    with c_er:
        if st.button("Erase", help="Delete this bout and auto-save"):
            _do_erase(fly_key, selected_idx, entry["bouts_csv"])
            _rerun()
    with c_comb:
        if st.button("Combine w/ next",
                     disabled=(selected_idx >= n_bouts - 1),
                     help="Merge with the next bout and auto-save"):
            _do_combine(fly_key, selected_idx, entry["bouts_csv"])
            _rerun()

    # Re-read in case erase/combine ran this cycle
    bouts_df = f["bouts_df"]
    n_bouts  = len(bouts_df)
    if n_bouts == 0:
        st.info("All bouts erased.")
        return
    selected_idx = max(0, min(ss.get("current_bout_idx", 0), n_bouts - 1))
    bout_row = bouts_df.iloc[selected_idx]

    # ── Boundary state ────────────────────────────────────────────────────────
    orig_start = int(bout_row["start_frame"])
    orig_end   = int(bout_row["end_frame"])
    _sk_s = f"sl_start_{selected_idx}_{orig_start}"
    _sk_e = f"sl_end_{selected_idx}_{orig_end}"
    ss.setdefault(_sk_s, orig_start)
    ss.setdefault(_sk_e, orig_end)
    cur_start = int(ss[_sk_s])
    cur_end   = int(ss[_sk_e])

    # ── View window ───────────────────────────────────────────────────────────
    _vc = f"vc_{selected_idx}"
    ss.setdefault(_vc, (orig_start + orig_end) // 2)
    nav_min     = min(VIEW_HALF, n_frames // 2)
    nav_max     = max(nav_min + 1, n_frames - 1 - VIEW_HALF)
    view_center = max(nav_min, min(nav_max, int(ss[_vc])))
    view_start  = max(0, view_center - VIEW_HALF)
    view_end    = min(n_frames - 1, view_center + VIEW_HALF)

    new_vc = st.slider(
        "Scroll time window",
        min_value=nav_min, max_value=nav_max,
        value=view_center, step=100,
        key=f"nav_{selected_idx}",
        help="Pan to frames outside the current view.",
    )
    if new_vc != view_center:
        ss[_vc] = new_vc
        _rerun()

    # ── Trace figure ──────────────────────────────────────────────────────────
    trace_fig = make_trace_fig(
        kp_pos, cur_start, cur_end,
        fps=FPS, view_start=view_start, view_end=view_end,
        show_wings=(mode == "Courtship"),
    )
    st.plotly_chart(trace_fig, width="stretch", key=f"trace_{selected_idx}")

    # ── Boundary range slider ─────────────────────────────────────────────────
    safe_s      = max(view_start, min(view_end - 1, cur_start))
    safe_e      = max(safe_s + 1, min(view_end, cur_end))
    _bk         = f"bounds_{selected_idx}_{cur_start}_{cur_end}_{view_start}_{view_end}"
    new_s, new_e = st.slider(
        "Bout boundaries",
        min_value=view_start, max_value=view_end,
        value=(safe_s, safe_e), step=1,
        key=_bk,
        help="Drag handles to adjust. Use the scroll slider to reach frames outside the window.",
    )
    if new_s != safe_s:
        ss[_sk_s] = new_s
        ss[f"ni_start_{selected_idx}"] = new_s
        _rerun()
    if new_e != safe_e:
        ss[_sk_e] = new_e
        ss[f"ni_end_{selected_idx}"] = new_e
        _rerun()

    # ── Fine-tuning inputs ────────────────────────────────────────────────────
    c_s, c_e, c_info = st.columns([1, 1, 2])
    with c_s:
        inp_s = st.number_input(
            "Start frame", min_value=0, max_value=cur_end - 1,
            value=cur_start, step=1, key=f"ni_start_{selected_idx}",
        )
    with c_e:
        inp_e = st.number_input(
            "End frame", min_value=cur_start + 1, max_value=n_frames - 1,
            value=cur_end, step=1, key=f"ni_end_{selected_idx}",
        )
    with c_info:
        n_fr  = cur_end - cur_start + 1
        dur_s = n_fr / FPS
        st.caption(f"**{cur_start}** → **{cur_end}** · **{n_fr} frames** ({dur_s:.3f} s)")
        if cur_start != orig_start or cur_end != orig_end:
            st.caption(f"Original: {orig_start} → {orig_end} ({orig_end - orig_start + 1} frames)")

    if inp_s != cur_start:
        ss[_sk_s] = inp_s
        _rerun()
    if inp_e != cur_end:
        ss[_sk_e] = inp_e
        _rerun()

    # ── Save / Reset ──────────────────────────────────────────────────────────
    c_save, c_reset = st.columns([2, 1])
    with c_save:
        if st.button("Save CSV", type="primary"):
            n_fr  = cur_end - cur_start + 1
            dur_s = n_fr / FPS
            bouts_df.at[selected_idx, "start_frame"] = cur_start
            bouts_df.at[selected_idx, "end_frame"]   = cur_end
            bouts_df.at[selected_idx, "n_frames"]    = n_fr
            bouts_df.at[selected_idx, "duration_s"]  = dur_s
            _fly(fly_key)["bouts_df"] = bouts_df
            curated = str(_curated_csv(entry["bouts_csv"]))
            _mark_modified(fly_key, selected_idx, bouts_csv=curated)
            save_bouts_csv(curated, bouts_df)
            st.toast("Saved.", icon="✅")
            _rerun()

    with c_reset:
        if selected_idx in modified:
            if st.button("Reset this bout"):
                curated = str(_curated_csv(entry["bouts_csv"]))
                df_disk = load_bouts_csv(curated)
                f["bouts_df"].iloc[selected_idx] = df_disk.iloc[selected_idx]
                f["modified"].discard(selected_idx)
                _save_modified_sidecar(curated, f["bouts_df"], f["modified"])
                ss.pop(_sk_s, None)
                ss.pop(_sk_e, None)
                _rerun()

    # ── Skeleton replay ───────────────────────────────────────────────────────
    with st.expander("Skeleton replay", expanded=True):
        step_val = st.select_slider(
            "Frame step  (smaller = smoother, slower to build)",
            options=[2, 4, 8, 16, 32], value=8, key="anim_step",
        )
        anim_key = (fly_key, selected_idx, cur_start, cur_end, step_val)
        if ss.get("anim_bout_key") != anim_key:
            with st.spinner("Building animation…"):
                ss["anim_fig"]     = make_animation_fig(
                    kp_pos, cur_start, cur_end, fps=FPS, step=step_val,
                )
            ss["anim_bout_key"] = anim_key
        st.plotly_chart(ss["anim_fig"], width="stretch")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Bout Curation", layout="wide")
    st.title("Bout Curation GUI")
    st.caption("Inspect and correct walking or courtship bout boundaries.")

    _init_ss()

    # Build fly entries using current mode + index path from session state
    index_path  = st.session_state.get("_index_path") or _DEFAULT_INDEX
    mode        = st.session_state.get("_mode", "Walking")
    fly_entries = []
    if Path(index_path).exists():
        fly_entries = build_fly_entries(index_path, mode)

    active_entry = render_sidebar(fly_entries)

    if active_entry is None:
        st.info("Select a fly from the sidebar to begin.")
        st.stop()

    _ensure_loaded(active_entry)

    # Mode may have been updated inside render_sidebar — read fresh
    render_bout_detail(active_entry, mode=st.session_state["_mode"])


if __name__ == "__main__":
    main()
