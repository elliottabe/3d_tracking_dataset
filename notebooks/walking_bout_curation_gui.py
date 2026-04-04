"""Walking Bout Curation GUI
============================
Streamlit app for manually inspecting and correcting walking bout boundaries
from a walking_bout_summary.csv produced by Sandbox_Strict.ipynb.

Run with:
    streamlit run walking_bout_curation_gui.py

Requirements:
    pip install streamlit plotly pandas numpy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
FPS = 800
SCALE = 10.0
CONTEXT_FRAMES = 250
VIEW_HALF = 2000  # half-width of the time-scroll window (4000 frames = 5 s at 800 fps)

LEG_TIPS = ["T1L_TaTip", "T1R_TaTip", "T2L_TaTip", "T2R_TaTip", "T3L_TaTip", "T3R_TaTip"]
LEG_COLORS = {
    "T1L_TaTip": "#74c476",  # light green  (L)
    "T1R_TaTip": "#1b7837",  # dark green   (R)
    "T2L_TaTip": "#762a83",  # purple       (L)
    "T2R_TaTip": "#e08214",  # orange       (R)
    "T3L_TaTip": "#2166ac",  # blue         (L)
    "T3R_TaTip": "#d73027",  # red          (R)
}
COLOR_DEFAULT  = "#4878cf"
COLOR_MODIFIED = "#f0a500"
COLOR_SELECTED_BORDER = "#222"

CONF_THRESHOLD = 0.8

_DEFAULT_CSV = (
    "/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/"
    "predictions3D/Predictions_3D_20260401-115616/walking_bouts_summary.csv"
)
_DEFAULT_INDEX = (
    "/home/user/src/JARVIS-HybridNet/projects/fly50_V5/predictions/"
    "predictions3D/session10_NC/predictions_index.csv"
)

# ── Skeleton topology (fly50 node names) ──────────────────────────────────────
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
_COXA_KPS = {f"T{n}{s}_ThxCx": LEG_COLORS.get(f"T{n}{s}_TaTip", "#999")
             for n in "123" for s in "LR"}


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_walking_bouts_csv(path):
    """Load walking_bout_summary.csv → DataFrame."""
    return pd.read_csv(path)


def save_bouts_csv(path, bouts_df):
    """Overwrite walking_bout_summary.csv with current DataFrame."""
    bouts_df.to_csv(path, index=False)


def load_3d_data(folder):
    csv_path = Path(folder) / "data3D.csv"
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
    df = df.iloc[:-1].reset_index(drop=True)
    seen, kp_names = set(), []
    for col in df.columns:
        base = col.split(".")[0]
        if base not in seen:
            seen.add(base)
            kp_names.append(base)
    return df, kp_names


def _build_kp_pos(df, kp_names):
    """Pre-extract all keypoint arrays from df.

    Returns {kp: np.array(n_frames, 4)} with columns [x/SCALE, y/SCALE, z/SCALE, confidence].
    Keypoints with <3 columns are excluded; those with <4 get a NaN confidence column.
    """
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


# ── Plotly figure builders ─────────────────────────────────────────────────────

def make_timeline_fig(bouts_df, modified_rows, selected_idx=None):
    """Gantt-style horizontal bar chart: one bar per bout."""
    fig = go.Figure()

    for i, row in enumerate(bouts_df.itertuples()):
        is_selected = i == selected_idx
        color = COLOR_MODIFIED if i in modified_rows else COLOR_DEFAULT

        dur = row.end_frame - row.start_frame
        hover = (
            f"Bout {row.bout_idx}<br>"
            f"Start: {row.start_frame}<br>"
            f"End: {row.end_frame}<br>"
            f"Duration: {dur} frames ({row.duration_s:.2f} s)<br>"
            f"Speed: {row.mean_speed_mm_s:.1f} mm/s"
            f"{'<br><i>MODIFIED</i>' if i in modified_rows else ''}"
        )

        fig.add_trace(go.Bar(
            x=[dur],
            y=[i],
            base=[row.start_frame],
            orientation="h",
            marker_color=color,
            marker_line_color=COLOR_SELECTED_BORDER if is_selected else color,
            marker_line_width=3 if is_selected else 0,
            opacity=1.0 if is_selected else 0.75,
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        ))

    # Invisible scatter markers so on_select fires via pointIndex
    mids = [(r.start_frame + r.end_frame) / 2 for r in bouts_df.itertuples()]
    fig.add_trace(go.Scatter(
        x=mids, y=list(range(len(bouts_df))),
        mode="markers",
        marker=dict(size=24, opacity=0.01, color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    n = len(bouts_df)
    fig.update_layout(
        height=max(220, min(30 * n + 80, 600)),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis_title="Frame",
        yaxis_title="Bout",
        barmode="overlay",
        plot_bgcolor="#f8f8f8",
        title="Click a bout to inspect · Blue=unmodified · Yellow=modified",
        title_font_size=12,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=[str(i) for i in range(n)],
        autorange="reversed",
    )
    return fig


def make_trace_fig(kp_pos, bout_start, bout_end, fps=FPS, view_start=None, view_end=None):
    """Three-panel figure: Y leg tips (top), Z leg tips (middle), mean confidence (bottom)."""
    n_frames = max(len(arr) for arr in kp_pos.values()) if kp_pos else 1
    if view_start is None:
        view_start = max(0, bout_start - CONTEXT_FRAMES)
    if view_end is None:
        view_end = min(n_frames - 1, bout_end + CONTEXT_FRAMES)

    plot_start, plot_end = int(view_start), int(view_end)
    frames = np.arange(plot_start, plot_end + 1)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Y — leg tip (mm)", "Z — leg tip (mm)", "Mean confidence"),
        vertical_spacing=0.08,
        row_heights=[0.38, 0.38, 0.24],
    )

    available_legs = [leg for leg in LEG_TIPS if leg in kp_pos]

    for leg in available_legs:
        arr     = kp_pos[leg]
        clipped = frames[frames < len(arr)]
        color   = LEG_COLORS.get(leg, "#888")

        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 1],
            mode="lines", name=leg,
            line=dict(color=color, width=1.5),
            legendgroup=leg,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=clipped, y=arr[clipped, 2],
            mode="lines", name=leg,
            line=dict(color=color, width=1.5),
            legendgroup=leg, showlegend=False,
        ), row=2, col=1)

    # Mean confidence across all keypoints
    conf_arrays = []
    for kp, arr in kp_pos.items():
        if arr.shape[1] >= 4:
            clipped_c = frames[frames < len(arr)]
            conf_arrays.append(arr[clipped_c, 3])
    if conf_arrays:
        min_len = min(len(a) for a in conf_arrays)
        clipped_frames = frames[frames < n_frames][:min_len]
        mean_conf = np.mean([a[:min_len] for a in conf_arrays], axis=0)

        fig.add_trace(go.Scatter(
            x=clipped_frames, y=mean_conf,
            mode="lines", name="mean conf",
            line=dict(color="#555", width=1.2),
            fill="tozeroy", fillcolor="rgba(100,100,100,0.08)",
            showlegend=False,
        ), row=3, col=1)

    dur_s = (bout_end - bout_start) / fps

    shapes = [
        dict(type="rect", xref="x", yref="paper",
             x0=bout_start, x1=bout_end, y0=0, y1=1,
             fillcolor="rgba(44,160,44,0.10)", layer="below",
             line=dict(width=0), editable=False),
        dict(type="line", xref="x", yref="paper",
             x0=bout_start, x1=bout_start, y0=0, y1=1,
             line=dict(color="#2ca02c", dash="dash", width=2),
             editable=False),
        dict(type="line", xref="x", yref="paper",
             x0=bout_end, x1=bout_end, y0=0, y1=1,
             line=dict(color="#d62728", dash="dash", width=2),
             editable=False),
        dict(type="line", xref="paper", yref="y3",
             x0=0, x1=1, y0=CONF_THRESHOLD, y1=CONF_THRESHOLD,
             line=dict(color="red", dash="dot", width=1),
             editable=False),
    ]

    fig.add_annotation(
        x=1, y=CONF_THRESHOLD, xref="paper", yref="y3",
        text=f"thr {CONF_THRESHOLD}", showarrow=False,
        font=dict(size=9, color="red"), xanchor="right",
    )

    fig.update_layout(
        shapes=shapes,
        height=680,
        margin=dict(l=55, r=20, t=50, b=40),
        hovermode="x",
        template="none",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=(f"Start {bout_start} → End {bout_end} "
                  f"({bout_end - bout_start + 1} frames, {dur_s:.2f} s)"),
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(range=[plot_start, plot_end], autorange=False)
    fig.update_xaxes(title_text="Frame", row=3, col=1)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=1, range=[0, 5.5])
    fig.update_yaxes(title_text="Z (mm)", row=2, col=1, range=[0, 3.0])
    fig.update_yaxes(title_text="Confidence", row=3, col=1, range=[0, 1.05])
    return fig


def _skeleton_proj(kp_pos, fi, available, coord_a, coord_b):
    """Build two coord lists (None-separated) for skeleton edges at frame fi."""
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
    """Side-by-side animated skeleton: top-down (XY) + side view (XZ)."""
    n_frames   = max(len(arr) for arr in kp_pos.values()) if kp_pos else 1
    plot_start = max(0, bout_start - CONTEXT_FRAMES)
    plot_end   = min(n_frames - 1, bout_end + CONTEXT_FRAMES)
    indices    = list(range(plot_start, plot_end + 1, step))

    available = set(kp_pos)
    kp_list   = sorted(available)

    pt_colors = [LEG_COLORS.get(kp, _COXA_KPS.get(kp, "#aaa")) for kp in kp_list]
    pt_sizes  = [8 if kp in LEG_COLORS else 5 for kp in kp_list]

    win = slice(plot_start, plot_end + 1)
    def _rng(arr, pad=0.5):
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        p = (hi - lo) * 0.05 + pad
        return [lo - p, hi + p]

    all_x = np.concatenate([kp_pos[kp][win, 0] for kp in kp_list])
    all_y = np.concatenate([kp_pos[kp][win, 1] for kp in kp_list])
    all_z = np.concatenate([kp_pos[kp][win, 2] for kp in kp_list])
    x_range = _rng(all_x)
    y_range = _rng(all_y)
    z_range = _rng(all_z)

    def frame_data(fi):
        xs = [float(kp_pos[kp][fi, 0]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        ys = [float(kp_pos[kp][fi, 1]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        zs = [float(kp_pos[kp][fi, 2]) if fi < len(kp_pos[kp]) else None for kp in kp_list]
        sk_x, sk_y  = _skeleton_proj(kp_pos, fi, available, 0, 1)
        sk_x2, sk_z = _skeleton_proj(kp_pos, fi, available, 0, 2)
        in_bout = bout_start <= fi <= bout_end
        border = "rgba(44,160,44,0.9)" if in_bout else "rgba(160,160,160,0.6)"
        mk = dict(color=pt_colors, size=pt_sizes, line=dict(color=border, width=1.5))
        return [
            go.Scatter(x=sk_x, y=sk_y, mode="lines",
                       line=dict(color="#888", width=1.5), hoverinfo="skip", showlegend=False),
            go.Scatter(x=xs, y=ys, mode="markers", marker=mk,
                       text=kp_list, hoverinfo="text", showlegend=False),
            go.Scatter(x=sk_x2, y=sk_z, mode="lines",
                       line=dict(color="#888", width=1.5), hoverinfo="skip", showlegend=False),
            go.Scatter(x=xs, y=zs, mode="markers", marker=mk,
                       text=kp_list, hoverinfo="text", showlegend=False),
        ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Top-down (XY)", "Side view (XZ)"],
                        column_widths=[0.5, 0.5])
    init = frame_data(indices[0])
    fig.add_trace(init[0], row=1, col=1)
    fig.add_trace(init[1], row=1, col=1)
    fig.add_trace(init[2], row=1, col=2)
    fig.add_trace(init[3], row=1, col=2)

    frames = [
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
    fig.frames = frames

    sliders = [{
        "steps": [
            {"args": [[f.name], {"frame": {"duration": 50, "redraw": True},
                                 "transition": {"duration": 0}}],
             "method": "animate", "label": ""}
            for f in frames
        ],
        "active": 0,
        "x": 0.0, "len": 1.0, "y": -0.05,
        "currentvalue": {"prefix": "Frame: ", "visible": True, "xanchor": "center"},
        "transition": {"duration": 0},
    }]

    fig.update_layout(
        height=540,
        margin=dict(l=50, r=20, t=60, b=130),
        plot_bgcolor="#f0f0f0",
        showlegend=False,
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "x": 0.5, "y": -0.12, "xanchor": "center", "yanchor": "top",
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
    fig.update_xaxes(title_text="X (mm)", range=x_range, row=1, col=2)
    fig.update_yaxes(title_text="Z (mm)", range=z_range, row=1, col=2)
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Walking Bout Curation", layout="wide")
    st.title("Walking Bout Curation GUI")
    st.caption(
        "Inspect and trim walking bouts from a `walking_bout_summary.csv`. "
        "Adjust start/end boundaries and save back to the same file."
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("File")
        mode = st.radio("Input mode", ["Single CSV", "Predictions index"], key="input_mode")

        if mode == "Single CSV":
            csv_path = st.text_input("walking_bout_summary.csv", value=_DEFAULT_CSV)
        else:
            index_path_str = st.text_input(
                "predictions_index.csv", value=_DEFAULT_INDEX, key="index_path_input"
            )
            if not Path(index_path_str).exists():
                st.error("predictions_index.csv not found.")
                st.stop()

            index_df = pd.read_csv(index_path_str)
            fly_options = index_df["fly_id"].tolist()

            # Reset fly selection when the index file changes
            if st.session_state.get("_index_path") != index_path_str:
                st.session_state["_index_path"] = index_path_str
                st.session_state["_selected_fly_id"] = fly_options[0] if fly_options else None

            cur_fly = st.session_state.get("_selected_fly_id")
            default_fly_idx = fly_options.index(cur_fly) if cur_fly in fly_options else 0
            selected_fly_id = st.selectbox("Fly", fly_options, index=default_fly_idx)
            if selected_fly_id != st.session_state.get("_selected_fly_id"):
                st.session_state["_selected_fly_id"] = selected_fly_id

            pred_folder = index_df.loc[
                index_df["fly_id"] == selected_fly_id, "prediction_folder"
            ].iloc[0]
            csv_path = str(Path(pred_folder) / "walking_bouts_summary.csv")

        if not Path(csv_path).exists():
            st.error(f"CSV file not found: {csv_path}")
            if mode == "Predictions index":
                st.caption("Run batch_bout_detection.py first to generate walking_bouts_summary.csv files.")
            st.stop()

        # Reload data when path changes
        if st.session_state.get("csv_path") != csv_path:
            st.session_state["csv_path"]      = csv_path
            st.session_state["bouts_df"]      = load_walking_bouts_csv(csv_path)
            st.session_state["modified_rows"] = set()
            st.session_state.pop("raw_df",   None)
            st.session_state.pop("raw_kp",   None)
            st.session_state.pop("kp_pos",   None)
            st.session_state["selected_bout_idx"] = 0
            st.session_state["bout_selectbox"]    = 0

        bouts_df      = st.session_state["bouts_df"]
        modified_rows = st.session_state["modified_rows"]
        n_bouts       = len(bouts_df)
        n_modified    = len(modified_rows)

        st.caption(f"{n_bouts} bouts · {n_modified} modified")
        st.markdown("Blue = unmodified · Yellow = modified")

        if st.button("Reload from disk", width='stretch'):
            st.session_state["bouts_df"]      = load_walking_bouts_csv(csv_path)
            st.session_state["modified_rows"] = set()
            st.session_state["selected_bout_idx"] = 0
            st.session_state["bout_selectbox"]    = 0
            _rerun()

    bouts_df      = st.session_state["bouts_df"]
    modified_rows = st.session_state["modified_rows"]

    # ── Load raw 3D data (cached per csv folder) ───────────────────────────────
    data_folder = str(Path(csv_path).parent)
    if st.session_state.get("raw_df") is None or st.session_state.get("kp_folder") != data_folder:
        with st.spinner("Loading data3D.csv…"):
            df_raw, kp_names = load_3d_data(data_folder)
        st.session_state["raw_df"]    = df_raw
        st.session_state["raw_kp"]    = kp_names
        st.session_state["kp_folder"] = data_folder
        st.session_state.pop("kp_pos", None)

    if "kp_pos" not in st.session_state:
        with st.spinner("Pre-processing keypoints…"):
            st.session_state["kp_pos"] = _build_kp_pos(
                st.session_state["raw_df"], st.session_state["raw_kp"]
            )

    kp_pos   = st.session_state["kp_pos"]
    n_frames = max(len(v) for v in kp_pos.values()) if kp_pos else len(st.session_state["raw_df"])

    # ── Header ─────────────────────────────────────────────────────────────────
    fly_ids = bouts_df["fly_id"].unique().tolist()
    st.subheader(", ".join(fly_ids) if len(fly_ids) <= 3 else f"{fly_ids[0]} … ({len(fly_ids)} sessions)")

    # ── Timeline ───────────────────────────────────────────────────────────────
    selected_idx = st.session_state.get("selected_bout_idx", 0)
    st.subheader("Timeline")
    timeline_fig = make_timeline_fig(bouts_df, modified_rows, selected_idx)

    tl_ev = st.plotly_chart(
        timeline_fig, width='stretch',
        key="timeline_chart", on_select="rerun", selection_mode="points",
    )
    _pts = getattr(getattr(tl_ev, "selection", None), "points", None) or []
    if _pts:
        pt     = _pts[0]
        curve  = pt.get("curve_number", pt.get("curveNumber"))
        pt_idx = pt.get("point_index",  pt.get("pointIndex"))
        # Invisible scatter is the last trace (index == n_bouts)
        if curve == len(bouts_df) and pt_idx is not None:
            new_sel = pt_idx
        elif curve is not None and 0 <= curve < len(bouts_df):
            new_sel = curve
        else:
            new_sel = None
        if new_sel is not None and new_sel != selected_idx:
            st.session_state["selected_bout_idx"] = new_sel
            st.session_state["bout_selectbox"]    = new_sel
            st.session_state.pop(f"vc_{new_sel}", None)
            _rerun()

    # Selectbox (also controlled by Prev/Next)
    if "bout_selectbox" not in st.session_state:
        st.session_state["bout_selectbox"] = selected_idx

    sel = st.selectbox(
        "Select bout",
        range(len(bouts_df)),
        format_func=lambda i: (
            f"Bout {bouts_df.iloc[i]['bout_idx']}  "
            f"[{int(bouts_df.iloc[i]['start_frame'])}–{int(bouts_df.iloc[i]['end_frame'])}]"
            f"  {bouts_df.iloc[i]['duration_s']:.2f} s"
            f"{'  ✏' if i in modified_rows else ''}"
        ),
        key="bout_selectbox",
    )
    st.session_state["selected_bout_idx"] = sel
    selected_idx = sel

    # ── Bout detail ────────────────────────────────────────────────────────────
    bout_row = bouts_df.iloc[selected_idx]

    st.divider()
    mod_label = "  ✏ **Modified**" if selected_idx in modified_rows else ""
    st.subheader(f"Bout {int(bout_row['bout_idx'])}{mod_label}")

    # Navigation — use on_click callbacks so session state is set before widgets render
    def _go_prev():
        new_idx = st.session_state["selected_bout_idx"] - 1
        st.session_state["selected_bout_idx"] = new_idx
        st.session_state["bout_selectbox"]    = new_idx

    def _go_next():
        new_idx = st.session_state["selected_bout_idx"] + 1
        st.session_state["selected_bout_idx"] = new_idx
        st.session_state["bout_selectbox"]    = new_idx

    col_prev, col_next, _ = st.columns([1, 1, 6])
    with col_prev:
        st.button("← Prev", disabled=(selected_idx == 0), on_click=_go_prev)
    with col_next:
        st.button("Next →", disabled=(selected_idx >= len(bouts_df) - 1), on_click=_go_next)

    # ── Live start/end state (keyed by bout + original boundaries) ─────────────
    orig_start = int(bout_row["start_frame"])
    orig_end   = int(bout_row["end_frame"])
    _sk_start  = f"sl_start_{selected_idx}_{orig_start}"
    _sk_end    = f"sl_end_{selected_idx}_{orig_end}"

    if _sk_start not in st.session_state:
        st.session_state[_sk_start] = orig_start
    if _sk_end not in st.session_state:
        st.session_state[_sk_end] = orig_end

    cur_start = int(st.session_state[_sk_start])
    cur_end   = int(st.session_state[_sk_end])

    # ── View window scroll ─────────────────────────────────────────────────────
    _vc_key = f"vc_{selected_idx}"
    if _vc_key not in st.session_state:
        st.session_state[_vc_key] = (orig_start + orig_end) // 2

    view_center = int(st.session_state[_vc_key])
    nav_min = min(VIEW_HALF, n_frames // 2)
    nav_max = max(nav_min + 1, n_frames - 1 - VIEW_HALF)
    view_center = max(nav_min, min(nav_max, view_center))
    view_start  = max(0, view_center - VIEW_HALF)
    view_end    = min(n_frames - 1, view_center + VIEW_HALF)

    new_vc = st.slider(
        "Scroll time window",
        min_value=nav_min, max_value=nav_max,
        value=view_center, step=100,
        key=f"nav_{selected_idx}",
        help="Scroll to access frames outside the current view.",
    )
    if new_vc != view_center:
        st.session_state[_vc_key] = new_vc
        _rerun()

    # ── Trace figure ───────────────────────────────────────────────────────────
    trace_fig = make_trace_fig(
        kp_pos, cur_start, cur_end,
        fps=FPS, view_start=view_start, view_end=view_end,
    )
    st.plotly_chart(trace_fig, width='stretch', key=f"trace_{selected_idx}")

    # ── Boundary range slider ──────────────────────────────────────────────────
    safe_start  = max(view_start, min(view_end - 1, cur_start))
    safe_end    = max(safe_start + 1, min(view_end, cur_end))
    _bounds_key = f"bounds_{selected_idx}_{cur_start}_{cur_end}_{view_start}_{view_end}"
    new_start, new_end = st.slider(
        "Bout boundaries",
        min_value=view_start, max_value=view_end,
        value=(safe_start, safe_end), step=1,
        key=_bounds_key,
        help="Drag handles to redefine bout start/end. Use the scroll slider above to reach frames outside the window.",
    )
    if new_start != cur_start:
        st.session_state[_sk_start] = new_start
        st.session_state[f"ni_start_{selected_idx}"] = new_start
        _rerun()
    if new_end != cur_end:
        st.session_state[_sk_end] = new_end
        st.session_state[f"ni_end_{selected_idx}"] = new_end
        _rerun()

    # ── Fine-tuning number inputs ──────────────────────────────────────────────
    col_s, col_e, col_info = st.columns([1, 1, 2])
    with col_s:
        inp_start = st.number_input(
            "Start frame", min_value=0, max_value=cur_end - 1,
            value=cur_start, step=1, key=f"ni_start_{selected_idx}",
        )
    with col_e:
        inp_end = st.number_input(
            "End frame", min_value=cur_start + 1, max_value=n_frames - 1,
            value=cur_end, step=1, key=f"ni_end_{selected_idx}",
        )
    with col_info:
        n_fr  = cur_end - cur_start + 1
        dur_s = n_fr / FPS
        st.caption(f"**{cur_start}** → **{cur_end}** · **{n_fr} frames** ({dur_s:.3f} s)")
        if cur_start != orig_start or cur_end != orig_end:
            st.caption(f"Original: {orig_start} → {orig_end} ({orig_end - orig_start + 1} frames)")

    if inp_start != cur_start:
        st.session_state[_sk_start] = inp_start
        _rerun()
    if inp_end != cur_end:
        st.session_state[_sk_end] = inp_end
        _rerun()

    # ── Save button ────────────────────────────────────────────────────────────
    col_save, col_reset = st.columns([2, 1])
    with col_save:
        if st.button("Save CSV", type="primary"):
            n_fr  = cur_end - cur_start + 1
            dur_s = n_fr / FPS
            st.session_state["bouts_df"].at[selected_idx, "start_frame"] = cur_start
            st.session_state["bouts_df"].at[selected_idx, "end_frame"]   = cur_end
            st.session_state["bouts_df"].at[selected_idx, "n_frames"]    = n_fr
            st.session_state["bouts_df"].at[selected_idx, "duration_s"]  = dur_s
            st.session_state["modified_rows"].add(selected_idx)
            save_bouts_csv(csv_path, st.session_state["bouts_df"])
            st.success(f"Saved {Path(csv_path).name}", icon="✅")
            _rerun()

    with col_reset:
        if selected_idx in modified_rows:
            if st.button("Reset this bout"):
                # Reload from disk and restore original values for this bout
                df_disk = load_walking_bouts_csv(csv_path)
                orig_row = df_disk.iloc[selected_idx]
                st.session_state["bouts_df"].iloc[selected_idx] = orig_row
                st.session_state["modified_rows"].discard(selected_idx)
                # Clear slider state so widgets reset to disk values
                st.session_state.pop(_sk_start, None)
                st.session_state.pop(_sk_end, None)
                _rerun()

    # ── Skeleton replay ────────────────────────────────────────────────────────
    with st.expander("Skeleton replay", expanded=True):
        step_val = st.select_slider(
            "Frame step  (smaller = smoother, slower to build)",
            options=[2, 4, 8, 16, 32],
            value=8,
            key="anim_step",
        )
        anim_key = (selected_idx, cur_start, cur_end, step_val)
        if st.session_state.get("anim_bout_key") != anim_key:
            with st.spinner("Building animation…"):
                st.session_state["anim_fig"] = make_animation_fig(
                    kp_pos, cur_start, cur_end,
                    fps=FPS, step=step_val,
                )
            st.session_state["anim_bout_key"] = anim_key
        st.plotly_chart(st.session_state["anim_fig"], width='stretch')


if __name__ == "__main__":
    main()
