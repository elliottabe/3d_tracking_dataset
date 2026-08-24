"""Convert the existing `assemble_figure` layout into a `figure.json`.

This is a one-time migration: it renders the current nested-subfigure layout,
reads every axes' true ROOT-figure rect, and writes the equivalent flat
absolute-rect design document.

Why not `ax.get_position()`: it returns coordinates relative to the axes'
parent SubFigure, and `assemble_figure` nests subfigures three deep. The
correct root-figure rect comes from the axes' window extent transformed
through the ROOT figure's `transFigure`.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figbuilder.figure import FigureSpec, GroupSpec, PanelSpec, save_figure
from utils.courtship_figure_panels import (
    DEFAULT_PANEL_LETTERS, assemble_figure,
)

#: Per-panel render parameters seeded into figure.json `spec`. The bundle's
#: `attrs` are metadata only; the render path reads `spec`. Ruling 6.
DEFAULT_PANEL_SPEC: Dict[str, Dict[str, object]] = {
    "wing":             {"fs": 800.0, "time_unit": "s", "min_segment_ms": 10.0},
    "scut":             {"fs": 800.0, "time_unit": "s"},
    "sine_phase":       {"fs": 800.0, "time_unit": "s"},
    "pitch":            {"fs": 800.0, "time_unit": "s"},
    "pulse_class":      {"fs": 800.0, "show_std": True},
    "angle_2d":         {"range_deg": [0.0, 90.0]},
    "wing_phase_polar": {"center_stat": "median"},
    "zheight":          {"kind": "violin"},
}

#: axes-dict key -> figbuilder panel type
TYPE_FOR_KEY: Dict[str, str] = {
    "wing": "courtship.wing_z",
    "scut": "courtship.scutellum_z",
    "sine_phase": "courtship.sine_inphase",
    "wing_phase_polar": "courtship.wing_polar",
    "angle_2d": "courtship.angle_density",
    "pulse_class": "courtship.pulse_class",
    "zheight": "courtship.zheight",
    "pitch": "courtship.male_pitch",
    "align_violin": "courtship.pitch_violin",
}


def root_rect(fig, ax) -> Tuple[float, float, float, float]:
    """Return an axes' rect in ROOT-figure fractions."""
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.transFigure.inverted())
    return (float(bb.x0), float(bb.y0), float(bb.width), float(bb.height))


def seed_layout(width_mm: float = 183.0, height_mm: float = 140.0,
                n_frames_strip: int = 6, n_render_strip: int = 4,
                n_video_frames: int = 4) -> FigureSpec:
    fig, axd = assemble_figure(
        fig_width_mm=width_mm, fig_height_mm=height_mm,
        n_frames_strip=n_frames_strip, n_render_strip=n_render_strip,
        n_video_frames=n_video_frames,
    )
    try:
        panels: List[PanelSpec] = []
        groups: List[GroupSpec] = []

        # Render parameters live in figure.json `spec` — nothing in the render
        # path reads the bundle's `attrs`. Seed `fs` etc. here rather than
        # relying on each adapter's default. See Ruling 6 in the SDD ledger.
        for key, ptype in TYPE_FOR_KEY.items():
            ax = axd[key]
            panels.append(PanelSpec(
                id=key, type=ptype, rect=root_rect(fig, ax),
                data={}, spec=dict(DEFAULT_PANEL_SPEC.get(key, {}))))

        for strip_key, gid in (("video", "video_strip"),
                               ("render", "render_strip")):
            axes = axd[strip_key]
            rects = [root_rect(fig, a) for a in axes]
            x0 = min(r[0] for r in rects)
            y0 = min(r[1] for r in rects)
            x1 = max(r[0] + r[2] for r in rects)
            y1 = max(r[1] + r[3] for r in rects)
            gutter_frac = ((x1 - x0) - sum(r[2] for r in rects)) / max(1, len(rects) - 1)
            groups.append(GroupSpec(id=gid, axis="x",
                                    gutter_mm=max(0.0, gutter_frac * width_mm),
                                    equal=True, rect=(x0, y0, x1 - x0, y1 - y0)))
            for i, r in enumerate(rects):
                panels.append(PanelSpec(id=f"{strip_key}_{i}", type="image",
                                        rect=r, group=gid,
                                        spec={"asset": "img"}))

        by_id = {p.id: p for p in panels}
        annotations: List[dict] = []
        for key, letter in DEFAULT_PANEL_LETTERS:
            target = key if key in by_id else f"{key}_0"
            if target not in by_id:
                continue
            annotations.append({
                "id": f"letter_{target}", "kind": "text",
                "managed": "panel_letter", "parent": target,
                "pos_mm": [-4.0, by_id[target].rect[3] * height_mm + 1.0],
                "text": letter,
                "style": {"font_size_pt": 8, "weight": "bold",
                          "font": "Arial, Helvetica, sans-serif"},
            })

        spec = FigureSpec(bundle="fig4_bundle.h5", panels=panels,
                          groups=groups, annotations=annotations,
                          style={"font.size": 6.0})
        spec.set_size(width_mm, height_mm)
        return spec
    finally:
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/paper_figures/fig4.json")
    ap.add_argument("--width-mm", type=float, default=183.0)
    ap.add_argument("--height-mm", type=float, default=140.0)
    args = ap.parse_args(argv)
    save_figure(seed_layout(args.width_mm, args.height_mm), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
