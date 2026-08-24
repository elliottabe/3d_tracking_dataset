"""Compare a figbuilder export against the hand-finished paper SVG.

EXPECTATION (state before looking, per CLAUDE.md):

    SCOPE FIRST, so this is not overclaimed. The reference
    figures/paper_figures/fig4_050426_ETTA.svg was made from the Session0
    2025_10_20 recording with SAM3 masks, DLT calibration and camera video.
    This node has a DIFFERENT dataset (the 04092026 combined h5) and no SAM3,
    no calibration and no courtship mp4 (Ruling 12). So this is NOT a
    content-identical comparison and must never be reported as one.

    What the gate DOES assert:

    1. LAYOUT. Every panel in the export sits at the rect seeded from the
       existing `assemble_figure`, i.e. the two figures share a layout
       skeleton even though their pixels differ. A panel at the wrong rect
       means the root-fraction conversion in Task 11 is wrong.
    2. CONTENT PRESENT. Panels the bundle contains render real data: the wing
       trace shows pulse/sine structure with segment shading, the scutellum
       trace tracks body height, the polar panel shows a phase distribution,
       and the render strip shows a recognisable fly at legible scale
       (not a blank frame, and not the camera-inside-the-animal framing that
       panel_render_strip's own 0.03 default produces on this model).
    3. ABSENCES ARE DECLARED. Panels that could not be built — video strip,
       pitch, align_violin — must be ABSENT and named in the bundle's
       `skipped` meta. A silently blank panel is a FAIL; a declared skip is a
       PASS.
    4. TEXT IS TEXT. The exported SVG contains real <text> elements, not
       glyph outlines — this is the whole point of replacing the Inkscape
       retyping step.

    Report what the image actually shows against each of these. If the render
    disagrees, say so plainly and do not claim the gate passed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


def render_svg_to_png(svg_path: str | Path, png_path: str | Path,
                      dpi: int = 200) -> Path:
    """Rasterize an SVG onto a white background."""
    svg_path, png_path = Path(svg_path), Path(png_path)
    exe = shutil.which("rsvg-convert")
    if exe is None:
        raise RuntimeError("rsvg-convert not found on PATH")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "-f", "png", "-d", str(dpi), "-p", str(dpi),
                    str(svg_path), "-o", str(png_path)],
                   check=True, capture_output=True)
    im = Image.open(png_path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        bg.save(png_path)
    return png_path


def side_by_side(png_a: str | Path, png_b: str | Path, out_path: str | Path,
                 labels: Sequence[str] = ("A", "B"), gutter: int = 24) -> Path:
    a, b = Image.open(png_a).convert("RGB"), Image.open(png_b).convert("RGB")
    w, h = a.width + gutter + b.width, max(a.height, b.height) + 24
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(a, (0, 24))
    canvas.paste(b, (a.width + gutter, 24))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), labels[0], fill="black")
    d.text((a.width + gutter + 4, 6), labels[1], fill="black")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference",
                    default="figures/paper_figures/fig4_050426_ETTA.svg")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out-dir", default="figures/2026-08-24-figbuilder-m0")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    ref_png = render_svg_to_png(args.reference, out / "reference.png", args.dpi)
    cand_png = render_svg_to_png(args.candidate, out / "candidate.png", args.dpi)
    combo = side_by_side(ref_png, cand_png, out / "side_by_side.png",
                         labels=("reference (Inkscape-finished)",
                                 "figbuilder export"))
    print(__doc__)
    print(f"wrote {combo}")
    print("Open this PNG with the Read tool and report what it shows "
          "against the stated expectation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
