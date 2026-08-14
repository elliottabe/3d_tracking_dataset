#!/usr/bin/env python3
"""Shared JARVIS-scheme keypoint colours: one colour per limb chain.

Source of truth: `third_party/JARVIS-HybridNet/jarvis/utils/skeleton.py::
get_skeleton(cfg)`. We IMPORT AND CALL that function directly (not a ported
copy) against a minimal `cfg` built from `data/fly50.json`'s `node_names`
(50, DETECTOR order) and `edges` (44 index pairs, converted to NAME pairs
here -- `get_skeleton` only ever calls `.index(name)` on `cfg.KEYPOINT_NAMES`,
never touches an edge index directly). `get_skeleton` needs exactly two
fields, `cfg.SKELETON` (bone name pairs) and `cfg.KEYPOINT_NAMES`; an
`OmegaConf.create({...})` DictConfig supplies both without needing a real
JARVIS project directory, and was verified (by hand, before writing this
module) to import and run standalone in the `3d_tracking` env.

`get_skeleton`'s own algorithm (walk the skeleton graph: cycles get a colour
each, then each chain seeded from a degree-1 endpoint walks forward taking
the next colour, unconnected points take a colour, everything else stays
grey) is reproduced FAITHFULLY, including its quirks -- e.g. on this
skeleton, `Abd_A4`/`Abd_tip` land on the default grey rather than a distinct
colour, because `get_skeleton` only starts a chain-walk from a keypoint that
is BOTH degree-1 AND the first element of some `SKELETON` bone tuple, and
`data/fly50.json`'s edge direction (`[Abd_A4, Abd_tip]`, matching the
natural thorax->tail direction) makes `Abd_tip` the second element only.
That is `get_skeleton`'s own behaviour on this data, not a bug introduced
here -- reproducing it by calling the real function (rather than
re-deriving a "corrected" version) is the point.

Colours are cached at import time and exposed keyed by keypoint NAME, never
index: 04_kp3d_filt.npz / model qpos outputs are in MODEL order while
fly50.json's `node_names` is DETECTOR order (see clip_io.py's module
docstring), and this project has already shipped two bugs from exactly that
class of positional mismatch (CLAUDE.md). Keying by name means every caller
gets the right colour regardless of which order its own keypoint array is in.
"""
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_JARVIS_DIR = _REPO / "third_party" / "JARVIS-HybridNet"
_FLY50_JSON = _REPO / "data" / "fly50.json"


def _load_fly50() -> dict:
    with open(_FLY50_JSON) as fh:
        return json.load(fh)


def _get_skeleton_colors_rgb():
    """Call JARVIS's own `get_skeleton(cfg)`. Returns (names, colors_rgb),
    both length 50, `names` in fly50.json's DETECTOR order, `colors_rgb[i]`
    an (R,G,B) 0-255 tuple -- `get_skeleton`'s own `base_colors`/`gray_color`
    convention (its docstring/usage in JARVIS is matplotlib/PIL-style RGB)."""
    if str(_JARVIS_DIR) not in sys.path:
        sys.path.insert(0, str(_JARVIS_DIR))
    from omegaconf import OmegaConf

    from jarvis.utils.skeleton import get_skeleton

    d = _load_fly50()
    names = list(d["node_names"])
    if len(names) != 50:
        raise ValueError(f"expected 50 node_names in fly50.json, got {len(names)}")
    name_pairs = [[names[a], names[b]] for a, b in d["edges"]]
    cfg = OmegaConf.create({"SKELETON": name_pairs, "KEYPOINT_NAMES": names})
    colors_rgb, _line_idxs = get_skeleton(cfg)
    if len(colors_rgb) != len(names):
        raise ValueError(
            f"get_skeleton returned {len(colors_rgb)} colours for {len(names)} "
            "keypoint names -- cfg.KEYPOINT_NAMES/cfg.SKELETON mismatch."
        )
    return names, [tuple(int(c) for c in rgb) for rgb in colors_rgb]


_NAMES, _COLORS_RGB = _get_skeleton_colors_rgb()
_RGB_BY_NAME = dict(zip(_NAMES, _COLORS_RGB))
_BGR_BY_NAME = {n: (b, g, r) for n, (r, g, b) in _RGB_BY_NAME.items()}


def _check_names(kp_names):
    missing = [n for n in kp_names if n not in _BGR_BY_NAME]
    if missing:
        raise ValueError(
            f"no JARVIS colour for keypoints (not in data/fly50.json "
            f"node_names): {missing}"
        )


def jarvis_kp_colors(kp_names=None) -> dict:
    """keypoint NAME -> BGR/0-255 tuple (cv2 / `viz.core.colors.PALETTE`
    convention), JARVIS's own per-limb-chain colour scheme.

    `kp_names` is optional and used only as a guard: if given, every name in
    it must have a JARVIS colour (i.e. be one of fly50.json's node_names) --
    the returned dict itself always covers all 50 names, keyed by name, so a
    caller in MODEL order or DETECTOR order gets the same answer either way.
    """
    if kp_names is not None:
        _check_names(kp_names)
    return dict(_BGR_BY_NAME)


def jarvis_kp_colors_rgb01(kp_names=None) -> dict:
    """keypoint NAME -> (R,G,B) 0-1 float tuple, for MuJoCo's `mjv_initGeom`
    (Acts 3-4's spheres). Derived from `jarvis_kp_colors`'s BGR/0-255 dict
    (single source of truth), not recomputed from `get_skeleton` again."""
    bgr = jarvis_kp_colors(kp_names)
    return {n: (r / 255.0, g / 255.0, b / 255.0) for n, (b, g, r) in bgr.items()}


def legend_entries(kp_names) -> list:
    """[(label, BGR tuple), ...] for an on-screen legend: one entry per leg
    chain (each gets its OWN colour under the JARVIS scheme, the whole point
    of Change 2) plus the head/thorax cycle and both wing chains. Every
    colour is looked up BY NAME from `jarvis_kp_colors` -- never hardcoded --
    so the legend cannot silently drift from the actual per-keypoint colours
    drawn on screen.
    """
    from viz.core.colors import leg_chains

    names = list(kp_names)
    colors = jarvis_kp_colors(names)
    chains = leg_chains(names)
    entries = []
    for leg in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R"):
        if leg in chains:
            entries.append((f"{leg} leg", colors[names[chains[leg][0]]]))
    if "Antenna_Base" in colors:
        entries.append(("head/thorax", colors["Antenna_Base"]))
    if "WingL_base" in colors:
        entries.append(("WingL", colors["WingL_base"]))
    if "WingR_base" in colors:
        entries.append(("WingR", colors["WingR_base"]))
    if "Abd_tip" in colors:
        entries.append(("abdomen (ungrouped)", colors["Abd_tip"]))
    return entries
