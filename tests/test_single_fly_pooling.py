"""Unpaired males must reach the single-fly panels and NOT the pair panels.

`pair_bouts` only pairs an adjacent (fly0, fly1), so a bout whose partner could
not be reconstructed contributes nothing -- and every Figure 4 panel derives
from pair results, including ones that need only one fly. After the 2026-08-28
re-run three Session0 bouts came back male-only (the mask-agreement gate found
the female unusable), so a good male fit was being thrown away.

The dangerous half of this change is the other direction: a male leaking into a
PAIR quantity would silently produce a male-vs-male pitch alignment.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "scripts" / "figures" / "export_fig4_bundle.py"


@pytest.fixture(scope="module")
def src():
    return SRC.read_text()


def test_pair_bouts_skips_an_orphan_rather_than_shifting_every_later_pair():
    """The premise. If an orphan shifted the pairing, male/female would be
    swapped for every subsequent bout in the dataset."""
    from utils.courtship_loader import pair_bouts
    keys = ['A0', 'A1', 'B1', 'C0', 'C1']
    info = {'source_flies': ['fly0', 'fly1', 'fly1', 'fly0', 'fly1'],
            'bucket': ['both'] * 5}
    assert pair_bouts(keys, info) == [('A0', 'A1'), ('C0', 'C1')]


def test_singles_are_pooled_only_into_single_fly_quantities(src):
    """phase_diffs, wing-angle density and pulse types read `pooled`."""
    for marker in ("    phase_diffs = []\n    for r in pooled:",
                   "    ext_pulse, ext_sine = [], []\n    for r in pooled:",
                   "get_pulse_type_labels(pooled, fs=fs)"):
        assert marker in src, f"single-fly site not wired: {marker!r}"


def test_zheight_pools_singles_via_extras(src):
    assert '_mean_z_by_label(extras.get("pooled") or results, "pulse"' in src
    assert '_mean_z_by_label(extras.get("pooled") or results, "sine"' in src


def test_the_exemplar_and_pair_only_panels_never_see_pooled(src):
    """The pitch traces and the pitch-alignment violin are pair quantities."""
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    body = ast.get_source_segment(src, fn)
    # the exemplar is chosen from `results`, never from pooled
    i = body.index("for _r in results:")
    assert "pooled" not in body[i:i + 400], "exemplar search must use pairs only"
    # build_fig4_panels still receives the PAIR list positionally
    assert "build_fig4_panels(results, ex, extras)" in body


def test_single_fly_dicts_expose_the_keys_the_pooled_panels_read(src):
    """_mean_z_by_label and the wing/pulse loops read these by name."""
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "analyze_unpaired_males"][0]
    body = ast.get_source_segment(src, fn)
    for k in ('"song0"', '"male_labels"', '"male_valid"', '"com_z"',
              '"by_song"', '"key0"'):
        assert k in body, f"single-fly result is missing {k}"


def test_pair_only_fields_are_none_so_misuse_breaks_loudly(src):
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "analyze_unpaired_males"][0]
    body = ast.get_source_segment(src, fn)
    for k in ('"song1": None', '"sex": None', '"colocated": None',
              '"valid_fly1": None'):
        assert k in body, f"{k} must be None on a single-fly result"
    assert '"single_fly": True' in body


def test_only_a_verified_male_slot_is_analysed(src):
    """Guessing the sex of a lone fly is exactly the mistake the wing-song CV
    metric made; the h5's own male_fly must decide."""
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "analyze_unpaired_males"][0]
    body = ast.get_source_segment(src, fn)
    assert "male_fly" in body and "sex_verified" in body
    assert "int(male_fly[i]) != slot" in body, "must skip a lone FEMALE"


def test_mean_z_by_label_accepts_a_single_fly_dict():
    """End to end on the one function that consumes them numerically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ex", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    single = {"key0": "bout_009", "single_fly": True,
              "male_valid": np.ones(6, bool),
              "male_labels": np.array(["pulse"] * 3 + ["sine"] * 3),
              "com_z": np.array([0.1, 0.2, 0.3, 1.0, 1.0, 1.0])}
    out = m._mean_z_by_label([single], "pulse")
    assert out.shape == (1,) and out[0] == pytest.approx(0.2)
