"""CPU-fast smoke test for scripts/viz_compare_3d_runs.py's two-cache /
two-front-end support (Task: compare ViTPose vs EfficientTrack arms, each
reading volumes from its OWN reprojected-volume cache).

No GPU, no real caches, no rendering -- just proves the wiring:

  1. ``run_compare``'s signature accepts ``cache_dir1``/``cache_dir2`` (new,
     for two separate front-end caches) and ``label1``/``label2`` (cosmetic
     legend/title labels), while staying backward compatible with the
     original ``cache_dir`` single-cache kwarg.
  2. ``pick_female_framesets`` (the female/courtship-recording frameset
     picker) returns the expected (idx, tag) pairs for a stubbed
     recording-name list + per-frameset error arrays, and returns [] when
     the target recording isn't present (documented skip path).
  3. ``recording_names`` extracts per-frameset datasetName from a stubbed
     dataset's ``.framesets`` list (mirrors V3FramesetDataset.framesets).
"""
from __future__ import annotations

import importlib.util
import inspect
import os

import numpy as np

from jarvis_jax.hydra_utils import CONFIG_DIR

# viz_compare_3d_runs.py lives under scripts/, not the jarvis_jax package --
# load it the same way tests/test_repro_cache_frontend.py does.
_SCRIPT_PATH = os.path.join(os.path.dirname(CONFIG_DIR), "scripts",
                            "viz_compare_3d_runs.py")
_spec = importlib.util.spec_from_file_location("viz_compare_3d_runs", _SCRIPT_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_run_compare_signature_has_two_cache_and_label_args():
    sig = inspect.signature(mod.run_compare)
    params = sig.parameters
    for name in ("cache_dir", "cache_dir1", "cache_dir2",
                 "run1", "run2", "out", "sharpen1", "sharpen2",
                 "label1", "label2"):
        assert name in params, f"run_compare missing param {name!r}"
    # backward compat: cache_dir1/cache_dir2/label1/label2 must be optional
    assert params["cache_dir"].default is None
    assert params["cache_dir1"].default is None
    assert params["cache_dir2"].default is None
    assert params["label1"].default == "run1"
    assert params["label2"].default == "run2"
    # all params keyword-only (called with kwargs throughout the module)
    for name, p in params.items():
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, name


def test_recording_names_reads_dataset_name_per_frameset():
    class _StubDS:
        framesets = [
            {"datasetName": "2026_05_20_10_00_00", "frames": [1, 2]},
            {"datasetName": "2026_05_27_11_56_05", "frames": [3, 4]},
            {"datasetName": "2026_05_27_11_56_05", "frames": [5, 6]},
        ]
    names = mod.recording_names(_StubDS())
    assert names == ["2026_05_20_10_00_00",
                      "2026_05_27_11_56_05",
                      "2026_05_27_11_56_05"]


def test_pick_female_framesets_max_diff_and_worst_by_either():
    names = ["recA", "recA", "female_rec", "female_rec", "female_rec", "recB"]
    #          idx0    idx1      idx2          idx3           idx4       idx5
    m1 = np.array([0.5, 0.6, 1.0, 2.0, 3.0, 0.9])
    m2 = np.array([0.4, 0.7, 1.1, 5.0, 3.1, 1.0])
    # among female_rec (idx 2,3,4): |m1-m2| = [0.1, 3.0, 0.1] -> max diff idx3
    #                                worst-by-either = max(m1,m2) = [1.1,5.0,3.1] -> idx3
    picks = mod.pick_female_framesets(names, m1, m2, "female_rec")
    assert picks == [(3, "female_diff"), (3, "female_worst")]


def test_pick_female_framesets_returns_empty_when_recording_absent():
    names = ["recA", "recB", "recC"]
    m1 = np.array([0.1, 0.2, 0.3])
    m2 = np.array([0.1, 0.2, 0.3])
    picks = mod.pick_female_framesets(names, m1, m2, "not_present_rec")
    assert picks == []


def test_pick_female_framesets_distinct_diff_and_worst_indices():
    names = ["female_rec"] * 3
    # idx0: biggest |diff|; idx1: biggest max(m1,m2) but small diff
    m1 = np.array([1.0, 4.0, 0.2])
    m2 = np.array([3.0, 4.1, 0.3])
    picks = mod.pick_female_framesets(names, m1, m2, "female_rec")
    assert picks[0] == (0, "female_diff")
    assert picks[1] == (1, "female_worst")


def test_col_map_assigns_colors_by_label():
    col = mod.col_map("ViTPose", "EfficientTrack")
    assert col["GT"] == "#00d000"
    assert col["ViTPose"] == "#e02020"
    assert col["EfficientTrack"] == "#2050ff"
