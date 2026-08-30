import numpy as np
from scripts.viz.voxel_resolution import segment_lengths_voxels

NAMES = ["Scutellum", "Abd_tip", "T1L_TaT3", "T1L_TaTip"]


def test_segment_lengths_in_voxels_scale_with_spacing():
    kp = np.zeros((4, 4, 3), np.float32)
    kp[:, 1] = [12.8, 0, 0]      # Scutellum -> Abd_tip = 12.8 world units
    kp[:, 3] = [1.59, 0, 0]      # tarsal segment = 1.59 world units
    kp[:, 2] = [0, 0, 0]
    a = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    b = segment_lengths_voxels(kp, NAMES, grid_spacing=0.25)
    assert np.isclose(a["Scutellum->Abd_tip"], 12.8, atol=0.1)
    assert np.isclose(a["T1L_TaT3->T1L_TaTip"], 1.59, atol=0.1)
    # finer spacing => MORE voxels per segment (the point of stage 2)
    assert np.isclose(b["T1L_TaT3->T1L_TaTip"] / a["T1L_TaT3->T1L_TaTip"], 4.0,
                      rtol=1e-3)


def test_nan_keypoints_are_ignored():
    kp = np.full((2, 4, 3), np.nan, np.float32)
    kp[0, 0] = [0, 0, 0]
    kp[0, 1] = [10, 0, 0]
    out = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    assert np.isclose(out["Scutellum->Abd_tip"], 10.0, atol=1e-3)


def test_short_bin_widened_to_include_2_17_voxel_segment_and_reports_n(tmp_path):
    """Reproduces the exact scenario from the review finding: with the
    original <2.0 voxel cutoff, only 2 of 9 canonical segments land in the
    short bin (1.98, 1.59) and the third canonical tarsal link
    (T1L_TaT3->T1L_TaTip at 2.17 voxels) falls into NEITHER bin -- thin
    enough that one noisy segment swings the short-vs-long verdict. Widening
    the cutoff to ~2.2 voxels must capture it, and the stats dict must
    report n alongside each mean so a thin bin is visible rather than
    implied."""
    from scripts.viz.voxel_resolution import SEGMENTS, plot_error_vs_segment

    names = sorted({n for pair in SEGMENTS for n in pair})
    idx = {n: i for i, n in enumerate(names)}
    pos = {
        # T1L tarsal chain -- adjacent distances are the canonical segment
        # lengths from the module docstring (3.0, 1.98, 1.59, 2.17 voxels).
        "T1L_FeTi": (0.0, 0.0, 0.0),
        "T1L_TiTa": (3.0, 0.0, 0.0),
        "T1L_TaT1": (4.98, 0.0, 0.0),
        "T1L_TaT3": (6.57, 0.0, 0.0),
        "T1L_TaTip": (8.74, 0.0, 0.0),
        # remaining segments, each an isolated pair -- long, per docstring.
        "Scutellum": (100.0, 0.0, 0.0),
        "Abd_tip": (112.8, 0.0, 0.0),
        "WingL_base": (200.0, 0.0, 0.0),
        "WingL_V12": (217.5, 0.0, 0.0),
        "WingL_V13": (217.5, 5.0, 0.0),
        "T3L_TiTa": (300.0, 0.0, 0.0),
        "T3L_TaT1": (306.0, 0.0, 0.0),
        "EyeL": (400.0, 0.0, 0.0),
        "EyeR": (404.0, 0.0, 0.0),
    }
    assert set(pos) == set(names)
    kp = np.zeros((1, len(names), 3), np.float32)
    for n, p in pos.items():
        kp[0, idx[n]] = p

    out_png = str(tmp_path / "err_vs_seg.png")
    stats = plot_error_vs_segment(kp, kp, names, out_png=out_png)

    lengths = stats["lengths_voxels"]
    assert np.isclose(lengths["T1L_TaT3->T1L_TaTip"], 2.17, atol=0.01)

    # widened short bin (<2.2 voxels) must include all three short segments:
    # 1.98, 1.59, and now 2.17 (previously stranded in neither bin).
    assert stats["short_segments_n"] == 3
    # long bin (>=4.0 voxels): the other 5 canonical segments.
    assert stats["long_segments_n"] == 5
