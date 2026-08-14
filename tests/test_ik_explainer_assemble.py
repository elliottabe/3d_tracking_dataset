import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import assemble


def _write_frames(dir_path: Path, n: int, value: int):
    """Write `n` solid-colour PNGs `f00000.png`.. so the frame's own pixel
    value identifies which synthetic 'act' and which index it came from."""
    dir_path.mkdir(parents=True, exist_ok=True)
    import cv2
    for i in range(n):
        img = np.full((4, 4, 3), value, np.uint8)
        cv2.imwrite(str(dir_path / f"f{i:05d}.png"), img)


def test_list_frames_raises_when_short(tmp_path):
    act_dir = tmp_path / "act1_views"
    _write_frames(act_dir, 10, 1)
    with pytest.raises(RuntimeError, match="expected exactly 15"):
        assemble._list_frames(act_dir, "act1_views", 15)


def test_list_frames_raises_when_long(tmp_path):
    """A resumed/duplicated render leaving extra files must also be refused,
    not silently truncated or padded."""
    act_dir = tmp_path / "act1_views"
    _write_frames(act_dir, 20, 1)
    with pytest.raises(RuntimeError, match="expected exactly 15"):
        assemble._list_frames(act_dir, "act1_views", 15)


def test_list_frames_ok_when_exact(tmp_path):
    act_dir = tmp_path / "act1_views"
    _write_frames(act_dir, 15, 1)
    files = assemble._list_frames(act_dir, "act1_views", 15)
    assert len(files) == 15
    assert files == sorted(files)


def test_build_edit_plan_total_length_matches_formula():
    """Same arithmetic as the real ACT_SPECS: n_acts sequences of length n_i,
    crossfade width c -> total = sum(n_i) - (n_acts-1)*c."""
    specs = [("a", 30), ("b", 20), ("c", 40)]
    n_cross = 5
    frame_lists = {name: [Path(f"{name}_{i}") for i in range(n)] for name, n in specs}
    order = [name for name, _ in specs]
    plan = assemble.build_edit_plan(frame_lists, order, n_cross)
    expected = sum(n for _, n in specs) - (len(specs) - 1) * n_cross
    assert len(plan) == expected


def test_build_edit_plan_order_and_crossfade_shape():
    """First/last acts contribute a full head/tail (no crossfade before act 0,
    none after the last act); interior boundaries are (path_a, path_b, t)
    triples referencing the correct source frames."""
    specs = [("a", 10), ("b", 10)]
    n_cross = 3
    frame_lists = {name: [Path(f"{name}_{i}") for i in range(n)] for name, n in specs}
    order = [name for name, _ in specs]
    plan = assemble.build_edit_plan(frame_lists, order, n_cross)

    # act "a" solo region: indices 0..6 (10 - 3 tail frames held back for the blend)
    assert plan[:7] == frame_lists["a"][:7]
    # the 3-frame crossfade: a's last 3 frames vs b's first 3 frames
    blend = plan[7:10]
    assert all(isinstance(op, tuple) for op in blend)
    for i, (path_a, path_b, t) in enumerate(blend):
        assert path_a == frame_lists["a"][10 - n_cross + i]
        assert path_b == frame_lists["b"][i]
    assert blend[0][2] == pytest.approx(0.0)
    assert blend[-1][2] == pytest.approx(1.0)
    # act "b" solo region: its first 3 frames were consumed by the crossfade above
    assert plan[10:] == frame_lists["b"][n_cross:]
    assert len(plan) == 7 + 3 + 7  # a-solo(7) + blend(3) + b-solo(7) == 17


def test_render_op_plain_frame_reads_the_file(tmp_path):
    import cv2
    p = tmp_path / "f00000.png"
    img = np.full((4, 4, 3), 77, np.uint8)
    cv2.imwrite(str(p), img)
    out = assemble._render_op(p)
    assert np.array_equal(out, img)


def test_render_op_crossfade_matches_draw_fade(tmp_path):
    import cv2
    from scripts.viz.ik_explainer import draw
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    img_a = np.zeros((4, 4, 3), np.uint8)
    img_b = np.full((4, 4, 3), 200, np.uint8)
    cv2.imwrite(str(pa), img_a)
    cv2.imwrite(str(pb), img_b)
    out = assemble._render_op((pa, pb, 0.5))
    expected = draw.fade(img_a, img_b, 0.5)
    assert np.array_equal(out, expected)


def test_real_act_specs_total_matches_expected_1530():
    """Locks the exact arithmetic the brief specifies for the shipped acts
    (task-14: Act 1 shortened 450 -> 270 frames; a later round lengthened
    crossfades 15 -> 30 frames for smoother transitions; total 2205 -> 2025
    -> 1980. task-15: Act 3 shortened 600 -> 360 frames (dropped its
    rest-pose fade-in phase); total 1980 -> 1740. A direct mid-task user
    follow-up then retired the merged scale+align design (see
    act3_align.py's TASK-15 PIVOT section) and shortened Act 3 again,
    360 -> 180 frames; total 1740 -> 1560. task-16: a second direct user
    pivot ("just the second half... where it just swings in and aligns")
    cut Act 3 again, 180 -> 90 frames (see act3_align.py's TASK-16 PIVOT
    section); total 1560 -> 1470. task-20: Act 1 extended 270 -> 330 frames,
    adding a 60-frame closing beat that crossfades raw detector 2D into the
    reprojection of the triangulated 3D; total 1470 -> 1530)."""
    assert assemble.EXPECTED_TOTAL == 1530
    assert sum(n for _, n in assemble.ACT_SPECS) == 1620
    assert assemble.N_CROSS == 30


def test_ffprobe_video_info_reports_exact_dims_not_macroblock_padded(tmp_path):
    """Regression test: write_video's default macro_block_size=16 pads
    1080 -> 1088 (1080 isn't a multiple of 16), which would silently violate
    the "exactly 1920x1080" contract. assemble_mp4 must pass
    macro_block_size=1, and ffprobe_video_info must read the real encoded
    dimensions by KEY (ffprobe's -of csv does not honour the requested field
    order -- verified directly against a real ffprobe binary)."""
    from viz.core.io import write_video
    out = tmp_path / "tiny.mp4"
    frames = [np.full((1080, 1920, 3), i * 40, np.uint8) for i in range(3)]
    write_video(str(out), frames, fps=30, macro_block_size=1)
    info = assemble.ffprobe_video_info(out)
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["n_frames"] == 3
    assert info["codec_name"] == "h264"


def test_stage_facts_rejects_non_monotonic_residuals(tmp_path, monkeypatch):
    """The residual-monotonicity guard in _stage_facts must actually fire on
    a non-monotonic sequence, not just pass on real (already-monotone) data."""
    clip = tmp_path / "clip"
    dirs_root = clip / "ik_explainer"
    (dirs_root / "predictions").mkdir(parents=True)

    def fake_out_dirs(clip_arg):
        return {"root": dirs_root, "predictions": dirs_root / "predictions"}

    monkeypatch.setattr(assemble.clip_io, "out_dirs", fake_out_dirs)

    np.savez(dirs_root / "predictions" / "06_stages.npz",
              stage_names=np.array(["default", "scaled", "root", "pose"]),
              residual_mm=np.array([1.0, 2.0, 0.5, 0.1]),  # non-monotonic (rises then falls)
              shared_scale=0.1, frame_for_stills=0)
    np.savez(dirs_root / "predictions" / "02_kp2d.npz",
              kp2d=np.zeros((5, 2, 3, 2)), cam_names=np.array(["Cam1", "Cam2"]))

    with pytest.raises(RuntimeError, match="not monotonically decreasing"):
        assemble._stage_facts(str(clip))
