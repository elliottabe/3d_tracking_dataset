import json
import numpy as np
from jarvis_jax.tracking.session_qc import aggregate_session_qc, _parse_name


def _bout_qc(p, iou, reproj, n, *, key="mesh_mask_iou"):
    json.dump({key: {"hard_median": iou, "soft_median": iou + 0.02, "n_frames": n},
               "per_camera_reproj_px": {"median": reproj, "n": n * 7},
               "loo_reproj_px": {"median": reproj + 1.0, "n_frames": n}, "n_frames": n}, open(p, "w"))


def test_aggregate_reads_bouts_and_writes_summary(tmp_path):
    """Flat-named qc.json paths (pre-rewrite convention) still aggregate on
    the CURRENT key (mesh_mask_iou)."""
    paths = []
    for i, (iou, rp, n) in enumerate([(0.74, 5.0, 100), (0.71, 6.5, 80)]):
        pth = tmp_path / f"bout_{i:05d}_fly0_qc.json"; _bout_qc(pth, iou, rp, n); paths.append(str(pth))
    out = tmp_path / "session_qc.json"
    summ = aggregate_session_qc(paths, str(out), plot_dir=str(tmp_path))
    assert summ["n_bouts_flies"] == 2
    assert abs(summ["iou_hard_median"] - 0.725) < 1e-6
    assert out.exists() and json.load(open(out))["n_bouts_flies"] == 2
    assert (tmp_path / "iou_by_bout.png").exists()


def test_aggregate_skips_missing(tmp_path):
    summ = aggregate_session_qc([str(tmp_path / "nope.json")], str(tmp_path / "s.json"))
    assert summ["n_bouts_flies"] == 0


def test_aggregate_real_run_root_layout_parses_bout_and_fly(tmp_path):
    """The real on-disk layout is <run_root>/bouts/bout_NNNNN/flyF/qc.json --
    every file is named "qc.json" (not bout_N_flyF_qc.json), so bout/fly MUST
    come from the path, not the basename. Regression test for the bug that
    made every session-dashboard row read bout -1 / fly -1."""
    run_root = tmp_path / "pose_mvq_p3a_r2"
    specs = [(8, 1, 0.80, 4.0, 120), (3, 0, 0.40, 9.0, 90)]
    paths = []
    for bout, fly, iou, reproj, n in specs:
        d = run_root / "bouts" / f"bout_{bout:05d}" / f"fly{fly}"
        d.mkdir(parents=True)
        p = d / "qc.json"
        _bout_qc(p, iou, reproj, n)
        paths.append(str(p))

    for (bout, fly, *_), p in zip(specs, paths):
        assert _parse_name(p) == (bout, fly)

    out = run_root / "qc" / "session_qc.json"
    summ = aggregate_session_qc(paths, str(out))
    assert summ["n_bouts_flies"] == 2
    rows = {(r["bout"], r["fly"]): r for r in summ["rows"]}
    assert set(rows) == {(8, 1), (3, 0)}
    assert (-1, -1) not in rows
    assert abs(rows[(8, 1)]["iou_hard"] - 0.80) < 1e-6
    assert abs(rows[(3, 0)]["iou_hard"] - 0.40) < 1e-6
    assert abs(summ["iou_hard_median"] - 0.60) < 1e-6  # median(0.80, 0.40)


def test_aggregate_falls_back_to_old_silhouette_iou_key(tmp_path):
    """qc.json from before the 2026-09-01 rename (key still `silhouette_iou`)
    must still contribute a real IoU number, not NaN."""
    run_root = tmp_path / "pose_v1"
    d = run_root / "bouts" / "bout_00028" / "fly0"; d.mkdir(parents=True)
    p = d / "qc.json"
    _bout_qc(p, 0.65, 3.0, 200, key="silhouette_iou")
    summ = aggregate_session_qc([str(p)], str(run_root / "qc" / "session_qc.json"))
    assert summ["n_bouts_flies"] == 1
    assert abs(summ["iou_hard_median"] - 0.65) < 1e-6
    assert summ["rows"][0]["bout"] == 28 and summ["rows"][0]["fly"] == 0


def test_parse_name_returns_minus_one_when_unmatched():
    assert _parse_name("/some/random/path/qc.json") == (-1, -1)
