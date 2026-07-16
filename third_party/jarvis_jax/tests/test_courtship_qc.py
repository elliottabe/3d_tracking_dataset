import json
import numpy as np
from jarvis_jax.tracking.session_qc import aggregate_session_qc


def _bout_qc(p, iou, reproj, n):
    json.dump({"silhouette_iou": {"hard_median": iou, "soft_median": iou + 0.02, "n_frames": n},
               "per_camera_reproj_px": {"median": reproj, "n": n * 7},
               "loo_reproj_px": {"median": reproj + 1.0, "n_frames": n}, "n_frames": n}, open(p, "w"))


def test_aggregate_reads_bouts_and_writes_summary(tmp_path):
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
