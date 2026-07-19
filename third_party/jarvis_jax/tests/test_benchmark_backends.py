# tests/test_benchmark_backends.py
"""Task 10: backend x fusion-variant benchmark harness.

Weight-free, CPU-fast: `load_inference_model`/`predict_batch` are monkeypatched
so the harness's own plumbing (dispatch, LOO/reproj accuracy math, throughput,
JSON report, skip accounting) is exercised without touching real checkpoints.
The real-weights benchmark run is a separate manual step (Task 11).
"""
import json
import os

import numpy as np


class _FakeRT:
    """Minimal ReprojectionTool stand-in, same convention as tests/test_qc.py:
    two orthographic-ish cameras (cam0 sees (x,y), cam1 sees (x,z)) so a 3-D
    point's DLT reprojection is exactly recoverable by hand."""

    def __init__(self):
        self._P = [
            np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
            np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ]
        self.num_cameras = 2

    def reproject_point(self, X):
        X = np.asarray(X, dtype=float)
        Xh = np.concatenate([X, [1.0]])
        out = np.zeros((self.num_cameras, 2))
        for i, P in enumerate(self._P):
            p = P @ Xh
            out[i] = (p / p[2])[:2]
        return out


_J = 3
# Fixed ground-truth 3-D keypoints shared by every synthetic frame, so the
# fake predict_batch (below) can return exactly this value and the resulting
# LOO/reproj accuracy is deterministically ~0 px -- a sanity check that the
# harness's reprojection math is wired correctly, not just that keys exist.
_KP3D_TRUE = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [-0.5, 0.5, 1.0]])


def _make_frame(rt, *, nc, bad=False):
    if bad:
        # Deliberately incomplete frame (missing centerHM/cameraMatrices/
        # kp2d_by_cam/vis_by_cam) -- must cause the harness to log + record a
        # skip for this bout, never silently drop it from consideration.
        return {"crops4": np.zeros((nc, 4, 4, 4), dtype=np.uint8)}
    kp2d_by_cam = {
        c: np.stack([rt.reproject_point(_KP3D_TRUE[j])[c] for j in range(_J)])
        for c in range(nc)
    }
    vis_by_cam = {c: np.ones(_J, dtype=bool) for c in range(nc)}
    return {
        "crops4": np.zeros((nc, 4, 4, 4), dtype=np.uint8),
        "centerHM": np.zeros((nc, 2), dtype=np.float32),
        "cameraMatrices": np.zeros((nc, 4, 3), dtype=np.float32),
        "masks": np.zeros((nc, 8, 8), dtype=np.float32),
        "kp2d_by_cam": kp2d_by_cam,
        "vis_by_cam": vis_by_cam,
    }


def _tiny_fake_bouts(n_bouts=2, n_frames=2):
    """Tiny synthetic bout set: n_bouts bouts x n_frames frames, 2 cameras,
    3 keypoints, plausible (if degenerate) camera geometry + 2-D detections."""
    bouts = []
    for b in range(n_bouts):
        rt = _FakeRT()
        frames = [_make_frame(rt, nc=rt.num_cameras) for _ in range(n_frames)]
        bouts.append({"bout_id": f"bout{b}", "rt": rt, "frames": frames})
    return bouts


def _fake_load_inference_model(*args, **kwargs):
    return {"fusion_mode": kwargs.get("fusion_mode", "none")}


def _fake_predict_batch(model, crops4, centerHM, cameraMatrices, masks=None):
    B = crops4.shape[0]
    kp3d = np.broadcast_to(_KP3D_TRUE, (B, _J, 3)).copy()
    conf = np.ones((B, _J), dtype=np.float32)
    center3D = np.zeros((B, 3), dtype=np.float32)
    return kp3d, conf, center3D


def test_run_benchmark_writes_report(tmp_path, monkeypatch):
    from jarvis_jax.eval import benchmark_backends

    monkeypatch.setattr(benchmark_backends, "load_inference_model",
                         _fake_load_inference_model)
    monkeypatch.setattr(benchmark_backends, "predict_batch", _fake_predict_batch)

    out = tmp_path / "bench.json"
    rep = benchmark_backends.run_benchmark(
        bouts=_tiny_fake_bouts(), backends=["hybridnet_jax"],
        fusion_variants=["none", "carve"], out_json=str(out),
    )

    assert os.path.exists(out)
    with open(out) as f:
        on_disk = json.load(f)
    assert on_disk.keys() == rep.keys()

    for k in ["hybridnet_jax:none", "hybridnet_jax:carve"]:
        assert set(rep[k]) >= {"mean_reproj_px", "p95_reproj_px", "fps", "n_frames"}
        assert rep[k]["n_frames"] == 4  # 2 bouts x 2 frames, none skipped
        assert rep[k]["mean_reproj_px"] < 1e-6
        assert rep[k]["p95_reproj_px"] < 1e-6
        assert rep[k]["fps"] > 0
    assert rep["skipped"] == []


def test_run_benchmark_records_skipped_bout_without_dropping_silently(
        tmp_path, monkeypatch, capsys):
    from jarvis_jax.eval import benchmark_backends

    monkeypatch.setattr(benchmark_backends, "load_inference_model",
                         _fake_load_inference_model)
    monkeypatch.setattr(benchmark_backends, "predict_batch", _fake_predict_batch)

    good_bouts = _tiny_fake_bouts(n_bouts=1, n_frames=2)
    rt = _FakeRT()
    bad_bout = {"bout_id": "bad_bout", "rt": rt,
                "frames": [_make_frame(rt, nc=rt.num_cameras, bad=True)]}
    bouts = good_bouts + [bad_bout]

    out = tmp_path / "bench.json"
    rep = benchmark_backends.run_benchmark(
        bouts=bouts, backends=["hybridnet_jax"], fusion_variants=["none"],
        out_json=str(out),
    )

    # Only the good bout's 2 frames are counted -- the bad bout must not
    # silently vanish from n_frames without a trace.
    assert rep["hybridnet_jax:none"]["n_frames"] == 2
    skipped_ids = [s["bout_id"] for s in rep["skipped"]]
    assert "bad_bout" in skipped_ids
    # Skip must also be logged (printed), not just recorded in the dict.
    assert "bad_bout" in capsys.readouterr().out
