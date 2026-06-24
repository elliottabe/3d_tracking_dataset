"""GPU test: batched/sharded HybridNet3D inference.

Verifies:
1. Output shapes (B,50,3) and (B,50) for kp3d and conf.
2. All outputs are finite.
3. Batch-invariance: processing a single item in a B=1 batch (padded to nd)
   matches the same item's prediction from a larger mixed batch, within atol=1e-3.
"""
import os, numpy as np, jax, pytest

gpu = any(d.platform == "gpu" for d in jax.devices())
needs_gpu = pytest.mark.skipif(not gpu, reason="needs GPU")

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
VIT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final"
RUN4 = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/run4/final"


@needs_gpu
def test_predict_batch_finite_and_batch_invariant():
    from jarvis_jax.predict.infer_3d import load_inference_model, predict_batch
    from jarvis_jax.data.v3_3d import V3FramesetDataset
    ds = V3FramesetDataset(ROOT, "val")
    nd = jax.device_count()
    B = max(2, nd)                                  # divisible by device count when nd<=2
    B = B - (B % nd)
    crops = np.stack([ds[i]["crops4"] for i in range(B)])
    chm = np.stack([ds[i]["centerHM"] for i in range(B)]).astype("float32")
    cams = np.stack([ds[i]["cameraMatrices"] for i in range(B)]).astype("float32")
    model = load_inference_model(VIT, RUN4, sharpen=3.0)
    kp, conf, c3d = predict_batch(model, crops, chm, cams)
    assert kp.shape == (B, 50, 3) and conf.shape == (B, 50)
    assert np.isfinite(np.asarray(kp)).all()
    # batch-invariance: first item matches a B=nd minibatch of the same item repeated
    rep = np.repeat(crops[:1], nd, 0)
    rchm = np.repeat(chm[:1], nd, 0); rcams = np.repeat(cams[:1], nd, 0)
    kp1, _, _ = predict_batch(model, rep, rchm, rcams)
    diff = np.abs(np.asarray(kp[0]) - np.asarray(kp1[0])).max()
    print(f"\nbatch-invariance max abs diff: {diff:.6f}")
    assert np.allclose(np.asarray(kp[0]), np.asarray(kp1[0]), atol=1e-3), \
        f"batch-invariance failed: max diff {diff:.4f} > 1e-3 (sharding/jit bug)"


@needs_gpu
def test_benchmark_throughput_reports_rate():
    import importlib.util, os
    import jarvis_jax, os as _os
    pkg_root = _os.path.dirname(_os.path.dirname(jarvis_jax.__file__))
    spec = importlib.util.spec_from_file_location(
        "predict_3d", _os.path.join(pkg_root, "scripts", "predict_3d.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    from jarvis_jax.predict.infer_3d import load_inference_model
    from jarvis_jax.data.v3_3d import V3FramesetDataset
    import jax
    ds = V3FramesetDataset(ROOT, "val")
    model = load_inference_model(VIT, RUN4, sharpen=3.0)
    nd = jax.device_count(); B = max(nd, 4) - (max(nd, 4) % nd)
    r = mod.benchmark_throughput(model, ds, batch=B, n=B * 2, nd=nd)
    assert r["frames_per_s"] > 0 and r["ms_per_frameset"] > 0
