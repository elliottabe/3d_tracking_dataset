import numpy as np
import jax, jax.numpy as jnp
import pytest
from flax import nnx

TINY = dict(crop=64, patch=16, embed_dim=32, num_keypoints=5, num_cameras=3, max_frames=4,
            n_instances=2, n_local=1, n_global=1, global_pool=2, dec_layers_3d=4, dec_layers_2d=1,
            dec_heads=4, mlp_ratio=2.0, refine_passes=1, patch_rgb=3, fourier_bands=4,
            roi_scale=24.0, backbone_depth=1, backbone_heads=4, remat=False)


def _inputs(B=2, T=2, C=3, H=64, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[8.1, 0.0, 0.0, 30.0], [0.0, -8.1, -0.2, 30.0], [0, 0, 0, 1.0]])
    Ms, ts = [], []
    for c in range(C):
        th = 2 * np.pi * c / C
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        Ms.append(P[:2, :3] @ Rz); ts.append(P[:2, 3])
    M = jnp.asarray(np.broadcast_to(np.stack(Ms), (B, C, 2, 3)).astype(np.float32))
    tl = jnp.asarray(np.broadcast_to(np.stack(ts), (B, T, C, 2)).astype(np.float32))
    crops = jnp.asarray(rng.normal(size=(B, T, C, H, H, 3)).astype(np.float32))
    cam_valid = jnp.ones((B, T, C), bool).at[1, :, 2].set(False)
    pm = jnp.zeros((B, T, C, H, H), bool).at[:, :, :, 20:40, 20:40].set(True)
    return crops, cam_valid, M, tl, pm


def test_output_shapes_and_aux():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(**TINY)
    m = MVQModel(cfg, rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    out = m(crops, cv, M, tl, pm, prompt_on=jnp.array([True, False]))
    B, T, C, I, K = 2, 2, 3, 2, 5
    assert out["xyz"].shape == (B, I, T, K, 3) and out["conf_logit"].shape == (B, I, T, K)
    assert out["exist_logit"].shape == (B, I)
    assert out["uv"].shape == (B, I, T, C, K, 2) and out["vis_logit"].shape == (B, I, T, C, K)
    assert out["aux_pass1"] is not None and set(out["aux_pass1"]) >= {"xyz", "uv", "conf_logit", "vis_logit", "exist_logit"}
    assert isinstance(out["aux_layers"], list) and all("xyz" in a for a in out["aux_layers"])
    # dec_layers_3d=4 gives one deep-supervision readout (layer index 1) per
    # decoder() call; refine_passes=1 means two calls (pass 1 + the refine
    # pass), so exactly 2 entries -- pins the non-empty case fix-round-1 asked for.
    assert len(out["aux_layers"]) == 2
    assert bool(jnp.isfinite(out["xyz"]).all()) and bool((out["uv"] >= 0).all()) and bool((out["uv"] <= 64).all())
    # heads.xyz init: small-scale, not zero (see decoder.py Heads NOTE). At
    # fresh init refine_in/e_pass are zero, so pass 2's query equals pass 1's
    # and the residual add exactly doubles pass 1's xyz -- pin both the small
    # magnitude and the exact doubling so a future init-scale change is caught.
    assert float(jnp.abs(out["xyz"]).max()) < 1.0
    doubling_err = jnp.abs(out["xyz"] - 2 * out["aux_pass1"]["xyz"])
    assert float(doubling_err.max()) < 1e-4


def test_invalid_camera_does_not_influence_outputs():
    """Zero/garbage in a masked view must not change the 3D output."""
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    m = MVQModel(MVQConfig(**TINY), rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    cv = cv.at[:, :, 1].set(False)
    a = m(crops, cv, M, tl, pm)["xyz"]
    crops2 = crops.at[:, :, 1].set(crops[:, :, 1] * 100.0 + 7.0)
    b = m(crops2, cv, M, tl, pm)["xyz"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_prompt_changes_only_instance_zero_when_on():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    m = MVQModel(MVQConfig(**TINY), rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    off = m(crops, cv, M, tl, pm, prompt_on=jnp.array([False, False]))["xyz"]
    on = m(crops, cv, M, tl, pm, prompt_on=jnp.array([True, True]))["xyz"]
    assert not np.allclose(np.asarray(off[:, 0]), np.asarray(on[:, 0]))
    # instance 1 sees the prompt only through self-attention; with fresh weights the
    # difference must be much smaller than instance 0's
    d0 = float(jnp.abs(off[:, 0] - on[:, 0]).mean()); d1 = float(jnp.abs(off[:, 1] - on[:, 1]).mean())
    assert d1 < d0


def test_zero_global_layers_is_approach_b():
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(**{**TINY, "n_global": 0})
    m = MVQModel(cfg, rngs=nnx.Rngs(0))
    crops, cv, M, tl, pm = _inputs()
    assert m(crops, cv, M, tl, pm)["xyz"].shape == (2, 2, 2, 5, 3)


def test_fusion_layerscale_zero_init_is_identity():
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.models.mvq.fusion import FusionStack
    cfg = MVQConfig(**TINY)
    fs = FusionStack(cfg, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 6, 16, 32)).astype(np.float32))
    valid = jnp.ones((2, 6), bool)
    np.testing.assert_allclose(np.asarray(fs(x, valid)), np.asarray(x), atol=1e-6)


def test_assemble_nan_policy():
    from jarvis_jax.models.mvq.model import assemble
    B, I, T, K, C = 1, 2, 1, 3, 2
    out = {"xyz": np.zeros((B, I, T, K, 3), np.float32), "conf_logit": np.zeros((B, I, T, K), np.float32),
           "exist_logit": np.array([[3.0, -3.0]], np.float32),
           "uv": np.zeros((B, I, T, C, K, 2), np.float32), "vis_logit": np.zeros((B, I, T, C, K), np.float32)}
    kp3d, conf3d, kp2d = assemble(out, center3D=np.array([[1.0, 2.0, 3.0]]),
                                  crop_origin=np.zeros((B, C, 2)), exist_thresh=0.5)
    assert np.isnan(kp3d[0, 1]).all() and (conf3d[0, 1] == 0).all()
    np.testing.assert_allclose(kp3d[0, 0, 0, 0], [1.0, 2.0, 3.0])

    # cam_valid: a frame with NO valid camera at all must be NaN/conf-0 for
    # every instance and keypoint in that frame, independent of exist_logit
    # (both instances "exist" here) -- the other half of spec Sec 4.6's NaN
    # policy that exist_thresh alone does not cover.
    T2 = 2
    out2 = {"xyz": np.ones((B, I, T2, K, 3), np.float32), "conf_logit": np.ones((B, I, T2, K), np.float32),
            "exist_logit": np.array([[3.0, 3.0]], np.float32),
            "uv": np.zeros((B, I, T2, C, K, 2), np.float32), "vis_logit": np.zeros((B, I, T2, C, K), np.float32)}
    cam_valid = np.ones((B, T2, C), bool)
    cam_valid[0, 0, :] = False
    kp3d2, conf3d2, _ = assemble(out2, center3D=np.array([[1.0, 2.0, 3.0]]),
                                 crop_origin=np.zeros((B, C, 2)), exist_thresh=0.5, cam_valid=cam_valid)
    assert np.isnan(kp3d2[0, :, 0]).all() and (conf3d2[0, :, 0] == 0).all()
    assert np.isfinite(kp3d2[0, :, 1]).all() and (conf3d2[0, :, 1] > 0).all()


def test_masked_attention_chunked_matches_unchunked():
    """Chunked (q_chunk=8) and unchunked (q_chunk=None) attention must agree
    to float32 tolerance for the same non-trivial key mask: attention is
    independent per query row, so chunking only trades memory for a bit of
    extra sequencing, never changes the numbers."""
    from jarvis_jax.models.mvq.fusion import masked_attention
    rng = np.random.default_rng(0)
    B, Nq, Nk, H, D = 2, 30, 17, 4, 32
    q = jnp.asarray(rng.normal(size=(B, Nq, D)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    key_valid = jnp.asarray(rng.uniform(size=(B, Nk)) > 0.3)
    key_valid = key_valid.at[:, 0].set(True)  # keep >=1 valid key per row, avoid an all-masked degenerate case
    full = masked_attention(q, k, v, key_valid, H, q_chunk=None)
    chunked = masked_attention(q, k, v, key_valid, H, q_chunk=8)
    np.testing.assert_allclose(np.asarray(full), np.asarray(chunked), atol=1e-5)


def test_masked_attention_chunked_grad_matches_unchunked():
    """jax.grad through the chunked (q_chunk=8) path must be finite and match
    the unchunked (q_chunk=None) gradient to 1e-5 -- regression test for the
    lax.map/remat fix: before it, chunking was numerically a no-op (the test
    above) but under grad it retained every chunk's logits simultaneously
    (worse than not chunking at all), which this test cannot see unless it
    actually differentiates."""
    from jarvis_jax.models.mvq.fusion import masked_attention
    rng = np.random.default_rng(0)
    B, Nq, Nk, H, D = 2, 30, 17, 4, 32
    q = jnp.asarray(rng.normal(size=(B, Nq, D)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(B, Nk, D)).astype(np.float32))
    key_valid = jnp.asarray(rng.uniform(size=(B, Nk)) > 0.3)
    key_valid = key_valid.at[:, 0].set(True)

    def loss(q, chunk):
        return masked_attention(q, k, v, key_valid, H, q_chunk=chunk).sum()

    g_full = jax.grad(lambda q: loss(q, None))(q)
    g_chunked = jax.grad(lambda q: loss(q, 8))(q)
    assert np.isfinite(np.asarray(g_chunked)).all()
    np.testing.assert_allclose(np.asarray(g_full), np.asarray(g_chunked), atol=1e-5)
