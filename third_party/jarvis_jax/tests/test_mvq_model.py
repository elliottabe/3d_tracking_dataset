import numpy as np
import jax, jax.numpy as jnp
import pytest
from flax import nnx

TINY = dict(crop=64, patch=16, embed_dim=32, num_keypoints=5, num_cameras=3, max_frames=4,
            n_instances=2, n_local=1, n_global=1, global_pool=2, dec_layers_3d=2, dec_layers_2d=1,
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
    assert bool(jnp.isfinite(out["xyz"]).all()) and bool((out["uv"] >= 0).all()) and bool((out["uv"] <= 64).all())


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
