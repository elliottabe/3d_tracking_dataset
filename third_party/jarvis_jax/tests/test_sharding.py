import jax, jax.numpy as jnp, numpy as np, pytest
from flax import nnx
from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.sharding import data_parallel_mesh, shard_batch, replicate

@pytest.mark.skipif(jax.device_count() < 2, reason="needs >=2 devices")
def test_sharded_forward_matches_single():
    cfg = ViTPoseConfig()
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    x = jnp.asarray(np.random.default_rng(0).standard_normal((4, 448, 448, 4)), jnp.float32)
    single = np.asarray(nnx.jit(lambda mm, xx: mm(xx, use_running_average=True))(m, x))
    mesh = data_parallel_mesh()
    xs = shard_batch(x, mesh)
    sh = np.asarray(nnx.jit(lambda mm, xx: mm(xx, use_running_average=True))(m, xs))
    assert sh.shape == single.shape
    assert np.abs(sh - single).max() < 1e-3


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >=2 devices")
def test_replicate_lets_sharded_batch_meet_device0_params():
    # Reproduces the resume bug: params committed to device 0 (as Orbax restore
    # leaves them) clash with a data-sharded batch in a jitted step. replicate()
    # across the mesh fixes it.
    from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
    cfg = ViTPoseConfig()
    mesh = data_parallel_mesh()
    nd = jax.device_count()
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    opt = make_optimizer(m, TrainConfig(total_steps=5, lr=1e-3, warmup_steps=1))
    step = make_train_step(0.0)
    img = shard_batch(jnp.zeros((nd, 448, 448, 4), jnp.uint8), mesh)
    kp = shard_batch(jnp.zeros((nd, 50, 2), jnp.float32), mesh)
    vis = shard_batch(jnp.zeros((nd, 50), bool), mesh)

    # Commit params/opt to device 0 (what Orbax restore does), then the jit step
    # must raise on the device mismatch.
    dev0 = jax.devices()[0]
    gm, sm = nnx.split(m); m = nnx.merge(gm, jax.device_put(sm, dev0))
    go, so = nnx.split(opt); opt = nnx.merge(go, jax.device_put(so, dev0))
    with pytest.raises(ValueError, match="incompatible devices"):
        float(step(m, opt, img, kp, vis))

    # replicate() across the mesh -> step runs.
    gm, sm = nnx.split(m); m = nnx.merge(gm, replicate(sm, mesh))
    go, so = nnx.split(opt); opt = nnx.merge(go, replicate(so, mesh))
    assert np.isfinite(float(step(m, opt, img, kp, vis)))
