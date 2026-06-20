import jax, jax.numpy as jnp, numpy as np, pytest
from flax import nnx
from jarvis_jax import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.sharding import data_parallel_mesh, shard_batch

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
