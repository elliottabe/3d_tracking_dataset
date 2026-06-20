import numpy as np
import jax.numpy as jnp
from flax import nnx
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step
from jarvis_jax.train.checkpoint import make_manager, save_step, restore_latest


def _pval(model):
    return float(jnp.ravel(
        nnx.state(model, nnx.Param)["decoder"]["head"]["bias"].value)[0])


def test_checkpoint_resume_roundtrip(tmp_path):
    cfg = ViTPoseConfig()
    m = ViTPose(cfg, rngs=nnx.Rngs(0))
    tcfg = TrainConfig(total_steps=50, lr=1e-3, warmup_steps=2)
    opt = make_optimizer(m, tcfg)
    step = make_train_step(0.0)
    img = jnp.asarray(np.random.RandomState(0).randint(0, 256, (2, 448, 448, 4), np.uint8))
    kp = jnp.zeros((2, 50, 2), jnp.float32).at[:, 0].set(jnp.array([100., 50.]))
    vis = jnp.zeros((2, 50), bool).at[:, 0].set(True)
    for _ in range(5):
        step(m, opt, img, kp, vis)
    p_before = _pval(m)

    mngr = make_manager(str(tmp_path / "ck"))
    save_step(mngr, 5, m, opt)
    mngr.wait_until_finished()
    assert mngr.latest_step() == 5

    # fresh build, then auto-resume
    m2 = ViTPose(cfg, rngs=nnx.Rngs(9))
    opt2 = make_optimizer(m2, tcfg)
    m2r, opt2r, start = restore_latest(mngr, m2, opt2)
    assert start == 5
    assert abs(_pval(m2r) - p_before) < 1e-6        # params restored
    # resumed training continues with finite loss
    assert np.isfinite(float(step(m2r, opt2r, img, kp, vis)))
