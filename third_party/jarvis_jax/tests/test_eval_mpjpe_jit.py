"""eval_mpjpe's forward was eager (~47s/batch for EfficientTrack b3+BiFPN,
~34min for a full val pass) -- jarvis_jax/train/train.py wraps it in
``@nnx.jit`` (module-level ``_eval_forward``) to compile once per input shape
and run fast thereafter. This is architecture-agnostic (the trainer calls the
model generically -- ``model(img, use_running_average=...)`` -> heatmaps --
see test_train_efficienttrack_2d.py's docstring), so a tiny ViTPose exercises
the same code path as the real EfficientTrack fix without the CPU cost of a
real b3+BiFPN forward.

The safety gate is numerical EQUIVALENCE: jit must not change the result.
We compute the same MPJPE two ways -- once via ``eval_mpjpe`` (jitted) and
once via a hand-rolled eager loop using the exact same math -- and assert
they agree to atol=1e-4.
"""
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vitpose import ViTPose
from jarvis_jax.data.device import normalize_image
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints, mpjpe
from jarvis_jax.train.train import eval_mpjpe

NUM_KEYPOINTS = 5
IMG_SIZE = 64
HEATMAP_SIZE = 32  # ClassicDecoder does a fixed 8x upsample of the token grid
                    # (img_size/patch)**0.5 = 4 -> 4*8=32, matching this value.
IN_SIZE = 448       # eval_mpjpe's default keypoint-space scale target


class _TinyDS:
    """Minimal stand-in for V3Dataset exposing exactly what eval_mpjpe/batches
    read: ``heatmap_size``, ``__len__``, and ``__getitem__`` -> (img4_u8,
    kp_xy, vis) per-sample tuples (see jarvis_jax/data/v3.py and the identical
    pattern in tests/test_train_efficienttrack_2d.py)."""

    def __init__(self, imgs, kps, viss, heatmap_size=HEATMAP_SIZE):
        self.imgs, self.kps, self.viss = imgs, kps, viss
        self.heatmap_size = heatmap_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.kps[i], self.viss[i]


def _make_model():
    cfg = ViTPoseConfig(img_size=IMG_SIZE, patch=16, embed_dim=48, depth=2,
                        num_heads=4, num_keypoints=NUM_KEYPOINTS,
                        heatmap_size=HEATMAP_SIZE)
    return ViTPose(cfg, rngs=nnx.Rngs(0))


def _synthetic_dataset(rng, n=5):
    # 5 samples, deliberately not a multiple of batch_size=2 so eval_mpjpe
    # exercises both a full batch and a partial last batch (an extra jit
    # compile for the odd shape -- still must match numerically).
    img4_u8 = rng.integers(0, 256, size=(n, IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
    kp_xy = rng.uniform(0, HEATMAP_SIZE, size=(n, NUM_KEYPOINTS, 2)).astype(np.float32)
    vis = rng.integers(0, 2, size=(n, NUM_KEYPOINTS)).astype(np.float32)
    vis[0, :] = 1.0  # guarantee at least one fully-visible sample
    return img4_u8, kp_xy, vis


def _eager_reference_mpjpe(model, ds, batch_size, in_size=IN_SIZE):
    """Same computation as eval_mpjpe, but with an un-jitted (eager) forward
    call -- the pre-fix code path. Used only as the equivalence oracle."""
    from jarvis_jax.data.v3 import batches
    model.eval()
    scale = in_size / float(ds.heatmap_size)
    total, count = 0.0, 0
    for img4_u8, kp_xy, vis in batches(ds, batch_size, shuffle=False, drop_last=False):
        img = normalize_image(jnp.asarray(img4_u8))
        pred = model(img, use_running_average=True)   # eager, no jit
        pk = heatmaps_to_keypoints(pred, in_size=in_size)
        gk = jnp.asarray(kp_xy) * scale
        n = int(vis.sum())
        if n == 0:
            continue
        total += float(mpjpe(pk, gk, jnp.asarray(vis))) * n
        count += n
    return total / max(count, 1)


def test_eval_mpjpe_jit_runs_and_is_finite():
    rng = np.random.default_rng(0)
    model = _make_model()
    img4_u8, kp_xy, vis = _synthetic_dataset(rng)
    ds = _TinyDS(img4_u8, kp_xy, vis)

    result = eval_mpjpe(model, ds, batch_size=2)

    assert np.isfinite(result)
    assert result >= 0.0


def test_eval_mpjpe_jit_matches_eager_reference():
    """The safety gate: jitting the forward must not change the numeric result."""
    rng = np.random.default_rng(0)
    model = _make_model()
    img4_u8, kp_xy, vis = _synthetic_dataset(rng)
    ds = _TinyDS(img4_u8, kp_xy, vis)

    jitted = eval_mpjpe(model, ds, batch_size=2)
    eager = _eager_reference_mpjpe(model, ds, batch_size=2)

    assert np.isfinite(jitted) and np.isfinite(eager)
    assert abs(jitted - eager) < 1e-4, (jitted, eager)
