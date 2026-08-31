"""Tests for the center-channel (instance-cue) ablation arm:
``jarvis_jax/data/center_channel.py::CenterChannelDataset``,
``jarvis_jax/data/device.py::normalize_image_center_channel``, and the
``normalize_fn`` wiring through ``make_train_step``/``eval_mpjpe``.

All CPU (JAX_PLATFORMS=cpu); no real dataset root or GPU needed.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jarvis_jax.data.center_channel import CenterChannelDataset, DEFAULT_CENTER_SIGMA_PX
from jarvis_jax.data.device import normalize_image, normalize_image_center_channel
from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.models.efficienttrack import EfficientTrack
from jarvis_jax.train.train import TrainConfig, make_optimizer, make_train_step


class _FakeDS:
    """Minimal stand-in for V3Dataset/V5Dataset -- only the fields
    CenterChannelDataset reads (bboxes/img_wh/crop) plus one pass-through
    attribute (sex) to check __getattr__ forwarding."""

    def __init__(self):
        self.crop = 448
        self.bboxes = [np.array([100.0, 50.0, 40.0, 60.0], dtype=np.float32)]  # x,y,w,h
        self.img_wh = [(1936, 448)]
        self.sex = ["male"]
        self.file_names = ["f"]

    def __len__(self):
        return 1

    def __getitem__(self, i):
        img4 = np.zeros((448, 448, 4), dtype=np.uint8)
        img4[..., 3] = 7          # sentinel -- must be fully overwritten
        img4[..., 0] = 3          # sentinel RGB -- must be left untouched
        kp = np.zeros((1, 2), dtype=np.float32)
        vis = np.array([True])
        return img4, kp, vis


# ---------------------------------------------------------------------------
# CenterChannelDataset
# ---------------------------------------------------------------------------

def test_replaces_channel_with_gaussian_at_gt_bbox_center():
    ds = CenterChannelDataset(_FakeDS(), sigma=5.0)
    img4, kp, vis = ds[0]
    assert img4.shape == (448, 448, 4)

    bbox = [100.0, 50.0, 40.0, 60.0]
    x0, y0 = crop_origin(bbox, 1936, 448, 448)
    fx, fy = bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0     # (120, 80)
    exp_cx, exp_cy = fx - x0, fy - y0

    ch3 = img4[..., 3]
    peak_y, peak_x = np.unravel_index(np.argmax(ch3), ch3.shape)
    assert (peak_y, peak_x) == (int(round(exp_cy)), int(round(exp_cx)))
    assert ch3.max() >= 250                     # peak of a unit Gaussian, uint8-quantized
    # RGB channels (and the sentinel written by _FakeDS) pass through unchanged.
    assert np.array_equal(img4[..., 0], np.full((448, 448), 3, dtype=np.uint8))


def test_forwards_unknown_attributes_to_wrapped_dataset():
    ds = CenterChannelDataset(_FakeDS())
    assert ds.sex == ["male"]
    assert ds.file_names == ["f"]
    assert len(ds) == 1


def test_predicted_centers_override_gt_bbox_center():
    """The eval-time 'what actually deploys' path: a predicted (CenterDetect)
    center, not the GT bbox center, drives the blob."""
    fake = _FakeDS()
    predicted = np.array([[300.0, 200.0]])       # far from the GT bbox center (120, 80)
    ds = CenterChannelDataset(fake, predicted_centers_xy=predicted, sigma=5.0)
    img4, kp, vis = ds[0]

    x0, y0 = crop_origin(fake.bboxes[0], 1936, 448, 448)
    ch3 = img4[..., 3]
    peak_y, peak_x = np.unravel_index(np.argmax(ch3), ch3.shape)
    assert (peak_y, peak_x) == (int(round(200.0 - y0)), int(round(300.0 - x0)))


def test_predicted_centers_shape_mismatch_raises():
    fake = _FakeDS()
    bad = np.zeros((2, 2), dtype=np.float32)     # len(ds)=1, not 2
    try:
        CenterChannelDataset(fake, predicted_centers_xy=bad)
        assert False, "expected a ValueError for a shape mismatch"
    except ValueError:
        pass


def test_default_sigma_matches_documented_constant():
    assert DEFAULT_CENTER_SIGMA_PX == 20.0
    ds = CenterChannelDataset(_FakeDS())
    assert ds._sigma == DEFAULT_CENTER_SIGMA_PX


# ---------------------------------------------------------------------------
# normalize_image_center_channel
# ---------------------------------------------------------------------------

def test_normalize_center_channel_rescales_4th_channel_by_255():
    img4 = np.zeros((1, 4, 4, 4), dtype=np.uint8)
    img4[..., 3] = 128
    mask_style = normalize_image(jnp.asarray(img4))
    center_style = normalize_image_center_channel(jnp.asarray(img4))

    assert float(mask_style[0, 0, 0, 3]) == 128.0             # mask convention: passed through raw
    assert abs(float(center_style[0, 0, 0, 3]) - 128.0 / 255.0) < 1e-6
    # RGB channels are identical between the two -- only ch3's scaling differs.
    assert np.allclose(np.asarray(mask_style[..., :3]), np.asarray(center_style[..., :3]))


# ---------------------------------------------------------------------------
# make_train_step / eval_mpjpe normalize_fn wiring
# ---------------------------------------------------------------------------

def test_make_train_step_invokes_custom_normalize_fn():
    calls = []

    def spy(img4_u8):
        calls.append(img4_u8.shape)
        return normalize_image(img4_u8)

    model = EfficientTrack(num_joints=1, in_channels=4, model_size="medium", rngs=nnx.Rngs(0))
    tcfg = TrainConfig(total_steps=1, warmup_steps=0)
    opt = make_optimizer(model, tcfg)
    step = make_train_step(mask_weight=0.0, heatmap_size=160, sigma=2.5, normalize_fn=spy)

    img4 = jnp.asarray(np.zeros((1, 320, 320, 4), dtype=np.uint8))
    kp = jnp.asarray(np.array([[[10.0, 10.0]]], dtype=np.float32))
    vis = jnp.asarray(np.ones((1, 1), dtype=bool))
    step(model, opt, jax.random.PRNGKey(0), img4, kp, vis)

    assert len(calls) == 1, "custom normalize_fn was not invoked by the train step"


def test_make_train_step_default_matches_explicit_normalize_image():
    """No normalize_fn passed -> byte-identical to passing normalize_image
    explicitly (both existing callers rely on this default staying put)."""
    img4 = jnp.asarray(np.random.RandomState(0).randint(0, 256, (1, 320, 320, 4)).astype(np.uint8))
    kp = jnp.asarray(np.array([[[10.0, 10.0]]], dtype=np.float32))
    vis = jnp.asarray(np.ones((1, 1), dtype=bool))
    key = jax.random.PRNGKey(0)

    model_a = EfficientTrack(num_joints=1, in_channels=4, model_size="medium", rngs=nnx.Rngs(0))
    model_b = EfficientTrack(num_joints=1, in_channels=4, model_size="medium", rngs=nnx.Rngs(0))
    tcfg = TrainConfig(total_steps=1, warmup_steps=0)
    opt_a = make_optimizer(model_a, tcfg)
    opt_b = make_optimizer(model_b, tcfg)

    step_default = make_train_step(mask_weight=0.0, heatmap_size=160, sigma=2.5)
    step_explicit = make_train_step(mask_weight=0.0, heatmap_size=160, sigma=2.5,
                                    normalize_fn=normalize_image)

    loss_default = float(step_default(model_a, opt_a, key, img4, kp, vis))
    loss_explicit = float(step_explicit(model_b, opt_b, key, img4, kp, vis))
    assert loss_default == loss_explicit
