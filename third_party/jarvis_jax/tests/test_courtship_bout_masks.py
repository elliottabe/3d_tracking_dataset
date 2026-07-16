import numpy as np
from jarvis_jax.tracking.bout_masks import unpack_one, load_bout_masks


def _synth_npz(tmp_path):
    # 2 flies, 2 cams, 3 frames, H=8, W=13 -> packed width ceil(13/8)=2
    H, W = 8, 13
    full = np.zeros((2, 2, 3, H, W), np.uint8)
    full[0, 1, 2, 2:5, 3:9] = 1                      # fly0 cam1 frame2 block
    packed = np.packbits(full, axis=-1)              # (2,2,3,H,2)
    valid = np.zeros((2, 2, 3), bool); valid[0, 1, 2] = True
    p = tmp_path / "sam3_masks.npz"
    np.savez(p, packed=packed, valid=valid, shape=np.array([H, W], np.int32),
             centroids=np.zeros((2, 2, 3, 2), np.float32))
    return str(p), full


def test_unpack_one_roundtrips_fullframe(tmp_path):
    p, full = _synth_npz(tmp_path)
    z = np.load(p)
    m = unpack_one(z["packed"], 0, 1, 2, int(z["shape"][1]))
    assert m.shape == (8, 13) and m.dtype == bool
    assert np.array_equal(m, full[0, 1, 2].astype(bool))    # exact full-frame match, trimmed to W


def test_load_bout_masks_shapes_and_valid(tmp_path):
    p, full = _synth_npz(tmp_path)
    out = load_bout_masks(p, fly=0)
    assert out["T"] == 3 and out["C"] == 2 and out["H"] == 8 and out["W"] == 13
    assert out["valid"][2, 1] and not out["valid"][0, 0]
    assert np.array_equal(np.asarray(out["masks"])[2, 1], full[0, 1, 2].astype(bool))
