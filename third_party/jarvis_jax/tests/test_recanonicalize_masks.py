import os, sys, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _write_npz(path, area0_byte, area1_byte, C=7, T=10, H=4, Wp=2):
    """packed (2,C,T,H,Wp) uint8: fly0 bytes=area0_byte, fly1 bytes=area1_byte
    (more set bits => larger silhouette). All frames/cams valid."""
    packed = np.zeros((2, C, T, H, Wp), np.uint8)
    packed[0] = area0_byte
    packed[1] = area1_byte
    valid = np.ones((2, C, T), bool)
    centroids = np.zeros((2, C, T, 2), np.float32)
    centroids[0] = 1.0                      # marker to track which fly axis moved
    centroids[1] = 2.0
    np.savez_compressed(path, packed=packed, valid=valid, centroids=centroids,
                        shape=np.array([H, Wp * 8], np.int32), version=np.array(1, np.int32),
                        cameras=np.array([f"Cam{i}" for i in range(C)]))


def test_recanonicalize_swaps_male_to_fly1(tmp_path):
    import recanonicalize_masks as rc
    npz = str(tmp_path / "sam3_masks.npz")
    _write_npz(npz, area0_byte=0xFF, area1_byte=0x01)      # fly0 much larger => male=0
    r = rc.recanonicalize_npz(npz, min_pairs=6, min_cams=2, vote_frames=10)
    assert r["status"] == "swapped" and r["male_detected_slot"] == 0
    with np.load(npz, allow_pickle=True) as d:
        # after swap, fly1 axis holds the formerly-larger masks + centroid marker 1.0
        assert d["packed"][1].sum() > d["packed"][0].sum()
        assert np.allclose(d["centroids"][1], 1.0) and np.allclose(d["centroids"][0], 2.0)
        meta = json.loads(str(d["sex_meta"]))
        assert meta["male_slot"] == 1 and meta["status"] == "swapped"
        assert meta["method"] == "mask_area_vote"


def test_recanonicalize_kept_when_male_already_fly1(tmp_path):
    import recanonicalize_masks as rc
    npz = str(tmp_path / "sam3_masks.npz")
    _write_npz(npz, area0_byte=0x01, area1_byte=0xFF)      # fly1 larger => already male=1
    r = rc.recanonicalize_npz(npz, min_pairs=6, min_cams=2, vote_frames=10)
    assert r["status"] == "kept"
    with np.load(npz, allow_pickle=True) as d:
        assert np.allclose(d["centroids"][0], 1.0)         # unchanged
        assert json.loads(str(d["sex_meta"]))["status"] == "kept"


def test_recanonicalize_idempotent(tmp_path):
    import recanonicalize_masks as rc
    npz = str(tmp_path / "sam3_masks.npz")
    _write_npz(npz, area0_byte=0xFF, area1_byte=0x01)
    rc.recanonicalize_npz(npz, min_pairs=6, min_cams=2, vote_frames=10)   # swaps
    r2 = rc.recanonicalize_npz(npz, min_pairs=6, min_cams=2, vote_frames=10)  # now male at 1
    assert r2["status"] == "kept"


def test_recanonicalize_dry_run_no_write(tmp_path):
    import recanonicalize_masks as rc
    npz = str(tmp_path / "sam3_masks.npz")
    _write_npz(npz, area0_byte=0xFF, area1_byte=0x01)
    r = rc.recanonicalize_npz(npz, min_pairs=6, min_cams=2, vote_frames=10, dry_run=True)
    assert r["status"] == "swapped"
    with np.load(npz, allow_pickle=True) as d:
        assert np.allclose(d["centroids"][0], 1.0)         # NOT swapped
        assert "sex_meta" not in d.files                    # NOT written
