import numpy as np
from jarvis_jax.predict import frame_sync as fs
from jarvis_jax.predict import sam3_driver as sd


def _clean_plan():
    return fs.SyncPlan(dict(delta_ns=1250000, canonical_len=10, predict_start=0,
                            predict_len=10, status="clean", first_drop_slot=None,
                            cameras={"Cam0": dict(start_slot=0, decoded_len=10,
                                                  true_span=10, frame_id_mode="reindex",
                                                  gaps=[])}))


def _reindex_plan():
    return fs.SyncPlan(dict(delta_ns=1250000, canonical_len=10, predict_start=0,
                            predict_len=10, status="reindex", first_drop_slot=3,
                            cameras={"Cam0": dict(start_slot=0, decoded_len=9,
                                                  true_span=10, frame_id_mode="hole",
                                                  gaps=[dict(slot=3, lost=1)])}))


def test_stamp_roundtrip_and_staleness(tmp_path):
    npz = str(tmp_path / "sam3_masks.npz")
    np.savez(npz, packed=np.zeros((1, 1, 2, 4, 1), np.uint8), valid=np.ones((1, 1, 2), bool),
             shape=np.array([4, 8]), centroids=np.zeros((1, 1, 2, 2), np.float32))
    # unstamped + reindex plan => stale; unstamped + clean plan => NOT stale
    assert sd.masks_are_stale(npz, _reindex_plan()) is True
    assert sd.masks_are_stale(npz, _clean_plan()) is False
    # after stamping with the reindex plan => not stale
    sd.append_sync_stamp_to_npz(npz, _reindex_plan())
    assert sd.masks_are_stale(npz, _reindex_plan()) is False


def test_no_plan_never_stale(tmp_path):
    npz = str(tmp_path / "sam3_masks.npz")
    np.savez(npz, packed=np.zeros((1, 1, 1, 1, 1), np.uint8))
    assert sd.masks_are_stale(npz, None) is False
