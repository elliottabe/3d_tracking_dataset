import numpy as np
from jarvis_jax.predict import frame_sync as fs
from jarvis_jax.predict import synced_reader as sr


def _plan_with_gap(cam_name="Cam2012000", start_slot=0, decoded=197, gap_slot=41, lost=3):
    d = dict(delta_ns=1250000, canonical_len=200, predict_start=0, predict_len=200,
             status="reindex", first_drop_slot=gap_slot,
             cameras={cam_name: dict(start_slot=start_slot, decoded_len=decoded,
                                     true_span=decoded + lost, frame_id_mode="hole",
                                     gaps=[dict(slot=gap_slot, lost=lost)])})
    return fs.SyncPlan(d)


def test_slot_positions_none_plan_is_positional():
    pos, present = sr.slot_positions(None, "Cam2012000", start_slot=100, T=5)
    assert pos == [100, 101, 102, 103, 104]
    assert present == [True] * 5


def test_slot_positions_maps_around_gap():
    plan = _plan_with_gap()
    pos, present = sr.slot_positions(plan, "Cam2012000", start_slot=39, T=8)
    # slots 39,40 present (pos==slot); 41,42,43 dropped; 44,45,46 present (pos = slot-3)
    assert present == [True, True, False, False, False, True, True, True]
    assert pos[0] == 39 and pos[1] == 40
    assert pos[2] is None and pos[3] is None and pos[4] is None
    assert pos[5] == 41 and pos[6] == 42 and pos[7] == 43
