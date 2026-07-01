import os
import numpy as np
import pytest

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"

# The three real schemas (transcribed from the real coco keypoint_names):
FULL_50 = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]
AMPUTEE_44 = [n for n in FULL_50 if n not in
              ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip")]
HEADLESS_47 = [n for n in FULL_50 if n not in ("Antenna_Base", "EyeL", "EyeR")]


def test_canonical_kp_names_match_stac_order():
    from jarvis_jax.cse.active_parts import CANONICAL_KP_NAMES
    assert CANONICAL_KP_NAMES == FULL_50
    assert len(CANONICAL_KP_NAMES) == 50


def test_part_table_shape_and_names():
    from jarvis_jax.cse.active_parts import PART_TABLE
    assert set(PART_TABLE) == {"head", "wing_left", "wing_right",
                               "legT1L", "legT1R", "legT2L", "legT2R", "legT3L", "legT3R"}
    # head has kps but NO joints (rigid weld) and its own mesh segs.
    assert PART_TABLE["head"]["joints"] == []
    assert set(PART_TABLE["head"]["kp_names"]) == {"Antenna_Base", "EyeL", "EyeR"}
    assert set(PART_TABLE["head"]["seg_ids"]) == {2, 3, 4, 5, 6, 7, 8}
    # legT1R: 11 hinge joints, 6 DISTAL kp names (ThxCx excluded from the diff set), 8 segs.
    assert len(PART_TABLE["legT1R"]["joints"]) == 11
    assert "T1R_ThxCx" not in PART_TABLE["legT1R"]["kp_names"]
    assert set(PART_TABLE["legT1R"]["kp_names"]) == {
        "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"}
    assert set(PART_TABLE["legT1R"]["seg_ids"]) == set(range(28, 36))
    # wings: 3 joints (yaw/roll/pitch), 2 kps, 1 seg.
    assert len(PART_TABLE["wing_left"]["joints"]) == 3
    assert set(PART_TABLE["wing_left"]["kp_names"]) == {"WingL_V12", "WingL_V13"}
    assert PART_TABLE["wing_left"]["seg_ids"] == [9]
    assert PART_TABLE["wing_right"]["seg_ids"] == [10]


def test_derive_active_parts_amputee_headless_full():
    from jarvis_jax.cse.active_parts import derive_active_parts
    amp = derive_active_parts(AMPUTEE_44)
    assert amp["off"] == ["legT1R"]
    assert "legT1R" not in amp["on"] and "head" in amp["on"]
    hl = derive_active_parts(HEADLESS_47)
    assert hl["off"] == ["head"]
    assert "head" not in hl["on"] and "legT1R" in hl["on"]
    full = derive_active_parts(FULL_50)
    assert full["off"] == []
    assert set(full["on"]) == {"head", "wing_left", "wing_right",
                               "legT1L", "legT1R", "legT2L", "legT2R", "legT3L", "legT3R"}


def test_derive_active_parts_override():
    from jarvis_jax.cse.active_parts import derive_active_parts
    d = derive_active_parts(FULL_50, override=["wing_left"])
    assert d["off"] == ["wing_left"]
    with pytest.raises(ValueError):
        derive_active_parts(FULL_50, override=["not_a_part"])
