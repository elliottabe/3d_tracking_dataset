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


_HAVE_MODEL = os.path.exists(XML) and os.path.exists(MESH)


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_amputee_locks_t1r_38_48():
    from jarvis_jax.cse.active_parts import build_active_mask, derive_active_parts
    off = derive_active_parts(AMPUTEE_44)["off"]         # ["legT1R"]
    mask = build_active_mask(AMPUTEE_44, XML, MESH, off)
    # T1R hinges are qpos 38..48 inclusive (11 DOF).
    assert set(mask["locked_qpos_idx"].tolist()) == set(range(38, 49))
    assert mask["qs_to_opt_mask"].shape == (93,)
    assert not mask["qs_to_opt_mask"][38:49].any()       # all locked
    assert mask["qs_to_opt_mask"][:38].all() and mask["qs_to_opt_mask"][49:].all()
    assert mask["qs_to_opt_mask"][:7].all()              # root never locked
    # excluded mesh segs are exactly T1R's 8 leg segs (28..35).
    assert set(mask["excluded_seg_ids"]) == set(range(28, 36))
    # excluded vertex set == vertices whose segment is a T1R seg.
    seg = np.load(MESH, allow_pickle=True)["vertex_segment"]
    expect = np.where(np.isin(seg, list(range(28, 36))))[0]
    assert np.array_equal(np.sort(mask["excluded_vertex_idx"]), np.sort(expect))
    assert len(mask["excluded_vertex_idx"]) > 0
    # rest_qpos is the model rest (all hinges 0).
    assert np.allclose(mask["rest_qpos"][38:49], 0.0)


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_headless_no_joints_but_geoms_excluded():
    from jarvis_jax.cse.active_parts import build_active_mask, derive_active_parts
    off = derive_active_parts(HEADLESS_47)["off"]        # ["head"]
    mask = build_active_mask(HEADLESS_47, XML, MESH, off)
    # head has NO joints -> nothing locked.
    assert mask["locked_qpos_idx"].size == 0
    assert mask["qs_to_opt_mask"].all()
    # but head-region geoms (segs 2..8) ARE excluded.
    assert set(mask["excluded_seg_ids"]) == {2, 3, 4, 5, 6, 7, 8}
    seg = np.load(MESH, allow_pickle=True)["vertex_segment"]
    expect = np.where(np.isin(seg, [2, 3, 4, 5, 6, 7, 8]))[0]
    assert np.array_equal(np.sort(mask["excluded_vertex_idx"]), np.sort(expect))


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_full_schema_is_identity():
    from jarvis_jax.cse.active_parts import build_active_mask
    mask = build_active_mask(FULL_50, XML, MESH, [])
    assert mask["qs_to_opt_mask"].all()
    assert mask["locked_qpos_idx"].size == 0
    assert np.allclose(mask["kps_to_opt_mask"], 1.0)
    assert mask["excluded_seg_ids"] == [] and mask["excluded_vertex_idx"].size == 0


@pytest.mark.skipif(not _HAVE_MODEL, reason="V1 model / mesh not present")
def test_build_active_mask_zeroes_present_off_markers_via_override():
    # SIMULATED case: full 50-name schema but legT1R toggled off by override ->
    # its distal markers ARE present, so their kps_to_opt coords must be zeroed.
    from jarvis_jax.cse.active_parts import build_active_mask, CANONICAL_KP_NAMES
    mask = build_active_mask(FULL_50, XML, MESH, ["legT1R"])
    for nm in ("T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip"):
        j = FULL_50.index(nm)
        assert np.allclose(mask["kps_to_opt_mask"][j * 3:j * 3 + 3], 0.0)
    # T1R_ThxCx (present, kept -- constrains the coxa root) NOT zeroed.
    j = FULL_50.index("T1R_ThxCx")
    assert np.allclose(mask["kps_to_opt_mask"][j * 3:j * 3 + 3], 1.0)


def test_clamp_locked_qpos_pins_to_rest():
    from jarvis_jax.cse.active_parts import clamp_locked_qpos
    mask = {"locked_qpos_idx": np.array([38, 39, 48], np.int32),
            "rest_qpos": np.zeros(93)}
    q = np.random.default_rng(0).normal(size=(5, 93))
    q2 = clamp_locked_qpos(q, mask)
    assert np.allclose(q2[:, [38, 39, 48]], 0.0)          # locked -> rest
    keep = [i for i in range(93) if i not in (38, 39, 48)]
    assert np.allclose(q2[:, keep], q[:, keep])           # everything else untouched
    assert not np.shares_memory(q2, q)                    # returns a copy


def test_apply_active_mask_to_inputs_gates_weights_and_dofs():
    from jarvis_jax.cse.active_parts import apply_active_mask_to_inputs
    inp = {"kps_to_opt": np.ones(6, np.float32), "qs_to_opt": np.ones(4, bool),
           "kp_data": np.ones((3, 2, 3), np.float32)}
    mask = {"kps_to_opt_mask": np.array([1, 1, 1, 0, 0, 0], np.float32),
            "qs_to_opt_mask": np.array([True, True, False, True]),
            "off_marker_stac_idx": np.array([1])}
    out = apply_active_mask_to_inputs(inp, mask)
    assert np.allclose(out["kps_to_opt"], [1, 1, 1, 0, 0, 0])
    assert out["qs_to_opt"].tolist() == [True, True, False, True]
    assert np.isnan(out["kp_data"][:, 1, :]).all()        # off marker NaN'd
    assert np.isfinite(out["kp_data"][:, 0, :]).all()     # present marker kept
    assert not np.shares_memory(out["kps_to_opt"], inp["kps_to_opt"])
