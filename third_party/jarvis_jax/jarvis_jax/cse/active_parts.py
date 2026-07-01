"""Per-recording active-parts config + mask (Phase 4: headless / amputation).

ONE anatomy per body type (V1 fruitfly_v1_free.xml, unchanged); a per-recording
active-parts mask expresses a missing body part by (a) dropping that part's
markers, (b) locking + post-solve-clamping that part's joints to rest, and (c)
excluding that part's mesh geoms from the silhouette. The mask is applied in the
cse layer only -- the stac_core_jaxls solver / run_stac_bout / run_single_fly are
reused unchanged. Parts are keyed by NAME so the table is anatomy-agnostic; joint
NAMES (not qpos indices) are stored and resolved against the real model at
build_active_mask time. Index by joint NAME (tarsus5_T1_right moves body
claw_T1_right -- a body/joint name mismatch that would break body-name indexing).
"""
from __future__ import annotations

import numpy as np

# Canonical 50 keypoint names in STAC / KEYPOINT_MODEL_PAIRS order (verified
# against stac-mjx/configs/anatomy/v1.yaml). Single source of truth for the diff.
CANONICAL_KP_NAMES = [
    "Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "EyeL", "EyeR",
    "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13", "Abd_A4", "Abd_tip",
    "T1L_ThxCx", "T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip",
    "T1R_ThxCx", "T1R_Tro", "T1R_FeTi", "T1R_TiTa", "T1R_TaT1", "T1R_TaT3", "T1R_TaTip",
    "T2L_Tro", "T2L_FeTi", "T2L_TiTa", "T2L_TaT1", "T2L_TaT3", "T2L_TaTip",
    "T2R_Tro", "T2R_FeTi", "T2R_TiTa", "T2R_TaT1", "T2R_TaT3", "T2R_TaTip",
    "T3L_Tro", "T3L_FeTi", "T3L_TiTa", "T3L_TaT1", "T3L_TaT3", "T3L_TaTip",
    "T3R_Tro", "T3R_FeTi", "T3R_TiTa", "T3R_TaT1", "T3R_TaT3", "T3R_TaTip",
]


def _leg_joints(prefix, side):
    """The 11 hinge joint NAMES of one leg (verified against fruitfly_v1_free.xml).

    prefix in {T1,T2,T3}; side in {left,right}. The distal tip joint is
    tarsus5_{prefix}_{side} (it moves body claw_{prefix}_{side}).
    """
    return [
        f"coxa_abduct_{prefix}_{side}", f"coxa_twist_{prefix}_{side}", f"coxa_{prefix}_{side}",
        f"femur_twist_{prefix}_{side}", f"femur_{prefix}_{side}", f"tibia_{prefix}_{side}",
        f"tarsus_{prefix}_{side}", f"tarsus2_{prefix}_{side}", f"tarsus3_{prefix}_{side}",
        f"tarsus4_{prefix}_{side}", f"tarsus5_{prefix}_{side}",
    ]


def _leg_kps(prefix, sideR):
    """The 6 DISTAL kp names of a leg (ThxCx excluded from the OFF-diff set)."""
    s = "R" if sideR else "L"
    return [f"{prefix}{s}_{suf}" for suf in ("Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")]


def _leg_segs(base):
    """8 mesh seg ids of a leg: coxa,femur,tibia,tarsus,tarsus2,tarsus3,tarsus4,claw."""
    return list(range(base, base + 8))


PART_TABLE = {
    "head": {"joints": [], "kp_names": ["Antenna_Base", "EyeL", "EyeR"],
             "seg_ids": [2, 3, 4, 5, 6, 7, 8]},
    "wing_left": {"joints": ["wing_yaw_left", "wing_roll_left", "wing_pitch_left"],
                  "kp_names": ["WingL_V12", "WingL_V13"], "seg_ids": [9]},
    "wing_right": {"joints": ["wing_yaw_right", "wing_roll_right", "wing_pitch_right"],
                   "kp_names": ["WingR_V12", "WingR_V13"], "seg_ids": [10]},
    "legT1L": {"joints": _leg_joints("T1", "left"), "kp_names": _leg_kps("T1", False),
               "seg_ids": _leg_segs(20)},
    "legT1R": {"joints": _leg_joints("T1", "right"), "kp_names": _leg_kps("T1", True),
               "seg_ids": _leg_segs(28)},
    "legT2L": {"joints": _leg_joints("T2", "left"), "kp_names": _leg_kps("T2", False),
               "seg_ids": _leg_segs(36)},
    "legT2R": {"joints": _leg_joints("T2", "right"), "kp_names": _leg_kps("T2", True),
               "seg_ids": _leg_segs(44)},
    "legT3L": {"joints": _leg_joints("T3", "left"), "kp_names": _leg_kps("T3", False),
               "seg_ids": _leg_segs(52)},
    "legT3R": {"joints": _leg_joints("T3", "right"), "kp_names": _leg_kps("T3", True),
               "seg_ids": _leg_segs(60)},
}


def derive_active_parts(kp_names, override=None):
    """Auto-derive off/on parts from a recording's coco keypoint_names.

    A part is OFF iff EVERY one of its (canonical) diff kp_names is absent from
    kp_names. With override given, off = override (validated). See Interfaces.
    """
    present = set(kp_names)
    if override is not None:
        bad = [p for p in override if p not in PART_TABLE]
        if bad:
            raise ValueError(f"override parts not in PART_TABLE: {bad}")
        off = list(override)
    else:
        off = []
        for part, spec in PART_TABLE.items():
            diff_kps = [n for n in spec["kp_names"] if n in CANONICAL_KP_NAMES]
            if diff_kps and all(n not in present for n in diff_kps):
                off.append(part)
    on = [p for p in PART_TABLE if p not in off]
    return {"off": off, "on": on}


def build_active_mask(kp_names, model_xml, mesh_npz, off_parts):
    """Resolve an off-part set against the real model+mesh into a mask dict.

    See Interfaces (Task 2). Marker order for kps_to_opt_mask/off_marker_stac_idx
    follows the passed kp_names (the STAC/bout kp order the mask will be applied
    against). Joints are resolved by NAME via mj_name2id, so a body/joint name
    mismatch (tarsus5_T1_right -> body claw_T1_right) is handled correctly.
    """
    import mujoco

    kp_names = list(kp_names)
    name2idx = {n: i for i, n in enumerate(kp_names)}

    m = mujoco.MjModel.from_xml_path(model_xml)
    nq = int(m.nq)
    rest_qpos = np.asarray(m.key_qpos[0], dtype=np.float64) if m.nkey > 0 else np.zeros(nq)

    qs_to_opt_mask = np.ones(nq, dtype=bool)
    locked = []
    kps_to_opt_mask = np.ones(len(kp_names) * 3, dtype=np.float32)
    off_marker_stac_idx = []
    excluded_seg_ids = []

    for part in off_parts:
        spec = PART_TABLE[part]
        # (b) lock the part's joints (by joint NAME -> qpos adr).
        for jname in spec["joints"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"joint '{jname}' (part {part}) not found in {model_xml}")
            adr = int(m.jnt_qposadr[jid])
            qs_to_opt_mask[adr] = False
            locked.append(adr)
        # (a) zero the part's markers that are actually PRESENT in kp_names.
        for kn in spec["kp_names"]:
            j = name2idx.get(kn)
            if j is not None:
                kps_to_opt_mask[j * 3:j * 3 + 3] = 0.0
                off_marker_stac_idx.append(j)
        # (c) collect the part's mesh seg ids.
        excluded_seg_ids.extend(spec["seg_ids"])

    # never lock the free root (qpos 0..6).
    qs_to_opt_mask[:7] = True
    locked = sorted(set(i for i in locked if i >= 7))

    excluded_seg_ids = sorted(set(excluded_seg_ids))
    if excluded_seg_ids:
        seg = np.load(mesh_npz, allow_pickle=True)["vertex_segment"]
        excluded_vertex_idx = np.where(np.isin(seg, excluded_seg_ids))[0].astype(np.int64)
    else:
        excluded_vertex_idx = np.zeros(0, dtype=np.int64)

    return {
        "kp_names": kp_names,
        "kps_to_opt_mask": kps_to_opt_mask,
        "qs_to_opt_mask": qs_to_opt_mask,
        "locked_qpos_idx": np.asarray(locked, dtype=np.int32),
        "rest_qpos": rest_qpos,
        "excluded_seg_ids": excluded_seg_ids,
        "excluded_vertex_idx": excluded_vertex_idx,
        "off_marker_stac_idx": np.asarray(sorted(set(off_marker_stac_idx)), dtype=np.int64),
        "off_parts": list(off_parts),
    }


def clamp_locked_qpos(qpos, mask):
    """Post-solve clamp: pin every locked joint's qpos back to rest (removes the
    phantom LM-variable drift). Returns a copy; no-op when nothing is locked."""
    q = np.array(qpos, copy=True)
    idx = np.asarray(mask["locked_qpos_idx"])
    if idx.size:
        q[:, idx] = np.asarray(mask["rest_qpos"])[idx]
    return q


def apply_active_mask_to_inputs(inputs, mask):
    """Pre-solve: gate marker weights + DOFs and NaN off-part markers. Copy-safe."""
    out = dict(inputs)
    out["kps_to_opt"] = (np.asarray(inputs["kps_to_opt"], np.float32)
                         * np.asarray(mask["kps_to_opt_mask"], np.float32))
    out["qs_to_opt"] = np.asarray(inputs["qs_to_opt"], bool) & np.asarray(mask["qs_to_opt_mask"], bool)
    off = np.asarray(mask.get("off_marker_stac_idx", np.zeros(0, np.int64)))
    if off.size and "kp_data" in inputs:
        kp = np.array(inputs["kp_data"], dtype=np.float64, copy=True)
        kp[:, off, :] = np.nan
        out["kp_data"] = kp
    return out
