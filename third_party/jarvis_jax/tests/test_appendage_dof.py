import os

import numpy as np
import mujoco
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

from jarvis_jax.tracking.appendage_dof import (build_appendage_dof_mask,
                                               appendage_vertex_indices)
from jarvis_jax.tracking.mask_fk_indices import mask_fk_indices

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"

# configs/outputs/default.yaml uses a `basename` resolver; register it or compose fails.
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)


def _model():
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        c = compose(config_name="pipeline", overrides=["paths=hyak"])
    return mujoco.MjModel.from_xml_path(c.anatomy.mjcf_path), c


def test_wing_selects_exactly_the_six_wing_joints():
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("wing",)))
    sel = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(mj.njnt)
           if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
           and m[int(mj.jnt_qposadr[j])]]
    assert len(sel) == 6 and all("wing" in s for s in sel), sel


def test_abdomen_pattern_does_not_catch_the_leg_coxa_abduct():
    """The documented reason the pattern is `abdomen` and not a bare `abd`.
    NOTE abdomen_abduct_* ARE abdomen joints -- the trap is `coxa_abduct`."""
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("abdomen",)))
    sel = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(mj.njnt)
           if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
           and m[int(mj.jnt_qposadr[j])]]
    assert sel and not any("coxa" in s for s in sel), sel


def test_free_root_joint_is_never_selected():
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("wing", "leg", "abdomen")))
    assert not m[:7].any(), "the free joint's 7 qpos must stay frozen"


def test_wing_vertex_selection_is_wing_only_BOTH_wings():
    """INDEX-SPACE TRAP: vertex_segment stores seg_ids VALUES (1..67) while
    seg_names is positional (0..66). Indexing seg_names by a vertex_segment
    value is off by one and made an earlier check report 'abdomen, wing_right'.
    Build the map by zipping seg_ids with seg_names."""
    _, c = _model()
    npz = c.anatomy.cse_mesh_npz
    z = np.load(npz, allow_pickle=True)
    id2name = dict(zip(np.asarray(z["seg_ids"]).tolist(),
                       [str(s) for s in z["seg_names"]]))
    vseg = np.asarray(z["vertex_segment"])
    idx = np.asarray(appendage_vertex_indices(npz, subset="fps_300", include=("wing",)))
    hit = sorted({id2name[int(vseg[int(v)])] for v in idx})
    assert hit == ["wing_left", "wing_right"], hit
    assert len(idx) == 100, f"expected 100 wing verts in fps_300, got {len(idx)}"


def test_returned_indices_are_FULL_array_space_not_subset_space():
    """The recovered docstring warns these are full-array indices (0..139352),
    unlike wing_side_vertices which returns fps-subset indices. Mixing them
    silently selects the wrong vertices."""
    _, c = _model()
    npz = c.anatomy.cse_mesh_npz
    z = np.load(npz, allow_pickle=True)
    idx = np.asarray(appendage_vertex_indices(npz, subset="fps_300", include=("wing",)))
    assert idx.max() > 300, "full-array indices should exceed the subset size"
    assert idx.max() < len(z["vertex_segment"])


def test_mask_fk_indices_with_no_filter_returns_fps_subset_unfiltered():
    """exclude_seg_ids=None is falsy, so mask_fk_indices takes the pass-through
    branch: the raw fps subset, unfiltered, already in full-array space (no
    existing caller reaches this branch -- every appendage_vertex_indices call
    builds a non-empty exclude list)."""
    _, c = _model()
    npz = c.anatomy.cse_mesh_npz
    z = np.load(npz, allow_pickle=True)
    fps = np.asarray(z["fps_300"], dtype=np.int64)
    out = np.asarray(mask_fk_indices(npz, subset="fps_300", exclude_seg_ids=None))
    np.testing.assert_array_equal(out, fps.astype(np.int32))
    assert len(out) == 300
    assert out.max() < len(z["vertex_segment"]), "must be full-array indices"
    assert out.dtype == np.int32
