import os
import re
import numpy as np
import pytest
from jarvis_jax.cse.silhouette_dof import (
    build_appendage_dof_mask, appendage_vertex_indices, APPENDAGE_PATTERNS,
)

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip_xml = pytest.mark.skipif(not os.path.exists(XML), reason="model xml absent")
skip_mesh = pytest.mark.skipif(not os.path.exists(MESH), reason="mesh npz absent")


@skip_xml
def test_dof_mask_excludes_root_and_covers_all_hinges():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m)
    assert mask.shape == (m.nq,)
    assert not mask[:7].any()               # free (root) joint DOFs excluded
    # V1 model: all 86 non-root hinge DOFs are wing/leg/abdomen
    assert int(mask.sum()) == m.nq - 7


@skip_xml
def test_dof_mask_wing_only_is_six():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m, include=("wing",))
    assert int(mask.sum()) == 6             # yaw/roll/pitch x left/right
    assert not mask[:7].any()


@skip_xml
def test_dof_mask_abdomen_does_not_catch_coxa_abduct():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m, include=("abdomen",))
    # 14 abdomen DOFs (abdomen_* + abdomen_abduct_*), NOT the leg coxa_abduct joints
    assert int(mask.sum()) == 14


@skip_mesh
def test_appendage_vertices_are_subset_of_fps_and_appendage_only():
    idx = appendage_vertex_indices(MESH, subset="fps_300")
    z = np.load(MESH, allow_pickle=True)
    fps = set(int(i) for i in z["fps_300"])
    assert set(int(i) for i in idx).issubset(fps)         # subset of fps_300
    assert len(idx) > 0 and len(idx) < len(fps)           # some excluded (body core)
    seg = np.asarray(z["vertex_segment"]); segids = np.asarray(z["seg_ids"])
    segname = {int(s): str(n) for s, n in zip(segids, z["seg_names"])}
    rex = re.compile("|".join(APPENDAGE_PATTERNS.values()))
    assert all(rex.search(segname[int(seg[i])].lower()) for i in idx)  # all appendage segs
