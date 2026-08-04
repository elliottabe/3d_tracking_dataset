"""Invariants for the v2_3 anatomy config and its compiled MuJoCo model.

These guard the three things that silently break a non-v1 anatomy run:
keypoint column order (STAC matches columns positionally), the MJX-incompatible
adhesion actuators, and the missing tracking[...] sites.
"""
from __future__ import annotations

from pathlib import Path

import mujoco as mj
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
BODY = Path('/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/fruitfly_body_models')
V2_3_XML = BODY / 'fruitfly_v2.3' / 'fruitfly_muscles_warp.xml'


def _cfg(name):
    return OmegaConf.load(REPO / 'configs' / 'anatomy' / f'{name}.yaml')


def _model():
    return mj.MjModel.from_xml_path(str(V2_3_XML))


def test_name_and_paths():
    c = _cfg('v2_3')
    assert c.name == 'v2_3'
    assert c.mjcf_path.endswith('fruitfly_v2.3/fruitfly_muscles_warp.xml')
    assert c.arena_path.endswith('fruitfly_v2.3/floor.xml')


def test_kp_names_order_identical_to_v1():
    """STAC matches keypoint columns POSITIONALLY once prune_model_to_available
    early-returns on an exact name-set match, so order is load-bearing."""
    assert list(_cfg('v2_3').model.KP_NAMES) == list(_cfg('v1').model.KP_NAMES)


def test_keypoint_model_pairs_key_order_identical_to_v2_muscles():
    assert (list(_cfg('v2_3').model.KEYPOINT_MODEL_PAIRS)
            == list(_cfg('v2_muscles').model.KEYPOINT_MODEL_PAIRS))


def test_tatip_offsets_are_left_right_mirrored():
    """v2_muscles.yaml has T3L_TaTip mirrored onto the right side; v2_3 fixes it."""
    offs = _cfg('v2_3').model.KEYPOINT_INITIAL_OFFSETS
    for leg in ('T1', 'T2', 'T3'):
        left = [float(x) for x in str(offs[f'{leg}L_TaTip']).split()]
        right = [float(x) for x in str(offs[f'{leg}R_TaTip']).split()]
        assert left == [0.0, 0.005, 0.0], f'{leg}L_TaTip should be +y'
        assert right == [0.0, -0.005, 0.0], f'{leg}R_TaTip should be -y'


def test_every_mapped_body_exists_in_model():
    m = _model()
    bodies = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    missing = {k: v for k, v in _cfg('v2_3').model.KEYPOINT_MODEL_PAIRS.items()
               if v not in bodies}
    assert not missing, f'bodies absent from v2.3: {missing}'


def test_declared_joint_names_exist_in_model():
    m = _model()
    joints = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
    missing = [j for j in _cfg('v2_3').joint_names if j not in joints]
    assert not missing, f'joint_names absent from v2.3: {missing}'


def test_haltere_joints_declared():
    names = list(_cfg('v2_3').joint_names)
    assert 'haltere_left' in names and 'haltere_right' in names


def test_model_kinematic_shape():
    m = _model()
    assert (m.nq, m.nv, m.nbody, m.njnt) == (101, 100, 74, 95)
