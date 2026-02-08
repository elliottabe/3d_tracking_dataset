"""Preprocess mocap data for mjx (adapted from fly_neuromech/fly_mimic)"""

import jax
from jax import jit
from jax import numpy as jp
from flax import struct

import mujoco
from mujoco import mjx
from mujoco.mjx._src import smooth

import utils.transformations as tr

from typing import Union, List, Optional



@struct.dataclass
class ReferenceClip:
    """This dataclass is used to store the trajectory in the env."""

    # qpos
    position: jp.ndarray = None
    quaternion: jp.ndarray = None
    joints: jp.ndarray = None

    # xpos
    body_positions: jp.ndarray = None

    # velocity (inferred)
    velocity: jp.ndarray = None
    joints_velocity: jp.ndarray = None
    angular_velocity: jp.ndarray = None

    # xquat
    body_quaternions: jp.ndarray = None
    
    clip_lengths: jp.ndarray = None


def process_clip(
    mocap_qpos,
    mjx_model,
    mjx_data,
    max_qvel: float = 20.0,
    dt: float = 0.02,
):
    """Process a set of joint angles into the features that
       the referenced trajectory is composed of.

    Args:
        mocap_qpos: Input qpos trajectory (T, nq)
        mjx_model: MJX model
        mjx_data: MJX data
        max_qvel: Maximum velocity clipping (not currently used)
        dt: Timestep in seconds

    Returns:
        ReferenceClip: Processed clip with positions, velocities, and orientations
    """

    # Feature logic for a single clip here
    clip = ReferenceClip()

    clip = extract_features(mjx_model, mjx_data, clip, mocap_qpos)
    # Padding for velocity corner case.
    mocap_qpos = jp.concatenate([mocap_qpos, mocap_qpos[-1, jp.newaxis, :]], axis=0)

    # Calculate qvel, clip.
    mocap_qvel = compute_velocity_from_kinematics(mocap_qpos, dt)
    vels = mocap_qvel[:, 6:]
    # clipped_vels = jp.clip(vels, -max_qvel, max_qvel)

    mocap_qvel = mocap_qvel.at[:, 6:].set(vels)
    clip = clip.replace(
        velocity=mocap_qvel[:, :3],
        angular_velocity=mocap_qvel[:, 3:6],
        joints_velocity=mocap_qvel[:, 6:],
    )

    return clip


@jit
def extract_features(mjx_model, mjx_data, clip, mocap_qpos):
    def f(mjx_data, qpos):
        mjx_data = set_position(mjx_model, mjx_data, qpos)
        qpos = mjx_data.qpos
        xpos = mjx_data.xpos
        xquat = mjx_data.xquat
        return mjx_data, (qpos[:3], qpos[3:7], qpos[7:], xpos, xquat)

    mjx_data, (position, quaternion, joints, body_positions, body_quaternions) = (
        jax.lax.scan(
            f,
            mjx_data,
            mocap_qpos,
        )
    )

    # Add features to ReferenceClip
    return clip.replace(
        position=position,
        quaternion=quaternion,
        joints=joints,
        body_positions=body_positions,
        body_quaternions=body_quaternions,
    )


def kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data):
    """jit compiled forward kinematics

    Args:
        mjx_model (mjx.Model):
        mjx_data (mjx.Data):

    Returns:
        mjx.Data: resulting mjx Data
    """
    return smooth.kinematics(mjx_model, mjx_data)


@jit
def set_position(
    mjx_model: mjx.Model, mjx_data: mjx.Data, qpos: jp.ndarray
) -> mjx.Data:
    """Sets the qpos and performs forward kinematics (zeros for qvel)

    Args:
        mjx_model (mjx.Model): _description_
        mjx_data (mjx.Data): _description_
        qpos (jp.Array): _description_

    Returns:
        mjx.Data: _description_
    """
    qvel = jp.zeros((mjx_model.nv,))
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    mjx_data = kinematics(mjx_model, mjx_data)
    return mjx_data


@jit
def compute_velocity_from_kinematics(
    qpos_trajectory: jp.ndarray, dt: float
) -> jp.ndarray:
    """Computes velocity trajectory from position trajectory.

    Args:
        qpos_trajectory (jp.ndarray): trajectory of qpos values T x ?
          Note assumes has freejoint as the first 7 dimensions
        dt (float): timestep between qpos entries

    Returns:
        jp.ndarray: Trajectory of velocities.
    """
    qvel_translation = (qpos_trajectory[1:, :3] - qpos_trajectory[:-1, :3]) / dt
    qvel_gyro = []
    for t in range(qpos_trajectory.shape[0] - 1):
        normed_diff = tr.quat_diff(qpos_trajectory[t, 3:7], qpos_trajectory[t + 1, 3:7])
        normed_diff /= jp.linalg.norm(normed_diff)
        angle = tr.quat_to_axisangle(normed_diff)
        qvel_gyro.append(angle / dt)
    qvel_gyro = jp.stack(qvel_gyro)
    qvel_joints = (qpos_trajectory[1:, 7:] - qpos_trajectory[:-1, 7:]) / dt
    return jp.concatenate([qvel_translation, qvel_gyro, qvel_joints], axis=1)
