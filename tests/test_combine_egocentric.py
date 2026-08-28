"""`egocentric_sites`: the array Figure 4 panel E needs.

`utils.song_analysis` computes the wing extension/horizontal angles only when a
bout carries `xpos_egocentric`; without it `horiz_angle_L/R` stay None and panel
E renders as empty axes. The legacy analysis h5 had the array, the per-bout
pipeline never writes it, so the combiner derives it.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.export.combine_ik_outputs import egocentric_sites, _quat_to_R

KP = ["Scutellum", "Antenna_Base", "WingL_base", "WingR_base",
      "WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"]


def _rot_z(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([c, np.zeros_like(c), np.zeros_like(c), s], -1)  # wxyz


def test_identity_rotation_is_just_a_translation_to_the_origin():
    T = 5
    p = np.random.default_rng(0).normal(size=(T, len(KP), 3))
    se3 = np.zeros((T, 7)); se3[:, 3] = 1.0                    # identity quat
    ego = egocentric_sites(p, se3, KP)
    assert np.allclose(ego, p - p[:, 0][:, None, :], atol=1e-5)


def test_origin_keypoint_sits_at_zero():
    T = 4
    p = np.random.default_rng(1).normal(size=(T, len(KP), 3))
    se3 = np.zeros((T, 7)); se3[:, 3] = 1.0
    ego = egocentric_sites(p, se3, KP)
    assert np.allclose(ego[:, KP.index("Scutellum")], 0.0, atol=1e-6)


def test_a_rigid_body_spinning_in_world_is_constant_in_its_body_frame():
    """The whole point: de-rotating by root_se3 must cancel the fly's yaw."""
    T = 60
    theta = np.linspace(0, 2 * np.pi, T)
    local = np.array([[0., 0., 0.], [13.8, 0., -1.7], [-1., 1.5, 0.5],
                      [-1., -1.5, 0.5], [-4., 3., 1.], [-5., 3.5, 1.],
                      [-4., -3., 1.], [-5., -3.5, 1.]])
    R = _quat_to_R(_rot_z(theta))                  # body->world
    world = np.einsum("tij,kj->tki", R, local) + np.array([100., -50., 3.])
    ego = egocentric_sites(world, np.concatenate(
        [np.zeros((T, 3)), _rot_z(theta)], axis=1), KP)
    # every site is constant over time, and equals its local offset
    assert np.allclose(ego.std(axis=0), 0.0, atol=1e-4)
    assert np.allclose(ego[0], local - local[0], atol=1e-4)


def test_returns_none_rather_than_a_bogus_array_when_inputs_are_missing():
    T = 3
    p = np.zeros((T, len(KP), 3))
    se3 = np.zeros((T, 7)); se3[:, 3] = 1.0
    assert egocentric_sites(p, None, KP) is None
    assert egocentric_sites(None, se3, KP) is None
    assert egocentric_sites(p, se3, ["nope"] + KP[1:]) is None   # no Scutellum
    assert egocentric_sites(p, se3[:, :4], KP) is None           # wrong width
    assert egocentric_sites(p, se3[:2], KP) is None              # length mismatch


def test_nan_frames_propagate_rather_than_being_invented():
    T = 6
    p = np.random.default_rng(2).normal(size=(T, len(KP), 3))
    p[3] = np.nan
    se3 = np.zeros((T, 7)); se3[:, 3] = 1.0
    ego = egocentric_sites(p, se3, KP)
    assert np.isnan(ego[3]).all()
    assert np.isfinite(ego[[0, 1, 2, 4, 5]]).all()


def test_the_angle_functions_are_invariant_to_any_rigid_transform():
    """The premise of deriving the frame here at all: both consumers build
    their own frame from normalised differences, so rotating/scaling/shifting
    the input cannot change the answer. Pure float64, no storage involved."""
    song = pytest.importorskip("utils.song_analysis")
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(30, len(KP), 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.sign(np.linalg.det(Q))
    moved = (pts @ Q.T) * 3.7 + np.array([10.0, -4.0, 2.0])
    idx = {n: i for i, n in enumerate(KP)}
    for fn in (song.compute_wing_extension_angles, song.compute_wing_horizontal_angles):
        d = np.nanmax(np.abs(np.concatenate(fn(pts, idx))
                             - np.concatenate(fn(moved, idx))))
        assert d < 1e-9, f"{fn.__name__} is not frame-invariant: {d} deg"


def test_stored_egocentric_gives_the_same_angles_as_world_coordinates():
    """End to end on the real output: `egocentric_sites` stores float32 to
    match the other h5 arrays, which costs well under 1e-3 deg against a
    0-90 deg signal."""
    song = pytest.importorskip("utils.song_analysis")
    T = 40
    rng = np.random.default_rng(3)
    theta = np.linspace(0, 1.5, T)
    local = np.array([[0., 0., 0.], [13.8, 0., -1.7], [-1., 1.5, 0.5],
                      [-1., -1.5, 0.5], [-4., 3., 1.], [-5., 3.5, 1.],
                      [-4., -3., 1.], [-5., -3.5, 1.]])
    local = local[None] + rng.normal(scale=0.2, size=(T, len(KP), 3))
    R = _quat_to_R(_rot_z(theta))
    world = np.einsum("tij,tkj->tki", R, local) + rng.normal(scale=5, size=(T, 1, 3))
    ego = egocentric_sites(world, np.concatenate(
        [np.zeros((T, 3)), _rot_z(theta)], axis=1), KP)
    idx = {n: i for i, n in enumerate(KP)}
    for fn in (song.compute_wing_extension_angles, song.compute_wing_horizontal_angles):
        d = np.nanmax(np.abs(np.concatenate(fn(ego, idx))
                             - np.concatenate(fn(world, idx))))
        assert d < 1e-3, f"{fn.__name__} differs by {d} deg"
