"""Per-frame weak-DOF polish: the pure selection/merge logic and its provenance.

The solver behaviour itself is pinned in stac-mjx
(tests/unit/test_jaxls_independent_frames.py); here we guard the parts that
decide WHAT gets written back: which candidate wins per frame, that only the
configured DOFs change, that unsolved (NaN) frames stay NaN, that the rest
start uses the model springref and never zero, and that the signature moves
with every setting that changes the result.
"""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from jarvis_jax.tracking.stac_polish import (
    build_starts,
    dof_mask,
    merge_dofs,
    merged_candidates,
    polish_signature,
    select_by_cost,
    viterbi_select,
)

NAMES = ["root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz",
         "wing_yaw_left", "wing_roll_left", "wing_pitch_left",
         "wing_yaw_right", "wing_roll_right", "wing_pitch_right",
         "abdomen_abduct", "abdomen", "abdomen_2", "coxa_T1_left", "femur_T1_left"]


def test_dof_mask_matches_globs_only():
    m = dof_mask(NAMES, ["wing_*", "abdomen*"])
    assert [n for n, k in zip(NAMES, m) if k] == NAMES[7:16]
    assert not m[:7].any() and not m[16:].any()


def test_select_by_cost_ignores_nan_and_defaults_to_candidate_zero():
    costs = np.array([[1.0, 5.0, np.nan, np.nan],
                      [2.0, 1.0, np.nan, np.nan],
                      [0.5, np.nan, 3.0, np.nan]])
    np.testing.assert_array_equal(select_by_cost(costs), [2, 1, 2, 0])


def test_merge_writes_only_masked_dofs_and_keeps_nan_frames():
    nq = len(NAMES)
    q_batch = np.arange(3 * nq, dtype=float).reshape(3, nq)
    q_pol = q_batch + 100.0
    q_pol[2] = np.nan                       # a frame the polish could not solve
    m = dof_mask(NAMES, ["wing_pitch_*"])
    out = merge_dofs(q_batch, q_pol, m)
    assert np.all(out[:2, m] == q_batch[:2, m] + 100.0)
    assert np.all(out[:, ~m] == q_batch[:, ~m])
    np.testing.assert_array_equal(out[2], q_batch[2])


def test_build_starts_uses_springref_not_zero_and_leaves_the_rest_alone():
    nq = len(NAMES)
    q = np.random.default_rng(0).normal(size=(4, nq))
    spring = np.zeros(nq); ipl, ipr = NAMES.index("wing_pitch_left"), NAMES.index("wing_pitch_right")
    spring[ipl] = spring[ipr] = np.radians(-57.3)
    s = build_starts(q, NAMES, spring, ["batch", "wing_rest_left", "wing_rest_right", "wing_rest_both"])
    np.testing.assert_array_equal(s["batch"], q)
    assert np.allclose(s["wing_rest_left"][:, ipl], np.radians(-57.3)) and np.all(s["wing_rest_left"][:, ipr] == q[:, ipr])
    assert np.allclose(s["wing_rest_right"][:, ipr], np.radians(-57.3)) and np.all(s["wing_rest_right"][:, ipl] == q[:, ipl])
    assert np.allclose(s["wing_rest_both"][:, [ipl, ipr]], np.radians(-57.3))
    other = np.ones(nq, bool); other[[ipl, ipr]] = False
    for v in s.values():
        np.testing.assert_array_equal(v[:, other], q[:, other])
    with pytest.raises(ValueError, match="unknown start"):
        build_starts(q, NAMES, spring, ["abdomen_rest"])


def test_signature_tracks_every_result_changing_setting():
    base = dict(enabled=True, dofs=["wing_*", "abdomen*"], starts=["batch", "wing_rest_both"],
                lambda_initial=5e-4, lambda_min=1e-8, cost_tolerance=1e-12,
                gradient_tolerance=1e-14, parameter_tolerance=1e-16, n_iter=3000)
    sig0 = polish_signature(OmegaConf.create(base))
    assert sig0 == polish_signature(OmegaConf.create({**base, "enabled": False}))   # enabled is not a result knob
    for k, v in (("dofs", ["wing_*"]), ("starts", ["batch"]), ("lambda_initial", 1.0),
                 ("lambda_min", 1e-5), ("cost_tolerance", 1e-5), ("n_iter", 500)):
        assert polish_signature(OmegaConf.create({**base, k: v})) != sig0, k


def test_merged_candidates_keep_batch_as_candidate_zero_and_graft_only_masked_dofs():
    nq = len(NAMES)
    q_batch = np.zeros((2, nq)); c1 = np.ones((2, nq)); c2 = np.full((2, nq), 2.0)
    m = dof_mask(NAMES, ["abdomen*"])
    merged = merged_candidates(q_batch, [q_batch, c1, c2], m)
    np.testing.assert_array_equal(merged[0], q_batch)
    assert np.all(merged[1][:, m] == 1.0) and np.all(merged[1][:, ~m] == 0.0)
    assert np.all(merged[2][:, m] == 2.0) and np.all(merged[2][:, ~m] == 0.0)


def test_viterbi_holds_a_basin_when_cost_differences_are_marginal():
    """Two candidates: A (edge-on) and B (flat), 30 deg apart on the polished
    DOF, whose marker costs alternate in favour by a hair every frame. A plain
    argmin flips every frame; the Viterbi choice stays put."""
    T = 40; nq = len(NAMES); m = dof_mask(NAMES, ["wing_pitch_right"])
    qA = np.zeros((T, nq)); qB = np.zeros((T, nq)); qB[:, m] = np.radians(30.0)
    costs = np.full((2, T), 1.0e-3); costs[0, ::2] -= 1e-5; costs[1, 1::2] -= 1e-5
    flip = select_by_cost(costs)
    assert np.sum(flip[1:] != flip[:-1]) == T - 1
    steady = viterbi_select(costs, [qA, qB], m, switch_weight=1e-3)
    assert np.sum(steady[1:] != steady[:-1]) == 0
    assert np.array_equal(viterbi_select(costs, [qA, qB], m, switch_weight=0.0), flip)


def test_viterbi_still_switches_when_the_gain_is_sustained():
    T = 40; nq = len(NAMES); m = dof_mask(NAMES, ["wing_pitch_right"])
    qA = np.zeros((T, nq)); qB = np.zeros((T, nq)); qB[:, m] = np.radians(30.0)
    costs = np.full((2, T), 1.0e-3); costs[0, :20] -= 5e-4; costs[1, 20:] -= 5e-4   # A clearly better first, then B
    idx = viterbi_select(costs, [qA, qB], m, switch_weight=1e-3)
    assert np.all(idx[:20] == 0) and np.all(idx[20:] == 1)
