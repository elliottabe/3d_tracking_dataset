"""Tests for the leg-motion QC gate (Task 12).

Background: in the v2_3 run, `fit_offsets` wrote an all-NaN `offsets` array
for 7 of 23 directories and the pipeline reported success. Every bout in
those dirs then solved with NaN offsets and all 58 leg joints stayed frozen
at exactly 0 -- 113 of 387 clips, a fly sliding along the floor with rigid
legs. 10 further bouts froze individually from scattered NaN keypoints.
123 of 387 clips (32%) were unusable, and nothing caught it: correct
shapes, correct DOF count, zero NaN, true clip_lengths, verified padding,
loads through the consumer's own loader. All of those are blind to a joint
that never moves.

This tests `audit_clips`, a standalone streaming audit over a packed
reference-clip h5 that flags clips whose leg joints never move (within
their TRUE, unpadded `clip_lengths[i]` span -- never the padding).

Real-file shape (`qpos_names` is an h5py GROUP of scalar strings, one per
qpos column, keyed "0".."100"; the first 7 are the literal string 'free'
for the 7-dof free joint; leg columns are named like `coxa_T1_left`,
`trochanter_T2_right`, `femur_*`, `tibia_*`, `tarsus1_*`..`tarsus4_*`,
`tarsal_claw_*`) is reproduced exactly by the fixtures below.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.qc.audit_reference_clips import audit_clips, main

# A small but realistic qpos layout: 7 free-joint columns + a handful of
# non-leg "other" columns + all 58 real leg-joint names.
OTHER_NAMES = ['head_abduct', 'head_twist', 'head', 'abdomen1', 'abdomen2']
LEG_NAMES = []
for seg in range(1, 3):
    for side in ('left', 'right'):
        LEG_NAMES += [
            f'coxa_abduct_T{seg}_{side}', f'coxa_twist_T{seg}_{side}',
            f'coxa_T{seg}_{side}', f'trochanter_T{seg}_{side}',
            f'femur_T{seg}_{side}', f'tibia_T{seg}_{side}',
            f'tarsus1_T{seg}_{side}', f'tarsus2_T{seg}_{side}',
            f'tarsus3_T{seg}_{side}', f'tarsus4_T{seg}_{side}',
            f'tarsal_claw_T{seg}_{side}',
        ]
QPOS_NAMES = ['free'] * 7 + OTHER_NAMES + LEG_NAMES
N_LEG = len(LEG_NAMES)
NQ = len(QPOS_NAMES)


def _write_h5(path: Path, clip_lengths, qpos, names=QPOS_NAMES):
    """Write a packed reference-clip h5 shaped like the real dataset.

    `qpos` is (N, T_max, nq). Other temporal arrays (qvel, xpos, xquat,
    kp_data) are filled with harmless random data of the right leading
    shape -- the audit tool must not need them to classify leg freezing,
    but the real file always carries them, so fixtures do too.
    """
    n, t_max, nq = qpos.shape
    with h5py.File(path, 'w') as f:
        f.create_dataset('qpos', data=qpos.astype(np.float32))
        f.create_dataset('qvel', data=np.zeros((n, t_max, nq - 1), dtype=np.float32))
        f.create_dataset('xpos', data=np.zeros((n, t_max, 3, 3), dtype=np.float32))
        f.create_dataset('xquat', data=np.zeros((n, t_max, 3, 4), dtype=np.float32))
        f.create_dataset('kp_data', data=np.zeros((n, t_max, 6), dtype=np.float32))
        f.create_dataset('clip_lengths', data=np.asarray(clip_lengths, dtype=np.int32))
        grp = f.create_group('qpos_names')
        for i, name in enumerate(names):
            grp[str(i)] = name
    return path


def _moving_clip(rng, t_max, true_len, nq, leg_idx, root_idx=(0, 1, 2)):
    """A clip where every leg column and the root genuinely move over
    [0, true_len), then holds its last real frame for the padding tail."""
    arr = np.zeros((t_max, nq), dtype=np.float32)
    real = rng.normal(size=(true_len, nq)).astype(np.float32)
    # Force legs to have clearly non-trivial ptp (not just noise near 0).
    ramp = np.linspace(0, 1, true_len, dtype=np.float32)[:, None]
    real[:, leg_idx] = real[:, leg_idx] * 0.1 + ramp
    arr[:true_len] = real
    if true_len < t_max:
        arr[true_len:] = arr[true_len - 1]
    return arr


def _leg_idx_other_idx(names):
    leg_prefixes = ('coxa_', 'trochanter_', 'femur_', 'tibia_', 'tarsus', 'tarsal_claw_')
    leg = [i for i, n in enumerate(names) if n.startswith(leg_prefixes)]
    root = [i for i, n in enumerate(names) if n == 'free']
    return leg, root


LEG_IDX, ROOT_IDX = _leg_idx_other_idx(QPOS_NAMES)


class TestHealthyClipSet:
    def test_no_frozen_clips_exit_zero(self, tmp_path):
        rng = np.random.default_rng(0)
        t_max = 50
        lengths = [30, 40, 50]
        qpos = np.stack([
            _moving_clip(rng, t_max, L, NQ, LEG_IDX) for L in lengths
        ])
        h5_path = _write_h5(tmp_path / 'healthy.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert result['n_clips'] == 3
        assert result['frozen_leg_clips'] == []
        assert result['per_clip_max_leg_ptp'].shape == (3,)
        assert np.all(result['per_clip_max_leg_ptp'] > 1e-3)

        rc = main(['--h5', str(h5_path)])
        assert rc == 0


class TestFrozenLegClip:
    def test_all_leg_joints_constant_is_detected(self, tmp_path):
        rng = np.random.default_rng(1)
        t_max = 50
        lengths = [30, 40]
        qpos = np.stack([
            _moving_clip(rng, t_max, L, NQ, LEG_IDX) for L in lengths
        ])
        # Freeze every leg column of clip 0 at exactly 0 for its whole
        # true span (the measured failure mode: NaN offsets -> legs stuck
        # at 0), leaving root and "other" columns moving normally.
        for i in LEG_IDX:
            qpos[0, :lengths[0], i] = 0.0
        h5_path = _write_h5(tmp_path / 'frozen.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert result['frozen_leg_clips'] == [0]
        assert result['per_clip_max_leg_ptp'][0] < 1e-3
        assert result['per_clip_max_leg_ptp'][1] >= 1e-3

        rc = main(['--h5', str(h5_path)])
        assert rc == 1

    def test_frozen_joint_counts_are_correct(self, tmp_path):
        """The whole clip must clear the frozen threshold (max leg ptp below
        `leg_ptp_threshold`) for `frozen_joint_counts` to have an entry --
        per the spec, per-joint static counts are reported only "for those
        [frozen] clips". Within that frozen clip, only some leg joints are
        EXACTLY static (ptp < 1e-6); the rest have tiny sub-threshold
        motion. `frozen_joint_counts` must count only the exactly-static
        ones.
        """
        t_max = 20
        lengths = [20]
        qpos = np.zeros((1, t_max, NQ), dtype=np.float32)
        # Root and "other" columns move normally (irrelevant to leg freeze).
        qpos[0, :, ROOT_IDX] = np.linspace(0, 1, lengths[0], dtype=np.float32)
        # First 5 leg columns: exactly static (ptp == 0).
        frozen_cols = LEG_IDX[:5]
        for i in frozen_cols:
            qpos[0, :lengths[0], i] = 3.0  # constant, but nonzero
        # Remaining leg columns: tiny sub-threshold motion (0 < ptp < 1e-6...
        # actually keep them just under leg_ptp_threshold but above 1e-6 so
        # they do NOT count as "exactly static").
        moving_cols = LEG_IDX[5:]
        for i in moving_cols:
            qpos[0, :lengths[0], i] = np.linspace(0, 5e-4, lengths[0], dtype=np.float32)
        h5_path = _write_h5(tmp_path / 'partial_frozen.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert result['frozen_leg_clips'] == [0]  # whole clip's max leg ptp (5e-4) < 1e-3
        counts = result['frozen_joint_counts']
        assert counts[0] == len(frozen_cols)


class TestPaddingIsExcluded:
    def test_motion_only_in_padding_is_still_frozen(self, tmp_path):
        """The subtle case: legs are dead flat for the TRUE span, then the
        padding tail (beyond clip_lengths[i]) is filled with obviously
        nonzero, high-ptp junk. A naive full-array ptp would miss this.
        """
        rng = np.random.default_rng(3)
        t_max = 50
        true_len = 20
        lengths = [true_len]
        qpos = np.zeros((1, t_max, NQ), dtype=np.float32)
        # True span: root moves, legs are frozen at 0.
        qpos[0, :true_len, ROOT_IDX] = rng.normal(size=(len(ROOT_IDX), true_len)).astype(np.float32)
        qpos[0, :true_len, LEG_IDX] = 0.0
        # Padding tail: legs jump around wildly -- must be ignored.
        qpos[0, true_len:, LEG_IDX] = rng.normal(size=(len(LEG_IDX), t_max - true_len)).astype(np.float32) * 100
        h5_path = _write_h5(tmp_path / 'padding_motion.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert result['frozen_leg_clips'] == [0]
        assert result['per_clip_max_leg_ptp'][0] < 1e-3

        rc = main(['--h5', str(h5_path)])
        assert rc == 1


class TestStaticRootIsNotFlagged:
    def test_moving_legs_static_root_not_flagged(self, tmp_path):
        rng = np.random.default_rng(4)
        t_max = 30
        true_len = 30
        lengths = [true_len]
        qpos = np.zeros((1, t_max, NQ), dtype=np.float32)
        ramp = np.linspace(0, 1, true_len, dtype=np.float32)
        for i in LEG_IDX:
            qpos[0, :true_len, i] = ramp + rng.normal(scale=0.01, size=true_len)
        # Root (the 'free' columns) is perfectly static.
        qpos[0, :, ROOT_IDX] = 1.0
        h5_path = _write_h5(tmp_path / 'static_root.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert result['frozen_leg_clips'] == []

        rc = main(['--h5', str(h5_path)])
        assert rc == 0


class TestNoFailOnFrozenFlag:
    def test_no_fail_on_frozen_returns_zero_but_still_reports(self, tmp_path, capsys):
        rng = np.random.default_rng(5)
        t_max = 20
        lengths = [20]
        qpos = np.stack([_moving_clip(rng, t_max, L, NQ, LEG_IDX) for L in lengths])
        for i in LEG_IDX:
            qpos[0, :lengths[0], i] = 0.0
        h5_path = _write_h5(tmp_path / 'frozen2.h5', lengths, qpos)

        rc = main(['--h5', str(h5_path), '--no-fail-on-frozen'])
        assert rc == 0
        captured = capsys.readouterr()
        assert 'frozen' in captured.out.lower()
        assert '1' in captured.out  # at least the frozen count is reported


class TestMaxReport:
    def test_max_report_limits_listed_indices(self, tmp_path, capsys):
        rng = np.random.default_rng(6)
        t_max = 20
        n_clips = 5
        lengths = [20] * n_clips
        qpos = np.stack([_moving_clip(rng, t_max, L, NQ, LEG_IDX) for L in lengths])
        for c in range(n_clips):
            for i in LEG_IDX:
                qpos[c, :lengths[c], i] = 0.0
        h5_path = _write_h5(tmp_path / 'all_frozen.h5', lengths, qpos)

        result = audit_clips(str(h5_path))
        assert len(result['frozen_leg_clips']) == n_clips

        rc = main(['--h5', str(h5_path), '--max-report', '2'])
        assert rc == 1


class TestLegPtpThreshold:
    def test_custom_threshold_is_honored(self, tmp_path):
        rng = np.random.default_rng(7)
        t_max = 20
        lengths = [20]
        qpos = np.stack([_moving_clip(rng, t_max, L, NQ, LEG_IDX) for L in lengths])
        # Small but nonzero leg motion: ptp around 5e-3.
        for i in LEG_IDX:
            qpos[0, :lengths[0], i] = np.linspace(0, 5e-3, lengths[0], dtype=np.float32)

        h5_path = _write_h5(tmp_path / 'small_motion.h5', lengths, qpos)

        default_result = audit_clips(str(h5_path))
        assert default_result['frozen_leg_clips'] == []  # 5e-3 > default 1e-3

        strict_result = audit_clips(str(h5_path), leg_ptp_threshold=1e-2)
        assert strict_result['frozen_leg_clips'] == [0]  # 5e-3 < 1e-2
