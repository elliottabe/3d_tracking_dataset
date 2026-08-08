"""Tests for scripts/sync_policy.stale_invalidation_enabled.

Production keeps stale-mask invalidation ON; benchmark/A-B runs turn it OFF so
pre-seeded (frozen) kp2d/kp3d/... inputs survive a sync-plan change instead of
being deleted and recomputed per-variant. See
docs/benchmark/2026-08-07-scale-ab/comparison.md.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.sync_policy import stale_invalidation_enabled


def test_default_true_when_sync_block_absent():
    cfg = OmegaConf.create({"paths": {"base_dir": "/x"}})
    assert stale_invalidation_enabled(cfg) is True


def test_default_true_when_sync_present_without_key():
    cfg = OmegaConf.create({"sync": {}})
    assert stale_invalidation_enabled(cfg) is True


def test_explicit_true():
    cfg = OmegaConf.create({"sync": {"invalidate_stale": True}})
    assert stale_invalidation_enabled(cfg) is True


def test_explicit_false():
    cfg = OmegaConf.create({"sync": {"invalidate_stale": False}})
    assert stale_invalidation_enabled(cfg) is False


def test_plain_dict_default_true():
    cfg = {"paths": {"base_dir": "/x"}}
    assert stale_invalidation_enabled(cfg) is True


def test_plain_dict_sync_without_key():
    cfg = {"sync": {}}
    assert stale_invalidation_enabled(cfg) is True


def test_plain_dict_explicit_false():
    cfg = {"sync": {"invalidate_stale": False}}
    assert stale_invalidation_enabled(cfg) is False


def test_plain_dict_explicit_true():
    cfg = {"sync": {"invalidate_stale": True}}
    assert stale_invalidation_enabled(cfg) is True


def test_returns_real_bool_not_truthy_object():
    cfg = OmegaConf.create({"sync": {"invalidate_stale": False}})
    result = stale_invalidation_enabled(cfg)
    assert result is False
    assert isinstance(result, bool)

    cfg_true = OmegaConf.create({"sync": {"invalidate_stale": True}})
    result_true = stale_invalidation_enabled(cfg_true)
    assert result_true is True
    assert isinstance(result_true, bool)


def test_string_false_is_treated_as_false():
    # A YAML-ish "false" string (e.g. from a CLI override that didn't get cast)
    # must not silently read as truthy just because it's a non-empty string.
    # Only the recognised falsy spellings disable the cascade; every other
    # string -- including "" -- falls back to the safe default True. (Falling
    # back to Python truthiness instead is what made "" silently disable
    # invalidation; see the fix in scripts/sync_policy.py.)
    cfg = {"sync": {"invalidate_stale": "false"}}
    assert stale_invalidation_enabled(cfg) is False


def test_string_true_is_treated_as_true():
    cfg = {"sync": {"invalidate_stale": "true"}}
    assert stale_invalidation_enabled(cfg) is True


def test_empty_string_falls_back_to_true():
    # bool("") is False in plain Python -- that would silently disable the
    # cascade on a malformed/empty config value. Disabling must be explicit,
    # so an empty string is NOT a recognized falsy spelling and must fall
    # back to the safe default (True).
    cfg = {"sync": {"invalidate_stale": ""}}
    assert stale_invalidation_enabled(cfg) is True


def test_unrecognized_string_falls_back_to_true():
    cfg = {"sync": {"invalidate_stale": "maybe"}}
    assert stale_invalidation_enabled(cfg) is True


@pytest.mark.parametrize("value", [0, 0.0, [], {}])
def test_odd_falsy_nonbool_values_fall_back_to_true(value):
    # 0 / 0.0 / [] / {} are all falsy under bool(), but none of them is a
    # real `False` or a recognized falsy string -- only those two may
    # disable the cascade. Everything else (including these) resolves to
    # the safe default True, so a stray non-bool config value can never
    # silently turn off stale-artifact deletion.
    cfg = {"sync": {"invalidate_stale": value}}
    assert stale_invalidation_enabled(cfg) is True
