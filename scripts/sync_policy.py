"""Whether the stale-mask cascade may delete previously-computed pose artifacts.

Production keeps this ON: when a recording's frame-sync plan changes, artifacts
derived from the old masks are stale and must recompute. Benchmark/A-B runs turn
it OFF so pre-seeded (frozen) inputs survive and every variant shares identical
2D -- see docs/benchmark/2026-08-07-scale-ab/comparison.md.
"""
from __future__ import annotations

_FALSY_STRINGS = {"false", "no", "off", "0"}
_TRUTHY_STRINGS = {"true", "yes", "on", "1"}


def stale_invalidation_enabled(cfg) -> bool:
    """True when run_bout may delete stale-mask-derived artifacts.

    Reads cfg.sync.invalidate_stale, defaulting to True when the key or the
    whole `sync` block is absent (production behaviour is unchanged).

    Accepts a plain dict or an OmegaConf DictConfig. The rule for the raw
    value is deliberately asymmetric, because disabling the cascade must
    always be explicit and unambiguous -- anything ambiguous resolves to the
    safe default (True, i.e. keep deleting stale artifacts):

      - an actual bool is returned as-is (True -> True, False -> False).
      - a recognized falsy string ("false", "no", "off", "0", case-
        insensitive, surrounding whitespace ignored) -> False.
      - a recognized truthy string ("true", "yes", "on", "1", same rules)
        -> True.
      - any OTHER string -- including "", or junk like "maybe" -- is NOT
        run through Python's bare `bool(str)` truthiness (that would make
        "" resolve to False, silently and unsafely turning the cascade off)
        -> True.
      - any other type (int, float, list, dict, ...), including falsy ones
        like 0, 0.0, [], {} -> True. Only a real `False` or a recognized
        falsy string may disable the cascade; every other value resolves to
        the safe default.

    The return value is always a real bool.
    """
    sync_block = getattr(cfg, "sync", None)
    if sync_block is None and isinstance(cfg, dict):
        sync_block = cfg.get("sync")
    if sync_block is None:
        return True

    value = getattr(sync_block, "invalidate_stale", None)
    if value is None and isinstance(sync_block, dict):
        value = sync_block.get("invalidate_stale")
    if value is None:
        return True

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _FALSY_STRINGS:
            return False
        if lowered in _TRUTHY_STRINGS:
            return True
        return True  # unrecognized string (incl. "") -> safe default

    return True  # non-bool, non-string (0, 0.0, [], {}, ...) -> safe default
