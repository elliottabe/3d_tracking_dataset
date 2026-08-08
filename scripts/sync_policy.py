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

    Accepts a plain dict or an OmegaConf DictConfig. A recognized falsy/truthy
    YAML-ish string (e.g. "false", "no", "0" / "true", "yes", "1", case-
    insensitive) is interpreted accordingly rather than relying on Python's
    "non-empty string is truthy" rule, since a stray unconverted CLI override
    string must not silently disable the flag's intent. Any other value falls
    back to plain Python truthiness. The return value is always a real bool.
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

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _FALSY_STRINGS:
            return False
        if lowered in _TRUTHY_STRINGS:
            return True
        return bool(value)

    return bool(value)
