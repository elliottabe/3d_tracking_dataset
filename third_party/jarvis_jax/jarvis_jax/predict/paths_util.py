"""Pure helpers for locating a recording's processed-output dir.

The processed tree mirrors the raw Video_recordings/<dataset>/SessionN/<rec>
hierarchy under <processed_root>/<dataset>/SessionN/<rec>, so raw data stays
read-only and derived outputs are transferable as one subtree.
"""
import os

from jarvis_jax.predict.sam3_driver import session_tag_for  # '.../SessionN/<rec>' -> 'SessionN/<rec>'


def dataset_for(session_dir, default=None):
    """The dataset component of '.../<dataset>/SessionN/<rec>' (3rd from end).
    Returns `default` when the path has fewer than 3 components."""
    parts = session_dir.rstrip("/").split("/")
    if len(parts) >= 3:
        return parts[-3]
    return default


def processed_dir_for(processed_root, session_dir, *, dataset=None):
    """'<processed_root>/<dataset>/SessionN/<rec>' for one recording."""
    ds = dataset or dataset_for(session_dir)
    return os.path.join(processed_root, ds, session_tag_for(session_dir))


def resolve_auto(value, fallback):
    """Return `fallback` when `value` is the sentinel 'auto' (case-insensitive),
    None, or empty; otherwise return `value` unchanged."""
    if value is None:
        return fallback
    if isinstance(value, str) and value.strip().lower() in ("", "auto"):
        return fallback
    return value
