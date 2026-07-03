"""Atomic writes + stage-skip markers for the resumable per-bout driver."""
from __future__ import annotations
import json, os
import numpy as np


def atomic_write(path, write_fn):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_fn(tmp)
    os.replace(tmp, path)
    return path


def atomic_save_npz(path, **arrays):
    return atomic_write(path, lambda p: _savez_exact(p, arrays))


def _savez_exact(tmp, arrays):
    # np.savez appends .npz if missing; writing to an open file object (rather
    # than a path string) bypasses that auto-append, so os.replace(tmp, path)
    # in atomic_write lands the bytes exactly at `path` (no `path.npz` stray).
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)


def atomic_save_json(path, obj):
    return atomic_write(path, lambda p: json.dump(obj, open(p, "w"), indent=2))


def stage_done(path) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def mark_done(bout_dir):
    atomic_write(os.path.join(bout_dir, "DONE"), lambda p: open(p, "w").write("ok"))


def bout_complete(bout_dir) -> bool:
    return os.path.exists(os.path.join(bout_dir, "DONE"))
