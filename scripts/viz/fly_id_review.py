"""Fly identity review server for courtship bouts.

Serves a keyboard-driven web UI (localhost only) to confirm or correct which
of fly0/fly1 is the male in every bout under a processed courtship root.
Decisions are saved immediately to <root>/id_review.json and to per-bout
sex.json (existing Session0 schema). Physical dir swaps are done separately
by scripts/viz/apply_fly_id_review.py.

Convention: fly0 = female, fly1 = male. `male_fly` always refers to the
CURRENT on-disk dir index.

Usage (on Hyak):
    python scripts/viz/fly_id_review.py \
        --root /gscratch/portia/eabe/data/Johnson_lab/processed/courtship
Then from your laptop:  ssh -L 8642:localhost:8642 <hyak-host>
and open http://localhost:8642  (VS Code Remote auto-forwards the port).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

DEFAULT_MALE_FLY = 1  # fly1 = male, fly0 = female
MANIFEST_NAME = "id_review.json"
VIDEO_NAME = "sidebyside.mp4"
DEFAULT_ROOT = Path("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")


# ---------------------------------------------------------------------------
# Scan + manifest
# ---------------------------------------------------------------------------

def bout_dir_from_key(root: Path, key: str) -> Path:
    """'Session1/recA/bout_00001' -> <root>/Session1/recA/pose/bouts/bout_00001."""
    session, rec, bout = key.split("/")
    return root / session / rec / "pose" / "bouts" / bout


def scan_bouts(root: Path) -> dict[str, dict]:
    """Find bout dirs under <root>/Session*/<rec>/pose/bouts/bout_*.

    A bout missing a fly dir or its video gets a warning (shown in the UI,
    excluded from apply).
    """
    bouts: dict[str, dict] = {}
    for bout_dir in sorted(root.glob("Session*/*/pose/bouts/bout_*")):
        if not bout_dir.is_dir():
            continue
        rec = bout_dir.parents[2].name
        session = bout_dir.parents[3].name
        key = f"{session}/{rec}/{bout_dir.name}"
        missing = [f"{fly}/{VIDEO_NAME}" for fly in ("fly0", "fly1")
                   if not (bout_dir / fly / VIDEO_NAME).is_file()]
        bouts[key] = {"warning": ("missing: " + ", ".join(missing)) if missing else None}
    return bouts


def read_sex_json(bout_dir: Path) -> dict | None:
    path = bout_dir / "sex.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def build_manifest(root: Path, existing: dict | None = None) -> dict:
    """Scan the tree and merge with an existing manifest.

    Existing decisions are never overwritten; new bouts enter as pending with
    the original assignment from sex.json (if present) else the convention
    default. Bouts that disappeared from disk are kept but flagged.
    """
    existing_bouts = (existing or {}).get("bouts", {})
    bouts: dict[str, dict] = {}
    for key, info in scan_bouts(root).items():
        prev = existing_bouts.get(key)
        if prev is not None:
            entry = dict(prev)
            entry["warning"] = info["warning"]
        else:
            sex = read_sex_json(bout_dir_from_key(root, key))
            if sex is not None and "male_fly" in sex:
                original, source = int(sex["male_fly"]), "sex.json"
            else:
                original, source = DEFAULT_MALE_FLY, "default"
            entry = {
                "original_male_fly": original,
                "reviewed_male_fly": original,
                "status": "pending",
                "source": source,
                "reviewed_at": None,
                "applied": False,
                "warning": info["warning"],
            }
        bouts[key] = entry
    for key, prev in existing_bouts.items():
        if key not in bouts:
            entry = dict(prev)
            entry["warning"] = "bout dir missing on disk"
            bouts[key] = entry
    return {"root": str(root), "convention": {"female": 0, "male": 1}, "bouts": bouts}


def save_manifest(root: Path, manifest: dict) -> None:
    path = root / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_manifest(root: Path) -> dict | None:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def record_decision(root: Path, manifest: dict, bout_key: str,
                    reviewed_male_fly: int, status: str) -> dict:
    """Record one review decision: update manifest (atomic) + bout sex.json.

    sex.json is only written for confirmed/swapped — an 'unsure' bout has no
    trustworthy identity to record.
    """
    if bout_key not in manifest["bouts"]:
        raise KeyError(bout_key)
    if status not in ("confirmed", "swapped", "unsure"):
        raise ValueError(f"bad status: {status}")
    if reviewed_male_fly not in (0, 1):
        raise ValueError(f"bad reviewed_male_fly: {reviewed_male_fly}")
    entry = manifest["bouts"][bout_key]
    entry["reviewed_male_fly"] = reviewed_male_fly
    entry["status"] = status
    entry["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(root, manifest)
    if status != "unsure":
        _write_sex_json(bout_dir_from_key(root, bout_key), entry)
    return entry


def _write_sex_json(bout_dir: Path, entry: dict) -> None:
    existing = read_sex_json(bout_dir) or {}
    sex = {
        "male_fly": entry["reviewed_male_fly"],
        "original_male_fly": existing.get("original_male_fly", entry["original_male_fly"]),
        "applied_swap": existing.get("applied_swap", False),
        "confidence": "user",
        "method": "manual-gui",
        "note": f"fly_id_review {entry['reviewed_at']} status={entry['status']}",
    }
    tmp = bout_dir / "sex.json.tmp"
    tmp.write_text(json.dumps(sex, indent=2))
    os.replace(tmp, bout_dir / "sex.json")
