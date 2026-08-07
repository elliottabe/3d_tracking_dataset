# Fly ID Review GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A keyboard-driven localhost web GUI to confirm/correct which of fly0/fly1 is the male in every courtship bout, plus a dry-run-by-default apply script that physically swaps fly dirs so the whole tree is uniformly fly0=female / fly1=male.

**Architecture:** One stdlib-only server script (`scripts/viz/fly_id_review.py`) that scans the data tree, maintains `<root>/id_review.json` + per-bout `sex.json`, and serves an embedded single-page UI with HTTP 206 byte-range video streaming. A second script (`scripts/viz/apply_fly_id_review.py`) reads the manifest and performs the physical swaps.

**Tech Stack:** Python 3.12 stdlib only (`http.server`, `json`, `pathlib`, `threading`). Tests: pytest 9 (already in the `3d_tracking` env). No new dependencies.

**Spec:** `docs/specs/2026-08-07-fly-id-review-design.md`

## Global Constraints

- Scripts must import **nothing outside the Python standard library** (tests may use pytest).
- Server binds `127.0.0.1` only; `/media` must reject paths resolving outside the data root.
- Convention constant: fly0 = female, fly1 = male (`DEFAULT_MALE_FLY = 1`). `male_fly` in any `sex.json` always refers to the **current on-disk dir index**.
- All manifest and `sex.json` writes are atomic: write `*.tmp` then `os.replace`.
- Rescans never overwrite existing manifest decisions.
- `sex.json` is written only for `confirmed`/`swapped` decisions — never for `unsure`.
- Data root default: `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship`.
- Bout key format everywhere: `"Session1/2026_04_02_16_03_48/bout_00001"` (session/recording/bout, no `pose/bouts`).
- Manifest bout entry fields (exact names): `original_male_fly`, `reviewed_male_fly`, `status` (`pending|confirmed|swapped|unsure`), `source` (`default|sex.json`), `reviewed_at` (ISO-8601 or null), `applied` (bool), `warning` (str or null).
- Videos are already H.264 + faststart (verified) — no re-encoding anywhere in this plan.
- Commit style: `feat(viz): ...` / `test(viz): ...` matching repo history.

## File Structure

- Create `scripts/viz/__init__.py` — empty; makes `scripts.viz` importable from tests (Task 1).
- Create `scripts/viz/fly_id_review.py` — scan/manifest/decision functions (Tasks 1–2), range parsing + HTTP server (Task 3), embedded UI + `main()` (Task 4).
- Create `scripts/viz/apply_fly_id_review.py` — validate/plan/swap/apply + `main()` (Task 5).
- Create `tests/test_fly_id_review.py` — unit + integration tests for the server script; also exports the `make_bout` fixture-helper reused by Task 5's tests.
- Create `tests/test_apply_fly_id_review.py` — apply-script tests on synthetic trees.

---

### Task 1: Scan + manifest core

**Files:**
- Create: `scripts/viz/__init__.py` (empty file)
- Create: `scripts/viz/fly_id_review.py`
- Test: `tests/test_fly_id_review.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces (used by every later task):
  - `DEFAULT_MALE_FLY: int = 1`, `MANIFEST_NAME = "id_review.json"`, `VIDEO_NAME = "sidebyside.mp4"`
  - `bout_dir_from_key(root: Path, key: str) -> Path`
  - `scan_bouts(root: Path) -> dict[str, dict]` — `{key: {"warning": str | None}}`
  - `read_sex_json(bout_dir: Path) -> dict | None`
  - `build_manifest(root: Path, existing: dict | None = None) -> dict`
  - `save_manifest(root: Path, manifest: dict) -> None` (atomic), `load_manifest(root: Path) -> dict | None`
  - Test helper `make_bout(root, session, rec, bout, flies=("fly0","fly1"), video=True, sex_json=None) -> Path` exported from `tests/test_fly_id_review.py`.

- [ ] **Step 1: Create the package init**

```bash
touch scripts/viz/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_fly_id_review.py`:

```python
"""Tests for scripts/viz/fly_id_review.py (fly identity review server)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viz.fly_id_review import (
    DEFAULT_MALE_FLY,
    MANIFEST_NAME,
    VIDEO_NAME,
    bout_dir_from_key,
    build_manifest,
    load_manifest,
    read_sex_json,
    save_manifest,
    scan_bouts,
)


def make_bout(root: Path, session: str, rec: str, bout: str,
              flies=("fly0", "fly1"), video=True, sex_json=None) -> Path:
    """Create a synthetic bout dir matching the real tree layout."""
    bout_dir = root / session / rec / "pose" / "bouts" / bout
    for fly in flies:
        d = bout_dir / fly
        d.mkdir(parents=True)
        if video:
            (d / VIDEO_NAME).write_bytes(b"\x00" * 32)
    if sex_json is not None:
        (bout_dir / "sex.json").write_text(json.dumps(sex_json))
    return bout_dir


# ---------------------------------------------------------------------------
# scan_bouts
# ---------------------------------------------------------------------------

def test_scan_finds_bouts_sorted(tmp_path):
    make_bout(tmp_path, "Session1", "recB", "bout_00002")
    make_bout(tmp_path, "Session0", "recA", "bout_00001")
    keys = list(scan_bouts(tmp_path))
    assert keys == ["Session0/recA/bout_00001", "Session1/recB/bout_00002"]


def test_scan_flags_missing_video_and_fly(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001", flies=("fly0",))
    make_bout(tmp_path, "Session1", "recA", "bout_00002", video=False)
    make_bout(tmp_path, "Session1", "recA", "bout_00003")
    bouts = scan_bouts(tmp_path)
    assert bouts["Session1/recA/bout_00001"]["warning"]  # fly1 missing entirely
    assert bouts["Session1/recA/bout_00002"]["warning"]  # videos missing
    assert bouts["Session1/recA/bout_00003"]["warning"] is None


def test_bout_dir_from_key_roundtrip(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00007")
    assert bout_dir_from_key(tmp_path, "Session1/recA/bout_00007") == bout_dir


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

def test_build_manifest_defaults(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m = build_manifest(tmp_path)
    e = m["bouts"]["Session1/recA/bout_00001"]
    assert e == {
        "original_male_fly": DEFAULT_MALE_FLY,
        "reviewed_male_fly": DEFAULT_MALE_FLY,
        "status": "pending",
        "source": "default",
        "reviewed_at": None,
        "applied": False,
        "warning": None,
    }
    assert m["convention"] == {"female": 0, "male": 1}
    assert m["root"] == str(tmp_path)


def test_build_manifest_reads_existing_sex_json(tmp_path):
    make_bout(tmp_path, "Session0", "recA", "bout_00001",
              sex_json={"male_fly": 0, "original_male_fly": 0,
                        "applied_swap": False, "method": "manual"})
    e = build_manifest(tmp_path)["bouts"]["Session0/recA/bout_00001"]
    assert e["original_male_fly"] == 0
    assert e["reviewed_male_fly"] == 0
    assert e["source"] == "sex.json"


def test_rescan_preserves_decisions_and_adds_new(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m1 = build_manifest(tmp_path)
    m1["bouts"]["Session1/recA/bout_00001"].update(
        {"status": "swapped", "reviewed_male_fly": 0, "reviewed_at": "2026-08-07T00:00:00+00:00"})
    make_bout(tmp_path, "Session1", "recA", "bout_00002")
    m2 = build_manifest(tmp_path, existing=m1)
    assert m2["bouts"]["Session1/recA/bout_00001"]["status"] == "swapped"
    assert m2["bouts"]["Session1/recA/bout_00001"]["reviewed_male_fly"] == 0
    assert m2["bouts"]["Session1/recA/bout_00002"]["status"] == "pending"


def test_rescan_flags_disappeared_bout(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m1 = build_manifest(tmp_path)
    import shutil
    shutil.rmtree(tmp_path / "Session1")
    m2 = build_manifest(tmp_path, existing=m1)
    assert m2["bouts"]["Session1/recA/bout_00001"]["warning"] == "bout dir missing on disk"


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_atomic(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    m = build_manifest(tmp_path)
    save_manifest(tmp_path, m)
    assert load_manifest(tmp_path) == m
    assert not (tmp_path / (MANIFEST_NAME + ".tmp")).exists()
    assert not list(tmp_path.glob("*.json.tmp"))


def test_load_manifest_missing_returns_none(tmp_path):
    assert load_manifest(tmp_path) is None


def test_read_sex_json_absent_or_corrupt(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    assert read_sex_json(bout_dir) is None
    (bout_dir / "sex.json").write_text("{not json")
    assert read_sex_json(bout_dir) is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_fly_id_review.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'scripts.viz.fly_id_review'`

- [ ] **Step 4: Write the implementation**

Create `scripts/viz/fly_id_review.py`:

```python
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
```

(Watch the `.tmp` naming: `path.with_name(path.name + ".tmp")` — the test asserts no `*.json.tmp` file survives.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/__init__.py scripts/viz/fly_id_review.py tests/test_fly_id_review.py
git commit -m "feat(viz): fly ID review scan + manifest core"
```

---

### Task 2: Decision recording + sex.json writing

**Files:**
- Modify: `scripts/viz/fly_id_review.py` (append after `load_manifest`)
- Test: `tests/test_fly_id_review.py` (append)

**Interfaces:**
- Consumes: Task 1's `save_manifest`, `read_sex_json`, `bout_dir_from_key`.
- Produces: `record_decision(root: Path, manifest: dict, bout_key: str, reviewed_male_fly: int, status: str) -> dict` — updates + saves the manifest, writes `sex.json` for confirmed/swapped, returns the updated entry. Raises `KeyError` (unknown bout) / `ValueError` (bad status or fly index). Used by the POST handler in Task 3.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fly_id_review.py`)

```python
from scripts.viz.fly_id_review import record_decision


# ---------------------------------------------------------------------------
# record_decision
# ---------------------------------------------------------------------------

def _fresh(tmp_path, **bout_kw):
    make_bout(tmp_path, "Session1", "recA", "bout_00001", **bout_kw)
    return build_manifest(tmp_path)


def test_record_confirmed_writes_manifest_and_sex_json(tmp_path):
    m = _fresh(tmp_path)
    e = record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "confirmed")
    assert e["status"] == "confirmed" and e["reviewed_male_fly"] == 1
    assert e["reviewed_at"] is not None
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["status"] == "confirmed"
    sex = read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001"))
    assert sex["male_fly"] == 1
    assert sex["original_male_fly"] == 1
    assert sex["applied_swap"] is False
    assert sex["method"] == "manual-gui"
    assert sex["confidence"] == "user"


def test_record_swap_writes_male_fly_0(tmp_path):
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, "Session1/recA/bout_00001", 0, "swapped")
    sex = read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001"))
    assert sex["male_fly"] == 0
    assert sex["original_male_fly"] == 1  # pre-review original preserved


def test_record_preserves_existing_sex_json_history(tmp_path):
    m = _fresh(tmp_path, sex_json={"male_fly": 1, "original_male_fly": 0,
                                   "applied_swap": True, "method": "manual",
                                   "confidence": "user", "note": "old"})
    record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "confirmed")
    sex = read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001"))
    assert sex["original_male_fly"] == 0   # history kept
    assert sex["applied_swap"] is True     # history kept


def test_record_unsure_skips_sex_json(tmp_path):
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "unsure")
    assert read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001")) is None
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["status"] == "unsure"


def test_record_rejects_bad_input(tmp_path):
    m = _fresh(tmp_path)
    with pytest.raises(KeyError):
        record_decision(tmp_path, m, "Session9/nope/bout_99999", 1, "confirmed")
    with pytest.raises(ValueError):
        record_decision(tmp_path, m, "Session1/recA/bout_00001", 2, "confirmed")
    with pytest.raises(ValueError):
        record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "pending")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fly_id_review.py -v -k record`
Expected: FAIL — `ImportError: cannot import name 'record_decision'`

- [ ] **Step 3: Write the implementation** (append to `scripts/viz/fly_id_review.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/fly_id_review.py tests/test_fly_id_review.py
git commit -m "feat(viz): fly ID review decision recording + sex.json writes"
```

---

### Task 3: HTTP server — byte-range media + JSON API

**Files:**
- Modify: `scripts/viz/fly_id_review.py` (append)
- Test: `tests/test_fly_id_review.py` (append)

**Interfaces:**
- Consumes: Task 1–2 functions.
- Produces:
  - `parse_range(header: str | None, file_size: int) -> tuple[int, int] | str | None` — inclusive byte offsets, the string `"unsatisfiable"` (→ HTTP 416), or `None` (→ plain 200).
  - `ReviewServer(addr: tuple[str, int], root: Path, manifest: dict)` — `ThreadingHTTPServer` subclass with `.root`, `.manifest`, `.lock` attributes.
  - `PAGE_HTML: str` placeholder is **not** created here — Task 4 defines it. Until then `do_GET("/")` returns 404; the route is added in Task 4.
  - Routes: `GET /api/bouts` (manifest JSON), `POST /api/decision` (`{"bout_key", "reviewed_male_fly", "status"}` → updated entry JSON, 400 on bad input), `GET /media/<relpath>` (206/200/416, path-guarded).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fly_id_review.py`)

```python
import http.client
import threading
import urllib.request

from scripts.viz.fly_id_review import ReviewServer, parse_range


# ---------------------------------------------------------------------------
# parse_range (pure)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("header,size,expected", [
    (None, 100, None),                       # no header -> whole file
    ("bytes=0-9", 100, (0, 9)),
    ("bytes=10-", 100, (10, 99)),            # open-ended
    ("bytes=-20", 100, (80, 99)),            # suffix
    ("bytes=0-500", 100, (0, 99)),           # end clamped
    ("bytes=100-", 100, "unsatisfiable"),    # start past EOF -> 416
    ("bytes=-0", 100, "unsatisfiable"),      # zero-length suffix -> 416
    ("bytes=5-3", 100, None),                # inverted -> ignore, whole file
    ("bites=0-9", 100, None),                # malformed -> ignore
    ("bytes=-", 100, None),                  # empty -> ignore
])
def test_parse_range(header, size, expected):
    assert parse_range(header, size) == expected


# ---------------------------------------------------------------------------
# server integration
# ---------------------------------------------------------------------------

MEDIA_BYTES = bytes(range(256)) * 4  # 1024 recognizable bytes


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "data"
    bout_dir = make_bout(root, "Session1", "recA", "bout_00001")
    (bout_dir / "fly0" / VIDEO_NAME).write_bytes(MEDIA_BYTES)
    manifest = build_manifest(root)
    save_manifest(root, manifest)
    srv = ReviewServer(("127.0.0.1", 0), root, manifest)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    yield f"http://{host}:{port}", srv, root
    srv.shutdown()


MEDIA_PATH = "/media/Session1/recA/pose/bouts/bout_00001/fly0/sidebyside.mp4"


def test_api_bouts_returns_manifest(server):
    url, srv, root = server
    with urllib.request.urlopen(url + "/api/bouts") as resp:
        body = json.loads(resp.read())
    assert "Session1/recA/bout_00001" in body["bouts"]


def test_media_full_200_with_accept_ranges(server):
    url, srv, root = server
    with urllib.request.urlopen(url + MEDIA_PATH) as resp:
        assert resp.status == 200
        assert resp.headers["Accept-Ranges"] == "bytes"
        assert resp.read() == MEDIA_BYTES


def test_media_range_206(server):
    url, srv, root = server
    req = urllib.request.Request(url + MEDIA_PATH, headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert resp.headers["Content-Range"] == f"bytes 10-19/{len(MEDIA_BYTES)}"
        assert resp.headers["Content-Length"] == "10"
        assert resp.read() == MEDIA_BYTES[10:20]


def test_media_suffix_range(server):
    url, srv, root = server
    req = urllib.request.Request(url + MEDIA_PATH, headers={"Range": "bytes=-16"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 206
        assert resp.read() == MEDIA_BYTES[-16:]


def test_media_unsatisfiable_416(server):
    url, srv, root = server
    req = urllib.request.Request(url + MEDIA_PATH, headers={"Range": "bytes=999999-"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 416


def test_media_traversal_forbidden(server):
    url, srv, root = server
    (root.parent / "secret.txt").write_text("nope")
    host, port = srv.server_address
    conn = http.client.HTTPConnection(host, port)  # raw: no client-side path collapse
    conn.request("GET", "/media/../secret.txt")
    assert conn.getresponse().status in (403, 404)
    conn.close()


def test_post_decision_roundtrip(server):
    url, srv, root = server
    payload = json.dumps({"bout_key": "Session1/recA/bout_00001",
                          "reviewed_male_fly": 0, "status": "swapped"}).encode()
    req = urllib.request.Request(url + "/api/decision", data=payload, method="POST")
    with urllib.request.urlopen(req) as resp:
        entry = json.loads(resp.read())
    assert entry["status"] == "swapped"
    assert load_manifest(root)["bouts"]["Session1/recA/bout_00001"]["reviewed_male_fly"] == 0
    assert read_sex_json(bout_dir_from_key(root, "Session1/recA/bout_00001"))["male_fly"] == 0


def test_post_decision_bad_input_400(server):
    url, srv, root = server
    payload = json.dumps({"bout_key": "nope/nope/nope",
                          "reviewed_male_fly": 1, "status": "confirmed"}).encode()
    req = urllib.request.Request(url + "/api/decision", data=payload, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400
```

Add `import urllib.error` next to the other new imports at the top of the test additions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fly_id_review.py -v -k "parse_range or server or media or post"`
Expected: FAIL — `ImportError: cannot import name 'ReviewServer'`

- [ ] **Step 3: Write the implementation** (append to `scripts/viz/fly_id_review.py`)

```python
# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def parse_range(header: str | None, file_size: int):
    """Parse a Range header. Returns inclusive (start, end), the string
    'unsatisfiable' (caller sends 416), or None (caller sends plain 200 --
    malformed headers are ignored per RFC 9110)."""
    if not header:
        return None
    m = _RANGE_RE.match(header.strip())
    if not m:
        return None
    start_s, end_s = m.groups()
    if start_s == "" and end_s == "":
        return None
    if start_s == "":  # suffix: last N bytes
        n = int(end_s)
        if n == 0 or file_size == 0:
            return "unsatisfiable"
        return (max(0, file_size - n), file_size - 1)
    start = int(start_s)
    if start >= file_size:
        return "unsatisfiable"
    end = int(end_s) if end_s else file_size - 1
    if end < start:
        return None
    return (start, min(end, file_size - 1))


class ReviewHandler(BaseHTTPRequestHandler):
    server: "ReviewServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the terminal quiet during review
        pass

    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", PAGE_HTML.encode())
        elif path == "/api/bouts":
            with self.server.lock:
                body = json.dumps(self.server.manifest).encode()
            self._send(HTTPStatus.OK, "application/json", body)
        elif path.startswith("/media/"):
            self._serve_media(path[len("/media/"):])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/decision":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length))
            with self.server.lock:
                entry = record_decision(
                    self.server.root, self.server.manifest,
                    req["bout_key"], int(req["reviewed_male_fly"]), req["status"])
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, explain=str(exc))
            return
        self._send(HTTPStatus.OK, "application/json", json.dumps(entry).encode())

    # -- media with byte-range support (the seek-performance requirement) --

    def _serve_media(self, relpath: str):
        root = self.server.root.resolve()
        target = (root / relpath).resolve()
        if not (target == root or str(target).startswith(str(root) + os.sep)):
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = target.stat().st_size
        rng = parse_range(self.headers.get("Range"), size)
        if rng == "unsatisfiable":
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ctype = "video/mp4" if target.suffix == ".mp4" else "application/octet-stream"
        with open(target, "rb") as f:
            if rng is None:
                self.send_response(HTTPStatus.OK)
                start, length = 0, size
            else:
                start, end = rng
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            f.seek(start)
            self._copy(f, length)

    def _copy(self, f, length: int, chunk: int = 64 * 1024):
        remaining = length
        try:
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser aborted mid-stream (e.g. a seek) -- normal

    def _send(self, status, ctype: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, root: Path, manifest: dict):
        super().__init__(addr, ReviewHandler)
        self.root = root
        self.manifest = manifest
        self.lock = threading.Lock()


PAGE_HTML = "<!doctype html><title>Fly ID Review</title>placeholder until Task 4"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/fly_id_review.py tests/test_fly_id_review.py
git commit -m "feat(viz): fly ID review HTTP server with 206 byte-range media"
```

---

### Task 4: Embedded UI + main()

**Files:**
- Modify: `scripts/viz/fly_id_review.py` (replace the `PAGE_HTML` stub; append `main`)
- Test: `tests/test_fly_id_review.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `PAGE_HTML: str` (full page), `main(argv: list[str] | None = None) -> None` with `--root` (default `DEFAULT_ROOT`) and `--port` (default 8642). The UI POSTs to `/api/decision` exactly as Task 3 defined.

UI behavior spec (from the design doc): two synced looping videos with M/F badges; header shows bout key, recording filter, status, warning, progress; keys — Enter/→ confirm, S swap, U unsure, ← back, J next unresolved, Space pause, R replay, 1/2 speed. Status is *derived*: `confirmed` if the displayed assignment equals the original, else `swapped`. Prefetch of bout N+1 via hidden preloading video elements.

- [ ] **Step 1: Write the failing test** (append to `tests/test_fly_id_review.py`)

```python
# ---------------------------------------------------------------------------
# page + main
# ---------------------------------------------------------------------------

def test_index_serves_ui(server):
    url, srv, root = server
    with urllib.request.urlopen(url + "/") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers["Content-Type"]
        html = resp.read().decode()
    for needle in ('id="v0"', 'id="v1"', "keydown", "/api/bouts", "/api/decision",
                   "prefetch", "playbackRate"):
        assert needle in html, needle


def test_main_scans_and_saves_manifest_before_serving(tmp_path, monkeypatch):
    import scripts.viz.fly_id_review as mod
    make_bout(tmp_path, "Session1", "recA", "bout_00001")

    served = {}

    class FakeServer:
        def __init__(self, addr, root, manifest):
            served["root"], served["manifest"] = root, manifest
            self.server_address = ("127.0.0.1", 0)
        def serve_forever(self):
            raise KeyboardInterrupt  # return immediately

    monkeypatch.setattr(mod, "ReviewServer", FakeServer)
    mod.main(["--root", str(tmp_path), "--port", "0"])
    assert "Session1/recA/bout_00001" in served["manifest"]["bouts"]
    assert load_manifest(tmp_path) is not None  # manifest persisted before serving
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fly_id_review.py -v -k "index or main"`
Expected: `test_index_serves_ui` FAILS (stub page lacks the needles); `test_main_...` FAILS (`module ... has no attribute 'main'`)

- [ ] **Step 3: Write the implementation**

Replace the `PAGE_HTML` stub in `scripts/viz/fly_id_review.py` with the full page, and append `main`:

```python
PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Fly ID Review</title>
<style>
  body { margin:0; font:14px system-ui, sans-serif; background:#111; color:#ddd; }
  header { display:flex; gap:1rem; align-items:center; padding:.5rem 1rem; background:#1b1b1b; }
  header .spacer { flex:1; }
  main { display:flex; gap:8px; padding:8px; }
  .card { flex:1; position:relative; }
  video { width:100%; background:#000; display:block; }
  .badge { position:absolute; top:8px; left:8px; font-size:28px; font-weight:800;
           padding:2px 14px; border-radius:6px; color:#fff; }
  .badge.male { background:#1565d8; }
  .badge.female { background:#c2337e; }
  .flylabel { position:absolute; top:8px; right:8px; color:#aaa; font-size:13px; }
  footer { padding:.4rem 1rem; color:#888; font-size:12px; }
  #status { font-weight:600; }
  .st-confirmed { color:#5dbb63; } .st-swapped { color:#e2a93b; }
  .st-unsure { color:#d05c5c; } .st-pending { color:#888; }
  select { background:#222; color:#ddd; border:1px solid #444; }
  kbd { background:#2a2a2a; border-radius:3px; padding:0 4px; }
</style></head>
<body>
<header>
  <span id="pos"></span>
  <select id="filter"><option value="">all recordings</option></select>
  <span id="status"></span>
  <span id="warn" style="color:#e2a93b"></span>
  <span class="spacer"></span>
  <span id="progress"></span>
</header>
<main>
  <div class="card"><video id="v0" muted loop autoplay playsinline></video>
    <div class="badge" id="b0"></div><div class="flylabel">fly0</div></div>
  <div class="card"><video id="v1" muted loop autoplay playsinline></video>
    <div class="badge" id="b1"></div><div class="flylabel">fly1</div></div>
</main>
<footer>
 <kbd>Enter</kbd>/<kbd>&rarr;</kbd> confirm &middot; <kbd>S</kbd> swap &middot;
 <kbd>U</kbd> unsure &middot; <kbd>&larr;</kbd> back &middot;
 <kbd>J</kbd> next unresolved &middot; <kbd>Space</kbd> pause &middot;
 <kbd>R</kbd> replay &middot; <kbd>1</kbd>/<kbd>2</kbd> speed
</footer>
<script>
let bouts = {}, keys = [], idx = 0, assign = {}; // assign[key] = displayed male fly index
const $ = id => document.getElementById(id);
const v0 = $("v0"), v1 = $("v1");
// prefetch elements for bout N+1 (seek-performance requirement #4)
const prefetch = [document.createElement("video"), document.createElement("video")];
prefetch.forEach(v => { v.preload = "auto"; v.muted = true; });

function mediaUrl(key, fly) {
  const [s, r, b] = key.split("/");
  return `/media/${s}/${r}/pose/bouts/${b}/fly${fly}/sidebyside.mp4`;
}
function visibleKeys() {
  const f = $("filter").value;
  return f ? keys.filter(k => k.startsWith(f + "/")) : keys;
}
function render() {
  const vis = visibleKeys();
  if (!vis.length) return;
  idx = Math.max(0, Math.min(idx, vis.length - 1));
  const key = vis[idx], e = bouts[key], male = assign[key];
  v0.src = mediaUrl(key, 0); v1.src = mediaUrl(key, 1);
  v0.play().catch(() => {}); v1.play().catch(() => {});
  $("b0").textContent = male === 0 ? "M" : "F";
  $("b0").className = "badge " + (male === 0 ? "male" : "female");
  $("b1").textContent = male === 1 ? "M" : "F";
  $("b1").className = "badge " + (male === 1 ? "male" : "female");
  $("pos").textContent = `${idx + 1}/${vis.length}  ${key}`;
  $("status").textContent = e.status;
  $("status").className = "st-" + e.status;
  $("warn").textContent = e.warning || "";
  const done = vis.filter(k => bouts[k].status !== "pending").length;
  $("progress").textContent = `${done}/${vis.length} reviewed`;
  if (idx + 1 < vis.length) {
    prefetch[0].src = mediaUrl(vis[idx + 1], 0);
    prefetch[1].src = mediaUrl(vis[idx + 1], 1);
  }
}
async function post(status) {
  const key = visibleKeys()[idx];
  const res = await fetch("/api/decision", { method: "POST",
    body: JSON.stringify({ bout_key: key, reviewed_male_fly: assign[key], status }) });
  if (!res.ok) { alert("save failed: " + await res.text()); return false; }
  bouts[key] = await res.json();
  return true;
}
async function decide(status) { if (await post(status)) { idx += 1; render(); } }

document.addEventListener("keydown", async ev => {
  if (ev.target.tagName === "SELECT") return;
  const vis = visibleKeys(), key = vis[idx];
  switch (ev.key) {
    case "Enter": case "ArrowRight":
      await decide(assign[key] === bouts[key].original_male_fly ? "confirmed" : "swapped");
      break;
    case "s": case "S":
      assign[key] = 1 - assign[key];
      await decide(assign[key] === bouts[key].original_male_fly ? "confirmed" : "swapped");
      break;
    case "u": case "U": await decide("unsure"); break;
    case "ArrowLeft": idx -= 1; render(); break;
    case "j": case "J": {
      const next = vis.findIndex((k, i) => i > idx &&
        (bouts[k].status === "pending" || bouts[k].status === "unsure"));
      if (next >= 0) { idx = next; render(); }
      break;
    }
    case " ": ev.preventDefault();
      if (v0.paused) { v0.play(); v1.play(); } else { v0.pause(); v1.pause(); }
      break;
    case "r": case "R":
      v0.currentTime = 0; v1.currentTime = 0; v0.play(); v1.play(); break;
    case "1": v0.playbackRate = v1.playbackRate = 0.5; break;
    case "2": v0.playbackRate = v1.playbackRate = 2.0; break;
  }
});
// keep the two videos in sync (they are the same bout, same length)
v0.addEventListener("timeupdate", () => {
  if (Math.abs(v0.currentTime - v1.currentTime) > 0.15) v1.currentTime = v0.currentTime;
});
(async () => {
  const m = await (await fetch("/api/bouts")).json();
  bouts = m.bouts;
  keys = Object.keys(bouts).sort();
  keys.forEach(k => { assign[k] = bouts[k].reviewed_male_fly; });
  const recs = [...new Set(keys.map(k => k.split("/").slice(0, 2).join("/")))];
  for (const r of recs) {
    const o = document.createElement("option");
    o.value = r; o.textContent = r;
    $("filter").appendChild(o);
  }
  $("filter").addEventListener("change", () => { idx = 0; render(); });
  render();
})();
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="processed courtship root (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = build_manifest(root, load_manifest(root))
    save_manifest(root, manifest)
    n_bouts = len(manifest["bouts"])
    n_pending = sum(1 for e in manifest["bouts"].values() if e["status"] == "pending")
    server = ReviewServer(("127.0.0.1", args.port), root, manifest)
    port = server.server_address[1]
    print(f"{n_bouts} bouts ({n_pending} pending) under {root}")
    print(f"open   http://localhost:{port}")
    print(f"remote? ssh -L {port}:localhost:{port} <this-host>  (VS Code auto-forwards)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
```

Note: `PAGE_HTML` is a plain (non-f) string — the JS template literals `${...}` must NOT be escaped or reformatted into an f-string.

- [ ] **Step 4: Run the full test file**

Run: `pytest tests/test_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 5: Manual smoke test (synthetic tree)**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
SMOKE=$(mktemp -d)
python - "$SMOKE" <<'EOF'
import sys, shutil
from pathlib import Path
root = Path(sys.argv[1])
src = Path("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session1/2026_04_02_16_03_48/pose/bouts")
for bout in ["bout_00001", "bout_00002"]:
    for fly in ["fly0", "fly1"]:
        d = root / "Session1" / "recA" / "pose" / "bouts" / bout / fly
        d.mkdir(parents=True)
        shutil.copy(src / bout / fly / "sidebyside.mp4", d / "sidebyside.mp4")
EOF
python scripts/viz/fly_id_review.py --root "$SMOKE" --port 8642
```

Open http://localhost:8642 (port-forwarded). Verify: both videos autoplay in sync and the scrub bar seeks instantly; badges show F on fly0 / M on fly1; `S` flips badges and advances; `Enter` confirms; progress counts up; refresh keeps decisions. Ctrl-C the server, `rm -rf "$SMOKE"`.

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/fly_id_review.py tests/test_fly_id_review.py
git commit -m "feat(viz): fly ID review embedded UI + main entrypoint"
```

---

### Task 5: apply_fly_id_review.py — physical swap script

**Files:**
- Create: `scripts/viz/apply_fly_id_review.py`
- Test: `tests/test_apply_fly_id_review.py`

**Interfaces:**
- Consumes: `MANIFEST_NAME`, `bout_dir_from_key`, `load_manifest`, `save_manifest`, `read_sex_json` from `scripts.viz.fly_id_review`; the `make_bout` helper from `tests.test_fly_id_review`.
- Produces:
  - `validate(manifest: dict, allow_pending: bool) -> list[str]` — human-readable blockers.
  - `plan_swaps(manifest: dict, allow_pending: bool = False) -> list[str]` — bout keys needing a dir swap.
  - `swap_bout(root: Path, key: str, entry: dict) -> None` — dirs + sex.json for one bout.
  - `main(argv: list[str] | None = None) -> None` — `--root`, `--apply`, `--allow-pending`; dry-run by default; `SystemExit(1)` on blockers.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_apply_fly_id_review.py`:

```python
"""Tests for scripts/viz/apply_fly_id_review.py (physical fly0/fly1 swap)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.viz.apply_fly_id_review import main, plan_swaps, swap_bout, validate
from scripts.viz.fly_id_review import (
    bout_dir_from_key, build_manifest, load_manifest, read_sex_json, save_manifest,
)
from tests.test_fly_id_review import make_bout


def entry(status="confirmed", reviewed=1, original=1, applied=False, warning=None):
    return {"original_male_fly": original, "reviewed_male_fly": reviewed,
            "status": status, "source": "default",
            "reviewed_at": "2026-08-07T00:00:00+00:00",
            "applied": applied, "warning": warning}


def manifest_of(root, **bouts):
    return {"root": str(root), "convention": {"female": 0, "male": 1}, "bouts": bouts}


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_blocks_pending_and_unsure(tmp_path):
    m = manifest_of(tmp_path, a=entry("pending"), b=entry("unsure"), c=entry("confirmed"))
    errors = validate(m, allow_pending=False)
    assert len(errors) == 2
    errors = validate(m, allow_pending=True)
    assert len(errors) == 1 and "unsure" in errors[0]  # unsure ALWAYS blocks


# ---------------------------------------------------------------------------
# plan_swaps
# ---------------------------------------------------------------------------

def test_plan_swaps_selects_reviewed_male_fly0(tmp_path):
    m = manifest_of(
        tmp_path,
        swap_me=entry("swapped", reviewed=0),
        keep=entry("confirmed", reviewed=1),
        already=entry("swapped", reviewed=0, applied=True),
        broken=entry("swapped", reviewed=0, warning="missing: fly1/sidebyside.mp4"),
        pend=entry("pending", reviewed=0, original=0),
    )
    assert plan_swaps(m) == ["swap_me"]
    assert plan_swaps(m, allow_pending=True) == ["pend", "swap_me"]


# ---------------------------------------------------------------------------
# swap_bout
# ---------------------------------------------------------------------------

def test_swap_bout_exchanges_dirs_and_updates_sex_json(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    (bout_dir / "fly0" / "MARKER_A").touch()
    (bout_dir / "fly1" / "MARKER_B").touch()
    e = entry("swapped", reviewed=0, original=1)
    swap_bout(tmp_path, "Session1/recA/bout_00001", e)
    assert (bout_dir / "fly0" / "MARKER_B").exists()  # old fly1 now fly0
    assert (bout_dir / "fly1" / "MARKER_A").exists()  # old fly0 now fly1
    sex = read_sex_json(bout_dir)
    assert sex["male_fly"] == 1
    assert sex["applied_swap"] is True
    assert sex["original_male_fly"] == 1  # entry's original preserved verbatim
    assert not (bout_dir / ".fly_swap_tmp").exists()


def test_swap_bout_stale_tmp_marker_raises(tmp_path):
    bout_dir = make_bout(tmp_path, "Session1", "recA", "bout_00001")
    (bout_dir / ".fly_swap_tmp").mkdir()
    with pytest.raises(RuntimeError, match="fly_swap_tmp"):
        swap_bout(tmp_path, "Session1/recA/bout_00001", entry("swapped", reviewed=0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _reviewed_tree(tmp_path):
    """Two bouts: bout_00001 needs a swap, bout_00002 confirmed as-is."""
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    make_bout(tmp_path, "Session1", "recA", "bout_00002")
    m = build_manifest(tmp_path)
    m["bouts"]["Session1/recA/bout_00001"].update(
        {"status": "swapped", "reviewed_male_fly": 0,
         "reviewed_at": "2026-08-07T00:00:00+00:00"})
    m["bouts"]["Session1/recA/bout_00002"].update(
        {"status": "confirmed", "reviewed_at": "2026-08-07T00:00:00+00:00"})
    save_manifest(tmp_path, m)
    return m


def test_main_dry_run_changes_nothing(tmp_path, capsys):
    _reviewed_tree(tmp_path)
    (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").touch()
    main(["--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "dry-run" in out and "bout_00001" in out
    assert (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").exists()
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["applied"] is False


def test_main_apply_swaps_and_is_idempotent(tmp_path, capsys):
    _reviewed_tree(tmp_path)
    bout_dir = bout_dir_from_key(tmp_path, "Session1/recA/bout_00001")
    (bout_dir / "fly0" / "M0").touch()
    main(["--root", str(tmp_path), "--apply"])
    assert (bout_dir / "fly1" / "M0").exists()  # swapped
    m = load_manifest(tmp_path)
    e = m["bouts"]["Session1/recA/bout_00001"]
    assert e["applied"] is True and e["reviewed_male_fly"] == 1
    main(["--root", str(tmp_path), "--apply"])  # second run: no-op
    assert (bout_dir / "fly1" / "M0").exists()  # NOT swapped back


def test_main_refuses_on_pending(tmp_path, capsys):
    make_bout(tmp_path, "Session1", "recA", "bout_00001")
    save_manifest(tmp_path, build_manifest(tmp_path))
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--apply"])
    assert "REFUSING" in capsys.readouterr().out


def test_main_allow_pending_treats_pending_as_original(tmp_path):
    make_bout(tmp_path, "Session1", "recA", "bout_00001",
              sex_json={"male_fly": 0})  # original says male in fly0 -> needs swap
    (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly0" / "M0").touch()
    save_manifest(tmp_path, build_manifest(tmp_path))
    main(["--root", str(tmp_path), "--apply", "--allow-pending"])
    assert (bout_dir_from_key(tmp_path, "Session1/recA/bout_00001") / "fly1" / "M0").exists()


def test_main_missing_manifest_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path), "--apply"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_apply_fly_id_review.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'scripts.viz.apply_fly_id_review'`

- [ ] **Step 3: Write the implementation**

Create `scripts/viz/apply_fly_id_review.py`:

```python
"""Apply reviewed fly identities: physically swap fly0/fly1 bout dirs.

Reads <root>/id_review.json (written by scripts/viz/fly_id_review.py). For
every bout whose reviewed male currently lives in fly0/, swaps the two fly
dirs so the whole tree is uniformly fly0 = female, fly1 = male, and updates
the bout's sex.json (male_fly: 1, applied_swap: true).

DRY-RUN BY DEFAULT -- prints the full plan. Pass --apply to execute.
Refuses to run while any bout is pending or unsure; --allow-pending treats
pending bouts as confirmed-original (unsure always blocks).

Touches ONLY pose/bouts/bout_*/fly{0,1} dirs, sex.json, and the manifest.

Usage:
    python scripts/viz/apply_fly_id_review.py                 # dry-run
    python scripts/viz/apply_fly_id_review.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.viz.fly_id_review import (
    DEFAULT_ROOT,
    MANIFEST_NAME,
    bout_dir_from_key,
    load_manifest,
    read_sex_json,
    save_manifest,
)

TMP_NAME = ".fly_swap_tmp"


def validate(manifest: dict, allow_pending: bool) -> list[str]:
    """Blockers that must be resolved before --apply may run."""
    errors = []
    for key, e in sorted(manifest["bouts"].items()):
        if e["status"] == "unsure":
            errors.append(f"{key}: unsure (resolve in the GUI first)")
        elif e["status"] == "pending" and not allow_pending:
            errors.append(f"{key}: pending (review it, or pass --allow-pending)")
    return errors


def plan_swaps(manifest: dict, allow_pending: bool = False) -> list[str]:
    """Bout keys whose reviewed male currently lives in fly0/ (need a swap)."""
    ok_status = {"confirmed", "swapped"} | ({"pending"} if allow_pending else set())
    return [key for key, e in sorted(manifest["bouts"].items())
            if e["reviewed_male_fly"] == 0
            and not e.get("applied")
            and e["status"] in ok_status
            and not e.get("warning")]


def swap_bout(root: Path, key: str, entry: dict) -> None:
    """Swap fly0/ <-> fly1/ for one bout and rewrite its sex.json."""
    bout_dir = bout_dir_from_key(root, key)
    tmp = bout_dir / TMP_NAME
    if tmp.exists():
        raise RuntimeError(
            f"{key}: stale {TMP_NAME} from an interrupted swap -- inspect manually")
    (bout_dir / "fly0").rename(tmp)
    (bout_dir / "fly1").rename(bout_dir / "fly0")
    tmp.rename(bout_dir / "fly1")
    existing = read_sex_json(bout_dir) or {}
    sex = dict(existing)
    sex.update({
        "male_fly": 1,
        "original_male_fly": existing.get("original_male_fly",
                                          entry["original_male_fly"]),
        "applied_swap": True,
        "confidence": "user",
        "method": "manual-gui",
    })
    tmp_json = bout_dir / "sex.json.tmp"
    tmp_json.write_text(json.dumps(sex, indent=2))
    os.replace(tmp_json, bout_dir / "sex.json")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true",
                        help="execute the swaps (default: dry-run)")
    parser.add_argument("--allow-pending", action="store_true",
                        help="treat pending bouts as confirmed-original")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    manifest = load_manifest(root)
    if manifest is None:
        raise SystemExit(f"no {MANIFEST_NAME} under {root} -- run fly_id_review.py first")

    errors = validate(manifest, args.allow_pending)
    if errors:
        print(f"REFUSING to apply -- {len(errors)} unresolved bout(s):")
        for line in errors:
            print(f"  {line}")
        raise SystemExit(1)

    swaps = plan_swaps(manifest, args.allow_pending)
    warned = sorted(k for k, e in manifest["bouts"].items() if e.get("warning"))
    print(f"{len(manifest['bouts'])} bouts total; {len(swaps)} need a fly0<->fly1 swap; "
          f"{len(warned)} excluded (warnings)")
    for key in swaps:
        print(f"  swap: {key}")
    for key in warned:
        e = manifest["bouts"][key]
        flag = " (WOULD NEED SWAP -- fix manually)" if e["reviewed_male_fly"] == 0 else ""
        print(f"  excluded: {key} [{e['warning']}]{flag}")
    if not args.apply:
        print("dry-run only -- pass --apply to execute")
        return

    for key in swaps:
        entry = manifest["bouts"][key]
        swap_bout(root, key, entry)
        entry["applied"] = True
        entry["reviewed_male_fly"] = 1  # male now lives in fly1
        save_manifest(root, manifest)   # crash-safe: persist after every bout
        print(f"  swapped: {key}")
    print(f"done: {len(swaps)} bout(s) swapped; tree is now fly0=female, fly1=male")


if __name__ == "__main__":
    main()
```

Note: `original_male_fly` in both the manifest and `sex.json` is a **historical record of the pre-swap dir index** — it is deliberately not touched when dirs are swapped (matches the existing Session0 convention: `male_fly: 1, original_male_fly: 0, applied_swap: true`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_apply_fly_id_review.py tests/test_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/apply_fly_id_review.py tests/test_apply_fly_id_review.py
git commit -m "feat(viz): apply script for reviewed fly identities (dry-run default)"
```

---

### Task 6: Real-tree verification (read-only + dry-run)

**Files:** none created — verification commands only. Run these on Hyak in the `3d_tracking` env from the repo root.

**Interfaces:** consumes both scripts as a user would.

- [ ] **Step 1: Full test suite green**

Run: `pytest tests/test_fly_id_review.py tests/test_apply_fly_id_review.py -v`
Expected: all PASS

- [ ] **Step 2: Scan the real tree and check counts**

```bash
python - <<'EOF'
from pathlib import Path
from scripts.viz.fly_id_review import build_manifest
root = Path("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")
m = build_manifest(root)  # read-only: no save
by_session = {}
for key, e in m["bouts"].items():
    s = key.split("/")[0]
    by_session.setdefault(s, []).append(e)
for s, entries in sorted(by_session.items()):
    from_sex = sum(1 for e in entries if e["source"] == "sex.json")
    warned = sum(1 for e in entries if e["warning"])
    print(f"{s}: {len(entries)} bouts, {from_sex} from sex.json, {warned} warnings")
EOF
```

Expected: Session0 → 30 bouts, 30 from sex.json. Session1 → ~130 bouts, 0 from sex.json. Investigate any warnings (they should correspond to genuinely incomplete bouts).

- [ ] **Step 3: Serve the real tree, verify range requests with curl**

```bash
python scripts/viz/fly_id_review.py --port 8642 &
sleep 2
curl -s -D - -o /dev/null -H "Range: bytes=0-1023" \
  http://localhost:8642/media/Session1/2026_04_02_16_03_48/pose/bouts/bout_00001/fly0/sidebyside.mp4
curl -s -o /dev/null -w "%{http_code}\n" \
  "http://localhost:8642/media/../../../etc/passwd"
kill %1
```

Expected: first curl shows `HTTP/1.1 206`, `Content-Range: bytes 0-1023/...`, `Content-Length: 1024`; second prints `403` or `404`.

- [ ] **Step 4: Browser smoke on the real tree**

Start the server again, port-forward, open http://localhost:8642. Verify: videos play and **seek without lag** (scrub back and forth); advancing to the next bout is instant (prefetch); Session0 bouts show their sex.json-derived assignment; progress reads `0/160`-ish reviewed. Ctrl-C when done. The manifest now exists at the root — that's expected and correct.

- [ ] **Step 5: Dry-run the apply script**

```bash
python scripts/viz/apply_fly_id_review.py
```

Expected: `REFUSING to apply` listing every pending bout — proof the strict gate works on the real tree. (After the user finishes their real review, they run it again and then with `--apply`.)

- [ ] **Step 6: Commit any doc touch-ups and report**

If steps 2–5 surfaced surprises (bout counts, warnings), note them in the final report to the user. No code commit expected from this task.

---

## Self-Review (completed)

- **Spec coverage:** scan/merge (T1), decision + sex.json semantics (T2), 206 ranges + traversal guard + localhost bind (T3), UI with all 9 spec keys + badges + filter + progress + prefetch + derived confirmed/swapped status (T4), apply with dry-run/strict/idempotent/tmp-marker/warning-exclusion (T5), real-tree counts + curl range + seek check + dry-run gate (T6). Faststart re-encode: correctly absent (nothing to do).
- **Placeholder scan:** `PAGE_HTML` stub in T3 is explicitly replaced in T4 (declared in T3's Interfaces block) — not a dangling placeholder.
- **Type consistency:** `record_decision(root, manifest, bout_key, reviewed_male_fly, status)` matches T3's POST handler; `swap_bout(root, key, entry)` matches T5's main; `make_bout` signature identical between T1 definition and T5 import; manifest field names identical across all tasks.
