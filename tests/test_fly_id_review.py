"""Tests for scripts/viz/fly_id_review.py (fly identity review server)."""
from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.viz.fly_id_review import (
    DEFAULT_MALE_FLY,
    MANIFEST_NAME,
    VIDEO_NAME,
    ReviewServer,
    bout_dir_from_key,
    build_manifest,
    load_manifest,
    parse_range,
    read_sex_json,
    record_decision,
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
                                   "confidence": "user", "note": "old",
                                   "montage": "old-info"})
    record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "confirmed")
    sex = read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001"))
    assert sex["original_male_fly"] == 0   # history kept
    assert sex["applied_swap"] is True     # history kept
    assert sex["montage"] == "old-info"    # unknown field survives (not dropped)
    assert sex["note"] != "old"            # note IS a fresh decision, overwritten


def test_record_unsure_skips_sex_json(tmp_path):
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "unsure")
    assert read_sex_json(bout_dir_from_key(tmp_path, "Session1/recA/bout_00001")) is None
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["status"] == "unsure"


def test_record_decision_resets_applied_flag(tmp_path):
    """A re-review after apply must clear 'applied' so plan_swaps reconsiders it."""
    m = _fresh(tmp_path)
    m["bouts"]["Session1/recA/bout_00001"]["applied"] = True  # simulate post-apply
    e = record_decision(tmp_path, m, "Session1/recA/bout_00001", 0, "swapped")
    assert e["applied"] is False
    assert load_manifest(tmp_path)["bouts"]["Session1/recA/bout_00001"]["applied"] is False


def test_record_rejects_bad_input(tmp_path):
    m = _fresh(tmp_path)
    with pytest.raises(KeyError):
        record_decision(tmp_path, m, "Session9/nope/bout_99999", 1, "confirmed")
    with pytest.raises(ValueError):
        record_decision(tmp_path, m, "Session1/recA/bout_00001", 2, "confirmed")
    with pytest.raises(ValueError):
        record_decision(tmp_path, m, "Session1/recA/bout_00001", 1, "pending")


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
def server(tmp_path, monkeypatch):
    root = tmp_path / "data"
    bout_dir = make_bout(root, "Session1", "recA", "bout_00001")
    (bout_dir / "fly0" / VIDEO_NAME).write_bytes(MEDIA_BYTES)
    manifest = build_manifest(root)
    save_manifest(root, manifest)
    srv = ReviewServer(("127.0.0.1", 0), root, manifest)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    host, port = srv.server_address
    # Disable proxy for localhost connections via environment (urllib reads per-request)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
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
        assert resp.headers["Cache-Control"] == "no-cache"  # apply can swap dirs mid-session
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


def test_post_decision_malformed_content_length_400(server):
    url, srv, root = server
    host, port = srv.server_address
    # Use raw http.client to send non-numeric Content-Length (urllib would correct it)
    conn = http.client.HTTPConnection(host, port, timeout=2)
    payload = b'{"bout_key": "Session1/recA/bout_00001"}'
    conn.putrequest("POST", "/api/decision")
    conn.putheader("Content-Length", "abc")  # invalid numeric value
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    conn.send(payload)
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()
    # Verify server thread still alive by making another request
    with urllib.request.urlopen(url + "/api/bouts") as resp2:
        assert resp2.status == 200


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


# ---------------------------------------------------------------------------
# 'bad' quality verdict
# ---------------------------------------------------------------------------

KEY = "Session1/recA/bout_00001"


def test_bad_status_is_accepted(tmp_path):
    """'bad' = the TRACKING is unusable, distinct from 'unsure' = cannot tell
    which fly is male. A review pass found many female flies still mistracked
    and had no way to record it."""
    m = _fresh(tmp_path)
    e = record_decision(tmp_path, m, KEY, 1, "bad")
    assert e["status"] == "bad"


def test_bad_does_not_write_sex_json(tmp_path):
    # Like 'unsure': a bout whose tracking is unusable has no trustworthy
    # identity to persist downstream.
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, KEY, 1, "bad")
    assert read_sex_json(bout_dir_from_key(tmp_path, KEY)) is None


def test_bad_survives_a_manifest_round_trip(tmp_path):
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, KEY, 0, "bad")
    assert load_manifest(tmp_path)["bouts"][KEY]["status"] == "bad"


def test_unsure_and_bad_are_distinct_verdicts(tmp_path):
    m = _fresh(tmp_path)
    record_decision(tmp_path, m, KEY, 1, "unsure")
    assert load_manifest(tmp_path)["bouts"][KEY]["status"] == "unsure"
    record_decision(tmp_path, m, KEY, 1, "bad")
    assert load_manifest(tmp_path)["bouts"][KEY]["status"] == "bad"


def test_unknown_status_still_rejected(tmp_path):
    m = _fresh(tmp_path)
    with pytest.raises(ValueError, match="bad status"):
        record_decision(tmp_path, m, KEY, 1, "terrible")
