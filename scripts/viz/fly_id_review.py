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
# Which pose tree to review. Set once by main() from --pose-dir. A non-default
# tree gets its own manifest (id_review_<dir>.json) so reviewing a rerun cannot
# clobber the decisions recorded against the previous one -- those 159 earlier
# decisions were only recoverable because they lived in a separate file.
POSE_DIR = "pose"
VIDEO_NAME = "sidebyside.mp4"
DEFAULT_ROOT = Path("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship")


# ---------------------------------------------------------------------------
# Scan + manifest
# ---------------------------------------------------------------------------

def bout_dir_from_key(root: Path, key: str) -> Path:
    """'Session1/recA/bout_00001' -> <root>/Session1/recA/<POSE_DIR>/bouts/bout_00001."""
    session, rec, bout = key.split("/")
    return root / session / rec / POSE_DIR / "bouts" / bout


def scan_bouts(root: Path) -> dict[str, dict]:
    """Find bout dirs under <root>/Session*/<rec>/<POSE_DIR>/bouts/bout_*.

    A bout missing a fly dir or its video gets a warning (shown in the UI,
    excluded from apply).
    """
    bouts: dict[str, dict] = {}
    for bout_dir in sorted(root.glob(f"Session*/*/{POSE_DIR}/bouts/bout_*")):
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


def _manifest_name() -> str:
    """id_review.json for the default tree, id_review_<dir>.json otherwise."""
    return MANIFEST_NAME if POSE_DIR == "pose" else f"id_review_{POSE_DIR}.json"


def save_manifest(root: Path, manifest: dict) -> None:
    path = root / _manifest_name()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)


def load_manifest(root: Path) -> dict | None:
    path = root / _manifest_name()
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    # A non-empty file without "bouts" is a SCHEMA MISMATCH, not an empty
    # manifest. Returning it anyway made build_manifest see zero prior
    # decisions and save_manifest then overwrite the file with 160 pending
    # entries -- silently discarding a whole review pass. Refuse instead.
    if isinstance(data, dict) and "bouts" not in data and data:
        raise SystemExit(
            f"{path} has no 'bouts' key -- refusing to overwrite it.\n"
            f"It looks like a bare {{bout_key: entry}} mapping. Wrap it as\n"
            f"  {{'root': ..., 'convention': {{'female': 0, 'male': 1}}, 'bouts': {{...}}}}\n"
            f"or move it aside first.")
    return data


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def record_decision(root: Path, manifest: dict, bout_key: str,
                    reviewed_male_fly: int, status: str) -> dict:
    """Record one review decision: update manifest (atomic) + bout sex.json.

    sex.json is only written for confirmed/swapped — an 'unsure' bout has no
    trustworthy identity to record, and a 'bad' one has no trustworthy TRACKING
    at all.

    'bad' is a quality verdict, not an identity one: the reviewer is saying the
    pose in this bout is not usable (typically the female is mistracked), which
    is a different judgement from "I cannot tell which fly is male". Recording
    it here means the exclusion travels with the data instead of living in
    someone's notes -- combine_ik_outputs and the reconstructability gate can
    both honour it.
    """
    if bout_key not in manifest["bouts"]:
        raise KeyError(bout_key)
    if status not in ("confirmed", "swapped", "unsure", "bad"):
        raise ValueError(f"bad status: {status}")
    if reviewed_male_fly not in (0, 1):
        raise ValueError(f"bad reviewed_male_fly: {reviewed_male_fly}")
    entry = manifest["bouts"][bout_key]
    entry["reviewed_male_fly"] = reviewed_male_fly
    entry["status"] = status
    entry["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Physical state lives in sex.json's applied_swap; resetting this flag lets
    # a post-apply re-review be re-selected by plan_swaps (swap_bout's
    # male_fly==1 recovery guard keeps a no-op re-confirm safe either way).
    entry["applied"] = False
    save_manifest(root, manifest)
    if status not in ("unsure", "bad"):
        _write_sex_json(bout_dir_from_key(root, bout_key), entry)
    return entry


def _write_sex_json(bout_dir: Path, entry: dict) -> None:
    existing = read_sex_json(bout_dir) or {}
    sex = dict(existing)
    sex.update({
        "male_fly": entry["reviewed_male_fly"],
        "original_male_fly": existing.get("original_male_fly", entry["original_male_fly"]),
        "applied_swap": existing.get("applied_swap", False),
        "confidence": "user",
        "method": "manual-gui",
        "note": f"fly_id_review {entry['reviewed_at']} status={entry['status']}",
    })
    tmp = bout_dir / "sex.json.tmp"
    tmp.write_text(json.dumps(sex, indent=2))
    os.replace(tmp, bout_dir / "sex.json")


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
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", page_html().encode())
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
        try:
            length = int(self.headers.get("Content-Length", 0))
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
            self.send_header("Cache-Control", "no-cache")
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


PAGE_HTML_TMPL = """<!doctype html>
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
  .st-bad { color:#b04ad0; font-weight:600; }
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
 <kbd>U</kbd> unsure &middot; <kbd>B</kbd> bad tracking &middot; <kbd>&larr;</kbd> back &middot;
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
  return `/media/${s}/${r}/${POSE_DIR}/bouts/${b}/fly${fly}/sidebyside.mp4`;
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
    case "b": case "B": await decide("bad"); break;
    case "ArrowLeft": idx -= 1; render(); break;
    case "j": case "J": {
      const next = vis.findIndex((k, i) => i > idx &&
        (bouts[k].status === "pending" || bouts[k].status === "unsure"));
      // 'bad' is a decision, so J skips it like confirmed/swapped.
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

def page_html() -> str:
    """Inject the active pose dir into the page (the JS builds media URLs)."""
    return PAGE_HTML_TMPL.replace("${POSE_DIR}", POSE_DIR)


def main(argv=None):
    global POSE_DIR
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="processed courtship root (default: %(default)s)")
    parser.add_argument("--pose-dir", default="pose",
                        help="pose tree under each recording (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args(argv)
    POSE_DIR = args.pose_dir
    root = args.root.resolve()
    manifest = build_manifest(root, load_manifest(root))
    save_manifest(root, manifest)
    n_bouts = len(manifest["bouts"])
    n_pending = sum(1 for e in manifest["bouts"].values() if e["status"] == "pending")
    server = ReviewServer(("127.0.0.1", args.port), root, manifest)
    port = server.server_address[1]
    print(f"{n_bouts} bouts ({n_pending} pending) under {root}/*/*/{POSE_DIR}")
    print(f"manifest: {root / _manifest_name()}")
    print(f"open   http://localhost:{port}")
    print(f"remote? ssh -L {port}:localhost:{port} <this-host>  (VS Code auto-forwards)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
