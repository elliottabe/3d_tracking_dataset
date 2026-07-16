#!/usr/bin/env python3
# Vendored from JohnsonLabJanelia/cluster_pose (private), harness/check_sync.py @ main
# (copied 2026-07-15). Detector + SyncPlan/SyncCam reader for dropped-frame realignment.
# Local additions below the original body: sync_inventory().
"""
check_sync.py — detect camera desync from per-frame PTP metadata and emit a "sync plan".

The rig hardware-triggers all cameras together; the JARVIS predictor assumes the i-th
decoded frame of every camera is the same trigger instant. A dropped frame in one
camera breaks that assumption for everything after it (silent 3D garbage). This tool
reads each camera's `CamXXXXXXX_meta.csv`, reconstructs a canonical trigger timeline,
locates dropped frames, and classifies the recording as:

    clean   — all cameras start together, no interior drops (byte-identical to today)
    trim    — no interior drops, but cameras start/stop a few frames apart
    reindex — one or more interior drops (needs per-camera re-alignment)

It emits a compact JSON plan the cluster-side predictor consumes to align cameras onto
the canonical timeline WITHOUT needing meta.csv (which is never staged to the cluster).

Frame-rate agnostic: the nominal inter-frame interval (`delta_ns`) is INFERRED from the
data per recording (robust median of inter-frame timestamp deltas), never hardcoded.
Use --fps / --delta-ns to override for short/noisy recordings.

meta.csv schema (verified): frame_id,timestamp,timestamp_sys,ptp_offset
    frame_id  — per-camera frame counter (hardware counter leaves holes on a drop;
                some cameras re-index it contiguously — diagnostic only)
    timestamp — PTP hardware clock in nanoseconds (the alignment key)
"""
import os, sys, csv, json, glob, argparse, statistics

PLAN_VERSION = 1
DROP_FACTOR  = 1.5        # a gap is an inter-frame delta >= DROP_FACTOR * delta_ns
META_GLOB    = "Cam*_meta.csv"


# ---------------------------------------------------------------------------
# Plan CONSUMER model (imported by predict3d_safe.py on the cluster; stdlib-only).
# ---------------------------------------------------------------------------
class SyncCam:
    """Per-camera view of the canonical PTP timeline. A camera is 'present' at canonical
    slot t unless t is before it started, after it ended, or inside an interior drop gap.
    pos(t) = mp4 frame position that delivers slot t (= count of present frames before t)."""
    def __init__(self, start_slot, decoded_len, gaps):
        self.start_slot = start_slot
        self.decoded_len = decoded_len
        self.gaps = sorted(gaps, key=lambda g: g["slot"])
        self.end_slot = start_slot + decoded_len + sum(g["lost"] for g in self.gaps)

    def has(self, t):
        if t < self.start_slot or t >= self.end_slot:
            return False
        for g in self.gaps:
            if g["slot"] <= t < g["slot"] + g["lost"]:
                return False
            if g["slot"] > t:
                break
        return True

    def pos(self, t):
        lost_before = sum(g["lost"] for g in self.gaps if g["slot"] < t)
        return (t - self.start_slot) - lost_before


class SyncPlan:
    """Canonical-timeline alignment plan for one recording (produced by analyze_recording)."""
    def __init__(self, d):
        self.delta_ns = d["delta_ns"]
        self.canonical_len = d["canonical_len"]
        self.predict_start = d["predict_start"]
        self.predict_len = d["predict_len"]
        self.status = d.get("status")
        self.first_drop_slot = d.get("first_drop_slot")
        self.cams = {name: SyncCam(c["start_slot"], c["decoded_len"], c["gaps"])
                     for name, c in d["cameras"].items()}

    @classmethod
    def load(cls, path):
        import json as _json
        with open(path) as f:
            return cls(_json.load(f))


def iter_meta(path):
    """Yield (frame_id:int, timestamp:int) per data row. Skips header + blank/short
    lines (some files lack a trailing newline / have a stray blank last line)."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)                       # header: frame_id,timestamp,...
        for row in r:
            if not row or len(row) < 2 or row[0] == "" or row[1] == "":
                continue
            yield int(row[0]), int(row[1])


def read_camera(path):
    """Single streaming pass. Returns raw per-camera facts; drops are classified later
    (once the recording-wide delta_ns is known)."""
    name = os.path.basename(path)[:-len("_meta.csv")]
    first_ts = last_ts = first_fid = last_fid = None
    n = 0
    deltas = []                             # inter-frame timestamp deltas (ns)
    prev = None
    for fid, ts in iter_meta(path):
        if first_ts is None:
            first_ts, first_fid = ts, fid
        else:
            deltas.append(ts - prev)
        prev = ts
        last_ts, last_fid = ts, fid
        n += 1
    if n == 0:
        raise ValueError(f"{name}: empty meta.csv")
    return dict(name=name, first_ts=first_ts, last_ts=last_ts,
                first_fid=first_fid, last_fid=last_fid, decoded_len=n, deltas=deltas)


def infer_delta_ns(cams):
    """Robust nominal inter-frame interval = median of each camera's median delta.
    Drops (<0.1% of frames) and jitter (~0.05%) don't move the median."""
    per_cam = [int(round(statistics.median(c["deltas"]))) for c in cams if c["deltas"]]
    if not per_cam:
        raise ValueError("no inter-frame deltas to infer frame rate from")
    return int(round(statistics.median(per_cam)))


def classify_camera(cam, delta_ns):
    """Find interior drops from the delta sequence and reduce to canonical-slot gaps.
    Mutates `cam` with: drops, true_span, frame_id_mode, and (later) start_slot/gaps."""
    thr = DROP_FACTOR * delta_ns
    drops = []
    cum_lost = 0
    for i, d in enumerate(cam["deltas"]):
        if d >= thr:
            lost = int(round(d / delta_ns)) - 1
            if lost > 0:
                pos = i + 1                          # mp4 position of the frame AFTER the gap
                # diagnostic frame_id at the gap (hole-mode counters skip by `lost`)
                fid_after = cam["first_fid"] + pos + cum_lost
                drops.append(dict(pos=pos, lost=lost, fid_after=fid_after))
                cum_lost += lost
    cam["drops"] = drops
    total_lost = sum(d["lost"] for d in drops)
    cam["total_lost"] = total_lost
    # Canonical span is EXACT from counts (drift-free): every present frame is one slot,
    # every dropped frame is one slot. Do NOT derive it from timestamps — a ~ppm clock
    # drift over ~500k frames accumulates a 1-2 slot error and breaks byte-identity.
    cam["true_span"] = cam["decoded_len"] + total_lost
    # frame_id contiguous over the decoded rows => camera re-indexed (holes repacked)
    cam["frame_id_mode"] = ("reindex"
                            if (cam["last_fid"] - cam["first_fid"] == cam["decoded_len"] - 1)
                            else "hole")
    # soft drift check: timestamp-implied span should match the count-based span to within
    # jitter+drift (tolerant, so normal ppm drift does NOT trip it — only gross corruption)
    ts_span = round((cam["last_ts"] - cam["first_ts"]) / delta_ns) + 1
    cam["span_consistent"] = abs(ts_span - cam["true_span"]) <= max(3, int(5e-4 * cam["true_span"]))
    return cam


def analyze_recording(rec_dir, delta_ns=None):
    """Analyze all cameras in `rec_dir` and return a plan dict (the sync plan)."""
    meta_paths = sorted(glob.glob(os.path.join(rec_dir, META_GLOB)))
    if not meta_paths:
        raise ValueError(f"no {META_GLOB} in {rec_dir}")
    cams = [read_camera(p) for p in meta_paths]

    if delta_ns is None:
        delta_ns = infer_delta_ns(cams)
    for c in cams:
        classify_camera(c, delta_ns)

    anchor = min(c["first_ts"] for c in cams)     # earliest trigger tick = canonical slot 0
    T = 0
    for c in cams:
        c["start_slot"] = int(round((c["first_ts"] - anchor) / delta_ns))
        c["end_slot"]   = c["start_slot"] + c["true_span"]
        T = max(T, c["end_slot"])
        # interior gaps in CANONICAL-slot coordinates (so the cluster needs no meta.csv)
        gaps, cum = [], 0
        for d in c["drops"]:
            g = c["start_slot"] + d["pos"] + cum
            gaps.append(dict(slot=g, lost=d["lost"]))
            cum += d["lost"]
        c["gaps"] = gaps

    any_drops = any(c["drops"] for c in cams)
    P0 = max(c["start_slot"] for c in cams)        # first slot where ALL cameras have started
    P1 = min(c["end_slot"]   for c in cams)        # first slot where SOME camera has ended
    starts_zero = all(c["start_slot"] == 0 for c in cams)
    ends_equal  = len(set(c["end_slot"] for c in cams)) == 1
    # clean = every camera spans the identical [0, T) with no interior gaps -> today's
    # positional read is already correct and byte-identical (no plan needed).
    if not any_drops and starts_zero and ends_equal:
        status = "clean"
    elif not any_drops:
        status = "trim"          # start/end skew only -> per-camera start vector, no re-index
    else:
        status = "reindex"       # interior drops -> full canonical re-alignment

    first_drop_slot = None
    gap_slots = [g["slot"] for c in cams for g in c["gaps"]]
    if gap_slots:
        first_drop_slot = min(gap_slots)

    plan = dict(
        version=PLAN_VERSION,
        recording=os.path.basename(os.path.normpath(rec_dir)),
        delta_ns=delta_ns,
        anchor_ts_ns=anchor,
        canonical_len=T,               # full extent (max end_slot); informational
        predict_start=P0,              # first canonical slot to predict (all cams present)
        predict_len=max(0, P1 - P0),   # number of slots to predict: window [P0, P1)
        status=status,
        first_drop_slot=first_drop_slot,
        cameras={c["name"]: dict(start_slot=c["start_slot"],
                                 decoded_len=c["decoded_len"],
                                 true_span=c["true_span"],
                                 frame_id_mode=c["frame_id_mode"],
                                 gaps=c["gaps"]) for c in cams},
    )
    # diagnostics kept out of the shipped plan but returned for callers/logging
    plan["_diag"] = dict(
        n_cams=len(cams),
        span_consistent=all(c["span_consistent"] for c in cams),
        total_lost={c["name"]: sum(d["lost"] for d in c["drops"]) for c in cams},
        drops={c["name"]: c["drops"] for c in cams if c["drops"]},
    )
    return plan


def verify_decoded(rec_dir, plan):
    """Cross-check each mp4's decoded frame count against the meta row count. Catches a
    corrupt/truncated mp4 (moov-atom failure) versus a genuine desync. Needs cv2."""
    import cv2
    out = {}
    for name, cam in plan["cameras"].items():
        mp4 = os.path.join(rec_dir, name + ".mp4")
        cap = cv2.VideoCapture(mp4)
        cvn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        cap.release()
        # CAP_PROP_FRAME_COUNT over-reports by a few; flag only large disagreement
        ok = cvn > 0 and abs(cvn - cam["decoded_len"]) <= max(5, int(0.001 * cam["decoded_len"]))
        out[name] = dict(cv2_count=cvn, meta_count=cam["decoded_len"], ok=ok)
    return out


def _plan_for_disk(plan):
    """Shipped plan = everything except the _diag block."""
    return {k: v for k, v in plan.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description="Detect camera desync and emit a sync plan.")
    ap.add_argument("rec_dir", help="recording folder with CamXXXX.mp4 + CamXXXX_meta.csv")
    ap.add_argument("--json", metavar="OUT", help="write the sync plan JSON to OUT")
    ap.add_argument("--verify-decoded", action="store_true",
                    help="cross-check mp4 decoded frame count vs meta (needs cv2)")
    ap.add_argument("--delta-ns", type=int, default=None,
                    help="override the inferred nominal inter-frame interval (ns)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override frame rate (fps); sets delta-ns = 1e9/fps")
    args = ap.parse_args()

    delta_ns = args.delta_ns
    if args.fps:
        delta_ns = int(round(1e9 / args.fps))

    plan = analyze_recording(args.rec_dir, delta_ns=delta_ns)
    diag = plan["_diag"]

    rate = 1e9 / plan["delta_ns"]
    print(f"{plan['recording']}: status={plan['status']}  "
          f"cams={diag['n_cams']}  fps~{rate:.1f}  "
          f"predict=[{plan['predict_start']},{plan['predict_start']+plan['predict_len']}) "
          f"len={plan['predict_len']}  first_drop_slot={plan['first_drop_slot']}")
    for name, cam in plan["cameras"].items():
        nd = len(cam["gaps"]); tl = diag["total_lost"][name]
        flag = "" if (cam["start_slot"] == 0 and nd == 0) else "  <-- desync"
        print(f"    {name}: start_slot={cam['start_slot']} decoded={cam['decoded_len']} "
              f"true_span={cam['true_span']} drops={nd} lost={tl} "
              f"mode={cam['frame_id_mode']}{flag}")
    if not diag["span_consistent"]:
        print("    WARNING: span != decoded+lost for some camera (timestamps suspect)")

    if args.verify_decoded:
        vd = verify_decoded(args.rec_dir, plan)
        for name, r in vd.items():
            if not r["ok"]:
                print(f"    VERIFY-DECODED MISMATCH {name}: cv2={r['cv2_count']} "
                      f"meta={r['meta_count']}")
        if all(r["ok"] for r in vd.values()):
            print("    verify-decoded: all cameras OK")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(_plan_for_disk(plan), f, separators=(",", ":"))
        print(f"    wrote plan -> {args.json}")

    # exit code: 0 clean/trim, 2 reindex, 3 inconsistent (for scripting/gating)
    if not diag["span_consistent"]:
        sys.exit(3)
    sys.exit(2 if plan["status"] == "reindex" else 0)


def sync_inventory(tree_dir):
    """Classify every recording (dir containing Cam*_meta.csv) under tree_dir.
    Returns a list of {recording, status, predict_len, first_drop_slot, n_cams}."""
    rows = []
    for root, _dirs, files in os.walk(tree_dir):
        if any(f.startswith("Cam") and f.endswith("_meta.csv") for f in files):
            try:
                plan = analyze_recording(root)
                rows.append(dict(recording=root, status=plan["status"],
                                 predict_len=plan["predict_len"],
                                 first_drop_slot=plan["first_drop_slot"],
                                 n_cams=plan["_diag"]["n_cams"]))
            except Exception as e:  # noqa: BLE001 -- inventory is best-effort
                rows.append(dict(recording=root, status=f"error:{type(e).__name__}",
                                 predict_len=0, first_drop_slot=None, n_cams=0))
    return rows


if __name__ == "__main__":
    main()
