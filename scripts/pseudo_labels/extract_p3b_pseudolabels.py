"""Extract P3b campaign pseudo-labels into a v12-format export (spec 2026-09-05 §3.2).

Run (the courtship campaign; the two sides are sized separately -- see ruling
#2 below -- so there is no single "target" for a two-sex pool):
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
        --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_p3b_20260905 \
        --male-n 4000

and the SINGLE-FLY route (spec §3.4, free-running females written by
`scripts/pseudo_labels/singlefly_p3b_pass.py`), where the pool has one host
sex, no masks and therefore no containment gate and no identity gate:
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
        --runs OutFiles/v2_singlefly/*/ --num-animals 1 --no-identity-gate \
        --target 2000 --min-female 0 \
        --out /gscratch/portia/eabe/data/Johnson_lab/red_data_3d_v12_pseudo_singlefly_20260905

Two phases, and the first one can stop the second:

1. **Census** over EVERY campaign bout (no sampling, no frames decoded): how
   many admitted anchors each stratum (host sex x contact/apart x recording)
   can supply. Written to `<out>/census.json` and printed. Controller ruling
   2026-09-05: if the female-host stratum can supply fewer than
   `--min-female` (3,000) anchors the run STOPS here rather than quietly
   rebalancing -- a female-poor pseudo set would train exactly the cohort
   this whole effort exists to fix on the male fly's data.
2. **Per-side draw + export** (rulings 2026-09-05): the FEMALE side takes
   every admissible anchor (`--female-n` unset), the MALE side is capped
   (`--male-n 4000`) and drawn round-robin over recordings with a hard
   per-recording share (`--per-rec-frac 0.20`) and per-bout cap
   (`--per-bout-cap 200`), keeping >= 25 % contact and >= 25 % apart where the
   yield allows. The export is therefore deliberately NOT 50/50; the manifest's
   `balance` block carries `female_host_weight = male_n / female_n`, which
   restores 50/50 at training time. Wall frames enter at their share of the
   pool (neither boosted nor suppressed). Then the T=2 partners of every
   picked anchor, then `write_pseudo_export`, then a loader round trip on the
   real export.

**`contact` is the minimum inter-fly KEYPOINT distance** (`--contact-kp-units`,
0.5 mm), NOT the spec's centroid separation: measured over the whole campaign
on 2026-09-05, no frame anywhere has two fly centroids within 15 units (a fly
is 2.5-3 mm long), while 33 % of frames have a keypoint pair within 5 units
and the admitted anchors' distribution matches all frames'. Both quantities
are recorded per frameset (`stratum.kp_dist_units`, `stratum.sep_units`) and
both cell censuses are in `census.json`.

EXPECTATION (state it before looking, CLAUDE.md): a pseudo export whose gate
histograms show every quantity well inside its threshold -- exist mass above
0.9, step p99 below 3 units, reproj median below 1.5 px, containment above
0.95 -- and whose sampling report hits the two side sizes exactly, >= 25 %
contact, >= 25 % apart, all 11 recordings, no bout above 2 %. A histogram piled
against a threshold means the GATE, not the data, is choosing the set: the
admitted frames would then be the ones that happen to sit at the cut, and
tightening the gate by a hair would change the whole training set.

Everything crossing a name boundary here is looked up BY NAME: keypoints
(campaign `keypoint_names_written` -> the export's `keypoint_names.json`),
cameras (`load_bout_arrays` verifies the kp2d camera axis against the
calibration's sorted-glob order, and the export writes `<rec>/<cam>/...`),
and flies (`mvq_meta.json`'s `fly_sex`, which comes from the human mask ID
review, never a slot index).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "third_party", "jarvis_jax"))

from jarvis_jax.data.pseudo_export import stratum_cell  # noqa: E402  (needs the path insert above)

DEFAULT_ROOTS = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/*/*/pose_mvq_p3b"
DEFAULT_EXPORT_NAMES = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"
TARGET = 10000
HIST_EDGES = {"exist": (0.0, 1.0), "step_units": (0.0, 10.0),
              "reproj_px": (0.0, 10.0), "contain_frac": (0.0, 1.0)}
N_BINS = 50


# ----------------------------------------------------------------- discovery
@dataclasses.dataclass
class BoutRef:
    recording: str
    session: str
    run_root: str
    bout_dir: str
    bout: int
    masks_npz: str
    session_dir: str
    calib_dir: str
    frame_start: int
    n_frames: int
    fly_sex: dict
    cameras: list
    # provenance the export repeats per annotation and per recording. Defaults
    # are the masked campaign's; the single-fly route (spec §3.4, free-running
    # females, `singlefly_p3b_pass.py`) writes its own, and claiming those
    # frames are courtship pairs whose identity came from a mask review would
    # be false in the manifest AND in every annotation.
    behavior: str = "courtship"
    recording_sex: str = "mixed"
    sex_source: str = "mvq_p3b_mask_identity"


def discover_bouts(run_globs, bout_glob="bout_*"):
    """Every `<run_root>/bouts/<bout>/` with its `mvq_meta.json` resolved.

    The mask npz is taken from the meta's own `masks_npz` (what the lift
    actually read) and falls back to the sibling `sam3_masks/<bout>/` layout;
    the calibration is the recording's own `<session_dir>/calibration`.
    """
    out = []
    for g in run_globs:
        for run_root in sorted(glob.glob(g)):
            rec_dir = os.path.dirname(run_root)
            rec = os.path.basename(rec_dir)
            session = os.path.basename(os.path.dirname(rec_dir))
            for bd in sorted(glob.glob(os.path.join(run_root, "bouts", bout_glob))):
                mp = os.path.join(bd, "mvq_meta.json")
                if not os.path.exists(mp):
                    continue
                m = json.load(open(mp))
                name = os.path.basename(bd)
                masks = m.get("masks_npz") or os.path.join(rec_dir, "sam3_masks", name,
                                                           "sam3_masks.npz")
                sd = m.get("session_dir", "")
                out.append(BoutRef(
                    recording=rec, session=session, run_root=run_root, bout_dir=bd,
                    bout=int(m.get("bout", int(name.split("_")[-1]))), masks_npz=masks,
                    session_dir=sd, calib_dir=os.path.join(sd, "calibration"),
                    frame_start=int(m.get("frame_start", 0)), n_frames=int(m["n_frames"]),
                    fly_sex=dict(m.get("fly_sex") or {}),
                    cameras=[str(c) for c in m["cameras"]],
                    behavior=str(m.get("behavior", "courtship")),
                    recording_sex=str(m.get("recording_sex", "mixed")),
                    sex_source=str(m.get("sex_source", "mvq_p3b_mask_identity"))))
    return out


# ------------------------------------------------------------------ scanning
@dataclasses.dataclass
class Row:
    recording: str
    bout: int
    t: int
    frame: int
    host_fly: int
    host_sex: str
    sep_units: float | None       # inter-fly CENTROID separation (kept for the record)
    kp_dist_units: float | None   # MINIMUM inter-fly keypoint distance -- what `contact` means
    contact: bool
    apart: bool
    wall: bool
    height_units: float
    centroid: tuple
    gates: dict
    partners: dict           # delta -> absolute partner frame
    role: str = "anchor"
    contact_centroid: bool = False   # the spec's original (unreachable) definition


def _hist(x, name):
    lo, hi = HIST_EDGES[name]
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    return np.histogram(np.clip(x, lo, hi), bins=N_BINS, range=(lo, hi))[0]


def scan_bout(ref, thr, *, use_identity=True, contact_units=15.0, contact_kp_units=5.0,
              n_flies=2, use_masks=True):
    """Gate ONE bout; returns (rows, per-gate histogram counts, reject counts).

    Rows are per (anchor frame, admitted fly): a v12 frameset -- and therefore
    a training window -- is per fly, so the host-sex stratification is over
    (frame, fly) pairs, not frames.

    `use_masks=False` is the single-fly route (spec §3.4): a free-running
    recording has no SAM3 masks at all, so the containment gate cannot run and
    `admit_bout` is passed `store=None`. It is an explicit argument, never
    inferred from a missing file: a courtship bout whose mask npz has gone
    astray must FAIL loudly, not quietly lose its containment gate.
    """
    from jarvis_jax.data.pseudo_gates import admit_bout, load_bout_arrays
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore

    rt = ReprojectionTool(ref.calib_dir)
    cameras = list(rt.cameras.keys())
    a = load_bout_arrays(ref.bout_dir, cameras=cameras, n_flies=n_flies)
    store = BoutMaskStore(ref.masks_npz, cameras) if use_masks else None
    r = admit_bout(a, store, rt.camera_matrices, thr, use_identity=use_identity)

    with np.errstate(invalid="ignore"):
        cent = np.nanmean(a.kp3d, axis=2)                     # (F,T,3) fly centroid
        if a.kp3d.shape[0] == 2:
            sep = np.linalg.norm(cent[0] - cent[1], axis=-1)
            # `contact` is the MINIMUM inter-fly KEYPOINT distance, not the
            # centroid separation: measured 2026-09-05 over the whole
            # campaign, NO frame has centroids within 15 units (a fly is
            # 2.5-3 mm long), while 33 % of frames have a keypoint pair
            # within 5 units -- the flies really are touching there. The
            # centroid figure is still recorded as `sep_units`.
            d = np.linalg.norm(a.kp3d[0][:, :, None, :] - a.kp3d[1][:, None, :, :], axis=-1)
            kdist = np.nanmin(d.reshape(d.shape[0], -1), axis=1)
        else:
            sep = np.full(a.kp3d.shape[1], np.nan)
            kdist = np.full(a.kp3d.shape[1], np.nan)
        reproj = np.nanmedian(r.quant["reproj_px"], axis=-1)  # (F,T) median over views

    def _row(f, t, partners, role):
        s = float(sep[t]) if np.isfinite(sep[t]) else None
        k = float(kdist[t]) if np.isfinite(kdist[t]) else None
        return Row(
            recording=ref.recording, bout=ref.bout, t=int(t), frame=ref.frame_start + int(t),
            host_fly=int(f), host_sex=ref.fly_sex.get(f"fly{f}", "unknown"),
            sep_units=s, kp_dist_units=k,
            contact=(k is not None and k < contact_kp_units),
            apart=(k is not None and k >= contact_kp_units),
            contact_centroid=(s is not None and s < contact_units),
            wall=False, height_units=float("nan"),
            centroid=tuple(float(v) for v in cent[f, t]),
            gates={"exist": float(r.quant["exist"][f, t]),
                   "step_units": float(r.quant["step_units"][f, t]),
                   "reproj_px": float(reproj[f, t]),
                   "contain_frac": float(r.quant["contain_frac"][f, t])},
            partners=partners, role=role)

    rows, partner_rows = [], {}
    for t in np.flatnonzero(r.frame):
        t = int(t)
        for f in range(a.kp3d.shape[0]):
            if not r.fly[f, t]:
                continue
            partners = {}
            for d in thr.deltas:
                d = int(d)
                # `partners[d]` gates the FRAME PAIR (Task 1's endpoint
                # semantics); `r.fly[f, tp]` gates the pose we can actually
                # write for THIS host fly at the partner frame -- a frameset
                # is per fly, so a partner whose host fly was rejected there
                # is not a usable pair for this row.
                for tp, ok in ((t + d, bool(r.partners[d][t]) if t < len(r.partners[d]) else False),
                               (t - d, bool(r.partners[d][t - d]) if t - d >= 0 else False)):
                    if ok and 0 <= tp < a.kp3d.shape[1] and r.fly[f, tp]:
                        partners[d] = ref.frame_start + tp
                        # The partner frameset carries the partner frame's OWN
                        # gate quantities and stratum, never the anchor's.
                        partner_rows.setdefault((f, tp), _row(f, tp, {}, "partner"))
                        break
            rows.append(_row(f, t, partners, "anchor"))

    hists = {"exist": _hist(r.quant["exist"], "exist"),
             "step_units": _hist(r.quant["step_units"], "step_units"),
             "reproj_px": _hist(reproj, "reproj_px"),
             "contain_frac": _hist(r.quant["contain_frac"], "contain_frac")}
    n = int(r.fly.size)
    rej = {k: int(np.asarray(v).sum()) for k, v in r.reasons.items()}
    rej["_n"] = n
    rej["_n_frames"] = int(a.kp3d.shape[1])
    rej["_n_anchors"] = int(r.frame.sum())
    rej["_n_admitted_fly"] = int(r.fly.sum())
    del store
    return rows, list(partner_rows.values()), hists, rej


# ------------------------------------------------------------------- strata
def add_wall_flags(rows, wall_height_units, *, also=()):
    """Fill `wall`/`height_units` per recording from a floor plane fitted to
    that recording's OWN admitted ANCHOR centroids (`coarse_track.fit_floor`).

    `also` (the T=2 partner rows) gets the same plane applied but does not
    enter the fit: the plane is a property of the arena, and partners are a
    dense, anchor-correlated resample of the same frames."""
    from jarvis_jax.tracking.coarse_track import fit_floor
    planes = {}
    by_rec = collections.defaultdict(list)
    extra = collections.defaultdict(list)
    for r in rows:
        by_rec[r.recording].append(r)
    for r in also:
        extra[r.recording].append(r)
    for rec, rr in by_rec.items():
        pts = np.array([r.centroid for r in rr], np.float64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plane = fit_floor(pts, n_fit=len(pts))
        except ValueError as e:
            print(f"[extract] {rec}: floor fit failed ({e}); wall flags left False", flush=True)
            continue
        for group in (rr, extra.get(rec, [])):
            if not group:
                continue
            hh = plane.height(np.array([r.centroid for r in group], np.float64))
            for r, hv in zip(group, hh):
                r.height_units = float(hv)
                r.wall = bool(hv > wall_height_units)
        h = plane.height(pts)
        planes[rec] = dict(plane.as_dict(), n_points=int(len(pts)),
                           wall_frac=float(np.mean(h > wall_height_units)))
    return planes


def _cell(r):
    return stratum_cell({"contact": r.contact, "apart": r.apart})


def _cell_centroid(r):
    """The spec's ORIGINAL contact definition (centroid separation), kept in
    the census for the record: it admits nothing anywhere in the campaign."""
    if r.sep_units is None:
        return "mid"
    return "contact" if r.contact_centroid else "apart"


def census(rows):
    """Admitted anchors per (host sex, contact/apart/mid, recording) + totals."""
    cell = _cell
    per = collections.Counter((r.host_sex, cell(r), r.recording) for r in rows)
    by_sex = collections.Counter(r.host_sex for r in rows)
    by_sex_rec = collections.Counter((r.host_sex, r.recording) for r in rows)
    by_sex_cell = collections.Counter((r.host_sex, cell(r)) for r in rows)
    by_bout = collections.Counter((r.recording, r.bout) for r in rows)
    by_rec_wall = collections.Counter(r.recording for r in rows if r.wall)
    return {
        "n_anchor_flysets": len(rows),
        "by_host_sex": dict(by_sex),
        "by_host_sex_recording": {f"{s}/{rec}": n for (s, rec), n in sorted(by_sex_rec.items())},
        "by_host_sex_cell": {f"{s}/{c}": n for (s, c), n in sorted(by_sex_cell.items())},
        "by_stratum": {f"{s}/{c}/{rec}": n for (s, c, rec), n in sorted(per.items())},
        # both contact definitions, per the 2026-09-05 ruling: `contact` above
        # is the min inter-fly KEYPOINT distance; `*_centroid` is the spec's
        # original centroid rule, recorded to show it admits nothing.
        "by_host_sex_cell_centroid": {
            f"{s}/{c}": n for (s, c), n in sorted(collections.Counter(
                (r.host_sex, _cell_centroid(r)) for r in rows).items())},
        "by_stratum_centroid": {
            f"{s}/{c}/{rec}": n for (s, c, rec), n in sorted(collections.Counter(
                (r.host_sex, _cell_centroid(r), r.recording) for r in rows).items())},
        "by_bout": {f"{rec}/bout_{b:05d}": n for (rec, b), n in sorted(by_bout.items())},
        "wall_by_recording": dict(by_rec_wall),
        "recordings": sorted({r.recording for r in rows}),
        "partner_availability": {str(d): int(sum(1 for r in rows if d in r.partners))
                                 for d in sorted({d for r in rows for d in r.partners})},
    }


def print_census(cen, min_female):
    print("\n=== CENSUS: admitted anchors per stratum (host sex x cell x recording) ===")
    print(f"{'recording':<24}{'F/contact':>10}{'F/apart':>9}{'F/mid':>8}"
          f"{'M/contact':>11}{'M/apart':>9}{'M/mid':>8}{'total':>8}")
    g = cen["by_stratum"]
    for rec in cen["recordings"]:
        v = [g.get(f"{s}/{c}/{rec}", 0) for s in ("female", "male") for c in ("contact", "apart", "mid")]
        print(f"{rec:<24}{v[0]:>10}{v[1]:>9}{v[2]:>8}{v[3]:>11}{v[4]:>9}{v[5]:>8}{sum(v):>8}")
    f = cen["by_host_sex"].get("female", 0)
    m = cen["by_host_sex"].get("male", 0)
    print(f"{'TOTAL':<24}{'':>10}{'':>9}{'':>8}{'':>11}{'':>9}{'':>8}{f + m:>8}")
    print(f"female-host anchors: {f}   male-host anchors: {m}   (min-female gate {min_female})")
    print("=" * 88, flush=True)


# ------------------------------------------------------------------ sampling
class QuotaShortfall(RuntimeError):
    pass


def round_robin_draw(cand, n, rng, *, key, cap_by, cap, taken, bout_count,
                     rec_count=None, rec_cap=None, strict=True):
    """Draw `n` rows, cycling recordings so every one appears, refusing a bout
    already at `cap`, a recording already at `rec_cap`, and any row already
    `taken`. Raises `QuotaShortfall` when the pool cannot fill the quota
    (never silently under-fills)."""
    pools = collections.defaultdict(list)
    for r in cand:
        if id(r) not in taken:
            pools[key(r)].append(r)
    for k in pools:
        rng.shuffle(pools[k])
    order = sorted(pools)
    picked = []
    while len(picked) < n and order:
        nxt = []
        for k in order:
            if len(picked) >= n:
                nxt.append(k)
                continue
            pool = pools[k]
            while pool:
                r = pool.pop()
                if id(r) in taken:
                    continue
                if bout_count[cap_by(r)] >= cap:
                    continue
                if rec_cap is not None and rec_count[r.recording] >= rec_cap:
                    pool.clear()
                    break
                taken.add(id(r))
                bout_count[cap_by(r)] += 1
                if rec_count is not None:
                    rec_count[r.recording] += 1
                picked.append(r)
                break
            if pool:
                nxt.append(k)
        if not nxt:
            break
        order = nxt
    if len(picked) < n and strict:
        raise QuotaShortfall(f"asked for {n}, pool gave {len(picked)}")
    return picked


def draw_side(rows, n, *, rng, per_rec_frac=0.20, per_bout_cap=200,
              contact_frac=0.25, apart_frac=0.25, taken=None, bout_count=None):
    """One host-sex side (ruling 2026-09-05 #2).

    `n is None` (or >= the pool) takes EVERY admissible row of the side, caps
    and all: that is the female side, where the whole point is to use all
    1,905 anchors. Otherwise the side is drawn round-robin over recordings
    with a hard per-recording cap (`per_rec_frac` of the side) and a per-bout
    cap, filling a >= `contact_frac` / >= `apart_frac` floor first and the
    rest freely. A cell that its pool cannot fill is reported, not faked.
    """
    taken = set() if taken is None else taken
    bout_count = collections.Counter() if bout_count is None else bout_count
    if n is None or n >= len(rows):
        for r in rows:
            taken.add(id(r))
            bout_count[(r.recording, r.bout)] += 1
        return list(rows), {"mode": "all", "n": len(rows), "shortfalls": {},
                            "per_bout_cap": None, "per_rec_cap": None}
    rec_cap = max(1, int(np.ceil(per_rec_frac * n)))
    rec_count = collections.Counter()
    picked, short = [], {}
    cells = [("contact", int(contact_frac * n)), ("apart", int(apart_frac * n))]
    cells.append(("free", n - sum(c for _, c in cells)))
    for cell, want in cells:
        cand = [r for r in rows if cell == "free"
                or (r.contact if cell == "contact" else r.apart)]
        got = round_robin_draw(cand, want, rng, key=lambda r: r.recording,
                               cap_by=lambda r: (r.recording, r.bout), cap=per_bout_cap,
                               taken=taken, bout_count=bout_count, rec_count=rec_count,
                               rec_cap=rec_cap, strict=False)
        if len(got) < want:
            short[cell] = {"wanted": want, "got": len(got), "pool": len(cand)}
        picked += got
    # a cell that under-filled leaves room: top up from whatever is left
    if len(picked) < n:
        picked += round_robin_draw([r for r in rows if id(r) not in taken], n - len(picked),
                                   rng, key=lambda r: r.recording,
                                   cap_by=lambda r: (r.recording, r.bout), cap=per_bout_cap,
                                   taken=taken, bout_count=bout_count, rec_count=rec_count,
                                   rec_cap=rec_cap, strict=False)
    return picked, {"mode": "capped", "n": len(picked), "wanted": n, "shortfalls": short,
                    "per_bout_cap": per_bout_cap, "per_rec_cap": rec_cap}


def sided_draw(rows, *, female_n, male_n, rng, per_rec_frac=0.20, per_bout_cap=200):
    """Female side then male side; returns (picked, report). The two sides are
    deliberately NOT equal in size (ruling #2): the female side is everything
    admissible and the male side is capped, with 50/50 restored at training
    time by `female_host_weight = male_n / female_n` (written into the
    manifest's `balance` block)."""
    taken = set()
    out, rep = [], {}
    for sex, n in (("female", female_n), ("male", male_n)):
        side = [r for r in rows if r.host_sex == sex]
        # a FRESH bout counter per side: the cap "applies per side" (ruling
        # #2), and a shared one would let the uncapped female side -- which
        # takes every admissible row -- exhaust the male side's per-bout
        # budget before it is drawn at all.
        got, r = draw_side(side, n, rng=rng, per_rec_frac=per_rec_frac,
                           per_bout_cap=per_bout_cap, taken=taken,
                           bout_count=collections.Counter())
        r["pool"] = len(side)
        r["contact"] = int(sum(1 for x in got if x.contact))
        r["apart"] = int(sum(1 for x in got if x.apart))
        r["wall"] = int(sum(1 for x in got if x.wall))
        r["per_recording"] = dict(collections.Counter(x.recording for x in got))
        r["max_recording_frac"] = round(
            max(r["per_recording"].values(), default=0) / max(len(got), 1), 4)
        r["max_bout"] = int(max(collections.Counter(
            (x.recording, x.bout) for x in got).values())) if got else 0
        rep[sex] = r
        out += got
    return out, rep


# --------------------------------------------------------------- io adapters
class SessionFrames:
    """`(rec, frame) -> ((C,H,W,3) uint8, (C,) present)` from the recording's
    mp4s through `predict.synced_reader`'s canonical-slot mapping.

    Captures stay open per recording and read forward with a cursor, so a
    sorted sweep over frames costs one seek per gap rather than one per frame.
    A `cap.set(POS_FRAMES)` on these H.264 files costs ~195 ms (it decodes
    from the preceding keyframe) while a sequential `cap.read()` costs 0.6 ms
    (measured 2026-09-05 on 2025_10_20_13_20_04/Cam2012630), so a gap up to
    `FORWARD_MAX` frames is DECODED THROUGH and discarded rather than sought;
    only a jump bigger than that (a new bout) pays for a seek. That is the
    difference between ~1.4 s and ~40 ms of video time per written frameset.
    """

    FORWARD_MAX = 300

    def __init__(self, sessions, cameras):
        self.sessions = dict(sessions)          # rec -> session_dir
        self.cameras = list(cameras)
        self._rec = None
        self._caps = []
        self._cursor = []
        self._plan = None
        self._hw = None

    def _open(self, rec):
        import cv2
        self.close()
        sd = self.sessions[rec]
        from jarvis_jax.predict.synced_reader import load_plan
        self._plan = load_plan(sd)
        self._caps = [cv2.VideoCapture(os.path.join(sd, f"{c}.mp4")) for c in self.cameras]
        self._cursor = [None] * len(self.cameras)
        for cap in self._caps:
            if cap.isOpened():
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if h > 0 and w > 0:
                    self._hw = (h, w)
                    break
        if self._hw is None:
            raise FileNotFoundError(f"no readable camera videos in {sd}")
        self._rec = rec

    def close(self):
        for cap in self._caps:
            cap.release()
        self._caps, self._cursor, self._rec, self._hw = [], [], None, None

    def __call__(self, rec, frame):
        import cv2
        from jarvis_jax.predict.synced_reader import _read_at, slot_positions
        if rec != self._rec:
            self._open(rec)
        H, W = self._hw
        out = np.zeros((len(self.cameras), H, W, 3), np.uint8)
        present = np.zeros(len(self.cameras), bool)
        for ci, cam in enumerate(self.cameras):
            pos, pr = slot_positions(self._plan, cam, int(frame), 1)
            if not pr[0]:
                continue
            cap, p = self._caps[ci], int(pos[0])
            cur = self._cursor[ci]
            if cur is None or not (0 <= p - cur <= self.FORWARD_MAX):
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                cur = p
            while cur < p:                      # decode through a small gap
                cap.grab()
                cur += 1
            fr, self._cursor[ci] = _read_at(cap, cur, p)
            if fr is not None:
                out[ci] = fr[:, :, ::-1]
                present[ci] = True
        return out, present


class CampaignMasks:
    """`(rec, frame) -> (F,C,H,W) bool` SAM3 masks, one bout store open at a time.

    Frames arrive sorted by (recording, frame) and bouts hold disjoint frame
    ranges, so a single-entry cache never thrashes; the store is the LAZY one
    (`BoutMaskStore`), so a bout costs its packed array (~3 GB worst case),
    not its unpacked one (~24 GB).
    """

    def __init__(self, index, cameras):
        self.index = index                      # (rec, frame) -> (npz, t)
        self.cameras = list(cameras)
        self._key = None
        self._store = None

    def __call__(self, rec, frame):
        from jarvis_jax.tracking.lift_mvq import BoutMaskStore
        hit = self.index.get((rec, int(frame)))
        if hit is None:
            return None
        npz, t = hit
        if npz != self._key:
            self._store = None
            self._store = BoutMaskStore(npz, self.cameras)
            self._key = npz
        s = self._store
        return np.stack([[s.mask_at(f, c, t) for c in range(len(self.cameras))]
                         for f in range(s.n_flies)])


# ------------------------------------------------------------------- driver
def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help=f"run roots (default glob {DEFAULT_ROOTS})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-names-from", default=DEFAULT_EXPORT_NAMES)
    ap.add_argument("--female-n", type=int, default=None,
                    help="female-host anchors to take (default: ALL admissible, ruling #2)")
    ap.add_argument("--male-n", type=int, default=4000,
                    help="male-host anchors to take (capped side)")
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--per-rec-frac", type=float, default=0.20,
                    help="max share of a capped side one recording may contribute")
    ap.add_argument("--per-bout-cap", type=int, default=200,
                    help="max framesets from one bout on a capped side (the 2 %% rule)")
    ap.add_argument("--contact-kp-units", type=float, default=5.0,
                    help="contact := min inter-fly KEYPOINT distance below this (0.5 mm)")
    ap.add_argument("--contact-units", type=float, default=15.0,
                    help="the spec's original CENTROID contact rule; recorded, not used "
                         "for stratification (it admits no frame in the campaign)")
    ap.add_argument("--wall-height-units", type=float, default=30.0)
    ap.add_argument("--no-identity-gate", action="store_true")     # single-fly (Task 4)
    ap.add_argument("--num-animals", type=int, default=2,
                    help="1 is the single-fly route (spec §3.4): no SAM3 masks exist for "
                         "those recordings, so the containment gate and the mask export "
                         "are both off")
    ap.add_argument("--target", type=int, default=None,
                    help="total anchors to draw when the pool has ONE host sex (the "
                         "single-fly export). Refused on a two-sex pool: sizing that is "
                         "what --female-n/--male-n do, and one number cannot balance it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gates-json", default=None, help="override GateThresholds fields")
    ap.add_argument("--bouts", default="bout_*", help="bout dir glob (smoke runs)")
    ap.add_argument("--census-only", action="store_true",
                    help="scan and write census.json, then stop (no frames decoded)")
    ap.add_argument("--min-female", type=int, default=3000,
                    help="stop after the census if the female-host stratum is smaller "
                         "(controller ruling 2026-09-05)")
    ap.add_argument("--no-masks", action="store_true", help="skip masks/<rec>/... (Plan B copy-paste needs them)")
    ap.add_argument("--no-images", action="store_true", help="write the JSON only (schema smoke test)")
    ap.add_argument("--figures-dir", default="figures/2026-09-mvq/v2_pseudo")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-write loader round trip on the real export")
    return ap


def main(argv=None):
    from jarvis_jax.data.pseudo_export import PseudoRecord, write_pseudo_export
    from jarvis_jax.data.pseudo_gates import GateThresholds

    a = build_parser().parse_args(argv)
    t0 = time.time()
    thr = GateThresholds(**json.loads(a.gates_json)) if a.gates_json else GateThresholds()
    refs = discover_bouts(a.runs or [DEFAULT_ROOTS], bout_glob=a.bouts)
    if not refs:
        raise SystemExit(f"no bouts under {a.runs or [DEFAULT_ROOTS]}")
    recs = sorted({r.recording for r in refs})
    # A single-fly recording has no SAM3 masks at all (spec §3.4), so neither
    # the containment gate nor the mask export can run. Keyed on the fly count
    # the caller asked for, never on a missing file: a courtship bout whose
    # masks are missing must fail, not silently lose a gate.
    # (`--no-masks` is a separate question -- it only skips WRITING masks/ into
    # the export, and must never quietly turn the containment GATE off.)
    single_fly = int(a.num_animals) == 1
    print(f"[extract] {len(refs)} bouts / {len(recs)} recordings; gates {thr}", flush=True)
    if single_fly:
        print("[extract] single-fly route: containment gate OFF (no masks exist), "
              "no masks/ written", flush=True)

    rows, pool, hists, rejects, checkpoints = [], {}, {}, {}, set()
    per_rec_t = collections.Counter()
    for i, ref in enumerate(refs, 1):
        ts = time.time()
        try:
            rr, pr, hh, rj = scan_bout(ref, thr, use_identity=not a.no_identity_gate,
                                       contact_units=a.contact_units,
                                       contact_kp_units=a.contact_kp_units,
                                       n_flies=a.num_animals, use_masks=not single_fly)
        except Exception as e:                       # a broken bout must not lose the census
            print(f"[extract] {ref.recording}/{ref.bout:05d}: SKIPPED ({type(e).__name__}: {e})",
                  flush=True)
            rejects.setdefault(ref.recording, collections.Counter())["_skipped_bouts"] += 1
            continue
        rows += rr
        for p in pr:
            pool[(p.recording, p.frame, p.host_fly)] = p
        for k, v in hh.items():
            hists.setdefault("all", {}).setdefault(k, np.zeros(N_BINS, np.int64))
            hists["all"][k] += v
            hists.setdefault(ref.recording, {}).setdefault(k, np.zeros(N_BINS, np.int64))
            hists[ref.recording][k] += v
        c = rejects.setdefault(ref.recording, collections.Counter())
        for k, v in rj.items():
            c[k] += v
        per_rec_t[ref.recording] += time.time() - ts
        print(f"[extract] [{i:3d}/{len(refs)}] {ref.recording}/bout_{ref.bout:05d} "
              f"T={ref.n_frames} anchors={rj['_n_anchors']} rows={len(rr)} "
              f"({time.time() - ts:.1f}s)", flush=True)

    planes = add_wall_flags(rows, a.wall_height_units, also=list(pool.values()))
    cen = census(rows)
    cen.update({"gates": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in dataclasses.asdict(thr).items()},
                "contact_definition": {
                    "used": "min inter-fly KEYPOINT distance",
                    "contact_kp_units": a.contact_kp_units,
                    "recorded_only": "inter-fly CENTROID separation",
                    "contact_units": a.contact_units},
                "n_bouts": len(refs), "floor_planes": planes,
                "reject_counts": {k: dict(v) for k, v in rejects.items()},
                "scan_seconds": round(time.time() - t0, 1)})
    _write_json(os.path.join(a.out, "census.json"), cen)
    _write_json(os.path.join(a.figures_dir, "census.json"), cen)
    print_census(cen, a.min_female)

    n_female = cen["by_host_sex"].get("female", 0)
    if a.census_only:
        print(f"[extract] --census-only: stopping after {time.time() - t0:.0f}s", flush=True)
        return cen
    if n_female < a.min_female:
        print(f"[extract] STOP: female-host anchors {n_female} < --min-female {a.min_female}. "
              f"Census written to {a.out}/census.json; not sampling (controller ruling: no "
              f"silent rebalance).", flush=True)
        return cen

    # ---- per-side draw (ruling #2: all of the female side, capped male side)
    female_n, male_n = a.female_n, a.male_n
    if a.target is not None:
        # `--target` sizes a ONE-SIDED pool (the single-fly export). On a
        # two-sex pool a single number cannot say how the two sides split, and
        # guessing would silently un-balance the very thing `female_host_weight`
        # exists to fix.
        sexes = sorted({r.host_sex for r in rows})
        if len(sexes) != 1:
            raise SystemExit(f"--target {a.target} but the admitted pool has host sexes "
                             f"{sexes}; size the sides with --female-n/--male-n instead")
        female_n = int(a.target) if sexes[0] == "female" else None
        male_n = int(a.target) if sexes[0] != "female" else 0
        print(f"[extract] --target {a.target}: one host sex ({sexes[0]}) in the pool",
              flush=True)
    rng = np.random.default_rng(a.seed)
    picked, sides = sided_draw(rows, female_n=female_n, male_n=male_n, rng=rng,
                               per_rec_frac=a.per_rec_frac, per_bout_cap=a.per_bout_cap)
    nf = sides["female"]["n"]
    nm = sides["male"]["n"]
    balance = {"female_n": nf, "male_n": nm,
               # a one-sided export has no host-sex ratio to restore; 0.0 (or a
               # division by an empty side) would read as "weight these to
               # nothing", which is the opposite of what it means
               "female_host_weight": (round(nm / nf, 4) if nf and nm else None),
               "note": "the export is deliberately NOT 50/50 -- the female side is every "
                       "admissible anchor and the male side is capped; training restores "
                       "50/50 by weighting female-host windows by female_host_weight "
                       "(train.female_host_weight). A one-sided (single-fly) export has "
                       "no ratio to restore and reports female_host_weight null."}
    srep = {"sides": sides, "balance": balance}
    for sex in ("female", "male"):
        s = sides[sex]
        print(f"[extract] {sex}-host: {s['n']} of {s['pool']} admissible "
              f"({s['mode']}), contact {s['contact'] / max(s['n'], 1):.1%}, apart "
              f"{s['apart'] / max(s['n'], 1):.1%}, wall {s['wall']}, max recording share "
              f"{s['max_recording_frac']:.1%}, max bout {s['max_bout']}", flush=True)
        if s["shortfalls"]:
            print(f"[extract] SHORTFALL {sex}: {json.dumps(s['shortfalls'])}", flush=True)
    print(f"[extract] balance: female_host_weight = {balance['female_host_weight']}", flush=True)

    # ---- partners (exempt from the cap and from decorrelation): each carries
    # its OWN frame's gate quantities and stratum, from `pool` (built in
    # scan_bout), never the anchor's.
    seen = {(r.recording, r.frame, r.host_fly) for r in picked}
    partner_rows = []
    for r in picked:
        for d, pf in r.partners.items():
            k = (r.recording, pf, r.host_fly)
            if k in seen:
                continue
            src = pool.get(k)
            if src is None:
                raise KeyError(f"partner row {k} missing from the scan pool")
            seen.add(k)
            partner_rows.append(src)
    print(f"[extract] picked {len(picked)} anchors + {len(partner_rows)} partner framesets",
          flush=True)

    # ---- records
    export_names = json.load(open(os.path.join(a.export_names_from, "annotations",
                                               "keypoint_names.json")))
    by_rec_ref = {}
    for ref in refs:
        by_rec_ref.setdefault(ref.recording, ref)
    bout_of = {(ref.recording, ref.bout): ref for ref in refs}
    mask_index = {}
    for ref in refs:
        for t in range(ref.n_frames):
            mask_index[(ref.recording, ref.frame_start + t)] = (ref.masks_npz, t)

    records, cache = [], {}
    written_names = None
    # bout-major so the per-bout npz cache is loaded exactly once
    for r in sorted(picked + partner_rows, key=lambda r: (r.recording, r.bout, r.frame, r.host_fly)):
        ref = bout_of[(r.recording, r.bout)]
        key = (r.recording, r.bout)
        if key not in cache:
            z3 = [np.load(os.path.join(ref.bout_dir, f"fly{f}", "kp3d.npz"), allow_pickle=True)
                  for f in range(a.num_animals)]
            z2 = [np.load(os.path.join(ref.bout_dir, f"fly{f}", "kp2d.npz"), allow_pickle=True)
                  for f in range(a.num_animals)]
            cache.clear()
            cache[key] = (np.stack([z["kp3d"] for z in z3]),
                          np.stack([z["kp2d"] for z in z2]),
                          np.stack([z["conf"] for z in z2]),
                          [str(s) for s in z3[0]["kp_names"]])
        kp3d, kp2d, conf, kp_names = cache[key]
        if written_names is None:
            written_names = kp_names
        elif kp_names != written_names:
            raise ValueError(f"{ref.bout_dir}: kp_names {kp_names[:3]}... differ from the "
                             f"rest of the campaign {written_names[:3]}... -- the export has "
                             f"ONE keypoint axis and it is resolved BY NAME")
        sexcode = np.array([0 if ref.fly_sex.get(f"fly{f}") == "female" else
                            (1 if ref.fly_sex.get(f"fly{f}") == "male" else -1)
                            for f in range(a.num_animals)], np.int8)
        vis = conf[:, r.t] >= thr.conf_min
        vis = vis & np.isfinite(kp3d[:, r.t]).all(-1)[:, None, :]
        records.append(PseudoRecord(
            recording=r.recording, frame=r.frame, host_fly=r.host_fly,
            kp3d=kp3d[:, r.t], kp2d=kp2d[:, r.t], vis=vis, sex=sexcode,
            stratum={"host_sex": r.host_sex, "contact": bool(r.contact), "apart": bool(r.apart),
                     "wall": bool(r.wall),
                     "kp_dist_units": None if r.kp_dist_units is None
                     else round(r.kp_dist_units, 2),
                     "sep_units": None if r.sep_units is None else round(r.sep_units, 2),
                     "height_units": round(r.height_units, 2)},
            gates=r.gates, partners=r.partners, role=r.role, bout=r.bout))

    cameras = list(refs[0].cameras)
    recordings = {rec: {"calib_dir": by_rec_ref[rec].calib_dir,
                        "calib_group": rec,
                        "fly_sex": by_rec_ref[rec].fly_sex,
                        "kp_names": None, "n_flies": a.num_animals,
                        # from the bout's own mvq_meta.json (the masked
                        # campaign's defaults; the single-fly route writes
                        # free_running / female / its own sex source)
                        "behavior": by_rec_ref[rec].behavior,
                        "sex": by_rec_ref[rec].recording_sex,
                        "sex_source": by_rec_ref[rec].sex_source}
                  for rec in sorted({r.recording for r in picked + partner_rows})}
    for rec in recordings:
        ref = by_rec_ref[rec]
        z = np.load(os.path.join(ref.bout_dir, "fly0", "kp3d.npz"), allow_pickle=True)
        recordings[rec]["kp_names"] = [str(s) for s in z["kp_names"]]
        checkpoints.add(json.load(open(os.path.join(ref.bout_dir, "mvq_meta.json")))["checkpoint"])

    frames = SessionFrames({rec: by_rec_ref[rec].session_dir for rec in recordings}, cameras)
    masks = None if (a.no_masks or single_fly) else CampaignMasks(mask_index, cameras)
    summary = write_pseudo_export(
        a.out, records, export_names=export_names, cameras=cameras, recordings=recordings,
        checkpoint=sorted(checkpoints)[0] if checkpoints else "unknown", gates=thr,
        weight=a.weight, frame_reader=frames, mask_reader=masks,
        write_images=not a.no_images,
        extra_manifest={"balance": balance,
                        "contact_definition": cen["contact_definition"],
                        # the exact run roots this export was extracted from:
                        # the campaign is a moving target (a re-lift renames
                        # `pose_mvq_p3b` aside), so the manifest names the tree,
                        # not just the checkpoint.
                        "source_roots": sorted({ref.run_root for ref in refs}),
                        "n_source_bouts": len(refs)})
    frames.close()

    # ---- reports
    srep.update({"realised": summary["n_framesets"],
                 "per_stratum": summary["per_stratum"], "per_recording": summary["per_recording"],
                 "per_role": summary["per_role"], "per_bout": summary["per_bout"],
                 "partner_availability": summary["partner_availability"],
                 "contact_frac": round(sum(1 for r in picked if r.contact) / max(len(picked), 1), 4),
                 "apart_frac": round(sum(1 for r in picked if r.apart) / max(len(picked), 1), 4),
                 "wall_frac": round(sum(1 for r in picked if r.wall) / max(len(picked), 1), 4),
                 "female_frac": round(sum(1 for r in picked if r.host_sex == "female")
                                      / max(len(picked), 1), 4),
                 "max_bout_frac": round(max(summary["per_bout"].values()) / max(
                     summary["n_framesets"], 1), 4) if summary["per_bout"] else 0.0,
                 "n_recordings": len(summary["per_recording"]),
                 "seconds": round(time.time() - t0, 1)})
    hist_out = {rec: {k: {"edges": list(np.linspace(*HIST_EDGES[k], N_BINS + 1)),
                          "counts": v.tolist()} for k, v in h.items()}
                for rec, h in hists.items()}
    hist_out["_rejected"] = {rec: dict(c) for rec, c in rejects.items()}
    for d in (a.out, a.figures_dir):
        _write_json(os.path.join(d, "sampling_report.json"), srep)
        _write_json(os.path.join(d, "gate_histograms.json"), hist_out)
    write_summary_md(os.path.join(a.out, "summary.md"), cen, srep, summary, thr, a)
    print(f"[extract] wrote {summary['n_framesets']} framesets / {summary['n_images']} images "
          f"to {a.out} in {time.time() - t0:.0f}s", flush=True)
    print(json.dumps({k: srep[k] for k in ("realised", "female_frac", "contact_frac",
                                           "apart_frac", "wall_frac", "max_bout_frac",
                                           "n_recordings")}, indent=1), flush=True)
    if not a.no_verify:
        srep["verify"] = verify_export(a.out, n=20)
        _write_json(os.path.join(a.out, "sampling_report.json"), srep)
    return srep


def verify_export(root, n=20, split="train", seed=0):
    """Open the REAL export with `V12WindowDataset` and read `n` random
    windows: keypoint-axis order, finite labels, a fly with 3D, and a crop
    that is not empty. This is the round trip the trainer performs, run on the
    artefact that was actually written -- not on a fixture."""
    from jarvis_jax.data.v12_windows import V12WindowDataset
    canon = json.load(open(os.path.join(root, "annotations", "keypoint_names.json")))
    ds = V12WindowDataset(root, split, T=1, train=False)
    if list(ds.keypoint_names) != list(canon):
        raise AssertionError("instances keypoint_names != annotations/keypoint_names.json")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    n_kp, n_cams = 0, 0
    for i in idx:
        s = ds[int(i)]
        rec, fly, frame = ds.windows[int(i)]
        if not np.isfinite(s["kp3d_local"]).all() or not np.isfinite(s["center3D"]).all():
            raise AssertionError(f"non-finite 3D in window {rec}/Frame_{frame}/fly{fly}")
        if not np.isfinite(s["kp2d"]).all():
            raise AssertionError(f"non-finite 2D in window {rec}/Frame_{frame}/fly{fly}")
        if not s["has3d"][0, 0].any():
            raise AssertionError(f"host has no triangulated 3D in {rec}/Frame_{frame}/fly{fly}")
        if int(s["cam_valid"].sum()) < 2:
            raise AssertionError(f"< 2 valid cameras in {rec}/Frame_{frame}/fly{fly}")
        if not s["crops"][0][s["cam_valid"][0]].any():
            raise AssertionError(f"all-black crops in {rec}/Frame_{frame}/fly{fly}")
        n_kp += int(s["has3d"][0, 0].sum())
        n_cams += int(s["cam_valid"].sum())
    out = {"windows_checked": len(idx), "n_windows": len(ds),
           "mean_kp_with_3d": round(n_kp / max(len(idx), 1), 2),
           "mean_valid_cameras": round(n_cams / max(len(idx), 1), 2),
           "keypoint_names_ok": True}
    print(f"[verify] {out}", flush=True)
    return out


def write_summary_md(path, cen, srep, summary, thr, a):
    L = ["# Pseudo-label export summary", "",
         f"- root: `{a.out}`", f"- checkpoint gates: `{thr}`",
         f"- contact := min inter-fly KEYPOINT distance < {a.contact_kp_units} units "
         f"(the spec's centroid rule < {a.contact_units} units admits 0 frames campaign-wide)",
         f"- weight {a.weight}; balance {json.dumps(srep['balance'])}",
         f"- framesets {summary['n_framesets']} (anchors "
         f"{summary['per_role'].get('anchor', 0)}, partners "
         f"{summary['per_role'].get('partner', 0)}), images {summary['n_images']}, "
         f"annotations {summary['n_annotations']}, mask rows {summary['n_mask_rows']}",
         f"- dropped for < 2 cameras: {summary['n_dropped_too_few_cameras']}",
         f"- elapsed {srep['seconds']} s", "",
         "## Per side (anchors)", "",
         "| side | pool | taken | contact | apart | wall | max recording share | max bout |",
         "|---|---|---|---|---|---|---|---|"]
    for sex in ("female", "male"):
        s = srep["sides"][sex]
        L.append(f"| {sex} | {s['pool']} | {s['n']} | {s['contact']} "
                 f"({s['contact'] / max(s['n'], 1):.1%}) | {s['apart']} "
                 f"({s['apart'] / max(s['n'], 1):.1%}) | {s['wall']} | "
                 f"{s['max_recording_frac']:.1%} | {s['max_bout']} |")
    L += ["", "## Framesets per stratum (anchors + partners)", "",
          "| stratum | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(srep["per_stratum"].items())]
    L += ["", "## Framesets per recording", "", "| recording | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(srep["per_recording"].items())]
    L += ["", "## Partner availability (framesets carrying a partner at delta)", "",
          "| delta | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(srep["partner_availability"].items(),
                                              key=lambda kv: int(kv[0]))]
    L += ["", f"- realised contact {srep['contact_frac']:.1%}, apart {srep['apart_frac']:.1%}, "
              f"wall {srep['wall_frac']:.1%}, female-host {srep['female_frac']:.1%}",
          f"- largest single bout: {srep['max_bout_frac']:.2%} of the export",
          f"- recordings represented: {srep['n_recordings']}",
          "", "Census (all admissible anchors, before sampling) is in `census.json`; "
          "gate histograms in `gate_histograms.json`."]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
