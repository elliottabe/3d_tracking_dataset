"""Coarse pass: a whole recording at stride 16, mask-free.

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §4.3. This is
the layer that puts §4.1 (`coarse_centres`: CenterDetect peaks -> 3D centres
-> window plans) and §4.2 (`lift_mvq.MVQRunner`: frames -> windows -> typed
slots) together over TIME, and writes the recording-level `coarse_tracks.npz`
that bout detection (§5) reads. It owns no geometry and no model: everything
here is bookkeeping over frames, plus the per-frame features (§5's inputs)
and the file format.

Per sampled frame:

    reader(frame) -> (frames (C,H,W,3) u8, present (C,))
    detector.peaks(frames)            -> (C,2,2) px peaks, (C,2) scores
    lift_peaks_to_centres(...)        -> up to num_animals 3D centres
    cluster_centres / plan_windows    -> 1-2 window centres (merged when close)
    runner.windows(...) -> runner.infer(...) -> runner.read_typed(want_sex)

FLY AXIS. `f = 0` is the FEMALE slot and `f = 1` the MALE slot, read from the
model's FIXED typed slots (`read_typed(want_sex=0/1)` -> slot 1 / slot 2), not
from a heuristic and not from a per-frame nearest-centre rule. A
single-animal recording (`num_animals=1`) has one row, read with
`want_sex=-1` (whichever typed slot exists) and REPORTS the sex it found in
`sex_prob`/`slot` rather than assuming one.

WINDOW-CENTRE REUSE. CenterDetect fails on ~20 % of real courtship frames
(the two flies touch and it collapses to one peak) and can fail outright.
A frame with NO usable centre REUSES the previous frame's window centres --
at 800 fps a fly moves ~0.6 mm between two coarse frames, well inside the
5.6 mm window -- and says so in `centre_source`:

    0 = CENTRE_DETECTED   this frame's own CenterDetect centres
    1 = CENTRE_REUSED     the last detected frame's centres
    2 = CENTRE_NONE       no centre and no history -> the whole row is NaN

The alternative (a zero/origin centre) would crop a 448-px window at the
world origin and come back with a confident, perfectly smooth fit of the
arena floor, which no residual or confidence metric would flag.

BATCHING SPANS FRAMES. `MVQRunner.infer` always pads to `runner.batch`, so
calling it once per frame with one window would compute `batch` windows and
throw away all but one. Windows are accumulated across frames and flushed
with `concat_windows` when the next frame's windows would not fit; every
window of one frame always lands in the SAME forward, since the typed-slot
read is per-`infer`-output.

UNITS AND NAMES (CLAUDE.md). Distances and centres are 0.1 mm world units,
pixels are FULL-IMAGE px, angles are degrees, `frame` is an absolute video
frame index (the canonical slot). Keypoints are looked up BY NAME through
`kp_names` (the model's own detector order) -- never by integer, which is the
trap that once turned a middle-left leg into a "collapsed right wing vein".

FILE FORMAT (`write_coarse_tracks`). One npz + one `.meta.json`, deliberately
a SUPERSET of the SAM3 coarse pass's schema (`scripts/coarse_pass.py`), so
`scripts/coarse_pass_gates.py` opens either file:

  SAM3 schema (F = num_animals, C = cameras, T = coarse frames)
    coarse_frame (T,) i64      absolute video frame index of each sample
    cameras (C,) str           CANONICAL camera order
    centroid (F,C,T,2) f32     the 3D centroid REPROJECTED to each camera, px
    valid (F,C,T) bool         finite centroid that lands inside the image --
                               NOTE this means "the reprojected 3D centroid is
                               inside camera c's frame", NOT "camera c saw the
                               fly" (there is no per-camera detection here,
                               only one 3D point reprojected everywhere), so
                               `n_valid_cams >= 3` is NOT an occlusion signal
                               the way it might be read from a SAM3 file where
                               masks make `valid` a real per-camera visibility
                               bit
    in_frame (F,C,T) i8        1 inside the image, 0 outside -- ONLY 0/1. The
                               SAM3 file's third state, `IN_FRAME_UNKNOWN = 2`
                               ("< 2 other valid cameras -- position not
                               determinable", `sam3_driver.py`), is collapsed
                               to 0 here because there is exactly one 3D point
                               and no notion of "not enough OTHER cameras" --
                               a fine pass that gap-repairs by pattern-matching
                               `IN_FRAME_YES`/`IN_FRAME_NO` runs of a SAM3 file
                               must NOT be applied unchanged to an mvq file
    border_dist (F,C,T) f32    px from the reprojected centroid to the nearest edge
    area (F,C,T) f32           ALL NaN -- mask-free, there is no area
    area_med (F,T) f32         ALL NaN (see `area`)
    border_med (F,T) f32       median border_dist over valid cameras
    n_valid_cams (F,T) i16
    X3d (F,T,3) f32            the 3D centroid (mean of the fly's finite keypoints)
    sep3d (T,) f32             inter-fly 3D distance (empty array if F < 2)
    sep2d_med (T,) f32         median per-camera 2D centroid separation
    filled (T,) bool           this coarse index was processed (always True here;
                               emptiness is carried by `valid`/`n_valid_cams`)

  mvq fields
    kp3d (F,T,K,3) f16         world keypoints (f16: 0.5 unit = 0.05 mm at
                               1000 units, far below the coarse pass's own error)
    kp_names (K,) str          the model's keypoint order, for BY-NAME indexing
    exist (F,T) f32            typed-slot existence sigmoid
    sex_prob (F,T) f32         P(female) of the slot that was read
    slot (F,T) i8              which model slot (1 female, 2 male; -1 = miss)
    centre_source (F,T) i8     0/1/2 as above
    wing_angle_deg (F,T) f32   max over L/R of the body-axis-to-wing angle
    heading_deg (T,) f32       male's facing direction vs the male->female vector
    speed (F,T) f32            units per coarse frame
    height (F,T) f32           above the fitted floor plane, units
    dist (T,) f32              == sep3d, under the mvq name
    trackable (F,T) bool       exist >= 0.5 and a finite centroid
    last_centres (F,3) f32     the last CENTRE_DETECTED/REUSED window centres
                               (NaN-padded, see `pad_last_centres`), so a
                               `--resume` can restore the reuse chain across
                               the process boundary instead of starting the
                               new chunk with no history (`init_centres`)

  meta json: session_dir, stride, cameras, W, H, num_animals, n_coarse,
    coarse0 (= coarse_frame[0] // stride), source="mvq", plus the floor plane
    and whatever the caller passes as `meta_extra` (checkpoint, timings).

NOTE for §5: because `area` is all-NaN, `coarse_pass_gates.py`'s area-ratio
gate can never pass on an mvq file -- it OPENS and its shapes are right, but
the SAM3 thresholds are not meaningful without masks. Bout detection on mvq
tracks uses the mvq features above (distance, wing angle, speed, heading).

TWO MORE SEMANTICS THAT CHANGE IN AN MVQ FILE (§5, read before gating on
them): `valid`/`n_valid_cams` mean "the reprojected 3D centroid lands inside
this camera's image", not "this camera saw the fly" -- there is one 3D point
reprojected to every camera, no per-camera detection -- so `n_valid_cams >=
3` is not an occlusion signal here the way it can be against a SAM3 file's
masks. And `in_frame` is 0/1 ONLY: SAM3's `IN_FRAME_UNKNOWN = 2` is collapsed
to 0 (`IN_FRAME_NO`), so the fine pass's gap-repair idiom that runs on
`IN_FRAME_YES`/`IN_FRAME_NO` codes must not be applied to mvq files unchanged.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from typing import NamedTuple

import numpy as np

from jarvis_jax.tracking.coarse_centres import (_project_batch, cluster_centres,
                                                lift_peaks_to_centres, plan_windows)
from jarvis_jax.tracking.lift_mvq import concat_windows
from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN

CENTRE_DETECTED, CENTRE_REUSED, CENTRE_NONE = 0, 1, 2
FLOOR_FIT_N = 2000          # spec §4.3: the first 2000 finite coarse centroids
TRACKABLE_EXIST = 0.5

# Keypoints the features are built from, BY NAME (never by index).
KP_HEAD, KP_TAIL = "Scutellum", "Abd_tip"
WING_PAIRS = (("WingL_base", "WingL_V13"), ("WingR_base", "WingR_V13"))


class FloorPlane(NamedTuple):
    """`normal . x + offset` = height above the floor, in world units.

    `normal` is a unit vector oriented so that a fly's height is POSITIVE (see
    `fit_floor`): the sign is not a convention anyone can read off the
    calibration, and getting it backwards would silently invert every
    height-based gate.

    `orientation` records WHICH mechanism picked that sign -- `"hint"` when
    `fit_floor`'s `up_hint` decided it directly, `"skew"` when the
    bottom-heaviness heuristic did (see `fit_floor`'s DEVIATION note, and its
    round-2 addendum on where that heuristic stops being trustworthy).
    `skew` is the heuristic's own statistic, recorded even when `up_hint`
    made the actual decision, so the two can be cross-checked in the meta
    json.
    """
    normal: np.ndarray
    offset: float
    orientation: str = "skew"
    skew: float = float("nan")

    def height(self, points):
        """(...,3) world points -> (...,) height above the plane, units."""
        return np.asarray(points, np.float64) @ np.asarray(self.normal, np.float64) + self.offset

    def as_dict(self):
        return {"normal": [float(v) for v in self.normal], "offset": float(self.offset),
                "orientation": str(self.orientation), "skew": float(self.skew)}


def _svd_normal(pts):
    """Unit normal of the least-variance direction of `(N,3)` points, via SVD
    about their mean. Not sign-resolved -- callers pick "up" themselves."""
    _, _, vh = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    return vh[-1] / np.linalg.norm(vh[-1])


FLOOR_SKEW_MARGINAL = 0.1   # see fit_floor's `skew_marginal` docs for how this was picked


def fit_floor(centroids, *, exist=None, n_fit=FLOOR_FIT_N, floor_pct=3.0, up_hint=None,
             skew_marginal=FLOOR_SKEW_MARGINAL):
    """Least-squares floor plane from the FIRST `n_fit` finite, TRACKABLE
    coarse centroids.

    Args:
        centroids: (F, N, 3) per-fly 3D centroids in frame order (what
            `coarse_pass` returns as `centroid`), or any (..., 3) array.
        exist: optional, the SAME (F, N) (or `(...,)` matching `centroids`
            minus its last axis) existence array `coarse_pass` returns as
            `exist` -- a centroid is used only if `exist >= TRACKABLE_EXIST`
            (0.5). Without a trackable gate, a run of low-confidence rows
            (the typed slot barely fired, or fired on background) is fit as
            if it were real floor/wall geometry; `None` (the default) fits
            every finite point, matching the ungated behaviour before this
            argument existed.
        n_fit: how many finite, trackable points to fit on. The plane is a
            property of the ARENA, so the first ~20 s of the recording (2000
            coarse centroids at stride 16 == ~1000 coarse frames == ~16 000
            video frames == ~20 s at 800 fps) is as good as all of it and
            orders of magnitude cheaper; a whole-recording fit would also be
            dragged by whatever the flies do at the end.
        floor_pct: the percentile of the along-normal coordinate the plane is
            placed at (low single digits = just under the lowest observed
            centroids; not 0/1, which is one outlier away from the true
            floor on a real, noisy cloud).
        up_hint: optional (3,) world-unit vector giving an ALREADY-KNOWN up
            direction (e.g. hand-picked from a floor-majority stretch of the
            same recording, or carried in the recording config from a prior
            run's diagnosis). When given, the fitted normal is oriented by
            `dot(normal, up_hint) > 0` and the skew heuristic below is
            SKIPPED entirely -- `FloorPlane.orientation == "hint"`. Use this
            whenever the recording is expected to be wall-heavy (see
            `skew_marginal`).
        skew_marginal: only used when `up_hint` is None. The sign is then
            chosen by the bottom-heaviness skew heuristic below, which
            ASSUMES floor points are the majority of the trackable sample and
            flips ~180 deg once wall points take over -- reproduced
            empirically at wall_frac 0.63-0.70 across 5+ seeds (round-2
            review finding). The heuristic's own statistic, `skew = (mean(s)
            - median(s)) / std(s)`, tracks this: measured `|skew|` stays
            above ~0.2 for wall_frac <= 0.5 and collapses under ~0.1 right in
            the 0.6-0.7 flip zone (both the sign AND the confidence come from
            the SAME quantity failing together, which is why "check a
            residual" does not catch this -- CLAUDE.md). `|skew| <
            skew_marginal` (default 0.1) is used as a proxy for "the
            heuristic is close to a coin flip here" and emits a
            `RuntimeWarning` recommending `up_hint` instead of trusting the
            sign silently.

    Returns:
        FloorPlane with a unit `normal` pointing UP and an `offset` that puts
        the plane at the BOTTOM of the fly cloud, so heights are ~0 for a fly
        walking on the glass and positive for one up a wall. `orientation`
        and `skew` record which mechanism picked the sign and the heuristic's
        own statistic (see `FloorPlane`).

    DEVIATION from the design note's literal wording ("least squares ... with
    the sign chosen so the median fly height is positive"), because that rule
    is not well posed and orients the plane BACKWARDS on real data:

      * A least-squares plane runs through the MEAN of the points it is fit
        to, so the median height above it is ~0 by construction -- it measures
        the mean height of the FLIES, not the floor they stand on.
      * Its sign is then decided by the skew of that near-zero residual.
        Flies rest on the glass and occasionally climb, never the reverse, so
        the cloud is bottom-heavy and the median residual is NEGATIVE
        (measured: -2.2 units on a 90 %-floor / 10 %-wall synthetic cloud).
        The literal rule therefore flips the normal DOWNWARD and every height
        reads backwards -- a sign error no smoothness or residual check sees.

    What is kept: the least-squares fit over the first `n_fit` finite,
    trackable centroids, and positive fly heights. What is fixed: "up" is the
    direction the cloud is bottom-heavy in (mean along the normal above the
    median -- flies climb up, not down) UNLESS `up_hint` says otherwise; a
    SECOND SVD pass refits the normal on only the bottom half (by the first
    pass's height) of those points, because a total-least-squares fit over
    floor+wall points together tilts the normal toward the wall in
    proportion to the wall's point fraction -- every point gets equal SVD
    weight regardless of whether it is "the floor" -- and the wall points
    are, almost by construction, the ones farthest from co-planar with the
    true floor; the offset sits at the `floor_pct` percentile of the REFIT
    normal's heights rather than at the mean, so a handful of below-floor
    outliers cannot single-handedly set it.

    ROUND-2 DEVIATION: the bottom-heaviness heuristic above is itself only a
    MAJORITY-RULE proxy for "up" -- it silently flips when wall points are
    the majority of the sample, which no per-run smoothness/residual metric
    would catch (both "correct" and "confidently backwards" look identical to
    everything except the sign). Orientation must not rest on that statistic
    alone: `up_hint` (an independent, externally-supplied up direction) is
    now the PREFERRED path, and the heuristic path now warns when its own
    confidence is marginal instead of picking a silent coin flip.

    Raises:
        ValueError: fewer than 3 finite, trackable points -- a plane through
            2 points is not determined, and returning an arbitrary one would
            give every height a meaningless value rather than an obvious
            failure.
    """
    a = np.asarray(centroids, np.float64)
    # Frame-major order: `centroids` is (F,N,3), so reshaping fly-major would
    # take "the first n_fit" all from fly 0. Transpose to (N,F,3) first.
    pts_all = np.transpose(a, (1, 0, 2)).reshape(-1, 3) if a.ndim == 3 else a.reshape(-1, 3)
    ok = np.isfinite(pts_all).all(axis=1)
    if exist is not None:
        e = np.asarray(exist, np.float64)
        # Same frame-major reshape as `pts_all`: (F,N) -> (N,F) -> flat.
        e = np.transpose(e, (1, 0)).reshape(-1) if a.ndim == 3 else e.reshape(-1)
        if e.shape[0] != pts_all.shape[0]:
            raise ValueError(f"exist has {e.shape[0]} entries but centroids reshape to "
                             f"{pts_all.shape[0]}; exist must carry the SAME (fly, frame) axes "
                             f"as centroids (coarse_pass's own `exist`/`centroid` arrays)")
        ok = ok & (e >= TRACKABLE_EXIST)
    pts = pts_all[ok]
    if pts.shape[0] < 3:
        raise ValueError(f"floor fit needs >= 3 finite, trackable (exist >= {TRACKABLE_EXIST}) "
                         f"coarse centroids, got {pts.shape[0]}")
    pts = pts[:int(n_fit)]

    hint = None if up_hint is None else np.asarray(up_hint, np.float64)

    def _orient(normal, s):
        """Flip `normal` (and its scores `s`) to point UP.

        With `up_hint`, "up" is simply `dot(normal, hint) > 0` -- no
        heuristic, no majority assumption. Without it, "up" is the direction
        the cloud is bottom-heavy in: flies rest on the floor and
        occasionally climb, never the reverse, so a correctly-oriented
        up-normal has mean(s) > median(s) -- see the ROUND-2 DEVIATION note
        on why this alone is not trusted blindly any more."""
        if hint is not None:
            return (normal, s) if np.dot(normal, hint) >= 0 else (-normal, -s)
        return (normal, s) if s.mean() >= np.median(s) else (-normal, -s)

    normal = _svd_normal(pts)
    s = pts @ normal
    normal, s = _orient(normal, s)

    # Refit the normal on the BOTTOM half only (by this first pass's height),
    # which is dominated by the true floor regardless of the wall fraction --
    # see the DEVIATION note above.
    low = s <= np.percentile(s, 50.0)
    if low.sum() >= 3:
        normal2 = _svd_normal(pts[low])
        if np.dot(normal2, normal) < 0:        # keep the refit continuous with pass 1
            normal2 = -normal2
        normal = normal2
        s = pts @ normal
        normal, s = _orient(normal, s)

    skew = float((s.mean() - np.median(s)) / (np.std(s) + 1e-9))
    if hint is not None:
        orientation = "hint"
    else:
        orientation = "skew"
        if abs(skew) < float(skew_marginal):
            warnings.warn(
                f"fit_floor: orientation skew={skew:.4f} is below the marginal threshold "
                f"{skew_marginal} -- the bottom-heaviness heuristic is close to a coin flip "
                f"here (this happens once wall points approach a MAJORITY of the trackable "
                f"sample); the normal's sign is UNCERTAIN. Pass up_hint=<a known up direction, "
                f"e.g. hand-picked from a floor-majority stretch of the recording> to resolve "
                f"it directly instead of trusting this heuristic.", RuntimeWarning, stacklevel=2)

    offset = -float(np.percentile(s, float(floor_pct)))
    return FloorPlane(normal.astype(np.float64), offset, orientation, skew)


def _angle_deg(a, b):
    """Angle between two (...,3) vector fields, degrees, NaN-propagating."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (a * b).sum(axis=-1) / (na * nb)
        cos = np.where((na > 0) & (nb > 0), cos, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _kp(kp3d, names, name):
    """One keypoint's (F,N,3) track, BY NAME. Raises if the model does not
    have it, rather than silently indexing whatever sits at that position."""
    try:
        k = names.index(name)
    except ValueError:
        raise ValueError(f"keypoint {name!r} is not in this model's keypoint names "
                         f"({len(names)} names, e.g. {names[:4]}); the coarse features "
                         f"index BY NAME on purpose") from None
    return kp3d[:, :, k]


def coarse_features(tracks, kp_names, *, floor):
    """Per-frame behavioural features for bout detection (spec §5).

    Args:
        tracks: `coarse_pass`'s output (needs `kp3d` (F,N,K,3), `centroid`
            (F,N,3), `exist` (F,N)).
        kp_names: the keypoint order of `tracks["kp3d"]` -- the model's own
            (detector) order, `MVQRunner.kp_names`.
        floor: a `FloorPlane` (see `fit_floor`).

    Returns dict of:
        dist (N,)             inter-fly 3D centroid distance, units (NaN if F < 2)
        heading_deg (N,)      angle between the MALE's facing direction
            (Abd_tip -> Scutellum, i.e. anterior) and the male->female vector.
            0 deg = the male is pointed straight AT the female, 180 deg =
            straight away. The anterior direction is used deliberately: the
            wing angle below measures against the posterior axis
            (Scutellum -> Abd_tip) because that is the axis a wing is held
            relative to, but a "heading" whose zero meant facing away would
            be read backwards by every downstream gate.
        speed (F,N)           ||centroid[t] - centroid[t-1]||, units per COARSE
            frame (one sample step, i.e. `stride` video frames); [:,0] is NaN.
        wing_angle_deg (F,N)  max over L/R of the angle between the body axis
            (Scutellum -> Abd_tip) and the wing vector (WingX_base -> WingX_V13).
            The max is nan-aware (`np.fmax`): one dropped wing does not erase
            the other.
        height (F,N)          centroid height above `floor`, units
        trackable (F,N)       exist >= 0.5 AND a finite centroid
    """
    kp3d = np.asarray(tracks["kp3d"], np.float64)
    centroid = np.asarray(tracks["centroid"], np.float64)
    exist = np.asarray(tracks["exist"], np.float64)
    names = [str(n) for n in kp_names]
    if kp3d.shape[2] != len(names):
        raise ValueError(f"kp3d has {kp3d.shape[2]} keypoints but kp_names has {len(names)}; "
                         f"the feature lookup is BY NAME and needs the model's own order")
    F, N = kp3d.shape[0], kp3d.shape[1]

    head = _kp(kp3d, names, KP_HEAD)
    tail = _kp(kp3d, names, KP_TAIL)
    body_axis = tail - head                                  # Scutellum -> Abd_tip
    wing = [ _angle_deg(body_axis, _kp(kp3d, names, tip) - _kp(kp3d, names, base))
             for base, tip in WING_PAIRS ]
    wing_angle_deg = np.fmax(wing[0], wing[1]).astype(np.float32)

    height = floor.height(centroid).astype(np.float32)

    speed = np.full((F, N), np.nan, np.float32)
    if N > 1:
        speed[:, 1:] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)

    dist = np.full(N, np.nan, np.float32)
    heading_deg = np.full(N, np.nan, np.float32)
    if F >= 2:
        dist = np.linalg.norm(centroid[0] - centroid[1], axis=-1).astype(np.float32)
        forward_male = (head - tail)[1]                      # anterior direction, fly 1 = male
        male_to_female = centroid[0] - centroid[1]
        heading_deg = _angle_deg(forward_male, male_to_female).astype(np.float32)

    with np.errstate(invalid="ignore"):
        trackable = (exist >= TRACKABLE_EXIST) & np.isfinite(centroid).all(axis=-1)
    return {"dist": dist, "heading_deg": heading_deg, "speed": speed,
            "wing_angle_deg": wing_angle_deg, "height": height,
            "trackable": np.asarray(trackable, bool)}


def coarse_pass(reader, runner, detector, *, frames, num_animals=2, merge_dist_units=30.0,
                min_views=3, max_resid_px=25.0, floor=None, progress=None, init_centres=None):
    """Run the coarse pass over `frames` (absolute video frame indices).

    Args:
        reader: callable `(frame_idx) -> (frames (C,H,W,3) uint8 RGB,
            present (C,) bool)` in the runner's CANONICAL camera order.
        runner: `lift_mvq.MVQRunner`.
        detector: anything with `.peaks(frames) -> (peaks (C,2,2) px,
            scores (C,2))` -- `coarse_centres.CenterDetector` in production,
            a fake in the tests. `peaks[c]` must be the SAME camera as
            `runner.cam_mats[c]`; both are the canonical order.
        frames: iterable of absolute frame indices (the stride-16 sample).
        num_animals: 1 or 2. 2 reads the FEMALE (slot 1) and MALE (slot 2)
            typed slots into rows 0 and 1; 1 reads whichever typed slot exists
            (`want_sex=-1`) into row 0 and reports its sex.
        merge_dist_units / min_views / max_resid_px: passed to §4.1.
        floor: an already-fit `FloorPlane` to carry into the result (a
            RESUMED pass keeps the floor of its first chunk so heights stay
            comparable across chunks); None leaves it to the caller's
            `fit_floor` over the finished tracks.
        progress: callable `(done, total, fps)` after every frame.
        init_centres: (W,3) window centres from the previous chunk, so the
            reuse rule survives a chunk boundary.

    Returns a dict of arrays (F = num_animals, N = len(frames), K = runner.K):
        frame (N,) i64, kp3d (F,N,K,3) f32 world, centroid (F,N,3) f32,
        exist (F,N) f32, sex_prob (F,N) f32, slot (F,N) i8, centre_source
        (F,N) i8, n_windows (N,) i8, plus kp_names, W, H, last_centres, floor.

    `centre_source` is per fly for the fine pass's benefit, but the reuse rule
    is per FRAME today (a frame either has centres or it does not), so both
    rows of a frame always agree.
    """
    frame_list = [int(f) for f in frames]
    F = int(num_animals)
    if F not in (1, 2):
        raise ValueError(f"num_animals must be 1 or 2, got {num_animals}")
    N, K = len(frame_list), runner.K

    kp3d = np.full((F, N, K, 3), np.nan, np.float32)
    centroid = np.full((F, N, 3), np.nan, np.float32)
    exist = np.full((F, N), np.nan, np.float32)
    sex_prob = np.full((F, N), np.nan, np.float32)
    slot = np.full((F, N), -1, np.int8)
    centre_source = np.full((F, N), CENTRE_NONE, np.int8)
    n_windows = np.zeros(N, np.int8)

    want = [SEX_UNKNOWN] if F == 1 else [SEX_FEMALE, SEX_MALE]
    prev_centres = None if init_centres is None else np.asarray(init_centres, np.float32)

    pend = []            # (t, row offset into the batch, n rows)
    batch = []           # per-frame windows dicts
    rows = 0
    W = H = None
    t_start = time.time()
    warned_drop = False  # emit the windows-dropped warning at most once per run

    def _read_batch():
        """One forward over the accumulated windows; fill every pending frame."""
        nonlocal pend, batch, rows
        if not pend:
            return
        out = runner.infer(concat_windows(batch))
        for t, off, nb in pend:
            for fi, want_sex in enumerate(want):
                best = None
                for b in range(off, off + nb):
                    r = runner.read_typed(out, b, want_sex=want_sex)
                    # More than one window can host the same typed slot (two
                    # flies, two crops); take the most confident, never the
                    # first, so a near-empty window cannot claim the fly.
                    if r is not None and (best is None or r["exist"] > best["exist"]):
                        best = r
                if best is None:
                    continue
                pts = np.asarray(best["kp3d"], np.float32)
                kp3d[fi, t] = pts
                ok = np.isfinite(pts).all(axis=-1)
                if ok.any():
                    centroid[fi, t] = pts[ok].mean(axis=0)
                exist[fi, t] = best["exist"]
                sex_prob[fi, t] = best["sex_prob"]
                slot[fi, t] = best["slot"]
        pend, batch, rows = [], [], 0

    for t, fidx in enumerate(frame_list):
        imgs, present = reader(fidx)
        imgs = np.asarray(imgs)
        if W is None:
            H, W = int(imgs.shape[1]), int(imgs.shape[2])

        peaks, scores = detector.peaks(imgs)
        centres, _n_views, _score = lift_peaks_to_centres(
            peaks, scores, runner.cam_mats, min_views=min_views,
            max_resid_px=max_resid_px, max_animals=F)
        centres = cluster_centres(centres)
        wc, _assign = plan_windows(centres, merge_dist_units=merge_dist_units)

        if wc.shape[0]:
            src = CENTRE_DETECTED
            prev_centres = wc
        elif prev_centres is not None and prev_centres.shape[0]:
            wc, src = prev_centres, CENTRE_REUSED
        else:
            # No centre and no history: the row stays NaN and says so. See the
            # module docstring on why a zero centre would be worse than a miss.
            if progress is not None:
                progress(t + 1, N, (t + 1) / max(time.time() - t_start, 1e-9))
            continue

        if wc.shape[0] > runner.batch:
            if not warned_drop:
                warnings.warn(
                    f"coarse_pass: frame {fidx} planned {wc.shape[0]} windows but "
                    f"runner.batch={runner.batch}; dropping {wc.shape[0] - runner.batch} "
                    f"window(s) (only the first offending frame is reported)", RuntimeWarning,
                    stacklevel=2)
                warned_drop = True
            wc = wc[:runner.batch]
        if rows + wc.shape[0] > runner.batch:
            _read_batch()

        batch.append(runner.windows(imgs, present, wc))
        pend.append((t, rows, int(wc.shape[0])))
        rows += int(wc.shape[0])
        centre_source[:, t] = src
        n_windows[t] = int(wc.shape[0])

        if progress is not None:
            progress(t + 1, N, (t + 1) / max(time.time() - t_start, 1e-9))

    _read_batch()

    return {"frame": np.asarray(frame_list, np.int64),
            "kp3d": kp3d, "centroid": centroid, "exist": exist, "sex_prob": sex_prob,
            "slot": slot, "centre_source": centre_source, "n_windows": n_windows,
            "kp_names": list(runner.kp_names), "W": W, "H": H,
            "last_centres": prev_centres, "floor": floor}


def concat_tracks(chunks):
    """Glue consecutive `coarse_pass` results (the resume/chunk path).

    Frame-axis arrays are concatenated in order; `kp_names`, `W` and `H` must
    agree (they come from the same runner and the same videos, so a
    disagreement is a bug, not something to paper over).
    """
    chunks = [c for c in chunks if c["frame"].shape[0]]
    if not chunks:
        raise ValueError("nothing to concatenate")
    first = chunks[0]
    for c in chunks[1:]:
        if list(c["kp_names"]) != list(first["kp_names"]):
            raise ValueError("chunks disagree on kp_names -- different checkpoints?")
        if (c["W"], c["H"]) != (first["W"], first["H"]):
            raise ValueError(f"chunks disagree on frame size: {(first['W'], first['H'])} "
                             f"vs {(c['W'], c['H'])}")
    out = dict(first)
    out["frame"] = np.concatenate([c["frame"] for c in chunks], axis=0)
    for k in ("kp3d", "centroid", "exist", "sex_prob", "slot", "centre_source"):
        out[k] = np.concatenate([c[k] for c in chunks], axis=1)
    out["n_windows"] = np.concatenate([c["n_windows"] for c in chunks], axis=0)
    out["last_centres"] = chunks[-1].get("last_centres")
    return out


def pad_last_centres(centres, num_animals):
    """`(W,3)` window centres (W in `0..num_animals`, `plan_windows`'s own
    output) -> a FIXED `(num_animals,3)` array, NaN-padded, for npz storage.

    `coarse_pass`'s `last_centres` is one row per window, and the number of
    windows varies frame to frame (two flies can share one window), so it
    cannot be stored at a fixed shape as-is; NaN-padding to `num_animals` (an
    upper bound: `plan_windows` never returns more rows than valid input
    centres, which is capped at `num_animals`) round-trips through
    `unpad_last_centres` losslessly.
    """
    out = np.full((int(num_animals), 3), np.nan, np.float32)
    if centres is None:
        return out
    c = np.asarray(centres, np.float32)
    if c.shape[0]:
        n = min(c.shape[0], int(num_animals))
        out[:n] = c[:n]
    return out


def unpad_last_centres(padded):
    """Inverse of `pad_last_centres`: `(num_animals,3)` NaN-padded -> the
    `(W,3)` window-centres array `coarse_pass`'s `init_centres` expects (NaN
    rows dropped), or `None` if every row was NaN (no history to resume)."""
    padded = np.asarray(padded, np.float32)
    valid = ~np.isnan(padded).any(axis=1)
    c = padded[valid]
    return c if c.shape[0] else None


def write_coarse_tracks(path, tracks, features, cameras, *, session_dir, stride,
                        num_animals, cam_mats, meta_extra=None):
    """`coarse_tracks.npz` + `.meta.json` in the SAM3 schema plus the mvq fields.

    Args:
        path: output `.../coarse_tracks.npz` (the meta json is written beside
            it as `.../coarse_tracks.meta.json`, which is exactly where
            `coarse_pass_gates.load_tracks` looks).
        tracks: `coarse_pass`'s output (must carry `W`/`H`).
        features: `coarse_features`'s output.
        cameras: the CANONICAL camera order -- the same order `cam_mats` is in.
        cam_mats: (C,4,3) `ReprojectionTool.camera_matrices`. The reprojection
            uses the same `p_h @ M` convention as `ReprojectionTool.
            reproject_point`, on the float32 stack (a ~1e-2 px difference from
            the float64 path, which is nothing against a coarse centroid).
        meta_extra: extra keys for the meta json (checkpoint, timings, ...).

    The `.meta.json` is written FIRST (also tmp-name + `os.replace`), then the
    npz. A kill between the two leaves a meta describing a T that is one
    write ahead of the npz still on disk -- harmless, because `load_partial`
    only trusts the npz's own arrays for shape/count and uses the meta for
    run-identity fields (`cameras`, `stride`, `W`/`H`, `checkpoint`, ...) that
    do not change between writes of the SAME run. The reverse order (npz
    then meta, the previous behaviour) could leave a *complete* npz with NO
    meta at all if killed in between, and `concat_tracks`/`load_partial` need
    the meta's `W`/`H` to resume -- a partial file with no meta used to raise
    there instead of resuming.
    """
    cameras = [str(c) for c in cameras]
    C = len(cameras)
    cam_mats = np.asarray(cam_mats)
    if cam_mats.shape != (C, 4, 3):
        raise ValueError(f"cam_mats must be (C={C},4,3) `ReprojectionTool.camera_matrices` in "
                         f"the same canonical order as `cameras`, got {cam_mats.shape}")
    X3d = np.asarray(tracks["centroid"], np.float32)              # (F,N,3)
    F, T = X3d.shape[0], X3d.shape[1]
    if F != int(num_animals):
        raise ValueError(f"tracks carry {F} flies but num_animals={num_animals}")
    W, H = tracks.get("W"), tracks.get("H")
    if W is None or H is None:
        raise ValueError("tracks carry no frame size (W/H); the in-frame and border-distance "
                         "arrays are meaningless without it")
    W, H = int(W), int(H)

    with np.errstate(invalid="ignore", divide="ignore"):
        uv = _project_batch(X3d.reshape(-1, 3), cam_mats).reshape(F, T, C, 2)
    uv = np.transpose(uv, (0, 2, 1, 3)).astype(np.float32)         # (F,C,T,2)
    finite = np.isfinite(uv).all(axis=-1)
    with np.errstate(invalid="ignore"):
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= W - 1)
                  & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1) & finite)
    valid = finite & inside
    in_frame = inside.astype(np.int8)
    bx = np.minimum(uv[..., 0], (W - 1) - uv[..., 0])
    by = np.minimum(uv[..., 1], (H - 1) - uv[..., 1])
    border = np.where(valid, np.minimum(bx, by), np.nan).astype(np.float32)

    area = np.full((F, C, T), np.nan, np.float32)                  # no masks, no area
    area_med = np.full((F, T), np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)            # all-NaN slices are normal
        border_med = np.nanmedian(np.where(valid, border, np.nan), axis=1).astype(np.float32)
        sep2d_med = np.full(T, np.nan, np.float32)
        if F >= 2:
            both = valid[0] & valid[1]
            d2 = np.linalg.norm(uv[0] - uv[1], axis=-1)
            sep2d_med = np.nanmedian(np.where(both, d2, np.nan), axis=0).astype(np.float32)
    n_valid_cams = valid.sum(axis=1).astype(np.int16)
    sep3d = np.asarray(features["dist"], np.float32) if F >= 2 else np.array([], np.float32)
    coarse_frame = np.asarray(tracks["frame"], np.int64)
    last_centres = pad_last_centres(tracks.get("last_centres"), F)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Meta FIRST (see docstring): a kill between this and the npz write below
    # leaves a meta that is at most one write ahead of the npz on disk, which
    # `load_partial` tolerates (it trusts the npz's own arrays for shape, the
    # meta only for run-identity fields that are constant across writes of
    # the same run) -- never a complete npz with no meta, which used to make
    # a resume unable to recover W/H at all.
    floor = tracks.get("floor")
    meta_path = path.rsplit(".npz", 1)[0] + ".meta.json"
    meta = {"session_dir": str(session_dir), "stride": int(stride), "cameras": cameras,
            "W": W, "H": H, "num_animals": int(num_animals), "n_coarse": int(T),
            "coarse0": int(coarse_frame[0] // int(stride)) if T else 0,
            "source": "mvq",
            "floor": floor.as_dict() if isinstance(floor, FloorPlane) else None,
            "centre_source_counts": {
                "detected": int((tracks["centre_source"][0] == CENTRE_DETECTED).sum()),
                "reused": int((tracks["centre_source"][0] == CENTRE_REUSED).sum()),
                "none": int((tracks["centre_source"][0] == CENTRE_NONE).sum())} if T else {},
            "frac_trackable": [float(np.mean(features["trackable"][f])) for f in range(F)]}
    meta.update(meta_extra or {})
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    os.replace(tmp_meta, meta_path)

    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        # --- SAM3 coarse-pass schema ---
        coarse_frame=coarse_frame, cameras=np.array(cameras),
        area=area, centroid=uv, valid=valid, border_dist=border, in_frame=in_frame,
        area_med=area_med, border_med=border_med, n_valid_cams=n_valid_cams,
        X3d=X3d, sep3d=sep3d, sep2d_med=sep2d_med, filled=np.ones(T, bool),
        # --- mvq fields ---
        kp3d=np.asarray(tracks["kp3d"], np.float16),
        kp_names=np.array([str(n) for n in tracks["kp_names"]]),
        exist=np.asarray(tracks["exist"], np.float32),
        sex_prob=np.asarray(tracks["sex_prob"], np.float32),
        slot=np.asarray(tracks["slot"], np.int8),
        centre_source=np.asarray(tracks["centre_source"], np.int8),
        n_windows=np.asarray(tracks["n_windows"], np.int8),
        wing_angle_deg=np.asarray(features["wing_angle_deg"], np.float32),
        heading_deg=np.asarray(features["heading_deg"], np.float32),
        speed=np.asarray(features["speed"], np.float32),
        height=np.asarray(features["height"], np.float32),
        dist=np.asarray(features["dist"], np.float32),
        trackable=np.asarray(features["trackable"], bool),
        last_centres=last_centres)
    os.replace(tmp, path)
    return meta
