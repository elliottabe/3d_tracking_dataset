"""The one place that turns raw synced frames into mvq windows, runs the
lifter, and reads its typed slots back out in the pipeline's own format.

Spec: `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` §4.2. Shared by
the coarse pass (§4.3), the fine pass (§4.4) and `scripts/viz/mvq_bout_video.py`
(the figure gate this code was extracted from, after it worked end to end on
bout 28). Everything here is deliberately one class: the geometry, the slot
choice and the name permutation are exactly the three places where an
independent reimplementation would silently disagree with the training loader
or with the pipeline.

ORDER DISCIPLINE (CLAUDE.md; both traps were live bugs in this pipeline):

  * CAMERA axis. `cameras` is the CANONICAL order (`cfg.recording.cameras`,
    which is also the calibration glob order). The constructor asserts it
    equals `ReprojectionTool`'s own key order, so `crops`, `M`, `t_local`,
    `origin` and every returned `kp2d` share ONE camera axis. Anything a
    caller hands in (SAM3 mask npz arrays, for instance) must be permuted
    into it BY NAME first -- `tracking.bout_masks._camera_permutation`.
  * KEYPOINT axis. The model speaks the v12 DETECTOR order
    (`meta["keypoint_names"]`, == `configs/detector/vitpose_v3.yaml`
    kp_names). The pipeline's `kp2d.npz`/`kp3d.npz` speak MODEL/XML order
    (`configs/anatomy/*.yaml model.KP_NAMES`). They are the same 50 names in
    DIFFERENT orders. `to_pipeline` is the only conversion and it goes BY
    NAME through `detector_to_model_perm`; nothing here indexes a keypoint by
    integer.

WINDOW GEOMETRY is `data/v12_windows.py::V12WindowDataset._build`'s inference
half, reproduced exactly (`tests/test_lift_mvq.py` asserts pixel-for-pixel
equality against the dataset on the fixture): every camera is cropped 448x448
at `crop_origin` of the projection of ONE window-level 3D centre, and the
local offset of that same projection travels with the window as `t_local`.
A runner that placed its crops even slightly differently from the loader
would be a domain shift the model was never trained on, invisible to every
residual/confidence metric.

CONFIDENCES (spec §2). mvq's `conf` head is the D4RT confidence (~0.02-0.2),
not a probability, and the pipeline thresholds confidence at 0.3-0.5. The
pipeline-facing confidences are therefore the per-view visibility sigmoid
(2D) and its mean over cameras (3D); the raw head value is carried beside
them as `conf3d_mvq_raw` so nothing is lost.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings

import jax.numpy as jnp
import numpy as np

from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.data.v12_windows import CROP, _affine_np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.models.mvq.checkpoint import load_mvq_model
from jarvis_jax.models.mvq.model import assemble
from jarvis_jax.models.mvq.policy import EXIST_THRESH, policy_instance
from jarvis_jax.train.matching import (N_SLOTS, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN,
                                       SLOT_FEMALE, SLOT_MALE)
# `_fwd` is the SAME jitted forward `train_mvq.evaluate` uses, on purpose: a
# second wrapper would be a second compilation of the same graph and one more
# place for an inference/eval divergence to hide.
from jarvis_jax.train.train_mvq import _fwd, normalize_crops

# Files whose bytes identify a checkpoint, hashed in this order into the gate
# signature. `_METADATA` pins the parameter TREE (paths and shapes -- a
# different architecture is a different checkpoint) and `_CHECKPOINT_METADATA`
# carries orbax's commit timestamp, so retraining a run at the SAME path with
# the SAME architecture still moves the signature. `ckpt/<step>/` keeps its
# tree metadata one level down, under `model/`.
_DIGEST_FILES = ("_CHECKPOINT_METADATA", "_METADATA", os.path.join("model", "_METADATA"))

# How the masked-bout lifter decides WHICH written fly an instance is:
#
#   "sex"   the model's own typed slots -- fly0 is the FEMALE slot, fly1 the
#           MALE slot, each taken from whichever window reads it most
#           confidently. The original P3a behaviour.
#   "mask"  the HUMAN id review carried by the SAM3 masks -- fly0 is mask fly
#           0 and fly1 is mask fly 1, and the model is only asked "which
#           instance is ON this mask?".
#
# `mask` is the default because it is the pipeline's own precedence (human >
# mvq, `sexing.canonicalize_bout`) and because the sex head is measurably
# unreliable on at least one recording: on 2025_10_20_13_20_04 it types the
# female as a male, which NaN'd fly0 on 40% of that recording's frames and let
# the male track jump onto her body on 2.5% of them (up to 23% in a bout). See
# `.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md`.
IDENTITY_MODES = ("mask", "sex")
DEFAULT_IDENTITY = "mask"

# Mask `sex_meta.method` that carries a human identity decision (==
# `sexing.HUMAN_REVIEW_METHOD`; a plain string here so this module still needs
# no import of sexing).
HUMAN_REVIEW_METHOD = "human_id_review"

# Mask assignment is RELATIVE ("which mask is this instance nearest?"), not an
# absolute radius around each mask centre. An absolute radius assumes the mask
# centre is a good estimate of the animal's centre, and on 20_04 the FEMALE's
# SAM mask is not: over the 30-bout re-lift her mask's DLT centre sits 13-35
# units (median) from her body as located by the model, on only 3-4 valid mask
# cameras, while the male's is 5-6 units off on all 7. A 10-unit radius
# therefore threw away instances that were correctly detected -- fly0 went to
# 1.00 NaN in bout 8, 0.99 in bout 19, 0.95 in bout 26 -- even though those
# same instances were ~50 units from the OTHER mask and so never ambiguous.
#
# So: an instance belongs to the mask it is nearer, provided it wins by
# `MASK_ASSIGN_MARGIN_UNITS` (below that the frame is ambiguous and geometry
# abstains) and is not absurdly far from it (`MASK_ASSIGN_MAX_UNITS`).
MASK_ASSIGN_MARGIN_UNITS = 8.0     # 0.8 mm; how much nearer its own mask an
                                   # instance must be than the other mask
MASK_ASSIGN_MAX_UNITS = 60.0       # 6 mm; a garbage cap only -- the 448-px
                                   # window's half-width is ~28 units, so an
                                   # instance beyond this is not in the crop
                                   # either mask placed

# WHICH WINDOW a fly is read out of, when more than one of them holds it.
#
# A frame with two separated masks plans TWO 448-px crops, one centred on each
# mask, and at courtship distances BOTH crops usually contain BOTH animals. The
# mask-assignment rule above is about geometry only ("which mask is this
# instance nearest?"), so before this preference existed a fly was taken from
# whichever window read it most confidently -- routinely the OTHER fly's crop.
#
# The 3D survives that: it is the same animal, triangulated from the same seven
# views, and the IK fits it. The CONFIDENCE does not. A fly at the edge of a
# crop centred on its partner has half its body at or past the crop boundary,
# so the model's per-view visibility for those keypoints is ~0 and `conf3d`
# (its mean over cameras) collapses with it. Measured 2026-09-05 on the p3b
# campaign: bout 10 of Session1/2026_04_02_14_54_28 read the FEMALE out of the
# MALE's window on 293 of 303 frames, and her head/thorax conf3d was 0.10 there
# against 0.71 from her own window; across 12 sampled p3b bouts, 7 female
# tracks sat at front-group conf3d <= 0.4 while the male read ~0.99 (and the
# older r2 lift shows the male doing the same when he is read from HER window).
# conf3d is a GATE downstream -- the offsets sampler needs min conf >= 0.7, the
# rigid-edge repair 0.5, the side-by-side display hides low-conf 2D points
# ("half the fly disappears"), the pseudo-label export thresholds on it -- so
# those frames were being discarded for a cropping accident.
#
#   "own"  (default) a fly is taken from the window centred on ITS OWN mask
#          whenever that window holds a candidate passing `exist_thresh` and
#          the same relative-nearest mask test; only when its own window holds
#          none does it fall back to instances from the other windows.
#   "any"  the pre-2026-09-05 rule: the best candidate from any window.
#
# It is a PREFERENCE, not a filter: nothing is NaN'd for being in the wrong
# crop, and `mvq_meta.json` records per frame which flies were read from their
# own window (`per_frame.own_window`) and how often each fell back
# (`n_from_other_window`). A MERGED window (two masks within
# `merge_dist_units` share one crop) is every fly's own window, so the two
# modes are identical there by construction.
WINDOW_PREF_MODES = ("own", "any")
DEFAULT_WINDOW_PREF = "own"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def resolved_step(step):
    """`step` as a CONCRETE int, or None for a `final/` dir.

    `"latest"` is refused by name. It is not a checkpoint identity -- it
    names a different step every time the training job saves -- so a gate
    signature built from it would compare EQUAL to a kp3d.npz produced by
    different weights, which is the exact failure the signature exists to
    catch. `MVQRunner` resolves it once at construction (see `step_label`)
    and stores the int; a config has to carry the int.
    """
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        raise ValueError(
            f"mvq.step must be a concrete int; resolve {step!r} before writing the config "
            f"(MVQRunner resolves 'latest' at construction and exposes it as `step`/"
            f"`step_label`) -- a signature built from a moving step would accept a "
            f"kp3d.npz produced by different weights") from None


def checkpoint_dir(checkpoint, step=None):
    """The directory a `(checkpoint, step)` pair actually loads from.

    `step is None`: `checkpoint` IS a `final/` dir. Otherwise it is the RUN
    dir and the weights live in `<run>/ckpt/<step>` -- the same convention as
    `models/mvq/checkpoint.py::load_mvq_model`.
    """
    step = resolved_step(step)
    return str(checkpoint) if step is None else os.path.join(str(checkpoint), "ckpt", str(step))


def checkpoint_sha256(checkpoint, step=None, *, n=16):
    """First `n` hex chars of sha256 over the checkpoint's metadata files.

    Hashing the parameter shards themselves would be gigabytes per call (this
    runs on every bout, and again on every staleness check); the metadata is
    a few hundred KB and moves whenever the tree or the save does.
    """
    d = checkpoint_dir(checkpoint, step)
    h = hashlib.sha256()
    found = []
    for rel in _DIGEST_FILES:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(rel.encode()); h.update(f.read())
            found.append(rel)
    if not found:
        raise FileNotFoundError(
            f"{d} carries none of {list(_DIGEST_FILES)} -- it is not an orbax "
            f"checkpoint directory, so the Stage-B gate signature cannot name "
            f"which weights produced a kp3d.npz")
    return h.hexdigest()[:n]


def concat_windows(ws):
    """Concatenate per-frame `windows()` dicts into one batch.

    A batch spans FRAMES -- each frame's images are read once and every fly
    in it becomes one window -- so callers build small dicts and glue them
    here. Windows that disagree about carrying a `prompt_mask` are REFUSED:
    concatenating them could only be done by inventing an empty prompt for
    the ones without, which is a prompted forward following nothing.
    """
    ws = [w for w in ws if w["crops"].shape[0]]
    if not ws:
        raise ValueError("nothing to concatenate")
    keys = set(ws[0])
    for w in ws[1:]:
        if set(w) != keys:
            raise ValueError(f"windows disagree on their keys ({sorted(keys)} vs {sorted(w)}) -- "
                             f"a prompt_mask present in some rows and absent in others cannot be "
                             f"batched")
    return {k: np.concatenate([w[k] for w in ws], axis=0) for k in keys}


def resolved_identity(identity):
    """`identity` as one of `IDENTITY_MODES` (None -> `DEFAULT_IDENTITY`).

    Refuses anything else by name rather than falling back to a default: a
    typo'd mode that silently became "sex" would produce a gate string saying
    one thing while the lifter did another, and nothing downstream re-derives
    which rule assigned the flies.
    """
    if identity is None:
        return DEFAULT_IDENTITY
    identity = str(identity)
    if identity not in IDENTITY_MODES:
        raise ValueError(f"mvq identity must be one of {list(IDENTITY_MODES)}, got "
                         f"{identity!r}")
    return identity


def resolved_window_pref(window_pref):
    """`window_pref` as one of `WINDOW_PREF_MODES` (None -> `DEFAULT_WINDOW_PREF`).

    Refused by name rather than defaulted, for the same reason `identity` is: a
    typo'd mode that silently became "any" would stamp a gate string saying
    "own" on a lift that ran the old rule, and nothing downstream re-derives
    which window each fly came from.
    """
    if window_pref is None:
        return DEFAULT_WINDOW_PREF
    window_pref = str(window_pref).strip().lower()
    if window_pref not in WINDOW_PREF_MODES:
        raise ValueError(f"mvq window_pref must be one of {list(WINDOW_PREF_MODES)}, "
                         f"got {window_pref!r}")
    return window_pref


def resolve_mask_identity(identity, mask_sex_meta, *, where=""):
    """The identity rule ONE bout will actually run, given what its masks carry.

    `identity="mask"` only means something when the masks carry a HUMAN id
    review; a bout whose masks do not falls back to the sex head. That
    fallback has to be resolved BEFORE the gate string is built, by every
    caller that builds one -- `lift_masked_bout`, `scripts/mvq_lift_bout.py`
    and `slurm_bout_array`'s `--mvq-lift skip` verification. Stamping the
    REQUESTED mode on a bout that ran the other rule would make
    `bout_lift_is_current` answer True for a lift that is not the one the
    config asks for, so a re-lift after a review is canonicalized into those
    masks would be skipped and the bout would keep its sex-head identity
    inside a run labelled `mask`.

    Returns `(mode, fallback_message or None)` -- the message is a warning the
    caller emits (a lifter warns, a submitter prints), never a silent switch.

    Raises:
        ValueError: `identity="mask"` with a review that puts the male at mask
            slot 0. `fly{f}` IS mask fly `f` on that route, so `male_fly: 1`
            would name the wrong fly. Raised here, before any work, rather
            than after a bout has been lifted.
    """
    identity = resolved_identity(identity)
    if identity != "mask":
        return identity, None
    ms = mask_sex_meta or {}
    tag = f"{where}: " if where else ""
    if str(ms.get("method")) != HUMAN_REVIEW_METHOD:
        return "sex", (
            f"{tag}identity='mask' was asked for, but the masks' sex_meta.method is "
            f"{ms.get('method')!r}, not {HUMAN_REVIEW_METHOD!r} -- there is no human "
            f"identity decision to honour, so this bout runs the sex head "
            f"(identity='sex') and is GATED as such: a run asking for the 'mask' gate "
            f"will not accept it, and re-canonicalizing a review into these masks makes "
            f"it stale so the re-lift actually happens.")
    if ms.get("male_slot") != 1:
        raise ValueError(
            f"{tag}identity='mask' writes fly{{f}} from MASK fly {{f}} and a sex.json "
            f"saying male_fly=1, but these masks' human review says male = mask slot "
            f"{ms.get('male_slot')!r}. Re-run scripts/canonicalize_sam_masks.py so the "
            f"male is mask slot 1, or lift this bout with identity='sex' -- writing it "
            f"as-is would name the wrong fly the male.")
    return "mask", None


def mvq_gate_signature(checkpoint, *, step=None, exist_thresh=None, identity=None,
                       containment=None, window_pref=None):
    """The Stage-B `gates` payload for an mvq-lifted kp3d.npz.

    With `pipeline.lifter: mvq` the DLT gates (view-conf, mask-agreement,
    wing-collapse, rigid-repair) never run -- Stages A and B are replaced by
    the lifter -- so what determines the file's contents is WHICH CHECKPOINT
    produced it, the existence threshold that decided which frames are NaN,
    and WHICH RULE decided which written fly each instance is (`identity`;
    the two modes disagree on ~40% of one recording's frames, so a run
    switched between them must not reuse the other's bouts).
    `scripts/run_bout.py::stage_b_gate_signature` returns
    `json.dumps(this, sort_keys=True)` for such a run, so a bout produced by
    one checkpoint is refused after the config points at another, exactly as
    a changed DLT gate is.

    `identity` must be the mode the bout ACTUALLY RAN, not the one the config
    asked for: a bout whose masks carry no human review falls back to "sex"
    for itself alone (`resolve_mask_identity`), and gating it as "mask" would
    make `bout_lift_is_current` accept a sex-head lift for a mask-identity
    run -- and skip the re-lift once a review is canonicalized into those
    masks. Callers that build this string per bout therefore resolve first.

    The consequence is deliberate: with `mvq.identity: mask` in the config,
    `run_bout.stage_b_gate_signature` (which has no bout index and never opens
    the mask npz) computes the "mask" string, so a fallback bout is REFUSED at
    Stage B rather than silently accepted. The fix for such a bout is to give
    its masks the human review, or to run that recording with
    `mvq.identity=sex`.

    `containment` is the per-keypoint mask-containment filter
    (`mask_containment_filter`), which DELETES keypoints from kp3d.npz -- so
    it changes the file's contents and belongs here for the same reason
    `exist_thresh` does. It is reported as the EFFECTIVE bool: the filter
    needs both flies' masks and the human id review those masks carry, so it
    can only run under `identity="mask"` and is False otherwise regardless of
    what was asked for. Its THRESHOLDS (`min_views`, `own_margin`,
    `max_step_units`) are recorded in `mvq_meta.json` rather than enrolled
    here -- the same choice already made for `collapse_dist_units` and the
    mask-assignment margins, and for the same reason: re-tuning a threshold
    must not charge a 12-minute STAC re-solve per bout-fly on every sweep
    point, while a lift that ran the filter and one that did not are
    genuinely different files.

    `window_pref` (`WINDOW_PREF_MODES`) is which window a fly is read from when
    several hold it. It does not move the 3D, but it decides the per-view
    visibilities -- and so `conf3d`, which IS in kp3d.npz and gates the offsets
    sampler, the rigid-edge repair, the display and the pseudo-label export.
    Same effective-value discipline as `containment`: the preference only
    exists inside the mask-identity assignment, so it is reported as "any"
    whenever `identity` is not "mask", where the rule genuinely does not run.
    Adding the key at all makes every lift produced before 2026-09-05 stale,
    which is intended -- those lifts carry the cross-window confidences.
    """
    if not checkpoint:
        raise ValueError("pipeline.lifter=mvq needs mvq.checkpoint set -- the Stage-B gate "
                         "signature must name the weights that produced kp3d.npz")
    step = resolved_step(step)                    # refuses "latest" by name
    # validated whatever the identity mode is (a typo must never pass), then
    # reported as the no-op it is when the mask assignment does not run
    window_pref = resolved_window_pref(window_pref)
    return {"lifter": "mvq",
            "checkpoint": os.path.abspath(str(checkpoint)),
            "step": "final" if step is None else step,
            "sha256": checkpoint_sha256(checkpoint, step),
            "exist_thresh": float(EXIST_THRESH if exist_thresh is None else exist_thresh),
            "identity": resolved_identity(identity),
            "containment": bool(resolved_containment(containment)
                                and resolved_identity(identity) == "mask"),
            "window_pref": (window_pref if resolved_identity(identity) == "mask"
                            else "any")}


def mvq_gate_string(checkpoint, *, step=None, exist_thresh=None, identity=None,
                    containment=None, window_pref=None):
    """`mvq_gate_signature` as the stable string stored inside kp3d.npz."""
    return json.dumps(mvq_gate_signature(checkpoint, step=step, exist_thresh=exist_thresh,
                                         identity=identity, containment=containment,
                                         window_pref=window_pref),
                      sort_keys=True)


def slot_read(out, bi, slot):
    """One instance slot of one window of an `infer` output, as a plain dict.

    The single place the per-slot arrays are named, so `MVQRunner.read_typed`
    (which picks a slot from the model's TYPING) and `pick_mask_pair` (which
    picks one from the mask GEOMETRY) hand their callers byte-identical
    payloads -- a second, hand-written copy of this dict is exactly where a
    `vis`/`conf_raw` mix-up would hide.
    """
    slot = int(slot)
    return {"slot": slot,
            "kp3d": out["kp3d"][bi, slot],
            "kp2d": out["kp2d"][bi, slot],
            "vis": out["vis"][bi, slot],
            "conf_raw": out["conf_raw"][bi, slot],
            "exist": float(out["exist"][bi, slot]),
            "sex_prob": float(out["sex_prob"][bi, slot])}


class MVQRunner:
    """frames -> windows -> world keypoints -> pipeline format, for ONE checkpoint.

    `run_dir_or_final` / `step` / `attn_impl` are `load_mvq_model`'s (a
    `final/` dir with no step, else the run dir plus an int step or
    `"latest"`; `attn_impl="xla"` to run a cuDNN-trained run on CPU).
    `"latest"` is resolved to a concrete int HERE, once, so the step that is
    loaded is the step that is reported and stamped into the gate signature --
    a live training run's `latest` can move between two resolutions.

    `cameras` is the canonical camera order and `calib_dir` the matching
    calibration directory; `batch` is the fixed batch the forward is compiled
    for (short batches are padded and the padding dropped).
    """

    def __init__(self, run_dir_or_final, *, step=None, attn_impl=None, calib_dir, cameras,
                 batch=32, exist_thresh=EXIST_THRESH, identity=None,
                 containment=None, window_pref=None):
        self.checkpoint = os.path.abspath(str(run_dir_or_final))
        # Which rule the caller will use to name the written flies. The runner
        # does not apply it (that is `lift_masked_bout`'s job); it is carried
        # here so `gates_string()` -- the string stamped into kp3d.npz -- names
        # the mode, and so a runner cannot be shared between two callers that
        # disagree about it.
        self.identity = resolved_identity(identity)
        # Only so `gates_string()` (which the CLI prints and
        # `slurm_bout_array` compares) names the same filter the lift will
        # run; `lift_masked_bout` resolves its own, per bout, against that
        # bout's masks and its own `mask_store`.
        self.containment = resolved_containment(containment)
        # Likewise carried only so `gates_string()` names the rule the lift
        # will run (`lift_masked_bout` applies it); a runner and a lift that
        # disagree about it would stamp a gate string for the other one.
        self.window_pref = resolved_window_pref(window_pref)
        if step == "latest":
            import orbax.checkpoint as ocp
            mgr = ocp.CheckpointManager(os.path.abspath(os.path.join(self.checkpoint, "ckpt")),
                                        options=ocp.CheckpointManagerOptions(read_only=True))
            step = int(mgr.latest_step())
        self.step = None if step is None else int(step)
        self.step_label = "final" if self.step is None else self.step
        self.model, self.meta = load_mvq_model(self.checkpoint, step=self.step,
                                               attn_impl=attn_impl)
        self.kp_names = list(self.meta["keypoint_names"])
        self.K = len(self.kp_names)
        self.I = int(self.meta["model"]["n_instances"])
        self.cameras = [str(c) for c in cameras]
        self.rt = ReprojectionTool(str(calib_dir))
        if list(self.rt.cameras.keys()) != self.cameras:
            raise ValueError(
                f"calibration glob order {list(self.rt.cameras.keys())} != canonical camera "
                f"order {self.cameras}; every camera axis in this runner assumes they are the "
                f"same, and a mismatch plots one camera's keypoints on another's image")
        self.C = len(self.cameras)
        self.cam_mats = np.asarray(self.rt.camera_matrices, np.float32)      # (C,4,3)
        self.M, self.t = _affine_np(self.rt.camera_matrices)                 # (C,2,3),(C,2) f64
        self.batch = int(batch)
        self.exist_thresh = float(exist_thresh)
        self._gates = None

    # ------------------------------------------------------------------ windows
    def windows(self, frames, present, centres, prompt_mask=None):
        """`V12WindowDataset._build`'s inference geometry, vectorised over centres.

        `frames` (C,H,W,3) uint8 RGB in `self.cameras` order (what
        `predict.synced_reader.read_window` yields), `present` (C,) bool,
        `centres` (B,3) in WORLD units. `prompt_mask`, when given, is a
        callable `(b, c) -> (H,W) bool` returning that window's host mask in
        FULL-FRAME coordinates, cropped here at the same origin as the image
        (a callable rather than an array because a whole bout's unpacked SAM3
        masks are ~12 GB per fly -- see `mvq_bout_video.BoutMaskStore`).

        Returns crops (B,1,C,448,448,3) u8, cam_valid (B,1,C) bool,
        M (B,C,2,3) f32, t_local (B,1,C,2) f32, origin (B,C,2) i32,
        centres (B,3) f32 -- the singleton axis is the model's T (window
        length); this runner only ever builds T=1 windows.
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[0] != self.C:
            raise ValueError(f"frames must be (C={self.C},H,W,3) in camera order {self.cameras}, "
                             f"got {frames.shape}")
        H, W = int(frames.shape[1]), int(frames.shape[2])
        centres = np.atleast_2d(np.asarray(centres, np.float64))
        if centres.shape[-1] != 3:
            raise ValueError(f"centres must be (B,3) world coordinates, got {centres.shape}")
        B = centres.shape[0]
        present = np.asarray(present, bool).reshape(-1)
        if present.shape[0] != self.C:
            raise ValueError(f"present must be (C={self.C},), got {present.shape}")

        uv = np.einsum("cij,bj->bci", self.M, centres) + self.t[None]        # (B,C,2)
        origin = np.zeros((B, self.C, 2), np.int32)
        crops = np.zeros((B, 1, self.C, CROP, CROP, 3), np.uint8)
        prompt = None if prompt_mask is None else np.zeros((B, 1, self.C, CROP, CROP), bool)
        for b in range(B):
            for c in range(self.C):
                x0, y0 = crop_origin([uv[b, c, 0], uv[b, c, 1], 0, 0], W, H, CROP)
                origin[b, c] = (x0, y0)
                crops[b, 0, c] = frames[c][y0:y0 + CROP, x0:x0 + CROP]
                if prompt is not None:
                    prompt[b, 0, c] = np.asarray(prompt_mask(b, c))[y0:y0 + CROP, x0:x0 + CROP]
        out = {"crops": crops,
               "cam_valid": np.broadcast_to(present, (B, self.C))[:, None].copy(),
               "M": np.broadcast_to(self.M.astype(np.float32)[None], (B, self.C, 2, 3)).copy(),
               "t_local": (uv - origin).astype(np.float32)[:, None],
               "origin": origin,
               "centres": centres.astype(np.float32)}
        if prompt is not None:
            out["prompt_mask"] = prompt
        return out

    def has_mask(self, w):
        """(B,) -- does this window carry a prompt in a camera that is itself
        valid? The SAME definition `train_mvq.evaluate` uses to decide
        `prompt_on`: a mask in a dropped camera is not a usable prompt, since
        the prompt token's mean is gated by `cam_valid` inside the model."""
        if "prompt_mask" not in w:
            return np.zeros(w["crops"].shape[0], bool)
        return np.asarray((w["prompt_mask"].any((3, 4)) & w["cam_valid"]).any((1, 2)))

    # ------------------------------------------------------------------ forward
    def infer(self, w, *, prompt_on=None):
        """One forward on a windows dict, padded to `self.batch`.

        `prompt_on` (bool or (B,) bool, default all-False) selects the
        prompted branch per window and REQUIRES `w["prompt_mask"]`. Padding
        repeats the last row (`train_mvq.evaluate`'s convention) and is
        dropped from every returned array, so a short batch reads exactly as
        a full one.

        Returns numpy, T squeezed out: kp3d (B,I,K,3) WORLD (NaN where the
        slot does not exist or the frame has no valid camera), kp2d
        (B,I,C,K,2) FULL-FRAME px, vis (B,I,C,K) per-view visibility sigmoid,
        exist (B,I), sex_prob (B,I) = P(female), conf_raw (B,I,K) the D4RT
        confidence head, xyz (B,I,K,3) ROI-LOCAL (what `policy_slot` scores).
        """
        B0 = int(w["crops"].shape[0])
        if B0 > self.batch:
            raise ValueError(f"{B0} windows > runner batch {self.batch}; build at most "
                             f"`batch` windows per infer call")
        if prompt_on is None:
            on = np.zeros(B0, bool)
        else:
            if "prompt_mask" not in w:
                raise ValueError("prompt_on was given but the window carries no prompt_mask; "
                                 "the prompted branch would follow an all-zero mask")
            on = np.broadcast_to(np.asarray(prompt_on, bool), (B0,))
        pad = self.batch - B0
        _pad = (lambda a: a) if pad == 0 else \
            (lambda a: np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0))
        crops, cam_valid = _pad(w["crops"]), _pad(w["cam_valid"])
        origin = _pad(w["origin"]).astype(np.float32)
        centres = _pad(w["centres"]).astype(np.float32)
        prompt = _pad(w["prompt_mask"]) if "prompt_mask" in w else \
            np.zeros((self.batch, 1, self.C, CROP, CROP), bool)
        out = _fwd(self.model, normalize_crops(jnp.asarray(crops)), jnp.asarray(cam_valid),
                   jnp.asarray(_pad(w["M"])), jnp.asarray(_pad(w["t_local"])),
                   jnp.asarray(prompt), jnp.asarray(_pad(on)))
        kp3d, conf3d, kp2d, sex_prob = assemble(out, centres, origin,
                                                exist_thresh=self.exist_thresh,
                                                cam_valid=np.asarray(cam_valid))
        return {"kp3d": kp3d[:B0, :, 0],
                "kp2d": kp2d[:B0, :, 0],
                "vis": _sigmoid(np.asarray(out["vis_logit"]))[:B0, :, 0].astype(np.float32),
                "exist": _sigmoid(np.asarray(out["exist_logit"]))[:B0].astype(np.float32),
                "sex_prob": sex_prob[:B0],
                "conf_raw": conf3d[:B0, :, 0],
                "xyz": np.asarray(out["xyz"])[:B0, :, 0]}

    # ------------------------------------------------------------------ reading slots
    def policy_slot(self, out, bi, *, prompted, has_mask):
        """`models/mvq/policy.py::policy_instance` for window `bi` -- the
        SHARED inference-time slot choice (slot 0 when there is a prompt to
        follow, else the typed candidate nearest the ROI centre, None on a
        miss). Kept here so no caller re-adds the T axis by hand, and so the
        policy filters on THIS runner's `exist_thresh`: with the module
        default it could otherwise return the very slot `read_typed` had just
        refused as too weak."""
        return policy_instance(out["exist"][bi], out["xyz"][bi][:, None],
                               prompted=prompted, has_mask=has_mask,
                               exist_thresh=self.exist_thresh)

    def read_typed(self, out, bi, want_sex):
        """The typed slot for one sex of window `bi`, or None below threshold.

        `want_sex` is a `train.matching` sex code and maps to the FIXED P3a
        slot table: 0 female -> slot 1, 1 male -> slot 2 (slot 0 is the
        prompted slot, 3 is "other"). No fallback and no guessing for a known
        sex -- a caller that wants the untyped nearest-candidate rule asks
        `policy_slot` for it explicitly, so a frame reported as "the female"
        is never quietly some other slot.

        SEX_UNKNOWN (-1) is the SINGLE-FLY rule (spec §4.2): take whichever
        typed slot exists -- the higher existence when both clear the
        threshold -- and REPORT its sex (`sex_prob`, `slot`) rather than
        assuming one. A single-fly recording has no second animal to
        disambiguate against, so demanding a particular typed slot there
        would drop every frame the model happened to type the other way.
        """
        if self.I != N_SLOTS:
            raise ValueError(
                f"this checkpoint has {self.I} instance slots, not the typed {N_SLOTS} "
                f"(0 prompted, 1 female, 2 male, 3 other) -- a legacy untyped run's slots "
                f"have no fixed meaning, so read_typed cannot name one")
        want = int(want_sex)
        if want not in (SEX_FEMALE, SEX_MALE, SEX_UNKNOWN):
            raise ValueError(f"want_sex must be {SEX_FEMALE} (female), {SEX_MALE} (male) or "
                             f"{SEX_UNKNOWN} (unknown -- the single-fly rule), got {want_sex!r}")
        if want == SEX_UNKNOWN:
            live = [s for s in (SLOT_FEMALE, SLOT_MALE)
                    if float(out["exist"][bi, s]) >= self.exist_thresh]
            if not live:
                return None
            slot = max(live, key=lambda s: float(out["exist"][bi, s]))
        else:
            slot = SLOT_FEMALE if want == SEX_FEMALE else SLOT_MALE
            if float(out["exist"][bi, slot]) < self.exist_thresh:
                return None
        return slot_read(out, bi, slot)

    # ------------------------------------------------------------------ pipeline format
    def to_pipeline(self, kp3d, kp2d, vis, conf_raw, model_names):
        """One fly's bout, in the pipeline's own arrays and keypoint order.

        In: kp3d (T,K,3), kp2d (T,C,K,2), vis (T,C,K), conf_raw (T,K), all in
        the model's own (detector) keypoint order. `model_names` is the
        pipeline's order (`cfg.model.KP_NAMES`). Out: the same arrays permuted
        BY NAME, the pipeline-facing confidences (`conf` = per-view
        visibility, `conf3d` = its mean over cameras -- spec §2, since the
        mvq confidence head is a D4RT score of ~0.02-0.2 and the pipeline
        thresholds at 0.3-0.5), the raw head value as `conf3d_mvq_raw`, and
        the Stage-B `gates` signature.
        """
        # local import: predict_2d pulls the ViTPose checkpoint loader, which
        # this module has no other reason to bring in.
        from jarvis_jax.tracking.predict_2d import detector_to_model_perm
        model_names = [str(n) for n in model_names]
        perm = detector_to_model_perm(self.kp_names, model_names)
        vis, kp2d = np.asarray(vis), np.asarray(kp2d)
        # `conf3d` averages over the CAMERA axis. A `vis` of the wrong rank
        # would average over time (or over nothing) and still come back with
        # a plausible (T,K) shape -- a confidence that is silently the wrong
        # statistic, which nothing downstream could notice.
        if vis.ndim != 3:
            raise ValueError(f"vis must be (T,C,K) per-view visibility, got {vis.shape}; "
                             f"conf3d is its mean over the CAMERA axis")
        if kp2d.shape[:-1] != vis.shape:
            raise ValueError(f"kp2d {kp2d.shape} and vis {vis.shape} disagree: kp2d must be "
                             f"(T,C,K,2) over the same frames, cameras and keypoints")
        vis = vis[..., perm]
        return {"kp2d": kp2d[..., perm, :],
                "conf": vis,
                "kp3d": np.asarray(kp3d)[..., perm, :],
                "conf3d": vis.mean(axis=-2),          # over CAMERAS, not time
                "conf3d_mvq_raw": np.asarray(conf_raw)[..., perm],
                "kp_names": np.array(model_names),
                "cameras": np.array(self.cameras),
                "gates": self.gates_signature()}

    def gates_signature(self):
        """This checkpoint's Stage-B `gates` payload (see `mvq_gate_signature`).

        Computed once and cached: it hashes the checkpoint's metadata files,
        and every bout-fly written by a pass asks for it.
        """
        if self._gates is None:
            self._gates = mvq_gate_signature(
                self.checkpoint, step=self.step, exist_thresh=self.exist_thresh,
                identity=self.identity,
                containment=getattr(self, "containment", False),
                window_pref=getattr(self, "window_pref", None))
        return dict(self._gates)

    def gates_string(self):
        """`gates_signature` as the string stored inside kp3d.npz -- byte-equal
        to what `run_bout.py::stage_b_gate_signature` computes for this run."""
        return json.dumps(self.gates_signature(), sort_keys=True)


# =============================================================================
# Masked-bout lifter (Task 7): SAM3-detected bouts -> pipeline bout dirs
# =============================================================================
# The coarse pass (§4.3) finds its window centres with CenterDetect. A
# courtship bout that SAM3 has already segmented has better centres for free:
# the DLT of each fly's per-camera mask centroid. Everything downstream of the
# centre -- the window geometry, the forward, the typed-slot read, the name
# permutation -- is the SAME code the coarse pass uses, so the two passes
# cannot drift apart.
#
# IDENTITY (`identity=`, see IDENTITY_MODES at the top of this module).
#
#   identity="sex"   the mask slots are NOT the identity: `fly0` is the model's
#                    FEMALE typed slot and `fly1` its MALE typed slot, per
#                    frame, from the sex head; the masks only place the crop.
#                    Writes `sex.json` with `method = "mvq_sex_head"`.
#   identity="mask"  (default) the masks' HUMAN id review IS the identity:
#                    `fly0` is mask fly 0 and `fly1` is mask fly 1, and the
#                    model is asked only WHICH INSTANCE IS ON THIS MASK.
#                    Writes `sex.json` with `method = "mask_human_id_review"`.
#
# Either way `canonicalize_bout` must not then re-decide the bout from the
# wing-song CV -- both methods are in its authoritative, no-swap set.
#
# Why "mask" is the default: the human id review outranks the model in this
# pipeline's own precedence, and the sex head is measurably wrong on at least
# one recording (20_04 -- see the module header constants and
# `.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b/female-miss-diagnosis.md`).

MVQ_SEX_METHOD = "mvq_sex_head"      # == sexing.MVQ_SEX_METHOD (kept as a plain
                                     # string here so this module needs no
                                     # import of sexing, which imports nothing
                                     # from jax and should stay that way)
MASK_ID_SEX_METHOD = "mask_human_id_review"    # == sexing.MASK_ID_SEX_METHOD

# Median per-keypoint 3D distance below which the two typed slots are judged to
# be the SAME physical fly (world units; 3.0 == 0.3 mm). Two real flies -- even
# a stacked mating pair -- are never that close over most of their 50
# landmarks: the male's and female's leg tips and wing veins stay body-lengths
# apart even when their thoraxes touch. The same instance read twice is ~0.
# NOT a knob: it is deliberately not in the Stage-B gate signature, so a run
# that changed it would be indistinguishable from one that did not. The value
# used is recorded in mvq_meta.json.
COLLAPSE_DIST_UNITS = 3.0


def pick_typed_pair(out, off, nb, runner, collapse_dist_units=COLLAPSE_DIST_UNITS):
    """Read the FEMALE (fi=0) and MALE (fi=1) typed slots for one pending
    frame's windows `[off, off + nb)` of an `infer` output, apply the
    collapse guard, and return the survivors.

    Shared by `lift_masked_bout._flush` (bout lift) and
    `coarse_track.coarse_pass._read_batch` (recording-wide coarse pass) so
    the two routes cannot silently drift apart on this rule -- the review
    that asked for this extraction found the guard present in one and
    missing from the other (`coarse_track._read_batch` had no collapse
    check at all).

    For each typed slot, take the window with the HIGHEST existence for that
    slot (never the first), so a near-empty window cannot claim the fly. Then:

    COLLAPSE GUARD. The two typed slots are chosen INDEPENDENTLY, so nothing
    above stops both of them reading the SAME physical fly -- most likely on
    a merged window, which is exactly the mounting frames. Two real flies are
    never within `collapse_dist_units` over most of their 50 keypoints (a
    stacked mating pair still has ~2 body-lengths of separated leg/wing
    landmarks); the same instance read twice is ~0. So a frame whose two
    slots agree that closely keeps only the more confident one, rather than
    shipping a duplicated fly that every jitter, confidence and residual
    metric would rate as excellent. Ties keep the FEMALE (fi=0): she is the
    fly this pipeline loses frames on, and on a tie the two reads are
    interchangeable.

    Returns:
        picks: {fi: (read_typed_result, absolute_window_index)} for fi in
            {0, 1} that a typed slot was found for -- 0, 1 or 2 entries.
        collapsed: bool, whether the collapse guard fired this frame.
        dropped_fi: the fi (0 or 1) the guard NaN'd, or None if it did not
            fire (or fewer than 2 typed slots were read at all).
    """
    picks = {}
    for fi, want_sex in enumerate((SEX_FEMALE, SEX_MALE)):
        best, best_b = None, -1
        for b in range(off, off + nb):
            r = runner.read_typed(out, b, want_sex=want_sex)
            if r is not None and (best is None or r["exist"] > best["exist"]):
                best, best_b = r, b
        if best is not None:
            picks[fi] = (best, best_b)

    collapsed, dropped_fi = False, None
    if len(picks) == 2:
        a3 = np.asarray(picks[0][0]["kp3d"], np.float64)
        b3 = np.asarray(picks[1][0]["kp3d"], np.float64)
        d = np.linalg.norm(a3 - b3, axis=-1)
        d = d[np.isfinite(d)]
        if d.size and float(np.median(d)) < float(collapse_dist_units):
            collapsed = True
            dropped_fi = 1 if picks[1][0]["exist"] <= picks[0][0]["exist"] else 0
            picks.pop(dropped_fi)
    return picks, collapsed, dropped_fi


def instance_centroid(kp3d):
    """(3,) mean of an instance's FINITE keypoints, or None if it has none.

    The same statistic `scripts/.../identity_check.py` compares against the
    mask centres, so "which instance is on this mask?" is answered here with
    the quantity the audit re-measures afterwards.
    """
    kp = np.asarray(kp3d, np.float64)
    m = np.isfinite(kp).all(axis=-1)
    return kp[m].mean(axis=0) if m.any() else None


ASSIGN_REASONS = ("nearest", "typed_preferred", "typed_fallback", "none")


def pick_mask_pair(out, off, nb, runner, mask_centres, *,
                   own_windows=None, window_pref=None,
                   margin_units=MASK_ASSIGN_MARGIN_UNITS,
                   max_units=MASK_ASSIGN_MAX_UNITS,
                   collapse_dist_units=COLLAPSE_DIST_UNITS):
    """`identity="mask"`: give each MASK fly the instance that is NEAREST it.

    The masks carry a human id review of which animal is the female (mask fly
    0) and which the male (mask fly 1). That decision is authoritative in this
    pipeline (`sexing`: human > mvq), so it -- not the model's sex head --
    says which written fly is which, and the model is asked only the question
    it is good at: which of the instances in this frame is the animal under
    this mask?

    RELATIVE, NOT ABSOLUTE (see `MASK_ASSIGN_MARGIN_UNITS`). Every instance of
    every window of the frame with `exist >= runner.exist_thresh` is a
    candidate. An instance belongs to mask `f` when it is nearer that mask's
    triangulated centre than the other's by at least `margin_units`, and
    within `max_units` of it. Requiring an absolute radius instead assumes the
    mask centre IS the animal's centre; on 20_04 the female's mask is poor
    enough (3-4 valid cameras, centre 13-35 units off her body) that a
    10-unit radius discarded instances that were correctly detected and were
    ~50 units from the OTHER mask -- never ambiguous, merely off-centre.

    Among the instances that qualify for one mask, that fly's TYPED slot wins
    if it is among them (`assign_reason` "typed_preferred"), else the nearest
    does ("nearest"); ties break on existence.

    OWN WINDOW FIRST (`window_pref`, see `WINDOW_PREF_MODES`). Both flies are
    usually inside BOTH crops at courtship distances, so "every instance of
    every window" repeatedly handed a fly the read from its PARTNER's crop --
    the same animal, the same 3D, but half its body at the crop edge and a
    per-view visibility (hence `conf3d`) near zero. Under `window_pref="own"`
    the qualifying instances from the window centred on THIS fly's own mask are
    considered first, and the other windows are consulted only when its own
    window offers none; the rule inside each group is unchanged (typed slot,
    then distance, then existence). `own_windows` names each fly's window as a
    LOCAL index into `[off, off + nb)` -- what `frame_windows`' per-fly
    assignment returns -- with -1 for "this fly has no window in this batch".
    A merged window is both flies' own window, so nothing changes there.

    TYPED FALLBACK. When no instance qualifies for a mask -- the usual cause
    is the two flies being nearly equidistant, i.e. geometry ABSTAINING rather
    than nothing being there -- but that mask's typed slot DID fire somewhere
    in the frame, the typed slot is used ("typed_fallback"), taken from its
    highest-existence window. It is refused when geometry positively assigned
    that instance to the OTHER mask (nearer the other mask by the margin): the
    fallback exists for "geometry could not decide", never to overrule a
    decision against this fly, which is exactly how a lifter ends up putting
    one fly's track on the other's body. Otherwise NaN ("none").

    A mask with no finite centre this frame gets nothing: its identity cannot
    be verified, and a guess would be indistinguishable from a swap.

    COLLAPSE GUARD. Geometry alone cannot give one instance to both masks (the
    margin test is exclusive), but a typed fallback can collide with a
    geometric pick. Two real flies are never within `collapse_dist_units` over
    most of their 50 keypoints; the same instance read twice is 0. Such a
    frame keeps the fly whose own mask the instance is NEARER and NaNs the
    other, rather than shipping a duplicated fly that every jitter, confidence
    and residual metric rates as excellent. Ties keep fly0 (the female -- the
    fly this pipeline loses frames on).

    Args:
        out: an `MVQRunner.infer` output.
        off, nb: this frame's window rows, `[off, off + nb)`.
        runner: for `I`, `exist_thresh`.
        mask_centres: (2,3) each mask fly's triangulated centre, world units.
        own_windows: (2,) LOCAL window index of each fly's own crop (-1 if it
            has none in this batch), or None for "no window information", which
            makes `window_pref` a no-op.
        window_pref: "own" (default) or "any" -- see above.

    Returns:
        picks: {fi: (slot_read result, absolute window index)}.
        dists: {fi: distance (world units) from the chosen instance's centroid
            to fly fi's OWN mask centre} -- recorded in mvq_meta.
        reasons: {fi: one of ASSIGN_REASONS} for both 0 and 1 (a fly with no
            pick reads "none").
        collapsed: bool, whether the guard fired.
        dropped_fi: the fi it NaN'd, or None.
        own: {fi: bool} for each fly PICKED -- whether its instance came out of
            its own window. False means the preference fell back (or is off).
    """
    typed = (SLOT_FEMALE, SLOT_MALE)
    mask_centres = np.asarray(mask_centres, np.float64)
    have = [bool(np.isfinite(mask_centres[f]).all()) for f in (0, 1)]
    margin, cap = float(margin_units), float(max_units)
    pref = resolved_window_pref(window_pref)
    # each fly's own window as an ABSOLUTE row of `out`, or -1 for "none in
    # this batch" (no centre, or a frame whose windows were truncated to
    # runner.batch) -- which simply leaves that fly with no preference
    if own_windows is None:
        own_row = [-1, -1]
    else:
        own_row = [(int(off) + int(w)) if 0 <= int(w) < int(nb) else -1
                   for w in own_windows]

    # every live instance of every window of this frame, with its distance to
    # each mask centre (inf where that mask has no centre this frame)
    cands = []                          # (b, slot, exist, [d0, d1])
    for b in range(int(off), int(off) + int(nb)):
        for s in range(int(runner.I)):
            e = float(out["exist"][b, s])
            if e < float(runner.exist_thresh):
                continue
            c = instance_centroid(out["kp3d"][b, s])
            if c is None:
                continue
            d = [float(np.linalg.norm(c - mask_centres[f])) if have[f] else np.inf
                 for f in (0, 1)]
            cands.append((b, s, e, d))

    picks, dists, reasons = {}, {}, {0: "none", 1: "none"}

    def _belongs(d, fi):
        """Is this instance the mask-`fi` animal? Nearer that mask than the
        other by the margin, and not absurdly far from it."""
        return d[fi] <= cap and d[fi] + margin <= d[1 - fi]

    def _prefer_own(cands_for_fi, fi, b_at):
        """The best of `cands_for_fi` (already sorted-comparable tuples),
        taken from fly `fi`'s OWN window if it has one there. `b_at` reads the
        window index out of a candidate tuple."""
        if pref == "own" and own_row[fi] >= 0:
            own_only = [c for c in cands_for_fi if b_at(c) == own_row[fi]]
            if own_only:
                return min(own_only)
        return min(cands_for_fi)

    for fi in (0, 1):
        if not have[fi]:
            continue
        mine = [(s != typed[fi], d[fi], -e, b, s) for b, s, e, d in cands
                if _belongs(d, fi)]
        if mine:
            not_typed, dd, _ne, b, s = _prefer_own(mine, fi, lambda c: c[3])
            picks[fi] = (slot_read(out, b, s), b)
            dists[fi] = dd
            reasons[fi] = "nearest" if not_typed else "typed_preferred"

    for fi in (0, 1):                   # typed fallback where geometry abstained
        if fi in picks or not have[fi]:
            continue
        live = [(-e, d[fi], b) for b, s, e, d in cands
                if s == typed[fi] and d[fi] <= cap and not _belongs(d, 1 - fi)]
        if not live:
            continue
        # highest existence wins, its own window first
        _ne, dd, b = _prefer_own(live, fi, lambda c: c[2])
        picks[fi] = (slot_read(out, b, typed[fi]), b)
        dists[fi] = dd
        reasons[fi] = "typed_fallback"

    collapsed, dropped_fi = False, None
    if len(picks) == 2:
        a3 = np.asarray(picks[0][0]["kp3d"], np.float64)
        b3 = np.asarray(picks[1][0]["kp3d"], np.float64)
        dd = np.linalg.norm(a3 - b3, axis=-1)
        dd = dd[np.isfinite(dd)]
        if dd.size and float(np.median(dd)) < float(collapse_dist_units):
            collapsed = True
            dropped_fi = 1 if dists[1] >= dists[0] else 0     # keep the nearer own-mask
            picks.pop(dropped_fi)
            dists.pop(dropped_fi)
            reasons[dropped_fi] = "none"
    own = {fi: (own_row[fi] >= 0 and b == own_row[fi]) for fi, (_r, b) in picks.items()}
    return picks, dists, reasons, collapsed, dropped_fi, own


def frame_windows(centres, ok=None, *, merge_dist_units=30.0):
    """One frame's crop windows from its per-fly 3D centres.

    `centres` (A,3) world units, `ok` (A,) bool (default: the finite rows). Two
    flies within `merge_dist_units` (30 units == 3 mm) share ONE 448-px crop --
    the model's own two-instance case, and the same rule `coarse_track` uses
    via `coarse_centres.plan_windows`, so the bout lifter and the coarse pass
    never disagree about what a window is.

    Returns `(window_centres (W,3) float32, assignment (A,) int)`; a fly with
    no centre gets assignment -1 and contributes no window. A frame with no
    centre at all yields `W == 0`: its flies are NaN, which is strictly better
    than a window at the world origin (that would crop the arena floor and come
    back with a confident, perfectly smooth fit of nothing).
    """
    from jarvis_jax.tracking.coarse_centres import plan_windows
    c = np.array(centres, np.float32, copy=True)
    if c.ndim != 2 or c.shape[-1] != 3:
        raise ValueError(f"centres must be (A,3) world coordinates, got {c.shape}")
    if ok is not None:
        c[~np.asarray(ok, bool).reshape(-1)] = np.nan
    return plan_windows(c, merge_dist_units=merge_dist_units)


class BoutMaskStore:
    """SAM3 masks for one bout, camera axis reordered BY NAME, unpacked LAZILY.

    `tracking.bout_masks.load_bout_masks` unpacks the WHOLE bout eagerly:
    (2007, 7, 448, 1936) bool is 12.2 GB for ONE fly and 24 GB for the pair,
    which does not fit in the job's memory alongside the model. This holds the
    packed array instead (3 GB) and unpacks one (fly, camera, frame) on demand
    with that module's OWN `unpack_one`, permuting the camera axis with that
    module's OWN `_camera_permutation` -- the same two functions
    `load_bout_masks` composes, so the by-name identity guarantee is identical.
    A legacy npz with no `cameras` name array is REFUSED rather than assumed
    positional (that assumption is the camera-scramble bug).
    """

    def __init__(self, npz_path, cameras):
        from jarvis_jax.tracking.bout_masks import _camera_permutation
        z = np.load(npz_path)
        if "cameras" not in z.files:
            raise RuntimeError(
                f"{npz_path} predates the `cameras` name array, so its camera axis "
                f"cannot be verified by name -- a permutation would silently crop "
                f"each fly out of the wrong camera. Re-run SAM3 for this bout.")
        self.npz_cameras = [str(c) for c in np.asarray(z["cameras"]).tolist()]
        self.cameras = [str(c) for c in cameras]
        self.perm = _camera_permutation(self.npz_cameras, self.cameras)
        self.packed = z["packed"]                                    # (A,C,T,H,Wb)
        self.valid = np.asarray(z["valid"])[:, self.perm]            # (A,C,T) canonical
        self.centroids = np.asarray(z["centroids"], np.float32)[:, self.perm]  # (A,C,T,2)
        self.H, self.W = int(z["shape"][0]), int(z["shape"][1])
        self.n_flies, self.T = self.packed.shape[0], self.packed.shape[2]

    def valid_at(self, fly, t):
        return np.asarray(self.valid[fly, :, t], bool)               # (C,)

    def centroid_at(self, fly, t):
        return np.asarray(self.centroids[fly, :, t], np.float64)     # (C,2)

    def mask_at(self, fly, cam_i, t):
        """(H,W) bool for canonical camera index `cam_i`."""
        from jarvis_jax.tracking.bout_masks import unpack_one
        return unpack_one(self.packed, fly, int(self.perm[cam_i]), t, self.W)


def bout_centres_3d(store, cam_mats, n_frames, t0=0):
    """(A, n_frames, 3) DLT of each fly's valid mask centroids + (A, n_frames) ok.

    One batched `triangulate_dlt_batched` call for the whole bout rather than
    a jit call per frame. A frame with fewer than 2 valid mask views has no
    usable center3D and is marked not-ok (its window is skipped -> NaN output).

    Raises:
        ValueError: `store.n_flies != 2`. `lift_masked_bout` hardcodes exactly
            TWO typed slots (female fly0, male fly1) throughout -- a mask
            store with any other fly count would silently misindex (a single
            fly) or silently drop flies (three or more) rather than raise,
            since nothing downstream checks the fly axis again.
    """
    from jarvis_jax.geometry.center3d import triangulate_dlt_batched
    A = store.n_flies
    if A != 2:
        raise ValueError(
            f"bout_centres_3d: store.n_flies={A}, but the masked-bout mvq lifter "
            f"(lift_masked_bout) hardcodes exactly TWO typed slots (female fly0, male "
            f"fly1) -- this route is for two-fly courtship bouts only. A single-fly "
            f"(free-running) recording is not lifted through this path.")
    cam_mats = np.asarray(cam_mats, np.float32)
    pts = np.zeros((A * n_frames, len(store.cameras), 2), np.float32)
    val = np.zeros((A * n_frames, len(store.cameras)), bool)
    for fly in range(A):
        for i in range(n_frames):
            r = fly * n_frames + i
            pts[r] = store.centroid_at(fly, t0 + i)
            val[r] = store.valid_at(fly, t0 + i)
    ok = val.sum(axis=1) >= 2
    cm = np.broadcast_to(cam_mats[None], (A * n_frames,) + cam_mats.shape)
    xyz = np.asarray(triangulate_dlt_batched(jnp.asarray(pts), jnp.asarray(cm),
                                             jnp.asarray(val)))
    xyz = np.where((ok & np.isfinite(xyz).all(axis=1))[:, None], xyz, np.nan)
    ok = ok & np.isfinite(xyz).all(axis=1)
    return xyz.reshape(A, n_frames, 3), ok.reshape(A, n_frames)


# =============================================================================
# Per-keypoint mask-containment filter (Task 9)
# =============================================================================
# WHAT THIS FIXES, measured on the 30-bout r2 lift of 2025_10_20_13_20_04.
# Identity is decided per FRAME (`pick_mask_pair`) and the collapse guard is a
# per-FRAME test, so both are blind to the failure that is left: in contact
# frames -- especially the 264 of 391 pose-jump frames where the female's slot
# is EMPTY -- the male instance absorbs PART of her body. 1.4% of frames have
# >= 5 male keypoints stepping > 0.5 mm in one frame; 1.7% have >= 5 male
# keypoints nearer her centroid than his; bout 1 frame 364 puts 17 male
# keypoints (T1 legs, trochanters) on the female and bout 3 frame 56 puts 26
# (WingL_base, Antenna_Base, EyeL, T1 legs). The whole instance is not wrong,
# so NaN-ing the frame would throw away a mostly-correct male; only those
# keypoints are.
#
# The test is the masks themselves, which carry the HUMAN id review: a male
# keypoint that reprojects INSIDE the female's mask and OUTSIDE his own, in
# `min_views` of the cameras where BOTH masks are valid, is on the wrong fly.
#
# WHY THE OWN-MASK TEST ALONE IS NOT USED. Legs routinely extend past a SAM
# mask and an occluded keypoint projects outside it too, so "outside its own
# mask" is common and benign -- dropping on that alone would delete real
# tarsi. The filter fires only on the CONJUNCTION (inside the other, outside
# its own), and the own mask is DILATED by `own_margin` px first, which makes
# "outside its own" harder to satisfy and so makes the drop conservative in
# the same direction.
CONTAINMENT_MIN_VIEWS = 3          # cameras that must agree before a keypoint
                                   # is called "on the other fly". 7 cameras,
                                   # of which typically 3-5 have BOTH masks
                                   # valid on 20_04; 3 is a majority of that
                                   # and no single bad SAM mask can reach it.
CONTAINMENT_OWN_MARGIN_PX = 6.0    # dilation of the OWN mask, px. SAM's fly
                                   # masks sit ~2-5 px inside the silhouette
                                   # at these scales, and a tarsus one leg
                                   # segment past the mask edge must NOT read
                                   # as "outside its own mask".
CONTAINMENT_MAX_STEP_UNITS = 5.0   # 0.5 mm in 1/800 s -- 400 mm/s for a
                                   # single landmark, which no fly part does.
                                   # A step above this is SUSPECT, not proof
                                   # (see `mask_containment_filter`).

CONTAINMENT_MODES = ("on", "off")
DEFAULT_CONTAINMENT = "on"


def resolved_containment(containment):
    """`containment` as a bool. None -> `DEFAULT_CONTAINMENT`.

    Accepts the CLI's strings ("on"/"off") and plain bools, because a YAML
    `containment: on` is parsed by PyYAML 1.1 rules as the BOOLEAN True, not
    the string "on" -- a config written the obvious way must not be refused.
    """
    if containment is None:
        containment = DEFAULT_CONTAINMENT
    if isinstance(containment, bool):
        return containment
    c = str(containment).strip().lower()
    if c not in CONTAINMENT_MODES:
        raise ValueError(f"mvq containment must be one of {list(CONTAINMENT_MODES)} "
                         f"(or a bool), got {containment!r}")
    return c == "on"


def _disk_offsets(radius):
    """(dy, dx) integer offsets of the pixels within `radius` of the origin."""
    r = int(np.ceil(float(radius)))
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    sel = (dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2) <= float(radius) ** 2
    return dy[sel].ravel().astype(np.int64), dx[sel].ravel().astype(np.int64)


def project_points(cam_mats, xyz):
    """(K,3) world points -> (C,K,2) full-frame px, `p_h @ M` + perspective divide.

    The SAME convention as `ReprojectionTool.reproject_point` and
    `tests/test_coarse_centres._project_all`, and `cam_mats` is
    `ReprojectionTool.camera_matrices`, whose camera axis IS the canonical
    order (`MVQRunner` asserts that). Non-finite inputs and points behind the
    camera come back NaN rather than a huge finite number that would land
    inside some mask by accident.
    """
    cam_mats = np.asarray(cam_mats, np.float64)
    xyz = np.asarray(xyz, np.float64)
    h = np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,), np.float64)], axis=-1)
    p = np.einsum("kj,cjm->ckm", h, cam_mats)                    # (C,K,3)
    w = p[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = p[..., :2] / w[..., None]
    return np.where(np.isfinite(uv) & (np.abs(w)[..., None] > 1e-9), uv, np.nan)


def _inside_mask(mask, uv, ok):
    """(K,) bool: the rounded pixel of `uv` is inside `mask`, for keypoints `ok`."""
    H, W = mask.shape
    out = np.zeros(uv.shape[0], bool)
    with np.errstate(invalid="ignore"):
        x = np.where(ok, np.rint(uv[:, 0]), 0).astype(np.int64)
        y = np.where(ok, np.rint(uv[:, 1]), 0).astype(np.int64)
    good = ok & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if good.any():
        out[good] = mask[y[good], x[good]]
    return out


def _inside_mask_dilated(mask, uv, ok, dy, dx):
    """(K,) bool: any mask pixel within the disk (`dy`,`dx`) of `uv`'s pixel.

    Equivalent to testing membership in the mask DILATED by that disk, but
    without materialising the dilated (448,1936) mask: 50 keypoints x ~113
    offsets is ~5.6k lookups, against ~867k pixels for a real dilation.
    Out-of-image offsets are CLIPPED to the border, which can only ever say
    "inside" for a point already at the border -- conservative in the
    drop-nothing direction, which is the direction this filter errs in.
    """
    H, W = mask.shape
    out = np.zeros(uv.shape[0], bool)
    if not ok.any():
        return out
    with np.errstate(invalid="ignore"):
        x = np.where(ok, np.rint(uv[:, 0]), 0).astype(np.int64)
        y = np.where(ok, np.rint(uv[:, 1]), 0).astype(np.int64)
    ys = np.clip(y[ok, None] + dy[None, :], 0, H - 1)
    xs = np.clip(x[ok, None] + dx[None, :], 0, W - 1)
    out[ok] = mask[ys, xs].any(axis=1)
    return out


def mask_containment_filter(kp3d_by_fly, kp2d_by_fly, store, cam_mats, *,
                            min_views=CONTAINMENT_MIN_VIEWS,
                            own_margin=CONTAINMENT_OWN_MARGIN_PX,
                            max_step_units=CONTAINMENT_MAX_STEP_UNITS,
                            conf_by_fly=None, t0=0, image_margin_px=0.0):
    """Drop the keypoints of one fly that are sitting on the OTHER fly.

    Pure: the inputs are not modified, copies come back.

    Args:
        kp3d_by_fly: (2,T,K,3) world units, row 0 the FEMALE (mask fly 0) and
            row 1 the MALE, exactly as `lift_masked_bout` holds them.
        kp2d_by_fly: (2,T,C,K,2) full-frame px on the CANONICAL camera axis.
        store: a `BoutMaskStore` (anything with `valid_at`, `mask_at`, whose
            camera axis is that same canonical order).
        cam_mats: (C,4,3) `ReprojectionTool.camera_matrices`.
        min_views: cameras that must agree (`CONTAINMENT_MIN_VIEWS`).
        own_margin: px the OWN mask is dilated by (`CONTAINMENT_OWN_MARGIN_PX`).
        max_step_units: the temporal gate's suspicion threshold
            (`CONTAINMENT_MAX_STEP_UNITS`).
        conf_by_fly: (2,T,C,K) per-view visibility; NaN'd alongside kp2d.
        t0: the store frame index of `kp3d_by_fly[:, 0]`.
        image_margin_px: how far outside the image a reprojection may fall and
            still be tested (0 = must be in-image).

    KEYPOINT ORDER. Nothing here indexes a keypoint by integer -- every test
    is per-keypoint and order-agnostic -- so this runs equally on the model's
    own order (where `lift_masked_bout` calls it, BEFORE `to_pipeline`) and on
    `cfg.model.KP_NAMES` (where the offline before/after check calls it).

    THE TWO RULES.

    1. CONTAINMENT. In every camera where BOTH flies' masks are valid, the
       keypoint's reprojection is tested against the other fly's mask (as-is)
       and its own (dilated by `own_margin`). "On the other fly" = inside the
       other AND outside its own, in >= `min_views` of those cameras. Being
       outside its OWN mask is never on its own a reason to drop: legs extend
       past the mask and occluded parts project outside it.
    2. TEMPORAL. A frame-to-frame step > `max_step_units` makes the keypoint
       SUSPECT in BOTH frames of the pair; it is dropped only if it also
       fails rule 1 in either frame (so the frame a keypoint jumped OFF is
       cleaned too, not just the frame it landed on) or the step exceeds
       `2 * max_step_units`, which is a pure spike no landmark can be. Steps
       are measured on the INPUT track, before rule 1 NaNs anything -- gating
       on the already-cleaned track would silently disable the pair test
       wherever rule 1 fired.

    Never NaNs a whole fly: frame-level identity belongs to `pick_mask_pair`
    and the collapse guard.

    Returns:
        (kp3d, kp2d, conf, report). `conf` is None when `conf_by_fly` is.
        A dropped keypoint is NaN in kp3d, in kp2d in EVERY view, and in conf
        (every `conf` consumer in the pipeline gates with `conf > thresh`,
        which NaN fails, so a NaN reads as "no measurement" exactly like the
        NaN coordinates beside it).
    """
    kp3d = np.array(kp3d_by_fly, np.float32, copy=True)
    kp2d = np.array(kp2d_by_fly, np.float32, copy=True)
    conf = None if conf_by_fly is None else np.array(conf_by_fly, np.float32, copy=True)
    if kp3d.ndim != 4 or kp3d.shape[0] != 2 or kp3d.shape[-1] != 3:
        raise ValueError(f"kp3d_by_fly must be (2,T,K,3), got {kp3d.shape}")
    A, T, K = kp3d.shape[:3]
    cam_mats = np.asarray(cam_mats, np.float64)
    C = cam_mats.shape[0]
    if kp2d.shape != (A, T, C, K, 2):
        raise ValueError(f"kp2d_by_fly must be (2,T,C,K,2) = {(A, T, C, K, 2)}, got "
                         f"{kp2d.shape}; kp2d's camera axis and `cam_mats` must be the "
                         f"SAME canonical order or this tests one camera's mask against "
                         f"another camera's reprojection")
    if conf is not None and conf.shape != (A, T, C, K):
        raise ValueError(f"conf_by_fly must be (2,T,C,K) = {(A, T, C, K)}, got {conf.shape}")

    dy, dx = _disk_offsets(own_margin)
    on_other = np.zeros((A, T, K), bool)
    n_views_tested = np.zeros((A, T, K), np.int16)
    n_frames_testable = 0
    n_unpacks = 0

    for t in range(T):
        both_valid = store.valid_at(0, t0 + t) & store.valid_at(1, t0 + t)
        cams = np.flatnonzero(np.asarray(both_valid, bool))
        if cams.size < min_views:
            continue
        n_frames_testable += 1
        fin = np.isfinite(kp3d[:, t]).all(-1)                       # (A,K)
        if not fin.any():
            continue
        uv = {f: project_points(cam_mats, np.nan_to_num(kp3d[f, t], nan=0.0))
              for f in range(A) if fin[f].any()}
        cache = {}
        for f in uv:
            inside_other = np.zeros((cams.size, K), bool)
            outside_own = np.zeros((cams.size, K), bool)
            for j, c in enumerate(cams):
                u = uv[f][c]
                ok = fin[f] & np.isfinite(u).all(-1)
                ok &= ((u[:, 0] >= -image_margin_px) & (u[:, 0] <= store.W - 1 + image_margin_px)
                       & (u[:, 1] >= -image_margin_px) & (u[:, 1] <= store.H - 1 + image_margin_px))
                if not ok.any():
                    continue          # nothing of this fly lands in this camera
                for g in (f, 1 - f):
                    if (g, c) not in cache:
                        cache[(g, c)] = store.mask_at(g, int(c), t0 + t)
                        n_unpacks += 1
                inside_other[j] = _inside_mask(cache[(1 - f, c)], u, ok)
                outside_own[j] = ok & ~_inside_mask_dilated(cache[(f, c)], u, ok, dy, dx)
                n_views_tested[f, t] += ok
            on_other[f, t] = fin[f] & ((inside_other & outside_own).sum(axis=0) >= min_views)

    # ---- rule 2, on the INPUT track (see the docstring)
    step = np.linalg.norm(np.diff(kp3d, axis=1), axis=-1)            # (A,T-1,K)
    with np.errstate(invalid="ignore"):
        suspect = step > float(max_step_units)
        pure_spike = step > 2.0 * float(max_step_units)
    fails_either = on_other[:, :-1] | on_other[:, 1:]
    pair_drop = suspect & (fails_either | pure_spike)
    spike = np.zeros((A, T, K), bool)
    if T > 1:
        spike[:, :-1] |= pair_drop
        spike[:, 1:] |= pair_drop
    spike &= np.isfinite(kp3d).all(-1) & ~on_other

    drop = on_other | spike
    kp3d[drop] = np.nan
    m = drop[:, :, None, :]                                          # (A,T,1,K)
    kp2d = np.where(m[..., None], np.float32(np.nan), kp2d)
    if conf is not None:
        conf = np.where(m, np.float32(np.nan), conf)

    finite_before = np.isfinite(np.asarray(kp3d_by_fly, np.float64)).all(-1)
    report = {
        "enabled": True,
        "min_views": int(min_views),
        "own_margin_px": float(own_margin),
        "max_step_units": float(max_step_units),
        "n_frames": int(T),
        "n_frames_testable": int(n_frames_testable),
        "n_mask_unpacks": int(n_unpacks),
        "n_kp_dropped_other_mask": {f"fly{f}": int(on_other[f].sum()) for f in range(A)},
        "n_kp_dropped_spike": {f"fly{f}": int(spike[f].sum()) for f in range(A)},
        "n_kp_finite_before": {f"fly{f}": int(finite_before[f].sum()) for f in range(A)},
        "frac_kp_dropped": {
            f"fly{f}": (float(drop[f].sum() / finite_before[f].sum())
                        if finite_before[f].any() else 0.0) for f in range(A)},
        "per_frame": {
            "n_other_mask": on_other.sum(axis=-1).astype(int).tolist(),   # (A,T)
            "n_spike": spike.sum(axis=-1).astype(int).tolist(),
            "n_views_tested_max": n_views_tested.max(axis=-1).astype(int).tolist(),
        },
    }
    return kp3d, kp2d, conf, report


def resolve_bout_frames(session_dir, bout_idx, bouts_csv=None):
    """(start_frame, end_frame, n_frames) for `bout_idx` from the session's bouts CSVs.

    On Session0 the unified CSV is a symlink into a `Predictions_3D_*` dir that
    no longer exists, so the per-fly CSVs (whose `fly_id` carries a `_fly<f>`
    suffix the unified one does not) are the fallback. `bouts_csv`, when given,
    is tried FIRST, under both key shapes. The result is cross-checked against
    the mask npz's own frame count by the caller.
    """
    from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for
    tag = session_tag_for(str(session_dir))
    tries = []
    if bouts_csv:
        tries += [(str(bouts_csv), tag)]
        tries += [(str(bouts_csv), f"{tag}_fly{f}") for f in (0, 1)]
    tries += [(os.path.join(session_dir, "courtship_bouts_unified_summary.csv"), tag)]
    tries += [(os.path.join(session_dir, f"courtship_bouts_fly{f}_summary.csv"),
               f"{tag}_fly{f}") for f in (0, 1)]
    errs = []
    for path, want in tries:
        try:
            rows = parse_bouts(path, want, bout_ids=[bout_idx])
        except OSError as e:                                   # broken symlink / absent
            errs.append(f"{os.path.basename(path)}: {e}")
            continue
        if rows:
            r = rows[0]
            return int(r["start"]), int(r["end"]), int(r["n"])
        errs.append(f"{os.path.basename(path)}: no row for bout {bout_idx} / fly_id {want!r}")
    raise KeyError(f"bout {bout_idx} not found in any bouts CSV under {session_dir}: {errs}")


def bout_lift_is_current(out_dir, gates_string, n_flies=2):
    """True when every `<out_dir>/fly*/kp3d.npz` exists and carries `gates_string`
    AND `<out_dir>/sex.json` exists with a `method` this lifter writes
    (`MVQ_SEX_METHOD` or `MASK_ID_SEX_METHOD` -- which one depends on the
    identity mode, and the gates string already pins THAT).

    The gates string names the checkpoint, the existence threshold and the
    identity mode, so this is the same staleness contract `run_bout.py`'s
    Stage B enforces: a bout lifted by different weights -- or by the other
    identity rule -- is NOT current and gets re-run, while a re-submitted
    array skips the work it already did.

    `sex.json` is REQUIRED too, not just the per-fly npz files:
    `lift_masked_bout` writes every fly's `kp3d.npz` BEFORE `sex.json` (it
    needs the per-fly `n_missing` counts collected during that loop to build
    the sex payload), so a requeued task killed between those writes leaves a
    bout with two complete-looking `kp3d.npz` files and no `sex.json`. Without
    this check that bout reads as "current" forever -- a re-submitted array
    skips it -- and the recording falls back to ONE shared body scale instead
    of a per-fly one (see MEMORY scale-from-first-bout-defect).
    """
    for fly in range(int(n_flies)):
        p = os.path.join(str(out_dir), f"fly{fly}", "kp3d.npz")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            return False
        try:
            with np.load(p) as z:
                if "gates" not in z.files or str(z["gates"]) != str(gates_string):
                    return False
        except (OSError, ValueError):
            return False
    sex_path = os.path.join(str(out_dir), "sex.json")
    if not (os.path.exists(sex_path) and os.path.getsize(sex_path) > 0):
        return False
    try:
        with open(sex_path) as f:
            sex = json.load(f)
    except (OSError, ValueError):
        return False
    if sex.get("method") not in (MVQ_SEX_METHOD, MASK_ID_SEX_METHOD):
        return False
    return True


def _own_frac_str(own_window, identity_resolved):
    """"fly0 0.97 / fly1 1.00" -- the fraction of each fly's WRITTEN frames
    that came from its own window, for the lifter's one-line summary. "n/a"
    under identity="sex", where the preference does not run."""
    if identity_resolved != "mask":
        return "n/a"
    out = []
    for f in (0, 1):
        w = own_window[f] >= 0
        out.append(f"fly{f} " + (f"{float((own_window[f][w] == 1).mean()):.2f}"
                                 if w.any() else "-"))
    return " / ".join(out)


def _mean_or_none(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else float(a.mean())


def lift_masked_bout(runner, frames_iter, centres, ok, *, out_dir, model_names,
                     merge_dist_units=30.0, collapse_dist_units=COLLAPSE_DIST_UNITS,
                     identity=None, window_pref=None,
                     mask_assign_margin_units=MASK_ASSIGN_MARGIN_UNITS,
                     mask_assign_max_units=MASK_ASSIGN_MAX_UNITS,
                     mask_store=None, containment=None,
                     containment_min_views=CONTAINMENT_MIN_VIEWS,
                     containment_own_margin_px=CONTAINMENT_OWN_MARGIN_PX,
                     containment_max_step_units=CONTAINMENT_MAX_STEP_UNITS,
                     mask_frame_offset=0,
                     force=False, progress_every=0,
                     meta_extra=None, mask_sex_meta=None, review_male_fly=None,
                     verbose=True):
    """Lift one whole bout into `<out_dir>/fly0|fly1/{kp2d,kp3d}.npz` + sex/meta.

    Args:
        runner: `MVQRunner` (or anything with the same `windows`/`infer`/
            `read_typed`/`to_pipeline`/`gates_string` surface).
        frames_iter: iterable of `(frames (C,H,W,3) uint8 RGB, present (C,))`
            in the runner's CANONICAL camera order, one item per bout frame --
            `predict.synced_reader.read_window(...)` in production.
        centres: (A,T,3) per-fly 3D window centres in world units (the DLT of
            the SAM3 mask centroids -- `bout_centres_3d`).
        ok: (A,T) bool, which of those centres are usable.
        out_dir: the BOUT directory, `<run>/bouts/bout_<idx:05d>`.
        model_names: `cfg.model.KP_NAMES` -- the pipeline's keypoint order.
        collapse_dist_units: a frame whose two written flies agree to within
            this median per-keypoint 3D distance is judged to be the SAME fly
            read twice; one of them is NaN'd and the frame flagged (see
            `COLLAPSE_DIST_UNITS`).
        identity: `"mask"` (default -- see IDENTITY_MODES) or `"sex"`; None
            takes the runner's own. `"mask"` needs the masks to carry a human
            id review (`mask_sex_meta["method"] == "human_id_review"`); a bout
            whose masks do not falls back to `"sex"` with a warning, and says
            so in `mvq_meta.json`. The RESOLVED mode is what the gates string
            names and what the skip check compares, so a fallback bout is not
            mistaken for a mask-identity one (see `resolve_mask_identity`).
        window_pref: `"own"` (default) or `"any"`; None takes the runner's.
            `identity="mask"` only -- which window a fly is read from when more
            than one holds it (`WINDOW_PREF_MODES`). "own" prefers the window
            centred on that fly's own mask, which is what keeps its `conf3d`
            off the floor; "any" is the pre-2026-09-05 rule. Enrolled in the
            gate signature, because conf3d is written into kp3d.npz and gates
            several downstream stages.
        mask_assign_margin_units / mask_assign_max_units: `identity="mask"`
            only -- how much NEARER its own mask an instance must be than the
            other mask to be assigned by geometry, and the garbage cap on that
            distance (`MASK_ASSIGN_MARGIN_UNITS` / `MASK_ASSIGN_MAX_UNITS`;
            see `pick_mask_pair`). Recorded in `mvq_meta.json` rather than
            enrolled in the gate signature, like `collapse_dist_units`.
        mask_store: the bout's `BoutMaskStore`. Required by the containment
            filter -- it is the only thing that can say whose body a keypoint
            is sitting on. A caller that builds `centres`/`ok` by hand and
            passes no store runs with containment OFF and is gated as such.
        containment: `"on"` (default) / `"off"` / a bool -- the per-keypoint
            mask-containment filter (`mask_containment_filter`), which NaNs
            the keypoints of one fly that reproject onto the OTHER fly.
            Effective only under `identity="mask"` (it needs the masks' human
            id review to know whose body is whose) and only with a
            `mask_store`; the EFFECTIVE bool is what the gate string names.
        containment_min_views / containment_own_margin_px /
        containment_max_step_units: that filter's thresholds
            (`CONTAINMENT_MIN_VIEWS` / `_OWN_MARGIN_PX` / `_MAX_STEP_UNITS`).
            Recorded in `mvq_meta.json`, not in the gate signature.
        mask_frame_offset: store frame index of bout frame 0 (0 in
            production; non-zero only when lifting a slice of a bout).
        force: re-run a bout whose kp3d.npz already carries this gates string.
        mask_sex_meta / review_male_fly: what the SAM3 masks / the id-review
            manifest believe about identity. Under `identity="mask"` the
            masks' human review IS the decision (and a mask npz whose
            `male_slot` is not 1 is refused, since `fly{f}` IS mask fly `f`
            here); under `identity="sex"` both are recorded only, and any
            disagreement logged, in `mvq_meta.json`.

    Writes, per fly (0 = the FEMALE, 1 = the MALE -- the canonical identity
    `canonicalize_bout` would enforce, from the mask review or the typed
    slots depending on the mode):
        kp2d.npz  kp2d (T,C,K,2) full-frame px, conf (T,C,K) per-view
                  visibility, cameras (C,), kp_names (K,)
        kp3d.npz  kp3d (T,K,3) world units, conf3d (T,K) = mean visibility
                  over cameras, conf3d_mvq_raw (T,K), kp_names (K,), gates
    and, per bout, `sex.json` (male_fly 1; method "mask_human_id_review" or
    "mvq_sex_head") and `mvq_meta.json` (which carries the per-frame
    `collapsed` flag, `identity_source`, `slot_used`, `sex_head_agrees` and
    `own_window` -- 1 read from this fly's own window, 0 from another, -1
    nothing written -- the per-fly `n_collapsed` and `n_from_other_window`
    counts, `collapsed_frac` and `sex_head_disagree_frac`).

    Returns a dict of the per-frame bookkeeping (in the MODEL's own keypoint
    order, before the name permutation) plus `skipped`.
    """
    model_names = [str(n) for n in model_names]
    identity = resolved_identity(identity if identity is not None
                                 else getattr(runner, "identity", None))
    runner_identity = getattr(runner, "identity", None)
    if runner_identity is not None and str(runner_identity) != identity:
        raise ValueError(
            f"the runner was built for identity={runner_identity!r} but this lift was "
            f"asked for {identity!r}; `runner.gates_string()` would then disagree with "
            f"the gates stamped into kp3d.npz, and a later staleness check could not "
            f"tell which rule assigned the flies")
    # Same contract for the window preference: it is in the gate string, so a
    # runner built for one mode and a lift asked for the other would stamp a
    # string describing neither.
    window_pref = resolved_window_pref(window_pref if window_pref is not None
                                       else getattr(runner, "window_pref", None))
    runner_window_pref = getattr(runner, "window_pref", None)
    if (runner_window_pref is not None
            and resolved_window_pref(runner_window_pref) != window_pref):
        raise ValueError(
            f"the runner was built for window_pref={runner_window_pref!r} but this lift "
            f"was asked for {window_pref!r}; `runner.gates_string()` would then disagree "
            f"with the gates stamped into kp3d.npz")
    out_dir = str(out_dir)
    # Resolve the EFFECTIVE mode BEFORE the gate string and the currency check.
    # Gating on the requested mode instead would stamp `identity: mask` on a
    # bout that actually ran the sex head, and then a re-lift after someone
    # canonicalizes a human review into those masks would be SKIPPED as
    # already current -- the bout would keep its sex-head identity forever
    # inside a run labelled mask.
    identity_resolved, _fallback = resolve_mask_identity(identity, mask_sex_meta,
                                                         where=out_dir)
    if _fallback:
        warnings.warn(_fallback, RuntimeWarning, stacklevel=2)
    # Same resolve-before-gating discipline as `identity`: the gate must name
    # the filter this bout ACTUALLY RAN. The filter needs the masks
    # themselves, so a bout that fell back to the sex head -- or a caller with
    # no `mask_store` -- runs with it off and is gated off, and a later re-lift
    # (once the masks carry a human review) is correctly seen as not-current.
    containment_want = resolved_containment(containment)
    containment_on = bool(containment_want and identity_resolved == "mask"
                          and mask_store is not None)
    if containment_want and containment is not None and not containment_on:
        warnings.warn(
            f"{out_dir}: containment={containment!r} was asked for but the filter is "
            f"OFF for this bout (identity_resolved={identity_resolved!r}, mask_store="
            f"{'absent' if mask_store is None else 'present'}); it needs both flies' "
            f"masks and the human id review those masks carry. The bout is GATED as "
            f"containment=false, so a re-lift once those exist is not skipped.",
            RuntimeWarning, stacklevel=2)
    gates_string = mvq_gate_string(runner.checkpoint, step=runner.step,
                                   exist_thresh=runner.exist_thresh,
                                   identity=identity_resolved,
                                   containment=containment_on,
                                   window_pref=window_pref)
    if not force and bout_lift_is_current(out_dir, gates_string):
        if verbose:
            print(f"[mvq-lift] skip {out_dir}: kp3d.npz already carries these gates "
                  f"(identity {identity_resolved})", flush=True)
        return {"skipped": True, "out_dir": out_dir, "gates": gates_string,
                "identity": identity, "identity_resolved": identity_resolved}

    centres = np.asarray(centres, np.float32)
    ok = np.asarray(ok, bool)
    if centres.ndim != 3 or centres.shape[-1] != 3 or ok.shape != centres.shape[:2]:
        raise ValueError(f"centres must be (A,T,3) and ok (A,T); got {centres.shape} "
                         f"/ {ok.shape}")
    # Hardcoded below: exactly two typed slots (female fly0, male fly1) --
    # `bout_centres_3d` already refuses a mask store with a different
    # `n_flies`, but `centres`/`ok` can also be built by hand (as the tests
    # and `scripts/mvq_lift_bout.py` sometimes do), so the same guard belongs
    # here too, at the one place every caller passes through.
    if centres.shape[0] != 2:
        raise ValueError(
            f"lift_masked_bout: centres has {centres.shape[0]} flies (A), but this "
            f"lifter hardcodes exactly TWO typed slots (female fly0, male fly1) -- a "
            f"different fly count would silently misindex (A=1) or silently drop flies "
            f"(A>=3) rather than raise. This route is for two-fly courtship bouts only.")
    T = centres.shape[1]
    C, K, I = len(runner.cameras), runner.K, runner.I

    # Row 0 is the FEMALE (mask fly 0 / female typed slot), row 1 the MALE.
    kp3d = np.full((2, T, K, 3), np.nan, np.float32)
    kp2d = np.full((2, T, C, K, 2), np.nan, np.float32)
    vis = np.zeros((2, T, C, K), np.float32)      # 0 => the pipeline's conf gate drops it
    conf_raw = np.zeros((2, T, K), np.float32)
    exist = np.full((2, T), np.nan, np.float32)
    sex_prob = np.full((2, T), np.nan, np.float32)
    slot = np.full((2, T), -1, np.int8)
    window = np.full((2, T), -1, np.int8)
    # identity="mask" only: was this fly read from the window centred on ITS
    # OWN mask? 1 yes, 0 another window, -1 nothing written this frame. The
    # audit quantity for the own-window preference -- a bout whose female is
    # mostly 0 is the p3b conf3d collapse.
    own_window = np.full((2, T), -1, np.int8)
    n_from_other_window = [0, 0]
    # each fly's own window for the frame, as a LOCAL index into that frame's
    # planned windows. Filled by the frame loop and read by `_flush`, which
    # runs a whole batch later and no longer has the plan.
    own_widx = np.full((2, T), -1, np.int8)
    all_exist = np.full((T, I), np.nan, np.float32)   # every slot, for the meta
    n_windows = np.zeros(T, np.int8)
    no_centre = np.zeros(T, bool)
    collapsed = np.zeros(T, bool)       # both written flies read the same instance
    n_collapsed = [0, 0]                # per fly, how often IT was the dropped one
    # identity="mask" only: how far the chosen instance's keypoint centroid sat
    # from that fly's own mask centre (world units). The audit quantity.
    mask_dist = np.full((2, T), np.nan, np.float32)
    # ... and WHY that instance was chosen (ASSIGN_REASONS). "typed_fallback"
    # is the row to watch: it means geometry abstained and the sex head, not
    # the human review, named that fly for that frame.
    assign_reason = [["none"] * T, ["none"] * T]

    pend, batch, rows = [], [], 0
    warned_drop = False
    t_start = time.time()
    n_seen = 0

    def _flush():
        nonlocal pend, batch, rows
        if not pend:
            return
        out = runner.infer(concat_windows(batch))
        for t, off, nb in pend:
            all_exist[t] = out["exist"][off]
            if identity_resolved == "mask":
                # The masks' human id review decides which written fly is
                # which; the model only says which instance is nearest which
                # mask. See `pick_mask_pair`.
                picks, dists, reasons, collapsed[t], drop, own = pick_mask_pair(
                    out, off, nb, runner, centres[:, t],
                    own_windows=own_widx[:, t], window_pref=window_pref,
                    margin_units=mask_assign_margin_units,
                    max_units=mask_assign_max_units,
                    collapse_dist_units=collapse_dist_units)
                for fi, d in dists.items():
                    mask_dist[fi, t] = d
                for fi, r in reasons.items():
                    assign_reason[fi][t] = r
                for fi, o in own.items():
                    own_window[fi, t] = 1 if o else 0
                    if not o:
                        n_from_other_window[fi] += 1
            else:
                # Typed-slot read + collapse guard: see `pick_typed_pair`'s
                # docstring for the full rationale (measured on Session0 bout
                # 28, whose `track_qc.json` flags 90/2007 frames as merged
                # tracks). Shared with `coarse_track.coarse_pass._read_batch`
                # so the two routes cannot silently drift apart on this rule.
                picks, collapsed[t], drop = pick_typed_pair(out, off, nb, runner,
                                                            collapse_dist_units)
            if drop is not None:
                n_collapsed[drop] += 1

            for fi, (best, best_b) in picks.items():
                kp3d[fi, t] = best["kp3d"]
                kp2d[fi, t] = best["kp2d"]
                vis[fi, t] = best["vis"]
                conf_raw[fi, t] = best["conf_raw"]
                exist[fi, t] = best["exist"]
                sex_prob[fi, t] = best["sex_prob"]
                slot[fi, t] = best["slot"]
                window[fi, t] = best_b - off
        pend, batch, rows = [], [], 0

    for t, (frames, present) in enumerate(frames_iter):
        if t >= T:
            raise ValueError(f"frames_iter yielded more than the {T} frames the centres "
                             f"describe -- the mask npz and the video read must cover "
                             f"the SAME bout frames")
        # `frame_windows`' per-fly window assignment does not RESTRICT
        # identity="mask": an instance is matched to a mask by DISTANCE over
        # every window of the frame, so a fly the model localises only in the
        # OTHER fly's crop is still available to its own mask. It does ORDER
        # the search -- `window_pref="own"` consults a fly's own window first
        # (see WINDOW_PREF_MODES) -- which is why the assignment is kept here
        # and handed to `pick_mask_pair` rather than discarded.
        wc, w_assign = frame_windows(centres[:, t], ok[:, t],
                                     merge_dist_units=merge_dist_units)
        n_seen = t + 1
        if wc.shape[0] == 0:
            no_centre[t] = True
            continue
        if wc.shape[0] > runner.batch:
            if not warned_drop:
                warnings.warn(
                    f"lift_masked_bout: frame {t} planned {wc.shape[0]} windows but "
                    f"runner.batch={runner.batch}; dropping "
                    f"{wc.shape[0] - runner.batch} window(s) (only the first "
                    f"offending frame is reported)", RuntimeWarning, stacklevel=2)
                warned_drop = True
            wc = wc[:runner.batch]
        # after the truncation, so a fly whose window was dropped reads -1
        # ("no own window") instead of indexing a window that is not there
        for _fi in (0, 1):
            _a = int(w_assign[_fi])
            own_widx[_fi, t] = _a if 0 <= _a < wc.shape[0] else -1
        if rows + wc.shape[0] > runner.batch:
            _flush()
        batch.append(runner.windows(frames, present, wc))
        pend.append((t, rows, int(wc.shape[0])))
        rows += int(wc.shape[0])
        n_windows[t] = int(wc.shape[0])
        if progress_every and (t + 1) % int(progress_every) == 0:
            el = time.time() - t_start
            print(f"[mvq-lift] frame {t + 1}/{T}  "
                  f"{(t + 1) / max(el, 1e-9):.1f} frames/s  "
                  f"collapsed {int(collapsed.sum())}", flush=True)
    _flush()
    if n_seen != T:
        raise ValueError(f"frames_iter yielded {n_seen} frames but the centres describe "
                         f"{T}; kp3d.npz must have exactly the masks' T or every "
                         f"downstream stage silently mis-indexes time")

    # ---- per-keypoint mask containment (Task 9), AFTER identity assignment
    # and BEFORE the write. Order matters both ways: run before the assignment
    # and there is no "own" fly to test against; run after `to_pipeline` and it
    # would have to be repeated per fly on permuted arrays. `kp3d`/`kp2d`/`vis`
    # are still in the model's own keypoint order here, which this filter is
    # agnostic to (it indexes no keypoint by name or by integer).
    containment_report = {"enabled": False, "reason": (
        "identity_resolved != mask" if identity_resolved != "mask" else
        "no mask_store" if mask_store is None else "asked off")}
    if containment_on:
        t_cont = time.time()
        kp3d, kp2d, vis, containment_report = mask_containment_filter(
            kp3d, kp2d, mask_store, runner.cam_mats,
            min_views=int(containment_min_views),
            own_margin=float(containment_own_margin_px),
            max_step_units=float(containment_max_step_units),
            conf_by_fly=vis, t0=int(mask_frame_offset))
        containment_report["seconds"] = round(time.time() - t_cont, 2)
        if verbose:
            d_o = containment_report["n_kp_dropped_other_mask"]
            d_s = containment_report["n_kp_dropped_spike"]
            fr = containment_report["frac_kp_dropped"]
            print(f"[mvq-lift] containment: dropped on-other-mask "
                  f"fly0 {d_o['fly0']} / fly1 {d_o['fly1']}, spike "
                  f"fly0 {d_s['fly0']} / fly1 {d_s['fly1']} keypoint-frames "
                  f"({100 * fr['fly0']:.2f}% / {100 * fr['fly1']:.2f}% of the finite "
                  f"ones), {containment_report['n_frames_testable']}/{T} frames "
                  f"testable, {containment_report['n_mask_unpacks']} mask unpacks, "
                  f"{containment_report['seconds']:.1f}s", flush=True)

    # ---- write the pipeline artifacts, keypoint axis permuted BY NAME
    from jarvis_jax.tracking.resume import atomic_save_json, atomic_save_npz
    os.makedirs(out_dir, exist_ok=True)
    n_missing = {}
    for fly in (0, 1):
        p = runner.to_pipeline(kp3d[fly], kp2d[fly], vis[fly], conf_raw[fly], model_names)
        if list(p["kp_names"]) != model_names:
            raise RuntimeError(
                f"to_pipeline returned keypoint order {list(p['kp_names'])[:4]}... which "
                f"is not cfg.model.KP_NAMES -- the pipeline would read every landmark as "
                f"a different body part")
        _check_eye_invariant(kp3d[fly], p["kp3d"], runner.kp_names, model_names)
        d = os.path.join(out_dir, f"fly{fly}")
        os.makedirs(d, exist_ok=True)
        atomic_save_npz(os.path.join(d, "kp2d.npz"), kp2d=p["kp2d"], conf=p["conf"],
                        cameras=p["cameras"], kp_names=p["kp_names"])
        atomic_save_npz(os.path.join(d, "kp3d.npz"), kp3d=p["kp3d"], conf3d=p["conf3d"],
                        conf3d_mvq_raw=p["conf3d_mvq_raw"], kp_names=p["kp_names"],
                        gates=np.asarray(gates_string))
        n_missing[f"fly{fly}"] = int((slot[fly] < 0).sum())

    # Did the SEX HEAD agree with who this fly turned out to be? 1 yes, 0 no,
    # -1 no instance written this frame. Under identity="sex" this is 1
    # wherever a fly was written (the typed slot IS the rule), so the number
    # that matters is the "mask" one: it measures how often the human review
    # and the model's typing disagree, which is the whole 20_04 defect.
    typed_slot = (int(SLOT_FEMALE), int(SLOT_MALE))
    sex_head_agrees = np.full((2, T), -1, np.int8)
    disagree_frac = {}
    # Only identity="mask" assigns instances to masks, so under "sex" these
    # would be a column of "none" that reads like a failure. None says "this
    # rule did not run" instead.
    reason_counts = ({f"fly{f}": {r: int(assign_reason[f].count(r))
                                  for r in ASSIGN_REASONS} for f in (0, 1)}
                     if identity_resolved == "mask" else None)
    for fly in (0, 1):
        wrote = slot[fly] >= 0
        sex_head_agrees[fly, wrote] = (slot[fly][wrote] == typed_slot[fly]).astype(np.int8)
        disagree_frac[f"fly{fly}"] = (
            float((slot[fly][wrote] != typed_slot[fly]).mean()) if wrote.any() else None)

    sex = _mvq_sex_json(sex_prob, exist, n_missing, T, mask_sex_meta,
                        identity_resolved, disagree_frac)
    atomic_save_json(os.path.join(out_dir, "sex.json"), sex)

    disagree = _identity_disagreements(sex, mask_sex_meta, review_male_fly,
                                       identity_resolved)
    for msg in disagree:
        print(f"[mvq-lift] identity NOTE: {msg}", flush=True)
    meta = {
        "checkpoint": runner.checkpoint,
        "step": runner.step_label,
        "gates": json.loads(gates_string),
        "exist_thresh": float(runner.exist_thresh),
        "merge_dist_units": float(merge_dist_units),
        "cameras": list(runner.cameras),
        "keypoint_names_mvq": list(runner.kp_names),
        "keypoint_names_written": model_names,
        # `identity` is what the gates string names (and what run_bout
        # recomputes); `identity_resolved` is what THIS bout actually did --
        # they differ only when masks with no human review forced a fallback.
        "identity": identity,
        "identity_resolved": identity_resolved,
        # which window each fly was read from, and how often the preference
        # had to fall back. Both take the EFFECTIVE value, like the gates dict:
        # under identity="sex" the mask assignment -- and so the preference --
        # does not run at all, so `window_pref` reads "any" (never the
        # requested mode) and the fallback counts are None.
        "window_pref": (window_pref if identity_resolved == "mask" else "any"),
        "n_from_other_window": ({"fly0": int(n_from_other_window[0]),
                                 "fly1": int(n_from_other_window[1])}
                                if identity_resolved == "mask" else None),
        "mask_assign_margin_units": float(mask_assign_margin_units),
        "mask_assign_max_units": float(mask_assign_max_units),
        "assign_reason_counts": reason_counts,
        "containment": containment_on,
        "containment_min_views": int(containment_min_views),
        "containment_own_margin_px": float(containment_own_margin_px),
        "containment_max_step_units": float(containment_max_step_units),
        "containment_report": containment_report,
        "sex_head_disagree_frac": disagree_frac,
        "fly_slots": {"fly0": int(SLOT_FEMALE), "fly1": int(SLOT_MALE)},
        "fly_sex": {"fly0": "female", "fly1": "male"},
        "n_frames": int(T),
        "n_missing": n_missing,
        "n_no_centre": int(no_centre.sum()),
        "collapse_dist_units": float(collapse_dist_units),
        "n_collapsed": {"fly0": int(n_collapsed[0]), "fly1": int(n_collapsed[1])},
        "collapsed_frac": float(collapsed.mean()) if T else 0.0,
        "mask_sex_meta": mask_sex_meta,
        "review_male_fly": review_male_fly,
        "identity_disagreements": disagree,
        "per_frame": {
            "n_windows": n_windows.astype(int).tolist(),
            "no_centre": no_centre.astype(int).tolist(),
            "collapsed": collapsed.astype(int).tolist(),
            "identity_source": [identity_resolved] * int(T),
            "slot": slot.astype(int).tolist(),
            "slot_used": slot.astype(int).tolist(),   # the brief's name for `slot`
            "sex_head_agrees": sex_head_agrees.astype(int).tolist(),
            "assign_reason": ([list(assign_reason[0]), list(assign_reason[1])]
                              if identity_resolved == "mask" else None),
            "mask_dist_units": np.round(np.nan_to_num(mask_dist, nan=-1.0), 3).tolist(),
            "window": window.astype(int).tolist(),
            "own_window": (own_window.astype(int).tolist()
                           if identity_resolved == "mask" else None),
            "exist": np.round(np.nan_to_num(exist, nan=-1.0), 4).tolist(),
            "sex_prob": np.round(np.nan_to_num(sex_prob, nan=-1.0), 4).tolist(),
            "exist_all_slots": np.round(np.nan_to_num(all_exist, nan=-1.0), 4).tolist(),
        },
    }
    meta.update(meta_extra or {})
    atomic_save_json(os.path.join(out_dir, "mvq_meta.json"), meta)
    if verbose:
        el = time.time() - t_start
        print(f"[mvq-lift] {out_dir}: {T} frames, identity {identity_resolved}, "
              f"missing {n_missing}, "
              f"sex-head disagree {disagree_frac}, "
              f"assign {reason_counts or 'n/a'}, "
              f"{int(no_centre.sum())} with no mask centre, "
              f"own-window {_own_frac_str(own_window, identity_resolved)}, "
              f"{int(collapsed.sum())} collapsed "
              f"({100 * (collapsed.mean() if T else 0):.1f}%, dropped "
              f"fly0 {n_collapsed[0]} / fly1 {n_collapsed[1]}), "
              f"{T / max(el, 1e-9):.1f} frames/s", flush=True)
    return {"skipped": False, "out_dir": out_dir, "gates": gates_string,
            "kp3d_mvq": kp3d, "kp2d_mvq": kp2d, "vis": vis, "conf_raw": conf_raw,
            "exist": exist, "sex_prob": sex_prob, "slot": slot, "window": window,
            "own_window": own_window,
            "n_from_other_window": ({"fly0": int(n_from_other_window[0]),
                                     "fly1": int(n_from_other_window[1])}
                                    if identity_resolved == "mask" else None),
            "window_pref": window_pref,
            "no_centre": no_centre, "n_windows": n_windows,
            "collapsed": collapsed,
            "identity": identity, "identity_resolved": identity_resolved,
            "identity_source": [identity_resolved] * int(T),
            "sex_head_agrees": sex_head_agrees,
            "sex_head_disagree_frac": disagree_frac,
            "assign_reason": assign_reason,
            "assign_reason_counts": reason_counts,
            "containment": containment_on,
            "containment_report": containment_report,
            "mask_dist": mask_dist,
            "n_collapsed": {"fly0": int(n_collapsed[0]), "fly1": int(n_collapsed[1])},
            "n_missing": n_missing, "sex": sex, "meta": meta}


def _check_eye_invariant(kp3d_mvq, kp3d_model, mvq_names, model_names):
    """EyeL-EyeR spacing is a RIGID head landmark pair: a by-name permutation
    cannot change it, and a by-index one almost certainly does. Cheap, and it
    is the check that would have caught the keypoint-order bug in CLAUDE.md."""
    if not ({"EyeL", "EyeR"} <= set(mvq_names) and {"EyeL", "EyeR"} <= set(model_names)):
        return
    iL, iR = mvq_names.index("EyeL"), mvq_names.index("EyeR")
    jL, jR = model_names.index("EyeL"), model_names.index("EyeR")
    d0 = np.nan_to_num(np.linalg.norm(kp3d_mvq[:, iL] - kp3d_mvq[:, iR], axis=-1))
    d1 = np.nan_to_num(np.linalg.norm(kp3d_model[:, jL] - kp3d_model[:, jR], axis=-1))
    if not np.allclose(d0, d1, atol=1e-4):
        bad = int(np.argmax(np.abs(d0 - d1)))
        raise RuntimeError(
            f"EyeL-EyeR distance moved through the keypoint permutation "
            f"({d0[bad]:.4f} -> {d1[bad]:.4f} units at frame {bad}); the written "
            f"kp3d.npz is NOT the same anatomy in a different order")


def _mvq_sex_json(sex_prob, exist, n_missing, T, mask_sex_meta,
                  identity=DEFAULT_IDENTITY, sex_head_disagree_frac=None):
    """`sex.json` in `sexing.canonicalize_bout`'s schema.

    Every key that function writes is present (its consumers -- notably
    `estimate_recording_scale._determine_identity`, which is what lets
    scale.json carry a per-fly body size -- read this file), plus the mvq
    evidence the heuristic fields have no room for.

    `method` names WHAT DECIDED IDENTITY, because that is what the
    authority chain in `sexing.canonicalize_bout` keys on:
    `mask_human_id_review` when the flies came from the masks' human review
    (identity="mask"), `mvq_sex_head` when they came from the model's typed
    slots (identity="sex"). Both are in that function's authoritative,
    no-swap set; the difference is which evidence a reader should trust and
    what `confidence` means below.
    """
    ms = mask_sex_meta or {}
    pf = {f"fly{f}": _mean_or_none(sex_prob[f]) for f in (0, 1)}
    ex = {f"fly{f}": _mean_or_none(exist[f]) for f in (0, 1)}
    mask_id = identity == "mask"
    if mask_id:
        # The human review IS the decision, so its confidence is not the sex
        # head's -- it is "a human looked at this bout". The sex head's
        # (dis)agreement is evidence, recorded below, not the verdict.
        method, confidence = MASK_ID_SEX_METHOD, "user"
        note = ("fly0 = SAM3 mask fly 0 (the female) and fly1 = mask fly 1 (the male), "
                "per the human ID review the masks carry; per frame the mvq instance "
                "nearest each mask's triangulated centre is written as that fly, and "
                "the model's sex head is recorded but does not decide")
    else:
        # The sex HEAD agreeing with the slot TYPING is the confidence signal:
        # the female slot should read P(female) > 0.5 and the male slot < 0.5.
        # When it does not, the identity is still the slot's (that is what was
        # trained), but the disagreement is recorded rather than smoothed over.
        agree = (pf["fly0"] is not None and pf["fly1"] is not None
                 and pf["fly0"] > 0.5 > pf["fly1"])
        method, confidence = MVQ_SEX_METHOD, ("high" if agree else "low")
        note = ("fly0 = the mvq FEMALE typed slot, fly1 = the mvq MALE typed slot "
                "(models/mvq typed decoder, slots 1/2); identity is per-frame from "
                "the model, not from mask slot order or a wing-song CV")
    return {
        "male_fly": 1,
        "original_male_fly": 1,
        "applied_swap": False,
        "confidence": confidence,
        "method": method,
        "authority": method,
        "heuristic_male_fly": None,
        "heuristic_method": f"not_run (mvq identity={identity})",
        "heuristic_agrees": None,
        "review_file": ms.get("review_file") if mask_id else None,
        "review_reviewed_at": ms.get("review_reviewed_at") if mask_id else None,
        "wing_cv_original": {"fly0": None, "fly1": None},
        "cv_ratio": None,
        "mask_song_cv": ms.get("song_cv"),
        "note": note,
        # --- mvq evidence (beyond canonicalize_bout's schema)
        "identity": identity,
        "sex_prob": pf,                     # mean P(female) of each written fly
        "exist": ex,                        # mean existence of each written fly
        # per fly, the fraction of WRITTEN frames whose instance was NOT the
        # sex head's typed slot for that fly -- 0 by construction when
        # identity="sex", and the size of the model's identity error when not
        "sex_head_disagree_frac": dict(sex_head_disagree_frac or {}),
        "n_frames": int(T),
        "n_missing": dict(n_missing),
    }


def _identity_disagreements(sex, mask_sex_meta, review_male_fly,
                            identity=DEFAULT_IDENTITY):
    """Human-readable notes where the identity sources disagree.

    Logged, never fatal. Under `identity="sex"` the mask/manifest slot order
    does not determine anything, so a mismatch is information; under
    `identity="mask"` a `male_slot != 1` has already been REFUSED upstream
    (fly{f} IS mask fly f there), and what is worth reporting instead is how
    often the model's sex head disagreed with the human review.
    """
    out = []
    ms = mask_sex_meta or {}
    if identity == "mask":
        d = sex.get("sex_head_disagree_frac") or {}
        for fly in ("fly0", "fly1"):
            v = d.get(fly)
            if v is not None and v > 0.05:
                out.append(
                    f"the mvq sex head disagrees with the human mask review on "
                    f"{100 * v:.1f}% of {fly}'s written frames (the instance on that "
                    f"mask was not the {'female' if fly == 'fly0' else 'male'}-typed "
                    f"slot); identity came from the review, as it should")
        if review_male_fly is not None and int(review_male_fly) != 1:
            out.append(
                f"the id-review manifest says male = fly{review_male_fly} in the tree "
                f"the reviewer watched; this run writes mask slot 1 as fly1, which the "
                f"masks' own sex_meta says is the male")
        return out
    if sex["confidence"] != "high":
        out.append(
            f"the mvq sex head does not separate the typed slots: mean P(female) "
            f"fly0={sex['sex_prob']['fly0']}, fly1={sex['sex_prob']['fly1']} "
            f"(expected fly0 > 0.5 > fly1)")
    if ms.get("male_slot") is not None and int(ms["male_slot"]) != 1:
        out.append(
            f"the SAM3 masks' sex_meta says male = mask slot {ms['male_slot']} "
            f"(method {ms.get('method')!r}); the mvq route reads TYPED slots, so "
            f"fly1 is the male regardless of mask slot order")
    if review_male_fly is not None and int(review_male_fly) != 1:
        out.append(
            f"the id-review manifest says male = fly{review_male_fly} in the tree the "
            f"reviewer watched; this run's fly1 is the mvq male typed slot")
    return out
