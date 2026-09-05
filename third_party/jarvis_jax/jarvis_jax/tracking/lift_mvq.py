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


def mvq_gate_signature(checkpoint, *, step=None, exist_thresh=None):
    """The Stage-B `gates` payload for an mvq-lifted kp3d.npz.

    With `pipeline.lifter: mvq` the DLT gates (view-conf, mask-agreement,
    wing-collapse, rigid-repair) never run -- Stages A and B are replaced by
    the lifter -- so what determines the file's contents is WHICH CHECKPOINT
    produced it and the existence threshold that decided which frames are
    NaN. `scripts/run_bout.py::stage_b_gate_signature` returns
    `json.dumps(this, sort_keys=True)` for such a run, so a bout produced by
    one checkpoint is refused after the config points at another, exactly as
    a changed DLT gate is.
    """
    if not checkpoint:
        raise ValueError("pipeline.lifter=mvq needs mvq.checkpoint set -- the Stage-B gate "
                         "signature must name the weights that produced kp3d.npz")
    step = resolved_step(step)                    # refuses "latest" by name
    return {"lifter": "mvq",
            "checkpoint": os.path.abspath(str(checkpoint)),
            "step": "final" if step is None else step,
            "sha256": checkpoint_sha256(checkpoint, step),
            "exist_thresh": float(EXIST_THRESH if exist_thresh is None else exist_thresh)}


def mvq_gate_string(checkpoint, *, step=None, exist_thresh=None):
    """`mvq_gate_signature` as the stable string stored inside kp3d.npz."""
    return json.dumps(mvq_gate_signature(checkpoint, step=step, exist_thresh=exist_thresh),
                      sort_keys=True)


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
                 batch=32, exist_thresh=EXIST_THRESH):
        self.checkpoint = os.path.abspath(str(run_dir_or_final))
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
        exist = float(out["exist"][bi, slot])
        return {"slot": int(slot),
                "kp3d": out["kp3d"][bi, slot],
                "kp2d": out["kp2d"][bi, slot],
                "vis": out["vis"][bi, slot],
                "conf_raw": out["conf_raw"][bi, slot],
                "exist": exist,
                "sex_prob": float(out["sex_prob"][bi, slot])}

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
            self._gates = mvq_gate_signature(self.checkpoint, step=self.step,
                                             exist_thresh=self.exist_thresh)
        return dict(self._gates)

    def gates_string(self):
        """`gates_signature` as the string stored inside kp3d.npz -- byte-equal
        to what `run_bout.py::stage_b_gate_signature` computes for this run."""
        return json.dumps(self.gates_signature(), sort_keys=True)
