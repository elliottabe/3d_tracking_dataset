"""Load a bout's kp2d/kp3d artifacts with the RIGHT keypoint and camera order.

Both orders are traps, and on 2026-08-31 both were fallen into in one sitting,
producing four rounds of confident, self-consistent, WRONG wing measurements
that no metric flagged:

  KEYPOINT ORDER. run_bout.py writes kp2d.npz/kp3d.npz AFTER
  ``reorder_detector_to_model`` (run_bout.py:986), so their keypoint axis is
  ``cfg.model.KP_NAMES`` (the XML/model order) -- NOT the detector's
  ``cfg.detector.kp_names``. The two differ. Indexing the artifacts with
  detector indices reads other body parts: detector "WingR_base/V12/V13"
  (28/29/30) land on model ``T2L_TiTa/TaT1/TaT3`` -- the middle-LEFT LEG. That
  produced a "collapsed right wing vein, 2.53u vs 21.58u, CV 4.1%, confidence
  0.96" which was really the length of a tarsal segment: a correct measurement
  of the wrong thing.

  CAMERA ORDER. kp2d's camera axis is the pipeline's canonical
  ``cfg.recording.cameras`` order (== the calibration order), because
  ``load_bout_masks(..., expected_cameras=cameras)`` reorders the mask npz BY
  NAME. The npz's own stored ``cameras`` array is a DIFFERENT order. Taking a
  camera index from the npz list and using it on kp2d (or on
  ``ReprojectionTool.camera_matrices``) plots one camera's keypoints on another
  camera's image.

The fix this module enforces: you never index by integer. Ask for a keypoint by
NAME and a camera by NAME, and the mapping is done once, here, from the configs
that actually define it.

    from viz.core.bout_artifacts import load_bout_kp
    b = load_bout_kp(bout_dir, fly=1, anatomy_cfg="configs/anatomy/v1.yaml",
                     cameras=cfg.recording.cameras)
    v12 = b.kp3d(  "WingL_V12")          # (T,3)
    xy  = b.kp2d("WingL_V12", "Cam2012630")   # (T,2) full-frame px
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


class KeypointOrderError(KeyError):
    """A keypoint name is absent from model.KP_NAMES."""


class CameraOrderError(KeyError):
    """A camera name is absent from the canonical camera list."""


@dataclass
class BoutKeypoints:
    kp_names: list
    cameras: list
    _kp2d: np.ndarray                      # (T,C,K,2) canonical cam, model kp
    _conf2d: np.ndarray                    # (T,C,K)
    _kp3d: np.ndarray | None               # (T,K,3)
    _conf3d: np.ndarray | None             # (T,K)
    _idx: dict = field(default_factory=dict)
    _cam: dict = field(default_factory=dict)

    def __post_init__(self):
        self._idx = {n: i for i, n in enumerate(self.kp_names)}
        self._cam = {c: i for i, c in enumerate(self.cameras)}

    def k(self, name):
        """Keypoint index by NAME. Raises rather than returning a wrong part."""
        try:
            return self._idx[name]
        except KeyError:
            raise KeypointOrderError_(name, self.kp_names) from None

    def c(self, cam):
        try:
            return self._cam[cam]
        except KeyError:
            raise CameraOrderError(
                f"camera {cam!r} not in the canonical order {self.cameras}") from None

    def kp2d(self, name, cam=None):
        i = self.k(name)
        return self._kp2d[:, self.c(cam), i, :] if cam is not None else self._kp2d[:, :, i, :]

    def conf2d(self, name, cam=None):
        i = self.k(name)
        return self._conf2d[:, self.c(cam), i] if cam is not None else self._conf2d[:, :, i]

    def kp3d(self, name):
        if self._kp3d is None:
            raise FileNotFoundError("this bout has no kp3d.npz")
        return self._kp3d[:, self.k(name), :]

    def conf3d(self, name):
        if self._conf3d is None:
            raise FileNotFoundError("this bout has no kp3d.npz")
        return self._conf3d[:, self.k(name)]

    def segment_length(self, a, b, *, use_3d=True):
        """|a-b| per frame -- for RIGID pairs this must be constant, which is
        the check that catches a keypoint-order bug when every jitter and
        confidence metric looks fine."""
        if not use_3d:
            raise ValueError("segment_length is 3-D only (2-D length is not rigid)")
        return np.linalg.norm(self.kp3d(a) - self.kp3d(b), axis=-1)


def KeypointOrderError_(name, names):
    wings = [n for n in names if "Wing" in n]
    return KeypointOrderError(
        f"keypoint {name!r} is not in model.KP_NAMES. These artifacts are in "
        f"MODEL order (post reorder_detector_to_model), whose wing names are "
        f"{wings}. If you passed a DETECTOR kp_names list, that is the "
        f"2026-08-31 bug: detector indices read other body parts.")


def model_kp_names(anatomy_cfg="configs/anatomy/v1.yaml"):
    """`model.KP_NAMES` -- the order kp2d.npz/kp3d.npz are actually stored in.

    Use this instead of reaching for `.model.KP_NAMES` yourself, so passing a
    DETECTOR config produces the explanation rather than an OmegaConf
    attribute error.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(anatomy_cfg) if isinstance(anatomy_cfg, str) else anatomy_cfg
    if "model" not in cfg or "KP_NAMES" not in cfg.model:
        raise KeypointOrderError(
            f"{anatomy_cfg} has no model.KP_NAMES. kp2d/kp3d are in MODEL order "
            f"(written after reorder_detector_to_model), so indexing them with a "
            f"detector kp_names list reads the WRONG body parts -- on 2026-08-31 "
            f"detector WingR_* indices read model T2L_* (a LEG). Pass an anatomy "
            f"config, e.g. configs/anatomy/v1.yaml.")
    return list(cfg.model.KP_NAMES)


def load_bout_kp(bout_dir, fly, *, anatomy_cfg="configs/anatomy/v1.yaml",
                 cameras, kp3d_file="kp3d.npz"):
    """Load ``<bout_dir>/fly<fly>/{kp2d,kp3d}.npz`` with named access.

    `cameras` MUST be the pipeline's canonical order (``cfg.recording.cameras``),
    which is what kp2d's camera axis is in. It is required, not defaulted, so a
    caller cannot silently inherit the mask npz's different order.
    """
    kp_names = model_kp_names(anatomy_cfg)
    cameras = list(cameras)

    d = os.path.join(bout_dir, f"fly{fly}")
    z2 = np.load(os.path.join(d, "kp2d.npz"))
    k2, c2 = z2["kp2d"], z2["conf"]
    if k2.shape[2] != len(kp_names):
        raise KeypointOrderError(
            f"kp2d has K={k2.shape[2]} but model.KP_NAMES has {len(kp_names)}")
    if k2.shape[1] != len(cameras):
        raise CameraOrderError(
            f"kp2d has C={k2.shape[1]} but {len(cameras)} camera names were given")
    k3 = c3 = None
    p3 = os.path.join(d, kp3d_file)
    if os.path.exists(p3):
        z3 = np.load(p3); k3, c3 = z3["kp3d"], z3["conf3d"]
    return BoutKeypoints(kp_names, cameras, k2, c2, k3, c3)


def centroids_canonical(masks_npz, cameras):
    """Mask-npz centroids reindexed BY NAME into the canonical camera order.

    The npz stores its own camera order; using its index against kp2d is the
    2026-08-31 camera bug. Returns (2,C,T,2) in `cameras` order.
    """
    z = np.load(masks_npz, allow_pickle=True)
    stored = [str(c) for c in z["cameras"]]
    missing = [c for c in cameras if c not in stored]
    if missing:
        raise CameraOrderError(f"cameras absent from {masks_npz}: {missing}")
    perm = [stored.index(c) for c in cameras]
    return z["centroids"][:, perm], z["valid"][:, perm], perm
