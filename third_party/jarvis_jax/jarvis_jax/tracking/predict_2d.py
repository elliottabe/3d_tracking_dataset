"""ViTPose 2-D inference on the SAM-cropped fly (V2VNet-free front end).

Load the retrained 4-ch ViTPose, crop each camera on the triangulated-centroid
reprojection (build_frameset), run the model on the (448,448,4) crops, decode
heatmap peaks (+ peak-value confidence), and map crop keypoints back to
full-frame pixels. No V2VNet / no 3-D volume.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.device import normalize_image
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from jarvis_jax.predict.session_frameset import build_frameset
from jarvis_jax.geometry.center3d import centroids_to_fullpx


def load_detector(ckpt: str, num_keypoints: int = 50):
    vit = load_vitpose(ckpt, ViTPoseConfig(num_keypoints=num_keypoints))
    vit.eval()
    return vit


def detector_to_model_perm(detector_kp_names, model_kp_names):
    """Permutation mapping detector output channels -> model/pipeline (XML) order.

    The retrained ViTPose emits keypoints in the tracking/COCO training order
    ``detector_kp_names`` (e.g. Antenna_Base, EyeL, EyeR, Scutellum, ...,
    all-left-legs then all-right-legs). The rest of the pipeline (triangulation,
    STAC, silhouette IK, QC) indexes channels by ``model_kp_names`` == the XML
    site order (cfg.model.KP_NAMES). These orders DIFFER, so the detector output
    must be reordered before it enters the pipeline -- otherwise every channel is
    mapped to the wrong model site (scrambled/flipped fits).

    Returns ``perm`` s.t. ``kp_model[..., k, :] = kp_detector[..., perm[k], :]``.
    Fails loud (ValueError) if the two name sets differ -- a silent mismatch here
    is exactly the class of bug this guards against.
    """
    det = list(detector_kp_names); mod = list(model_kp_names)
    if set(det) != set(mod):
        missing = sorted(set(mod) - set(det)); extra = sorted(set(det) - set(mod))
        raise ValueError(
            "detector kp_names and model KP_NAMES are not the same set of "
            f"landmarks -- cannot reorder. in model not detector: {missing}; "
            f"in detector not model: {extra}")
    di = {n: i for i, n in enumerate(det)}
    return [di[n] for n in mod]


def reorder_detector_to_model(kp2d, conf, detector_kp_names, model_kp_names):
    """Reorder detector output (kp2d (...,K,2), conf (...,K)) from the detector's
    channel order into model/XML order. K==0 (empty bout) is passed through.
    Fails loud if K disagrees with the name lists."""
    K = kp2d.shape[-2]
    if K == 0:
        return kp2d, conf
    if K != len(model_kp_names) or K != len(detector_kp_names):
        raise ValueError(
            f"keypoint count {K} != detector names ({len(detector_kp_names)}) / "
            f"model names ({len(model_kp_names)})")
    perm = detector_to_model_perm(detector_kp_names, model_kp_names)
    return kp2d[..., perm, :], conf[..., perm]


class DetectorOrderMismatch(ValueError):
    """A checkpoint's recorded training keypoint order disagrees with the order
    the config declares it emits."""


def _training_kp_names_for_ckpt(ckpt_path):
    """Recover the keypoint order a detector checkpoint was TRAINED on.

    `cfg.detector.kp_names` is a hand-maintained assertion about what channel
    order a checkpoint emits. Nothing checked it, and a wrong assertion
    scrambles anatomy silently: reorder_detector_to_model is name-based, so a
    mislabelled channel list permutes every keypoint into the wrong slot while
    residuals, NaN counts and IoU all stay plausible. This is the same failure
    class as the historical keypoint-order bug that LOO/IoU was blind to.

    Resolution chain, all on-disk artifacts:
        <ckpt>/..            -> the run directory (ckpt is <run>/final)
        <run>/.hydra/overrides.yaml -> paths.data_root
        <data_root>/annotations/keypoint_names.json -> the true training order

    Returns the name list, or None when the chain cannot be resolved (older
    runs predate `keypoint_names.json`; red_data_unified_V4 has none).
    """
    import json
    import os
    run_dir = os.path.dirname(os.path.normpath(str(ckpt_path)))
    ov = os.path.join(run_dir, ".hydra", "overrides.yaml")
    if not os.path.exists(ov):
        return None
    data_root = None
    with open(ov) as f:
        for line in f:
            line = line.strip().lstrip("- ").strip()
            if line.startswith("paths.data_root="):
                data_root = line.split("=", 1)[1].strip()
    if not data_root:
        return None
    names_path = os.path.join(data_root, "annotations", "keypoint_names.json")
    if not os.path.exists(names_path):
        return None
    with open(names_path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("keypoint_names") or d.get("names") or next(iter(d.values()))
    return list(d)


def verify_detector_kp_order(ckpt_path, declared_kp_names, *, strict=True):
    """Check `declared_kp_names` against the checkpoint's own training order.

    Raises DetectorOrderMismatch when the two disagree -- that is a silent
    anatomy scramble, so it must stop the run rather than warn. Returns the
    recovered training order on success.

    When the checkpoint predates `annotations/keypoint_names.json` the order
    cannot be verified from artifacts; that WARNS rather than raises, because
    refusing to run on an old-but-known-good checkpoint would be worse than
    the risk it carries. `strict=False` downgrades a real mismatch to a warning
    too -- for tooling that deliberately inspects a mismatched pair.
    """
    import warnings
    trained = _training_kp_names_for_ckpt(ckpt_path)
    declared = list(declared_kp_names)
    if trained is None:
        warnings.warn(
            f"detector keypoint order UNVERIFIED for {ckpt_path}: no "
            f"annotations/keypoint_names.json reachable via its "
            f".hydra/overrides.yaml. Trusting cfg.detector.kp_names "
            f"({len(declared)} names) on faith.", UserWarning, stacklevel=2)
        return None
    if trained == declared:
        return trained
    if set(trained) == set(declared):
        first = next(i for i, (a, b) in enumerate(zip(trained, declared)) if a != b)
        detail = (f"same {len(trained)} landmarks, PERMUTED: first disagreement at "
                  f"channel {first} (trained {trained[first]!r} vs declared "
                  f"{declared[first]!r})")
    else:
        detail = (f"different landmark sets: only in training "
                  f"{sorted(set(trained) - set(declared))[:5]}; only in config "
                  f"{sorted(set(declared) - set(trained))[:5]}")
    msg = (f"detector keypoint order MISMATCH for {ckpt_path}: {detail}. "
           f"cfg.detector.kp_names does not describe this checkpoint, so "
           f"reorder_detector_to_model would permute every keypoint into the "
           f"wrong slot and the error would be invisible to residual/NaN/IoU "
           f"checks. Fix cfg.detector.kp_names to the checkpoint's training "
           f"order before running.")
    if strict:
        raise DetectorOrderMismatch(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
    return trained


def peaks_and_conf(hm, *, decode_sharpen=1.0):
    """hm (B,Hh,Wh,K) logits -> (kp2d_crop (B,K,2) in 448px, conf (B,K) peak value).
    decode_sharpen>1 sharpens the soft-argmax centroid (kills high-freq wobble
    from ViTPose's diffuse tails; see docs/superpowers/plans 2D-wobble)."""
    kp = heatmaps_to_keypoints(hm, in_size=448, sharpen=decode_sharpen)  # (B,K,2)
    conf = jax.nn.relu(hm).max(axis=(1, 2))                    # (B,K)
    return kp, conf


def _forward(vit, crops4_u8, *, decode_sharpen=1.0):
    """(B,448,448,4) uint8 -> (kp2d_crop (B,K,2), conf (B,K)) on device."""
    x = normalize_image(jnp.asarray(crops4_u8))
    hm = vit(x, use_running_average=True)
    return peaks_and_conf(hm, decode_sharpen=decode_sharpen)


def predict_bout_2d(vitpose, frames_iter, masks, centroids, valid, cam_mats,
                    *, crop: int = 448, batch: int = 64, decode_sharpen: float = 1.0,
                    distractor_masks=None, distractor_dilate=0,
                    target_protect=None, zero_mask_channel: bool = False):
    """Per (frame,cam): crop -> ViTPose -> full-frame 2-D kp + conf.

    frames_iter: iterable of length T, each -> (C,H,W,3) uint8 RGB (all cameras
    for that frame). masks (T,C,H,W) bool, centroids (T,C,2), valid (T,C),
    cam_mats (C,4,3). Returns kp2d (T,C,K,2) full-frame, conf (T,C,K) (0 where
    the frame had <2 valid views or the crop was empty).

    distractor_dilate: px to dilate the distractor mask (and, symmetrically, the
    target mask that is excluded from it) before filling. A SAM3 mask covers the
    BODY only, so limbs survive the fill and the crop keeps a fly-shaped object
    the detector prefers when the real target is small or edge-on. 0 = previous
    behaviour.

    zero_mask_channel: zero the 4th (SAM-mask) channel of every crop before
    the forward pass. Set it from the same config block that names the ckpt
    for any checkpoint trained with train.mask_ablation=true, whose 4th
    channel was always 0 during training.

    CORRECTED 2026-09-02 (docs/benchmark/2026-09-02-vitpose-maskoff-ab/).
    This docstring previously claimed "v5vf_maskoff: 5.29px overall / 9.93px
    female val MPJPE, vs 5.49 / 19.08 ... the channel roughly doubled female
    error". Re-measured on the full red_data_3d_v5_valfix val split:
      * the OVERALL numbers hold -- 5.291 (maskoff, zeroed) vs 5.483
        (maskon, populated);
      * the FEMALE numbers do NOT. 9.932 / 19.080 are reproducible, but only
        as the MPJPE of the single recording 2026_05_27_11_56_05: 14 of 1871
        val annotations (0.7%), from a recording whose manifest split is
        "mixed". Across all 304 female annotations the gap is 4.854 vs
        5.250 -- 8.2%, not ~2x -- and it disappears on the only female
        recording never seen in training (3.748 vs 3.658).

    Also measured: for v5vf_maskoff this flag is a NO-OP for accuracy.
    Every one of the 196608 weights in patch_embed.proj.kernel[:, :, 3, :]
    is exactly 0.0, so the model is mathematically invariant to the 4th
    channel and its predictions are BITWISE identical either way (max abs
    difference 0.0 over 1871x50x2 values). The same holds for the other
    train.mask_ablation=true run, mask_ablation_zeroed_v5s70, so it is a
    property of the recipe. Keep the flag anyway -- it is correct, and it is
    what protects a future ablation checkpoint that is not perfectly
    invariant -- but "omitting it silently degrades accuracy" is not true of
    this checkpoint. A normally-trained checkpoint DOES use the channel:
    zeroing v5vf_maskon costs +36% overall (5.483 -> 7.453).

    The mask itself is still required regardless: it sets the crop centre and
    drives the distractor gray-fill below. Only the 4th CHANNEL is zeroed.

    distractor_masks (T,C,H,W) bool | None: the OTHER animal(s)' masks. Passed
    through to build_frameset, which replaces those pixels with the crop mean
    so the detector sees one clean fly plus the target-mask channel -- matching
    JARVIS's dataset2D training crop. This was NOT wired up, and build_frameset's
    own docstring warns that omitting it "feeds two overlapping flies on
    courtship frames and collapses the 3D reconstruction". Measured on the
    courtship set: with both flies in the crop and nothing marking the target,
    keypoints landed on the WRONG fly in >90% of unambiguous frames in 5 bouts
    and 10-90% in 33 more. None for single-animal assays.
    """
    T = masks.shape[0]; C = masks.shape[1]
    K = None                                         # discovered from first forward
    crops, centerHMs, keep = [], [], []            # collect valid (frame) crops
    per_frame = []                                  # (t, centerHM) for reassembly
    for t, frame_imgs in enumerate(frames_iter):
        c4, centerHM, nval = build_frameset(
            np.asarray(frame_imgs), masks[t], centroids[t], valid[t], cam_mats,
            crop=crop,
            distractor_masks=(None if distractor_masks is None
                              else distractor_masks[t]),
            distractor_dilate=distractor_dilate,
            target_protect=target_protect)
        if c4 is None:
            per_frame.append((t, None, None))
            continue
        crops.append(c4); centerHMs.append(centerHM); per_frame.append((t, len(crops) - 1, centerHM))
    # infer in batches of frames (each frame = C crops)
    kp2d = conf = None
    if crops:
        allc = np.concatenate(crops, 0)             # (n_valid_frames*C, 448,448,4)
        if zero_mask_channel:
            # This checkpoint was TRAINED with channel 3 == 0
            # (train.mask_ablation=true / data.mask_zero.ZeroMaskDataset), so it
            # has never seen a SAM mask there. Feeding one is silently OOD --
            # no shape error, just degraded keypoints -- which is why the flag
            # lives beside `ckpt` in configs/detector/*.yaml and travels with
            # it. The mask is still USED, for the crop centre and the
            # distractor gray-fill above; only the 4th CHANNEL is zeroed.
            allc = allc.copy()
            allc[..., 3] = 0
        outs_kp, outs_cf = [], []
        for i in range(0, allc.shape[0], batch):
            k, cf = _forward(vitpose, allc[i:i + batch], decode_sharpen=decode_sharpen)
            outs_kp.append(np.asarray(k)); outs_cf.append(np.asarray(cf))
        kp_crop = np.concatenate(outs_kp, 0); cf = np.concatenate(outs_cf, 0)
        K = kp_crop.shape[1]
        kp2d = np.zeros((T, C, K, 2), np.float32); conf = np.zeros((T, C, K), np.float32)
        for t, ci, centerHM in per_frame:
            if ci is None:
                continue
            block_kp = kp_crop[ci * C:(ci + 1) * C]  # (C,K,2) crop px
            block_cf = cf[ci * C:(ci + 1) * C]
            # centerHM (C,2) -> (C,1,2) so it broadcasts per-camera across K keypoints
            kp2d[t] = centroids_to_fullpx(block_kp, centerHM[:, None, :], crop)
            conf[t] = block_cf
    else:
        kp2d = np.zeros((T, C, 0, 2), np.float32); conf = np.zeros((T, C, 0), np.float32)
    return kp2d, conf
