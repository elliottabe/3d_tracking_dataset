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
                    *, crop: int = 448, batch: int = 64, decode_sharpen: float = 1.0):
    """Per (frame,cam): crop -> ViTPose -> full-frame 2-D kp + conf.

    frames_iter: iterable of length T, each -> (C,H,W,3) uint8 RGB (all cameras
    for that frame). masks (T,C,H,W) bool, centroids (T,C,2), valid (T,C),
    cam_mats (C,4,3). Returns kp2d (T,C,K,2) full-frame, conf (T,C,K) (0 where
    the frame had <2 valid views or the crop was empty).
    """
    T = masks.shape[0]; C = masks.shape[1]
    K = None                                         # discovered from first forward
    crops, centerHMs, keep = [], [], []            # collect valid (frame) crops
    per_frame = []                                  # (t, centerHM) for reassembly
    for t, frame_imgs in enumerate(frames_iter):
        c4, centerHM, nval = build_frameset(
            np.asarray(frame_imgs), masks[t], centroids[t], valid[t], cam_mats, crop=crop)
        if c4 is None:
            per_frame.append((t, None, None))
            continue
        crops.append(c4); centerHMs.append(centerHM); per_frame.append((t, len(crops) - 1, centerHM))
    # infer in batches of frames (each frame = C crops)
    kp2d = conf = None
    if crops:
        allc = np.concatenate(crops, 0)             # (n_valid_frames*C, 448,448,4)
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
