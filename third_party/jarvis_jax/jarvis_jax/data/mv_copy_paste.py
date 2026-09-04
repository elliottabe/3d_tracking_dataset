"""Affine-exact multi-view copy-paste (P3a spec §6). A donor fly (the host of
another window of the SAME calibration group) is translated by a 3D offset D
in the target's ROI-local frame; under affine cameras that is the per-camera
2D translation `M_c D + t_local_tgt[c] - t_local_src[c]` in crop px, so pixels,
2D labels and 3D labels stay mutually consistent in every view. Pure numpy +
cv2; runs in the loader thread pool."""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from jarvis_jax.train.matching import SEX_UNKNOWN


@dataclasses.dataclass(frozen=True)
class CopyPasteParams:
    p: float = 0.5
    opposite_sex_p: float = 0.7
    contact_p: float = 0.3
    contact_sep: tuple = (8.0, 30.0)      # world units (0.1 mm): the stacked-pair regime -- real
                                          # mounting pairs sit at ~24-30 units; the low end gives
                                          # heavier overlap than reality (amended 2026-09-04, see
                                          # docs/benchmark/2026-09-mvq/p3a-notes.md)
    far_sep: tuple = (15.0, 60.0)
    max_tries: int = 8
    gain_clip: tuple = (0.7, 1.4)


def body_plane_axes(kp3d_local, has3d):
    """(2,3) orthonormal axes spanning the two largest principal directions of
    the labelled points (the floor for a walking fly, the wall for a climber)."""
    pts = np.asarray(kp3d_local, np.float64)[np.asarray(has3d, bool)]
    if pts.shape[0] < 3:
        return np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    pts = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    return vt[:2].astype(np.float32)


def sample_offset(rng, axes, params: CopyPasteParams):
    lo, hi = params.contact_sep if rng.uniform() < params.contact_p else params.far_sep
    sep = rng.uniform(lo, hi); ang = rng.uniform(0, 2 * np.pi)
    return (sep * (np.cos(ang) * axes[0] + np.sin(ang) * axes[1])).astype(np.float32)


def view_shifts(M, D, t_local_src, t_local_tgt):
    """(C,2) crop-px translation that moves the donor's crop content to where the
    donor would appear in the TARGET crop after a 3D offset D.

    Derivation: kp2d_local = M @ X_local + t_local in both frames (loader
    convention), so the donor's own pixel is uv_src = M @ X + t_local_src and
    the pasted pixel is uv_tgt = M @ (X + D) + t_local_tgt. The shift added to
    uv_src to get uv_tgt is therefore M @ D + t_local_tgt - t_local_src."""
    return (np.einsum("cij,j->ci", np.asarray(M, np.float64), np.asarray(D, np.float64))
            + np.asarray(t_local_tgt, np.float64) - np.asarray(t_local_src, np.float64)).astype(np.float32)


def _translate(img, shift, nearest=False):
    h, w = img.shape[:2]
    A = np.array([[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]], np.float64)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, A, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def composite(tgt, src, D, params: CopyPasteParams):
    """Paste `src`'s host fly into `tgt` as fly 1. Both are T=1 samples from
    V12WindowDataset (numpy). `tgt` must have exactly one labelled fly (slot 0)
    and no unlabelled animal (raises ValueError otherwise -- the loader hook
    is responsible for only offering such targets). Returns a NEW sample dict,
    or None when rejected: a target-valid camera the donor lacks (cam_valid
    mismatch), or the shift pushing the donor's ACTUAL pasted content -- its
    SAM mask -- entirely off a target-valid camera that had donor mask
    content to begin with (the donor would not be visible there at all).
    Individual label points that scatter past the crop edge under the shift,
    or that fall in a camera the donor mask never painted, are marked
    invisible per-keypoint (`vis2d` below) rather than rejecting the whole
    paste: label spread routinely puts a few of many keypoints near/over the
    crop edge at ordinary contact-range offsets even though the mask -- what
    is actually composited into the pixels -- stays fully inside. A pasted
    keypoint invisible in EVERY view this way also loses its 3D label
    (`has3d`/`kp3d_local`): no label may claim evidence from pixels that
    never actually landed anywhere."""
    if tgt["fly_valid"].shape[0] < 2 or bool(tgt["fly_valid"][1]) or int(tgt["unlabelled_sex"]) != SEX_UNKNOWN:
        raise ValueError("composite: target must have exactly one labelled fly and no unlabelled animal "
                         "(loader hook enforces this)")
    crops = tgt["crops"]; T, C, H, W, _ = crops.shape
    assert T == 1, "copy-paste is T=1 only (P3a)"
    tv, sv = tgt["cam_valid"][0], src["cam_valid"][0]
    if np.any(tv & ~sv):
        return None
    shifts = view_shifts(tgt["M"], D, src["t_local"][0], tgt["t_local"][0])         # (C,2)
    kp2d_d = src["kp2d"][0, 0] + shifts[:, None, :]                                  # (C,K,2)
    vis_d = src["vis2d"][0, 0].copy()                                                # (C,K)
    inside = (kp2d_d >= 0).all(-1) & (kp2d_d[..., 0] <= W - 1) & (kp2d_d[..., 1] <= H - 1)
    shifted_masks = [None] * C
    painted = np.zeros(C, bool)              # camera actually got donor mask pixels composited into it
    for c in range(C):
        if not tv[c]:
            continue
        src_mask_c = src["prompt_mask"][0, c]
        if not src_mask_c.any():
            continue                                    # no donor mask content in this view anyway
        m = _translate(src_mask_c.astype(np.uint8), shifts[c], nearest=True).astype(bool)
        if not m.any():
            return None                                  # shift pushed it entirely off the crop
        shifted_masks[c] = m
        painted[c] = True
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tgt.items()}
    lo, hi = params.gain_clip
    for c in range(C):
        if not tv[c]:
            continue
        m = shifted_masks[c]
        if m is None:
            continue
        donor = _translate(src["crops"][0, c], shifts[c])
        gain = float(np.clip((np.median(crops[0, c, ::4, ::4]) + 1.0)
                              / (np.median(src["crops"][0, c, ::4, ::4]) + 1.0), lo, hi))
        donor = np.clip(donor.astype(np.float32) * gain, 0, 255)
        alpha = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]          # 1-px feather
        out["crops"][0, c] = (crops[0, c] * (1 - alpha) + donor * alpha).astype(np.uint8)
        # donor on top: host keypoints under it are occluded; host prompt loses those pixels
        hk = np.round(tgt["kp2d"][0, 0, c]).astype(int)
        ok = (hk[:, 0] >= 0) & (hk[:, 0] < W) & (hk[:, 1] >= 0) & (hk[:, 1] < H)
        covered = np.zeros(hk.shape[0], bool); covered[ok] = m[hk[ok, 1], hk[ok, 0]]
        out["vis2d"][0, 0, c] &= ~covered
        out["prompt_mask"][0, c] &= ~m
    # final per-camera visibility for the pasted fly: geometrically inside the
    # crop AND the camera is one the donor mask was actually painted into
    # (a camera the donor never covered, or that got rejected-empty above and
    # skipped, shows no pixel evidence of the pasted fly either).
    final_vis = vis_d & inside & tv[:, None] & painted[:, None]
    dead = vis_d.any(0) & ~final_vis.any(0)     # donor labelled it, but it landed nowhere visible
    out["kp2d"][1, 0] = kp2d_d.astype(np.float32)
    out["vis2d"][1, 0] = final_vis
    out["has3d"][1, 0] = src["has3d"][0, 0] & ~dead
    out["kp3d_local"][1, 0] = (src["kp3d_local"][0, 0] + D) * out["has3d"][1, 0][:, None]
    out["fly_valid"][1] = True
    out["fly_sex"][1] = src["fly_sex"][0]
    out["unlabelled_sex"] = np.int8(SEX_UNKNOWN)
    return out
