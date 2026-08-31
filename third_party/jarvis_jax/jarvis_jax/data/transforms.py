"""Pure NumPy transforms for the V3 keypoint loader: cropping, RGB
normalization, keypoint coordinate transforms, and Gaussian heatmap rendering.
"""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_origin(bbox, img_w, img_h, crop=448):
    """Top-left (x0, y0) of a `crop`x`crop` window centered on the bbox center,
    clamped so the window stays inside the image. bbox is [x, y, w, h]."""
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    x0 = int(round(cx - crop / 2.0))
    y0 = int(round(cy - crop / 2.0))
    x0 = max(0, min(x0, img_w - crop))
    y0 = max(0, min(y0, img_h - crop))
    return x0, y0


def normalize_rgb(rgb01):
    """(H,W,3) RGB in [0,1] -> ImageNet-normalized float32."""
    return ((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def transform_keypoints(kps, x0, y0, crop=448, heatmap_size=224):
    """Map full-image keypoints into heatmap coords.

    kps: (50,3) = [x, y, v]. Returns (hm_xy (50,2) float32, vis (50,) bool).
    A keypoint is visible if v>0 and its heatmap coord is inside the grid.
    """
    scale = heatmap_size / float(crop)
    xy = kps[:, :2].astype(np.float32)
    v = kps[:, 2]
    hm_xy = np.empty((kps.shape[0], 2), dtype=np.float32)
    hm_xy[:, 0] = (xy[:, 0] - x0) * scale
    hm_xy[:, 1] = (xy[:, 1] - y0) * scale
    inb = (
        (hm_xy[:, 0] >= 0) & (hm_xy[:, 0] <= heatmap_size - 1)
        & (hm_xy[:, 1] >= 0) & (hm_xy[:, 1] <= heatmap_size - 1)
    )
    vis = (v > 0) & inb
    return hm_xy, vis


def gaussian_blob(shape, cx, cy, sigma):
    """(H,W) float32 Gaussian, peak 1.0, centered at (cx, cy) with std `sigma`
    (px, in the SAME units as cx/cy/shape). ``shape`` is an (H, W) tuple or a
    single int for a square blob.

    Single-point single-channel primitive factored out of
    ``gaussian_heatmaps``'s per-channel loop body (identical formula,
    verified by that function's existing tests -- this is a
    behaviour-preserving extraction, not a new numeric recipe) so every
    single-peak Gaussian render in this repo (per-keypoint heatmap targets
    here, the CenterDetect per-instance target render in
    ``jarvis_jax/train/losses.py``, and the "target fly" instance-channel
    render in ``jarvis_jax/data/center_channel.py``) shares one
    implementation instead of three copies of ``np.exp(-(dx**2+dy**2)/2s^2)``.
    """
    h, w = (shape, shape) if isinstance(shape, int) else shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    two_s2 = 2.0 * float(sigma) * float(sigma)
    return np.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / two_s2).astype(np.float32)


def gaussian_heatmaps(hm_xy, vis, heatmap_size=224, sigma=7.0):
    """Render (heatmap_size, heatmap_size, K) Gaussian heatmaps (peak 1.0).

    hm_xy: (K, 2) array with columns [x (col), y (row)].

    sigma is 7.0 and MUST STAY 7.0. It was changed to 2.0 on 2026-08-29 and
    reverted on 2026-08-30 after the retrain measured a 6.5x regression:

        same data/sampling/aug, eval @1500   sigma=7.0 -> 28.98 px
                                             sigma=2.0 -> 86.17 px
        trained out, 30k steps               sigma=7.0 ->  6.448 px
                                             sigma=2.0 -> 55.528 px

    The argument for 2.0 was anatomical and is reproduced here because it is
    seductive and WRONG: 1 voxel is 2.7-3.2 heatmap px, the distal tarsal
    segment T1L_TaT3->T1L_TaTip is 1.59 voxels ~ 4.6 heatmap px, so sigma=7 is
    ~1.5x longer than the whole segment and adjacent tarsal targets overlap
    almost completely -- apparently the cause of the 0.352 peak concentration
    on the shipped detector.

    Why it does not follow: **sigma sets the optimisation basin, not the
    resolution ceiling.** The target's width governs how far a mispredicted
    peak can be and still get gradient pointing home; it does not cap how
    sharply the trained model can peak. At sigma=2 the model learned peaks that
    were SHARPER (concentration 0.885 vs 0.352) and fired them on the wrong
    legs -- 70.6% of its >30 px misses landed nearer the mirror keypoint's GT
    than their own. Sharpening the target bought peak quality and destroyed the
    correspondence that makes a peak mean anything.

    sigma/heatmap_size ~ 3% is an OPTIMISATION convention, not an anatomical
    one: 7/224 = 3.1% is textbook, 2/224 = 0.9% is far below anything standard.
    The tarsal-resolution problem is real but belongs downstream, in the
    coarse-to-fine stage-2 refinement volumes -- not in a tighter 2D target.
    See docs/specs/2026-08-29-coarse-to-fine-3d-design.md section 1.2.
    """
    k = hm_xy.shape[0]
    hm = np.zeros((heatmap_size, heatmap_size, k), dtype=np.float32)
    for j in range(k):
        if not vis[j]:
            continue
        cx, cy = hm_xy[j]  # hm_xy columns are [x (col), y (row)]
        hm[:, :, j] = gaussian_blob(heatmap_size, cx, cy, sigma)
    return hm
