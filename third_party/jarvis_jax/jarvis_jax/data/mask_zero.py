"""Mask-channel ablation wrapper for 2D keypoint datasets.

``ZeroMaskDataset`` wraps ANY dataset with the ``V3Dataset``/``V5Dataset``
``__getitem__`` contract (img4 (H,W,4) uint8, kp_xy (K,2) float32, vis (K,)
bool -- see ``jarvis_jax/data/v3.py`` and ``jarvis_jax/data/v5_2d.py``) and
zeroes the 4th (SAM-mask) channel of every sample it returns. Everything else
(``__len__``, and every attribute/method the training loop needs --
``sampling_weights``, ``balanced_weights``, ``class_counts``, ``sex``,
``behavior``, ``file_names``, ``heatmap_size``, ...) is forwarded verbatim to
the wrapped dataset via ``__getattr__``.

This is a wrapper rather than a change to ``V3Dataset``/``V5Dataset``
themselves, or a 3rd ``in_ch`` model variant, deliberately: it holds model
capacity and parameter count FIXED (still 4 input channels; the mask channel
is just always zero) so a checkpoint trained against it isolates the mask
CHANNEL's information content from any capacity difference -- see
``.superpowers/sdd/2026-08-29-coarse-to-fine-3d/mask-channel-ablation.md``.
It also makes the ablation available at EVAL time with zero risk to the two
real dataset classes (used by every other pipeline that trains/evaluates a
detector): a wrapper that never existed before this diagnostic cannot change
behaviour for a caller that never imports it.
"""
from __future__ import annotations


class ZeroMaskDataset:
    """Wrap ``ds``, zeroing channel index 3 (the SAM mask) of every sample."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        img4, kp_xy, vis = self._ds[i]
        img4 = img4.copy()
        img4[..., 3] = 0
        return img4, kp_xy, vis

    def __getattr__(self, name):
        # Only reached for attributes not found on ZeroMaskDataset itself
        # (i.e. not __len__/__getitem__/__init__) -- forwards everything
        # else (sex, behavior, file_names, heatmap_size, sampling_weights,
        # balanced_weights, class_counts, ...) to the wrapped dataset.
        return getattr(self._ds, name)
