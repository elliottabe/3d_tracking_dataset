"""Stage-2 per-joint refinement volumes.

WHY. Stage 1's 48^3 grid at spacing 1 puts 1 voxel at ~0.17 mm and 2.7-3.2
heatmap px. The distal tarsal segment T1L_TaT3->T1L_TaTip is 1.59 voxels, so
leg kinematics are quantization-limited and the 3D stage samples the 2D
heatmaps ~3x below their own resolution. Stage 2 re-samples a SMALL volume
around each joint's coarse estimate at spacing 0.25, where 1 voxel ~ 0.8
heatmap px -- matched to the 2D detail that already exists.

COST. cube=24 at spacing 0.25 spans +/-3 world units, adequate against a
~1-unit stage-1 error, and 50 joints x 24^3 = 691k voxels is 8x CHEAPER than
one 50 x 48^3 stage-1 volume (5.53M). Weights are shared across joints by
folding the joint axis into the batch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jarvis_jax.hybridnet.model import soft_argmax_3d
from jarvis_jax.hybridnet.reproject import reproject_heatmaps


def refine_volumes(heatmaps, coarse_kp3d, centerHM, camera_matrices, *,
                   cube: int = 24, spacing: float = 0.25,
                   heatmap_size: int = 224):
    """(B,J,cube,cube,cube) — one volume per joint, centred on its own estimate."""
    B, C, J = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]

    def one_joint(j):
        # Reproject ONLY joint j, with the grid recentred on that joint.
        # NOTE: a `j:j+1` slice with a vmap-traced `j` is rejected by this
        # jax version ("Slice entries must be static integers") even though
        # the slice width is static -- integer-index then reinsert the axis
        # with `None` instead, which lowers to a supported dynamic gather.
        hm_j = heatmaps[:, :, j][:, :, None]                  # (B,C,1,hm,hm)
        centre = coarse_kp3d[:, j]                            # (B,3)
        vol = reproject_heatmaps(
            hm_j, centre, centerHM, camera_matrices,
            grid_size=cube, grid_spacing=spacing, heatmap_size=heatmap_size)
        return vol[:, 0]                                      # (B,cube,cube,cube)

    vols = jax.vmap(one_joint, out_axes=1)(jnp.arange(J))
    return vols                                               # (B,J,cube,...)


class RefineNet(nnx.Module):
    """Small shared 3D CNN over per-joint refinement volumes.

    Deliberately tiny: it runs 50 times per sample, and its job is local peak
    sharpening, not global reasoning — stage 1 already did the localization and
    cross-view disambiguation.
    """

    def __init__(self, *, channels: int = 32, rngs: nnx.Rngs):
        k = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="SAME", rngs=rngs)
        self.c1 = nnx.Conv(1, channels, **k)
        self.n1 = nnx.BatchNorm(channels, rngs=rngs)
        self.c2 = nnx.Conv(channels, channels, **k)
        self.n2 = nnx.BatchNorm(channels, rngs=rngs)
        self.out = nnx.Conv(channels, 1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                            padding="VALID",
                            kernel_init=nnx.initializers.normal(0.001),
                            bias_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, x, use_running_average: bool = False):
        x = jax.nn.relu(self.n1(self.c1(x), use_running_average=use_running_average))
        x = jax.nn.relu(self.n2(self.c2(x), use_running_average=use_running_average))
        return self.out(x)


def refine_keypoints(net: RefineNet, heatmaps, coarse_kp3d, centerHM,
                     camera_matrices, *, cube: int = 24, spacing: float = 0.25,
                     heatmap_size: int = 224, sharpen: float = 3.0,
                     use_running_average: bool = False):
    """Coarse (B,J,3) -> refined (B,J,3). Offsets are RELATIVE to the coarse
    estimate, so a refinement can never move a joint outside its window."""
    B, J = coarse_kp3d.shape[0], coarse_kp3d.shape[1]
    vols = refine_volumes(heatmaps, coarse_kp3d, centerHM, camera_matrices,
                          cube=cube, spacing=spacing, heatmap_size=heatmap_size)
    flat = vols.reshape(B * J, cube, cube, cube, 1)
    logits = net(flat, use_running_average=use_running_average)
    logits = logits.reshape(B * J, 1, cube, cube, cube)
    roi = cube * spacing * 2.0
    delta, _ = soft_argmax_3d(logits, grid_spacing=spacing, roi_cube=roi,
                              sharpen=sharpen)
    return coarse_kp3d + delta.reshape(B, J, 3)
