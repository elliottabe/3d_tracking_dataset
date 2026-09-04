from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jarvis_jax.models.dinov3 import DINOv3, DINOv3Config
from jarvis_jax.models.mvq.decoder import QueryDecoder, fourier, gather_refine_context
from jarvis_jax.models.mvq.fusion import FusionStack
from jarvis_jax.models.mvq.geometry import ray_from_pixel, token_pixel_centres


@dataclasses.dataclass(frozen=True)
class MVQConfig:
    crop: int = 448
    patch: int = 16
    embed_dim: int = 768
    num_keypoints: int = 50
    num_cameras: int = 7
    max_frames: int = 8
    n_instances: int = 3
    n_local: int = 2
    n_global: int = 2
    global_pool: int = 2
    dec_layers_3d: int = 8
    dec_layers_2d: int = 4
    dec_heads: int = 12
    mlp_ratio: float = 4.0
    refine_passes: int = 1
    patch_rgb: int = 9
    fourier_bands: int = 8
    roi_scale: float = 24.0
    camera_slot_embed: bool = True
    backbone: str = "dinov3_b16"
    backbone_depth: int = 12
    backbone_heads: int = 12
    remat: bool = True
    # Query-chunk size for the 2D cross-attention path's masked_attention call
    # (None = unchunked). Profiled 2026-09-04: chunking (q_chunk=8, the shipped
    # default before this field existed) cost +13% step time to save 2.2GB --
    # a bad trade once the attention-chunk remat fix (fusion.py) made the
    # unchunked path fit comfortably at the 4-8 samples/GPU this model trains
    # at. Kept configurable (not deleted) because a future larger n_instances/
    # num_cameras config could make Nq large enough that chunking is worth its
    # cost again -- see tests/test_mvq_model.py's chunked==unchunked tests,
    # which exercise q_chunk=8 explicitly regardless of this default.
    q_chunk: int | None = None

    @property
    def grid(self):
        return self.crop // self.patch


class MVQModel(nnx.Module):
    def __init__(self, cfg: MVQConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.embed_dim
        base = DINOv3Config.vitl16() if cfg.backbone == "dinov3_l16" else DINOv3Config.vitb16()
        self.backbone = DINOv3(dataclasses.replace(base, embed_dim=D, depth=cfg.backbone_depth,
                                                   num_heads=cfg.backbone_heads), rngs=rngs)
        nf = 2 * cfg.fourier_bands + 1
        self.geom = nnx.Linear(6 * nf, D, rngs=rngs)
        self.e_frame = nnx.Param(jax.random.normal(rngs.params(), (cfg.max_frames, D)) * 0.02)
        self.e_cam = nnx.Param(jnp.zeros((cfg.num_cameras, D)))
        self.fusion = FusionStack(cfg, rngs=rngs)
        self.decoder = QueryDecoder(cfg, rngs=rngs)
        self.prompt_ln = nnx.LayerNorm(D, rngs=rngs)

    def _bank(self, crops, cam_valid, M, t_local):
        cfg = self.cfg; B, T, C, H, W, _ = crops.shape; g = cfg.grid; N = g * g
        toks = self.backbone(crops.reshape(B * T * C, H, W, 3), remat=cfg.remat).reshape(B, T, C, N, -1)
        centres = token_pixel_centres(g, g, cfg.patch)                                    # (N,2)
        def rays(Mb, tlb):                                                                # per sample, per frame
            p0, d = ray_from_pixel(Mb, tlb, jnp.broadcast_to(centres, (C, N, 2)))
            return jnp.concatenate([p0 / cfg.roi_scale, jnp.broadcast_to(d[:, None], p0.shape)], -1)  # (C,N,6)
        feats = jax.vmap(lambda Mb, tl: jax.vmap(lambda tlt: rays(Mb, tlt))(tl))(M, t_local)      # (B,T,C,N,6)
        toks = toks + self.geom(fourier(jnp.clip(feats, -3, 3) / 3.0, cfg.fourier_bands))
        toks = toks + self.e_frame[...][:T][None, :, None, None, :]
        if cfg.camera_slot_embed:
            toks = toks + self.e_cam[...][None, None, :, None, :]
        toks = toks * cam_valid[..., None, None]
        toks = self.fusion(toks.reshape(B, T * C, N, -1), cam_valid.reshape(B, T * C))
        return toks.reshape(B, T, C, N, -1)

    def _prompt(self, grid_tokens, prompt_mask, cam_valid):
        cfg = self.cfg; B, T, C, N, D = grid_tokens.shape; g = cfg.grid
        m = prompt_mask.reshape(B, T, C, g, cfg.patch, g, cfg.patch).mean(axis=(4, 6)) > 0.5   # (B,T,C,g,g)
        m = (m.reshape(B, T, C, N) & cam_valid[..., None]).astype(jnp.float32)
        num = (grid_tokens * m[..., None]).sum(axis=(1, 2, 3)); den = jnp.maximum(m.sum(axis=(1, 2, 3)), 1.0)
        return self.prompt_ln(num / den[:, None])

    def __call__(self, crops, cam_valid, M, t_local, prompt_mask=None, *, prompt_on=None):
        cfg = self.cfg; B, T, C = crops.shape[:3]; N = cfg.grid ** 2
        grid_tokens = self._bank(crops, cam_valid, M, t_local)                               # (B,T,C,N,D)
        bank = grid_tokens.reshape(B, T * C * N, -1)
        bank_valid = jnp.repeat(cam_valid.reshape(B, T * C), N, axis=1)
        femb = self.e_frame[...][:T]
        ptok = self._prompt(grid_tokens, prompt_mask, cam_valid) if prompt_mask is not None else None
        out, per_layer, _ = self.decoder(bank, bank_valid, M, t_local, femb, ptok, prompt_on, None, 0)
        aux_layers, aux_pass1 = list(per_layer), None
        for p in range(cfg.refine_passes):
            aux_pass1 = out
            ctx = gather_refine_context(out["xyz"], M, t_local, cam_valid,
                                        grid_tokens.reshape(B, T, C, cfg.grid, cfg.grid, -1), crops,
                                        cfg.patch_rgb, cfg.fourier_bands, cfg.crop)
            ctx = jax.lax.stop_gradient(ctx)
            out2, per2, _ = self.decoder(bank, bank_valid, M, t_local, femb, ptok, prompt_on, ctx, 1)
            out2["xyz"] = out["xyz"] + out2["xyz"]
            out2["uv"] = jnp.clip(out["uv"] + (out2["uv"] - cfg.crop / 2.0), 0.0, cfg.crop)
            aux_layers.extend(per2); out = out2
        out["aux_pass1"], out["aux_layers"] = aux_pass1, aux_layers
        return out


def assemble(out, center3D, crop_origin, exist_thresh=0.5, cam_valid=None):
    """numpy: model outputs -> world kp3d (NaN where absent), conf3d, full-frame kp2d.

    Two independent halves of spec Sec 4.6's NaN policy: an instance that
    doesn't exist (`exist_logit` below threshold) is NaN across every frame
    and keypoint; a FRAME with no valid camera at all (`cam_valid`, when
    given) is NaN across every instance and keypoint for that frame -- there
    is no observation to have triangulated a keypoint from. Per-keypoint
    visibility gating (a keypoint occluded in every view but the frame
    otherwise fine) is deferred to the P4 lifter, not this assembly step.
    """
    xyz, conf = np.asarray(out["xyz"]), 1 / (1 + np.exp(-np.asarray(out["conf_logit"])))
    exist = 1 / (1 + np.exp(-np.asarray(out["exist_logit"]))) >= exist_thresh                 # (B,I)
    kp3d = xyz + np.asarray(center3D)[:, None, None, None, :]
    kp3d = np.where(exist[:, :, None, None, None], kp3d, np.nan)
    conf3d = np.where(exist[:, :, None, None], conf, 0.0)
    if cam_valid is not None:
        frame_ok = np.asarray(cam_valid).any(axis=2)                                          # (B,T)
        kp3d = np.where(frame_ok[:, None, :, None, None], kp3d, np.nan)
        conf3d = np.where(frame_ok[:, None, :, None], conf3d, 0.0)
    kp2d = np.asarray(out["uv"]) + np.asarray(crop_origin)[:, None, None, :, None, :]
    return kp3d.astype(np.float32), conf3d.astype(np.float32), kp2d.astype(np.float32)
