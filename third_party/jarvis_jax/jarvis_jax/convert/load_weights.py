"""Load a timm MAE ViT-B/16 npz (Task 8) into the NNX ``ViT`` backbone.

Handles the three cross-framework conversions:

* patch-embed conv: timm NCHW ``(out, in, kh, kw)`` -> NNX NHWC ``(kh, kw, in, out)``
  with 3->4 channel inflation (the 4th mask channel is zero-initialized, so a zero
  4th input channel is a no-op and parity with the 3-channel timm model holds).
* pos-embed: 197 (14x14 grid + cls) -> ``num_tokens+1`` (e.g. 28x28 + cls) via
  bicubic interpolation of the grid tokens.
* per-block LN / qkv / proj / MLP and the final norm: timm ``nn.Linear`` weights are
  ``(out, in)``; ``nnx.Linear`` kernels are ``(in, out)`` -> assign the transpose.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _interp_pos_embed(pe, old_grid, new_grid):
    """pe: (1, old_grid*old_grid + 1, D) -> (1, new_grid*new_grid + 1, D)."""
    cls, grid = pe[:, :1], pe[:, 1:]
    d = grid.shape[-1]
    grid = grid.reshape(1, old_grid, old_grid, d)
    if new_grid != old_grid:
        grid = jax.image.resize(grid, (1, new_grid, new_grid, d), method="bicubic")
    grid = grid.reshape(1, new_grid * new_grid, d)
    return jnp.concatenate([jnp.asarray(cls), grid], axis=1)


def load_vit_from_npz(vit, npz_path, cfg):
    """Assign converted timm/MAE params into ``vit`` in place and return it."""
    z = np.load(npz_path, allow_pickle=True)
    g = cfg.img_size // cfg.patch                          # target grid (e.g. 28)
    old_g = int(round((z["pos_embed"].shape[1] - 1) ** 0.5))  # source grid (14)

    # patch embed: NCHW (out,in,kh,kw) -> NHWC (kh,kw,in,out); inflate in 3->4 (4th=0)
    w = z["patch_embed.proj.weight"]                       # (768,3,16,16)
    w = np.transpose(w, (2, 3, 1, 0))                      # (16,16,3,768)
    w4 = np.zeros((w.shape[0], w.shape[1], cfg.in_ch, w.shape[-1]), w.dtype)
    w4[:, :, : w.shape[2], :] = w
    vit.patch_embed.proj.kernel.value = jnp.asarray(w4)
    vit.patch_embed.proj.bias.value = jnp.asarray(z["patch_embed.proj.bias"])

    vit.cls_token.value = jnp.asarray(z["cls_token"])
    vit.pos_embed.value = _interp_pos_embed(jnp.asarray(z["pos_embed"]), old_g, g)

    def lin(dst, w_key, b_key):                            # torch (out,in) -> (in,out)
        dst.kernel.value = jnp.asarray(z[w_key].T)
        dst.bias.value = jnp.asarray(z[b_key])

    def ln(dst, w_key, b_key):
        dst.scale.value = jnp.asarray(z[w_key])
        dst.bias.value = jnp.asarray(z[b_key])

    for i, blk in enumerate(vit.blocks):
        p = f"blocks.{i}."
        ln(blk.norm1, p + "norm1.weight", p + "norm1.bias")
        lin(blk.attn.qkv, p + "attn.qkv.weight", p + "attn.qkv.bias")
        lin(blk.attn.proj, p + "attn.proj.weight", p + "attn.proj.bias")
        ln(blk.norm2, p + "norm2.weight", p + "norm2.bias")
        lin(blk.mlp.fc1, p + "mlp.fc1.weight", p + "mlp.fc1.bias")
        lin(blk.mlp.fc2, p + "mlp.fc2.weight", p + "mlp.fc2.bias")
    ln(vit.norm, "norm.weight", "norm.bias")
    return vit
