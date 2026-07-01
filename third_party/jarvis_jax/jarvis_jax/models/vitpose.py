from flax import nnx
from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.models.vit import ViT
from jarvis_jax.models.decoder import ClassicDecoder


class ViTPose(nnx.Module):
    """ViTPose: ViT backbone + ClassicDecoder for keypoint regression.

    Takes (B, 448, 448, 4) RGB-D images and produces (B, 224, 224, 50) heatmaps.
    """

    def __init__(self, cfg: ViTPoseConfig, *, backbone=None, rngs: nnx.Rngs):
        self.cfg = cfg
        if backbone is None:
            self.backbone = ViT(cfg, rngs=rngs)             # backward-compatible default
        else:
            from jarvis_jax.models.backbone import build_backbone
            self.backbone = build_backbone(backbone, cfg, rngs=rngs)
        self.decoder = ClassicDecoder(cfg.embed_dim, cfg.num_keypoints, rngs=rngs)

    def __call__(self, x, *, use_running_average: bool = False):
        """Forward pass.

        Args:
            x: (B, 448, 448, 4) input images (RGB-D).
            use_running_average: passed to decoder's BatchNorm layers (False = training).

        Returns:
            (B, 224, 224, 50) heatmap logits.
        """
        tokens = self.backbone(x)
        return self.decoder(tokens, use_running_average=use_running_average)
