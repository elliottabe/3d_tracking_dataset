import dataclasses

@dataclasses.dataclass(frozen=True)
class ViTPoseConfig:
    img_size: int = 448
    patch: int = 16
    in_ch: int = 4
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    num_keypoints: int = 50
    heatmap_size: int = 224

    @property
    def num_tokens(self) -> int:
        return (self.img_size // self.patch) ** 2
