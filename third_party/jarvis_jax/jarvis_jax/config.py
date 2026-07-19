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

    # SAM3 mask-fusion hook (HybridNet3D, Task 7) -- optional, for
    # discoverability only. HybridNet3D reads these via getattr() with these
    # exact defaults, so it works unchanged with any bare cfg object that
    # lacks them (e.g. ad-hoc `_Cfg` classes elsewhere in this codebase).
    fusion_mode: str = "none"        # 'none' | 'carve' | 'input_mask'
    gate_temperature: float = 1.0
    gate_floor: float = 0.0

    @property
    def num_tokens(self) -> int:
        return (self.img_size // self.patch) ** 2
