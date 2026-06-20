from jarvis_jax.config import ViTPoseConfig

def test_config_defaults():
    c = ViTPoseConfig()
    assert (c.img_size, c.patch, c.in_ch) == (448, 16, 4)
    assert (c.embed_dim, c.depth, c.num_heads, c.mlp_ratio) == (768, 12, 12, 4)
    assert (c.num_keypoints, c.heatmap_size) == (50, 224)
    assert c.num_tokens == 784  # (448/16)^2
