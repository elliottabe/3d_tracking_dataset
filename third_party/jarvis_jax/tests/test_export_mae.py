import os, numpy as np, subprocess, sys, tempfile, pytest
timm = pytest.importorskip("timm")

def test_export_produces_expected_keys(tmp_path):
    out = tmp_path / "mae.npz"
    subprocess.run([sys.executable, "-m", "jarvis_jax.convert.export_mae_timm",
                    "--out", str(out)], check=True)
    z = np.load(out, allow_pickle=True)
    keys = set(z.files)
    assert "patch_embed.proj.weight" in keys      # (768,3,16,16)
    assert z["patch_embed.proj.weight"].shape == (768, 3, 16, 16)
    assert "pos_embed" in keys                     # (1,197,768)
    assert z["pos_embed"].shape == (1, 197, 768)
    assert "blocks.0.attn.qkv.weight" in keys
    assert str(z["__model_id__"]) != ""
