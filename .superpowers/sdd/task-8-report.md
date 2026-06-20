# Task 8 Report: timm MAE ViT-B Weight Exporter

## Summary

Implemented a torch+timm CLI exporter that downloads MAE ViT-B/16 ImageNet weights and saves them to a `.npz` file for later loading into the NNX model (Task 9).

## Precondition: Lazy `__init__.py`

**Problem:** `jarvis_jax/__init__.py` eagerly imported `ViTPose` (which triggers jax import), breaking `import jarvis_jax` in the jarvis (torch-only) env.

**Fix:** Replaced with a lazy `__getattr__` version:
```python
from jarvis_jax.config import ViTPoseConfig
__all__ = ["ViTPoseConfig", "ViTPose"]

def __getattr__(name):
    if name == "ViTPose":
        from jarvis_jax.models.vitpose import ViTPose
        return ViTPose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Verified:**
- `jarvis` env: `import jarvis_jax; print(jarvis_jax.ViTPoseConfig)` → works (no jax import error)
- `3d_tracking` env: `from jarvis_jax import ViTPose` → `<class 'jarvis_jax.models.vitpose.ViTPose'>`

## Files Created

- `third_party/jarvis_jax/jarvis_jax/convert/__init__.py` — empty package marker
- `third_party/jarvis_jax/jarvis_jax/convert/export_mae_timm.py` — CLI exporter
- `third_party/jarvis_jax/tests/test_export_mae.py` — pytest test (skips if timm absent)

## TDD Flow

1. Wrote failing test: confirmed `ModuleNotFoundError: No module named 'jarvis_jax.convert'`
2. Implemented exporter
3. Test passed in jarvis env: `1 passed in 26.17s`

## Test Results

### jarvis env (torch+timm)
```
tests/test_export_mae.py::test_export_produces_expected_keys PASSED
1 passed in 26.17s
```

### 3d_tracking env (jax, regression)
```
8 passed, 9 warnings in 108.61s
```
(test_export_mae.py is skipped via `pytest.importorskip("timm")` in 3d_tracking; the 8 = prior 7 model tests + 1 skip counted as pass — actually 7 prior tests all pass, test_export_mae is collected but skipped)

## Artifact: `/tmp/mae_vitb.npz`

**Model used:** `vit_base_patch16_224.mae` (the MAE pretrained model — primary path, no fallback needed)

**Total arrays:** 151

**Key sample:**

| Key | Shape |
|-----|-------|
| `patch_embed.proj.weight` | `(768, 3, 16, 16)` |
| `pos_embed` | `(1, 197, 768)` |
| `blocks.0.attn.qkv.weight` | `(2304, 768)` |
| `blocks.0.norm1.weight` | `(768,)` |
| `norm.weight` | `(768,)` |
| `cls_token` | `(1, 1, 768)` |
| `__model_id__` | `vit_base_patch16_224.mae` |

Note: `qkv.weight` in timm is stored transposed relative to torch convention — shape is `(2304, 768)` not `(768, 2304)`. Task 9 weight loader will need to handle this (or verify orientation matches).

## Notes for Task 9

- The `pos_embed` shape `(1, 197, 768)` = 1 + 196 patches (14×14 grid for 224px). The NNX model uses 784+1=785 tokens for ViTPose (4× more patches for heatmap resolution) — Task 9 will need to interpolate pos_embed.
- `cls_token` is present at `(1, 1, 768)`.
- `blocks.N.attn.qkv.weight` is fused Q+K+V: shape `(2304, 768)` = `3*768 × 768`.
- The `.mae` timm variant drops the classification head (`num_classes=0`); no `head.*` keys in the npz.
