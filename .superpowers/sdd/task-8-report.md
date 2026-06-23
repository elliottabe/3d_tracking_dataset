### Task 8 Report (Hydra series): viz + diag entrypoints via Hydra

**Status:** DONE — 12/12 tests pass; no regressions.

**Commit:** `eff3a35` — `feat(jax-hydra): viz + diag entrypoints via Hydra`

---

#### Library functions extracted

| Script | Library fn |
|--------|-----------|
| `scripts/viz_compare_3d_runs.py` | `run_compare(*, root, cache_dir, run1, run2, out, sharpen1, sharpen2)` |
| `scripts/diag_shrinkage.py` | `run_diag(*, cache_dir, run, out)` |
| `scripts/diag_sharpen_sweep.py` | `run_sweep(*, cache_dir, run, sharpens)` |

Each script: removed `import argparse`; added `import hydra`, `from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers`; `register_resolvers()` at module level; `main_from_cfg(cfg)`; `@hydra.main` wrapper.

#### Config mappings

- `viz_compare`: `root=cfg.paths.data_root`, `cache_dir=cfg.paths.cache_dir`, `run1=cfg.viz.run1`, `run2=cfg.viz.run2`, `out=cfg.viz.out`, `sharpen1=cfg.viz.sharpen1`, `sharpen2=cfg.viz.sharpen2`
- `diag_shrinkage`: `cache_dir=cfg.paths.cache_dir`, `run=cfg.viz.run2`, `out=cfg.viz.out`
- `diag_sharpen_sweep`: `cache_dir=cfg.paths.cache_dir`, `run=cfg.viz.run2`, `sharpens=cfg.viz.sharpens`

#### RED → GREEN evidence

```
# RED:
FAILED tests/test_configs.py::test_viz_main_from_cfg_maps_config
  AttributeError: module 'viz_compare_3d_runs' has no attribute 'main_from_cfg'
1 failed in 3.98s

# GREEN (target test):
1 passed in 1.83s

# Full suite:
12 passed in 2.34s
```

#### Concerns

None. IDE import-resolution warnings for `jarvis_jax.*` in the scripts are pre-existing (IDE doesn't have the package on its path) and do not affect runtime.
