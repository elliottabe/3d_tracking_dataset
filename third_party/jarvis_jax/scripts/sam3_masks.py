"""Hydra entrypoint: SAM3 mask+identity front-end for a session's bouts (D2).

Reuses the JARVIS PyTorch SAM3 tracker; writes per-bout sam3_masks.npz +
manifest.json for the JAX D3 pipeline.

Env prelude (3d_tracking + SAM3): set the SAM3 cu13 LD_LIBRARY_PATH + LD_PRELOAD
and keep sam3_compile=false. Example:
    conda activate 3d_tracking
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
    python scripts/sam3_masks.py paths=hyak sam3=default sam3.limit=1
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# Pre-load huggingface_hub.file_download BEFORE the SAM3 import chain (jarvis ->
# timm -> torch) runs. That chain leaves `tqdm` in a state where huggingface_hub's
# LAZY file_download import later fails with "type object 'tqdm' has no attribute
# 'set_lock'", so hf_hub_download (needed by sam3.model_builder) becomes
# unimportable and every SAM3 array task dies at import. Loading it here, while
# tqdm is still intact, caches the module so sam3's later
# `from huggingface_hub import hf_hub_download` resolves. (Verified fix.)
import huggingface_hub.file_download  # noqa: E402,F401

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
from jarvis_jax.predict.sam3_driver import run_sam3_masks
register_resolvers()


def main_from_cfg(cfg):
    import torch
    from jarvis_jax.predict.sam3_driver import resolve_gpus, run_sam3_masks_multi
    from jarvis_jax.predict.paths_util import dataset_for, processed_dir_for, resolve_auto
    from jarvis_jax.predict.bouts_resolve import resolve_bout_summary
    s = cfg.sam3
    dataset = s.dataset if s.dataset else dataset_for(s.session_dir)
    processed_dir = processed_dir_for(cfg.paths.processed_root, s.session_dir, dataset=dataset)
    out = resolve_auto(s.out, os.path.join(processed_dir, "sam3_masks"))
    bouts_csv = resolve_auto(s.bouts_csv, None)
    if bouts_csv is None:
        bouts_csv = resolve_bout_summary(
            recording_dir=s.session_dir, processed_dir=processed_dir, dataset=dataset)
    bout_ids = [int(x) for x in str(s.bout_ids).split(",") if str(x).strip()] or None
    jarvis_root = s.jarvis_root if s.jarvis_root else None
    sam3_kwargs = {"sam3_version": s.sam3_version, "gpu_id": s.sam3_gpu,
                   "compile": s.sam3_compile, "text_prompt": s.sam3_text,
                   "checkpoint_path": s.sam3_checkpoint}
    common = dict(project=s.project, session_dir=s.session_dir, bouts_csv=bouts_csv,
                  out=out, num_animals=s.num_animals, limit=s.limit, bout_ids=bout_ids,
                  reuse_masks=s.reuse_masks, jarvis_root=jarvis_root, sam3=sam3_kwargs)
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpus = resolve_gpus(s.gpus, env=os.environ, device_count=device_count)
    if len(gpus) > 1:
        return run_sam3_masks_multi(gpus=gpus, **common)
    return run_sam3_masks(
        manifest_name=s.manifest_name, lowmem=s.lowmem,
        overlay=bool(s.get("overlay", True)),
        overlay_cams=int(s.get("overlay_cams", 3)),
        overlay_frames=int(s.get("overlay_frames", 300)),
        **common)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
