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

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
from jarvis_jax.predict.sam3_driver import run_sam3_masks
register_resolvers()


def main_from_cfg(cfg):
    s = cfg.sam3
    bout_ids = [int(x) for x in str(s.bout_ids).split(",") if str(x).strip()] or None
    jarvis_root = s.jarvis_root if s.jarvis_root else None
    return run_sam3_masks(
        project=s.project,
        session_dir=s.session_dir,
        bouts_csv=s.bouts_csv,
        out=s.out,
        num_animals=s.num_animals,
        limit=s.limit,
        bout_ids=bout_ids,
        reuse_masks=s.reuse_masks,
        jarvis_root=jarvis_root,
        sam3={"sam3_version": s.sam3_version, "gpu_id": s.sam3_gpu,
              "compile": s.sam3_compile, "text_prompt": s.sam3_text,
              "checkpoint_path": s.sam3_checkpoint},
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
