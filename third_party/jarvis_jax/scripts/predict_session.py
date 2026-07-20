"""Hydra entrypoint: D3 session/bout 3D inference -> per-fly Predictions_3D CSVs.

Reuses run_predict_session (D3 Task 4); writes per-fly data3D_fly{N}.csv in a
Predictions_3D-compatible layout under predict_session.out.

Env prelude (cv2 needs libstdc++ from the conda env):
    conda activate 3d_tracking
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

Example one-bout run:
    cd third_party/jarvis_jax && python scripts/predict_session.py \\
        paths=hyak model=hybridnet predict_session=default \\
        run_id=run4 \\
        predict_session.masks_dir=/path/to/sam3_out \\
        predict_session.out=/tmp/d3_b4 \\
        predict_session.bout_ids=4
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import hydra
from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, run_dir_for
from jarvis_jax.predict.session_predict import run_predict_session
register_resolvers()


def main_from_cfg(cfg):
    from jarvis_jax.predict.paths_util import dataset_for, processed_dir_for, resolve_auto
    ps = cfg.predict_session
    dataset = ps.dataset if ps.dataset else dataset_for(ps.session_dir)
    processed_dir = processed_dir_for(cfg.paths.processed_root, ps.session_dir, dataset=dataset)
    out = resolve_auto(ps.out, os.path.join(processed_dir, "predictions"))
    masks_dir = resolve_auto(ps.masks_dir, os.path.join(processed_dir, "sam3_masks"))
    bout_ids = [int(x) for x in str(ps.bout_ids).split(",") if str(x).strip()] or None
    run_dir = run_dir_for(cfg)
    return run_predict_session(
        session_dir=ps.session_dir, masks_dir=masks_dir, out=out,
        project=ps.project, jarvis_root=(ps.jarvis_root or None),
        v2v_final=os.path.join(run_dir, "final"),
        vitpose_ckpt=cfg.paths.vitpose_ckpt, sharpen=cfg.model.sharpen,
        num_animals=ps.num_animals, batch=ps.batch, bout_ids=bout_ids, limit=ps.limit,
        num_keypoints=int(ps.get("num_keypoints", 50)),
        front_end=ps.get("front_end", "vitpose"),
        efficienttrack_ckpt=(ps.get("efficienttrack_ckpt", "") or None),
        fusion_mode=ps.get("fusion_mode", "none"))


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
