"""Hydra entrypoint: python -m jarvis_jax.scripts.train_mvq run_id=<name> [overrides]"""
import os
import hydra
from omegaconf import OmegaConf

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers, build_dataclass, run_dir_for

register_resolvers()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    if cfg.model.get("arch") != "mvq":
        raise ValueError("run with model=mvq train=mvq")
    mcfg = build_dataclass(MVQConfig, cfg.model)
    tnode = OmegaConf.to_container(cfg.train, resolve=True)
    tnode["window_lengths"] = tuple(tnode["window_lengths"]); tnode["val_cohorts"] = tuple(tnode["val_cohorts"])
    tcfg = MVQTrainConfig(**{k: v for k, v in tnode.items() if k in MVQTrainConfig.__dataclass_fields__})
    weights = LossWeights(**tnode["loss"]); aug = MVAugParams(**tnode["mv_aug"])
    run_dir = run_dir_for(cfg)
    return run_training(cfg.paths.data_root, out_dir=os.path.join(run_dir, "final"),
                        ckpt_dir=os.path.join(run_dir, "ckpt"), mcfg=mcfg, tcfg=tcfg, aug=aug, weights=weights)


if __name__ == "__main__":
    main()
