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
    if "loss" not in cfg.train or "mv_aug" not in cfg.train:
        raise ValueError("run with train=mvq")
    mcfg = build_dataclass(MVQConfig, cfg.model)
    tnode = OmegaConf.to_container(cfg.train, resolve=True)
    tnode["window_lengths"] = tuple(tnode["window_lengths"]); tnode["val_cohorts"] = tuple(tnode["val_cohorts"])
    if "pair_deltas" in tnode:                                  # yaml list -> tuple of ints (v2, spec §4)
        tnode["pair_deltas"] = tuple(int(v) for v in tnode["pair_deltas"])
    if "copy_paste_contact_sep" in tnode:                       # yaml list -> tuple (CopyPasteParams field)
        tnode["copy_paste_contact_sep"] = tuple(float(v) for v in tnode["copy_paste_contact_sep"])
    if "sex_label_overrides" in tnode:                          # OmegaConf DictConfig -> plain dict
        tnode["sex_label_overrides"] = dict(tnode["sex_label_overrides"] or {})
    tcfg = MVQTrainConfig(**{k: v for k, v in tnode.items() if k in MVQTrainConfig.__dataclass_fields__})
    weights = LossWeights(**tnode["loss"]); aug = MVAugParams(**tnode["mv_aug"])
    run_dir = run_dir_for(cfg)
    # paths.runs_root drives run_dir_for (keeps .hydra/ beside the run, which
    # later phases rely on); paths.runs_root=${paths.mvq_runs_root} must be
    # passed explicitly at launch. Warn loudly rather than silently landing
    # an mvq run under some OTHER model's runs_root.
    mvq_root = cfg.paths.get("mvq_runs_root", None)
    if mvq_root is not None and cfg.paths.runs_root != mvq_root:
        print(f"[mvq] WARNING: paths.runs_root={cfg.paths.runs_root} != paths.mvq_runs_root={mvq_root} "
              f"(pass paths.runs_root='${{paths.mvq_runs_root}}' to use it) -- this run WILL land in {run_dir}",
              flush=True)
    # v2 Step 6 launch check: `+share_check_steps=200` runs that many steps of the
    # REAL pipeline, prints each loss term's weighted share of `total` and exits
    # WITHOUT eval, checkpoint or run-dir json. It exists because spec §4 sets
    # `loss.other_fly_repulsion=20` from a WARM-STARTED T=1 run, and v2 trains from
    # scratch at T=2 with negatives -- a different loss balance. Read the printed
    # `other_fly_repulsion ... share` line before spending a day of 8 GPUs on it.
    n_share = int(cfg.get("share_check_steps", 0) or 0)
    if n_share > 0:
        print(f"[mvq] LOSS-SHARE CHECK: {n_share} steps, no eval/checkpoint "
              f"(drop +share_check_steps to train)", flush=True)
    return run_training(cfg.paths.data_root, out_dir=os.path.join(run_dir, "final"),
                        ckpt_dir=None if n_share else os.path.join(run_dir, "ckpt"),
                        mcfg=mcfg, tcfg=tcfg, aug=aug, weights=weights, share_check_steps=n_share)


if __name__ == "__main__":
    main()
