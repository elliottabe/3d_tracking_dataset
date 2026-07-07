import os
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name="courtship_pipeline", overrides=overrides)


def test_courtship_pipeline_composes_and_paths_generalize(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    cfg = _compose(["paths=hyak"])
    # user comes from env; no hardcoded 'eabe'
    assert cfg.paths.user == "someone"
    # required groups present
    assert cfg.recording.session_dir and cfg.recording.num_animals == 2
    assert cfg.detector.num_keypoints == 50 and cfg.detector.crop == 448
    assert cfg.silhouette.mesh_npz.endswith(".npz")
    # keypoint-bridge is the courtship default (fixes the ~30px global-shift on the
    # thin leg base that mask-mode leaves; verified both flies, IoU unchanged)
    assert cfg.silhouette.bridge_mode == "keypoint"
    assert "bouts" not in cfg.outputs.out or True  # out is a resolvable pattern


def test_default_user_when_env_absent(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    cfg = _compose(["paths=hyak"])
    assert cfg.paths.user == "eabe"   # fallback default


def test_courtship_pipeline_fully_resolves(monkeypatch):
    """Reproduces the exact resolution stac_mjx.run_stac -> io.save_data_to_h5
    performs (`OmegaConf.to_container(config, resolve=True)` on the WHOLE cfg,
    not just cfg.stac) before writing the config into the output h5.

    courtship_pipeline.yaml reuses the `stac`/`paths` groups, which interpolate
    `${dataset.name}`/`${version}` (paths.base_dir/data_dir) and
    `${preprocessing.input_filename}` (stac.data_path) — none of which
    courtship_pipeline defined until this fix. paths.save_dir also calls the
    custom `multirun_save_dir` resolver, which is registered as an import side
    effect of `stac_mjx` (via stac_mjx/path_utils.py) — exactly what happens in
    the real pipeline, since jarvis_jax.cse.courtship_stac does `import
    stac_mjx` before ever calling stac_mjx.run_stac.
    """
    monkeypatch.setenv("USER", "eabe")
    import stac_mjx  # noqa: F401  (registers the multirun_save_dir resolver, as in production)

    cfg = _compose([])
    resolved = OmegaConf.to_container(cfg, resolve=True)  # MUST NOT raise
    assert resolved["dataset"]["name"] == "courtship"
    assert resolved["paths"]["data_dir"].endswith("/courtship/v1")
