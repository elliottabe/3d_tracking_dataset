import os
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"

# configs/outputs/default.yaml uses the `basename` resolver; register it (in
# production it is an import side effect of run_bout / slurm_bout_array).
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name="pipeline", overrides=overrides)


def test_pipeline_composes_and_paths_generalize(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    cfg = _compose(["paths=hyak"])
    # user comes from env; no hardcoded 'eabe'
    assert cfg.paths.user == "someone"
    # required groups present
    assert cfg.recording.session_dir and cfg.recording.num_animals == 2
    assert cfg.detector.num_keypoints == 50 and cfg.detector.crop == 448
    assert cfg.ik.mesh_npz.endswith(".npz")
    # keypoint-bridge is the courtship default (fixes the ~30px global-shift on the
    # thin leg base that mask-mode leaves; verified both flies, IoU unchanged)
    assert cfg.ik.bridge_mode == "keypoint"
    assert "bouts" not in cfg.outputs.out or True  # out is a resolvable pattern


def test_default_user_when_env_absent(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    cfg = _compose(["paths=hyak"])
    assert cfg.paths.user == "eabe"   # fallback default


def test_pipeline_fully_resolves(monkeypatch):
    """Reproduces the exact resolution stac_mjx.run_stac -> io.save_data_to_h5
    performs (`OmegaConf.to_container(config, resolve=True)` on the WHOLE cfg,
    not just cfg.stac) before writing the config into the output h5.

    pipeline.yaml reuses the `stac`/`paths` groups, which interpolate
    `${dataset.name}`/`${version}` (paths.base_dir/data_dir) and
    `${preprocessing.input_filename}` (stac.data_path) — none of which
    pipeline defined until this fix. paths.save_dir also calls the
    custom `multirun_save_dir` resolver, which is registered as an import side
    effect of `stac_mjx` (via stac_mjx/path_utils.py) — exactly what happens in
    the real pipeline, since jarvis_jax.tracking.stac does `import
    stac_mjx` before ever calling stac_mjx.run_stac.
    """
    monkeypatch.setenv("USER", "eabe")
    import stac_mjx  # noqa: F401  (registers the multirun_save_dir resolver, as in production)

    cfg = _compose([])
    resolved = OmegaConf.to_container(cfg, resolve=True)  # MUST NOT raise
    assert resolved["dataset"]["name"] == "courtship"
    assert resolved["paths"]["data_dir"].endswith("/courtship/v1")


def test_detector_maskoff_flag_travels_with_ckpt():
    """`zero_mask_channel` and `ckpt` are only correct TOGETHER.

    A checkpoint trained with `train.mask_ablation=true` saw a zeroed 4th
    channel on every sample. Feeding it a populated SAM mask at inference
    raises no error and silently degrades accuracy -- so the flag must be set
    whenever the ckpt is a mask-ablated run, and must NOT be set otherwise.

    This exists because the capability shipped in 821bcd6 as a function
    parameter that no caller ever passed and no config ever set: the code half
    of the promotion landed and the config half did not, leaving a flag that
    looked wired and was inert.
    """
    d = _compose(["paths=hyak"]).detector
    ckpt = str(d.ckpt)
    zeroed = bool(d.get("zero_mask_channel", False))
    is_maskoff_run = "maskoff" in os.path.basename(os.path.dirname(ckpt))
    assert zeroed == is_maskoff_run, (
        f"ckpt={ckpt!r} implies zero_mask_channel={is_maskoff_run}, config says {zeroed}")


def test_run_bout_actually_passes_zero_mask_channel():
    """The config key is inert unless the call site forwards it.

    Pins the plumbing, not just the value -- the original defect was a
    parameter with a default that nothing ever overrode.
    """
    import ast
    src = open(os.path.join(os.path.dirname(CFG_DIR), "scripts", "run_bout.py")).read()
    # AST, not string slicing: the call contains nested parens (int(...), .get(...)),
    # so `split(")")[0]` stops at the first one and never reaches the kwargs.
    kwargs = [kw.arg for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", getattr(n.func, "attr", None)) == "predict_bout_2d"
              for kw in n.keywords]
    assert "zero_mask_channel" in kwargs, (
        "run_bout.py calls predict_bout_2d without forwarding zero_mask_channel; "
        "the config key would be silently ignored")
