import importlib.util, sys

SPEC = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/scripts/slurm_bout_array.py"


def _load():
    spec = importlib.util.spec_from_file_location("slurm_bout_array", SPEC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_build_array_script_has_requeue_and_bout_id():
    m = _load()
    # Real bouts are 1-based and non-contiguous (no bout_00000); the array
    # spec must be the exact comma-joined bout ids, not a 0-based range.
    idxs = [1, 2, 3, 5, 8, 13, 21, 34]
    s = m.build_jax_array_script(job_name="c", partition="ckpt-g2", account="portia",
                                 cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                 conda_env="3d_tracking", idxs=idxs, run_dir="/tmp/rd",
                                 config_name="pipeline", overrides="")
    assert "--array=1,2,3,5,8,13,21,34" in s and "--requeue" in s
    assert "0-" not in s.split("--array=")[1].split("\n")[0]  # not a 0-based range
    assert "+bout_ids=${SLURM_ARRAY_TASK_ID}" in s
    assert "++bout_ids=${SLURM_ARRAY_TASK_ID}" in s   # force-override: pipeline.yaml
                                                        # already defines bout_ids, so a plain
                                                        # `+bout_ids=` would raise Hydra's
                                                        # ConfigCompositionException
    assert "unset LD_LIBRARY_PATH" in s        # JAX env


def test_sam3_script_uses_pytorch_env():
    m = _load()
    idxs = [1, 2, 3, 5, 8, 13, 21, 34]
    s = m.build_sam3_array_script(job_name="s", partition="ckpt-g2", account="portia",
                                  cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                  conda_env="3d_tracking", idxs=idxs, session_dir="/x",
                                  masks_out="/y")
    assert "cu13/lib" in s and "LD_PRELOAD" in s and "--array=1,2,3,5,8,13,21,34" in s


def test_precompute_script_uses_real_bout_id():
    """bout_00000 does not exist (bouts are 1-based); precompute must target a
    real discovered bout id, not the old hardcoded ++bout_ids=0."""
    m = _load()
    s = m.build_precompute_script(job_name="p", partition="ckpt-g2", account="portia",
                                  cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                  conda_env="3d_tracking", bout_id=1, run_dir="/tmp/rd",
                                  config_name="pipeline", overrides="")
    assert "++bout_ids=1" in s
    assert "++bout_ids=0" not in s
