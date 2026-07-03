import importlib.util, sys

SPEC = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/scripts/slurm_courtship_array.py"


def _load():
    spec = importlib.util.spec_from_file_location("slurm_courtship_array", SPEC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_build_array_script_has_requeue_and_bout_id():
    m = _load()
    s = m.build_jax_array_script(job_name="c", partition="ckpt-g2", account="portia",
                                 cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                 conda_env="3d_tracking", n_bouts=30, run_dir="/tmp/rd",
                                 config_name="courtship_pipeline", overrides="")
    assert "--array=0-29" in s and "--requeue" in s
    assert "+bout_ids=${SLURM_ARRAY_TASK_ID}" in s
    assert "unset LD_LIBRARY_PATH" in s        # JAX env


def test_sam3_script_uses_pytorch_env():
    m = _load()
    s = m.build_sam3_array_script(job_name="s", partition="ckpt-g2", account="portia",
                                  cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                  conda_env="3d_tracking", n_bouts=30, session_dir="/x",
                                  masks_out="/y")
    assert "cu13/lib" in s and "LD_PRELOAD" in s and "--array=0-29" in s
