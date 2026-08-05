"""Pure-builder tests for scripts/slurm_dir_array.py (Task 13).

Covers discovery of Predictions_3D_* dirs and the sbatch-script builders
only -- never calls `sbatch` (see module docstring / task-13-brief.md).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.slurm_dir_array import (
    discover_dirs,
    _array_spec,
    build_stage_array_script,
    build_finalize_script,
)


# ---------------------------------------------------------------------------
# discover_dirs
# ---------------------------------------------------------------------------

def test_discover_dirs_sorted_and_recursive(tmp_path):
    # Two sessions, each containing a Predictions_3D_* dir, plus decoys that
    # must not match.
    d1 = tmp_path / "session0" / "Predictions_3D_bbb"
    d2 = tmp_path / "session1" / "Predictions_3D_aaa"
    decoy = tmp_path / "session0" / "NotAPredictionsDir"
    for d in (d1, d2, decoy):
        d.mkdir(parents=True)

    found = discover_dirs(tmp_path)

    assert found == sorted([d1, d2])


def test_discover_dirs_single_dir_passthrough(tmp_path):
    single = tmp_path / "Predictions_3D_only"
    single.mkdir()
    # Even though it contains no further Predictions_3D_* dirs, passing it
    # directly as base_dir must return exactly itself, not search inside it.
    nested = single / "Predictions_3D_nested"
    nested.mkdir()

    found = discover_dirs(single)

    assert found == [single]


def test_discover_dirs_empty(tmp_path):
    (tmp_path / "unrelated").mkdir()
    assert discover_dirs(tmp_path) == []


# ---------------------------------------------------------------------------
# _array_spec
# ---------------------------------------------------------------------------

def test_array_spec_contiguous_0_based():
    assert _array_spec(23) == "0-22"
    assert _array_spec(1) == "0-0"


def test_array_spec_max_concurrent_suffix():
    assert _array_spec(23, max_concurrent=8) == "0-22%8"
    assert _array_spec(23, max_concurrent=None) == "0-22"


# ---------------------------------------------------------------------------
# build_stage_array_script
# ---------------------------------------------------------------------------

def _stage_kwargs(**overrides):
    kwargs = dict(
        job_name="dirarr_free_running_v2_3",
        partition="gpu-l40s",
        account="portia",
        cpus=8,
        mem=64,
        time_limit="12:00:00",
        requeue=False,
        conda_env="3d_tracking",
        n_dirs=23,
        max_concurrent=None,
        out_dir="/data/free_running",
        manifest_path="/data/free_running/dir_manifest.txt",
        dataset="free_running",
        anatomy="v2_3",
        paths="hyak",
        postprocessing="v2_3",
        force=False,
        steps=["preprocess", "stac", "postprocess"],
    )
    kwargs.update(overrides)
    return kwargs


def test_stage_array_script_has_0_22_array_and_manifest_sed():
    s = build_stage_array_script(**_stage_kwargs())
    assert "#SBATCH --array=0-22" in s
    assert 'sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "/data/free_running/dir_manifest.txt"' in s


def test_stage_array_script_max_concurrent_suffix():
    s = build_stage_array_script(**_stage_kwargs(max_concurrent=8))
    assert "#SBATCH --array=0-22%8" in s


def test_stage_array_script_contains_all_three_stage_commands_in_order():
    s = build_stage_array_script(**_stage_kwargs())
    pre = s.index("batch_process_predictions.py")
    stac = s.index("batch_run_stac.py")
    post = s.index("batch_postprocess_predictions.py")
    assert pre < stac < post
    assert '--base-dir "$DIR"' in s
    assert "--anatomy v2_3" in s
    assert "--dataset free_running" in s
    assert "--postprocessing v2_3" in s
    # Bare `postprocessing=v2_3` (no --flag, no `@dataset...`) must never appear.
    assert "postprocessing=v2_3" not in s.replace("--postprocessing v2_3", "")


def test_stage_array_script_steps_restricts_commands():
    s = build_stage_array_script(**_stage_kwargs(steps=["stac"]))
    assert "batch_process_predictions.py" not in s
    assert "batch_run_stac.py" in s
    assert "batch_postprocess_predictions.py" not in s


def test_stage_array_script_force_flag_passthrough():
    s_on = build_stage_array_script(**_stage_kwargs(force=True))
    s_off = build_stage_array_script(**_stage_kwargs(force=False))
    assert "--force" in s_on
    assert "--force" not in s_off


def test_stage_array_script_has_set_e_and_requeue():
    s_req = build_stage_array_script(**_stage_kwargs(requeue=True))
    s_norequeue = build_stage_array_script(**_stage_kwargs(requeue=False))
    assert "set -e" in s_req
    assert "#SBATCH --requeue" in s_req
    assert "#SBATCH --requeue" not in s_norequeue


def test_stage_array_script_env_preamble():
    s = build_stage_array_script(**_stage_kwargs())
    assert "module load cuda/12.9.1" in s
    assert "source ~/.bashrc" in s
    assert "micromamba activate 3d_tracking" in s
    assert "unset LD_LIBRARY_PATH" in s
    assert "unset JAX_PLATFORMS" in s
    assert 'export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"' in s
    assert "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9" in s
    assert ("export JAX_COMPILATION_CACHE_DIR=/gscratch/portia/eabe/data/"
            "Johnson_lab/torch_cache/jax_cache_v2_3") in s
    assert "#SBATCH --account=portia" in s
    assert "#SBATCH --nodes=1" in s
    assert "#SBATCH --ntasks-per-node=1" in s
    assert "#SBATCH --gpus=1" in s
    assert "#SBATCH --open-mode=append" in s
    assert "-o /data/free_running/slurm-stage-%A_%a.out" in s


# ---------------------------------------------------------------------------
# build_finalize_script
# ---------------------------------------------------------------------------

def _finalize_kwargs(**overrides):
    kwargs = dict(
        job_name="dirfin_free_running_v2_3",
        partition="gpu-l40s",
        account="portia",
        cpus=8,
        mem=64,
        time_limit="03:00:00",
        requeue=False,
        conda_env="3d_tracking",
        out_dir="/data/free_running",
        dataset="free_running",
        anatomy="v2_3",
        paths="hyak",
        base_dir="/data/free_running",
        combined_h5="/data/free_running/v2_3/ik_output_combined_v2_3_free_running_interpolated.h5",
        out_h5="/gscratch/portia/eabe/fly_neuromech/data/datasets/Fruitfly_v2_3_walk_1000hz_interp_padded.h5",
        dependency="afterok:12345",
    )
    kwargs.update(overrides)
    return kwargs


def test_finalize_script_dependency_string():
    s = build_finalize_script(**_finalize_kwargs())
    assert "#SBATCH --dependency=afterok:12345" in s


def test_finalize_script_no_dependency_line_when_empty():
    s = build_finalize_script(**_finalize_kwargs(dependency=""))
    assert "--dependency=" not in s


def test_finalize_script_combine_uses_plus_base_dir():
    s = build_finalize_script(**_finalize_kwargs())
    assert "+base_dir=/data/free_running" in s
    assert "base_dir=/data/free_running" in s  # sanity: substring present at all
    # never a bare (non-`+`) base_dir= before combine_data.py's own args
    assert "combine_data.py paths=hyak dataset=free_running anatomy=v2_3 +base_dir=/data/free_running" in s


def test_finalize_script_commands_in_order_combine_pack_audit():
    s = build_finalize_script(**_finalize_kwargs())
    combine = s.index("combine_data.py")
    pack = s.index("pack_reference_clips.py")
    audit = s.index("audit_reference_clips.py")
    assert combine < pack < audit


def test_finalize_script_audit_is_last_command():
    s = build_finalize_script(**_finalize_kwargs())
    tail = s.strip().splitlines()[-1]
    assert "audit_reference_clips.py" in tail


def test_finalize_script_pack_and_audit_paths():
    s = build_finalize_script(**_finalize_kwargs())
    assert ("--input /data/free_running/v2_3/ik_output_combined_v2_3_free_running_interpolated.h5"
            in s)
    assert ("--output /gscratch/portia/eabe/fly_neuromech/data/datasets/"
            "Fruitfly_v2_3_walk_1000hz_interp_padded.h5") in s
    assert ("--h5 /gscratch/portia/eabe/fly_neuromech/data/datasets/"
            "Fruitfly_v2_3_walk_1000hz_interp_padded.h5") in s


def test_finalize_script_requeue_and_set_e():
    s_req = build_finalize_script(**_finalize_kwargs(requeue=True))
    s_norequeue = build_finalize_script(**_finalize_kwargs(requeue=False))
    assert "set -e" in s_req
    assert "#SBATCH --requeue" in s_req
    assert "#SBATCH --requeue" not in s_norequeue


def test_finalize_script_gpu_and_single_task():
    s = build_finalize_script(**_finalize_kwargs())
    assert "#SBATCH --gpus=1" in s
    assert "#SBATCH --array=" not in s
