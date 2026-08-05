#!/usr/bin/env python3
"""
Submit ONE SLURM array job over 23 independent `Predictions_3D_*` directories
(e.g. under the v2_3 `free_running` root), instead of one separate job per
folder (`scripts/slurm_run.py`'s pattern). A dependent finalize job then runs
the session-level combine -> pack -> QC gate.

    [stage array, 1 task/dir]  --afterok-->  [finalize, 1 task]
     preprocess -> stac ->                    combine_data.py ->
     postprocess (per dir)                     pack_reference_clips.py ->
                                                audit_reference_clips.py (gate)

Task -> dir mapping is via a manifest file (`<base-dir>/dir_manifest.txt`,
one discovered dir per line) rather than embedding all paths in the script;
each array task reads its own line with
`sed -n "$((SLURM_ARRAY_TASK_ID+1))p"`.

Usage:
    # See both scripts + the dependency, without submitting:
    python scripts/slurm_dir_array.py \\
        --base-dir /gscratch/portia/eabe/data/Johnson_lab/free_running --dry-run

    # Real submit, capped at 8 concurrent tasks, forcing overwrite:
    python scripts/slurm_dir_array.py \\
        --base-dir /gscratch/portia/eabe/data/Johnson_lab/free_running \\
        --max-concurrent 8 --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Partition -> default requeue policy: preemptible ckpt* partitions requeue by
# default (mirrors scripts/slurm_run.py).
DEFAULT_OUT_H5 = Path(
    "/gscratch/portia/eabe/fly_neuromech/data/datasets/"
    "Fruitfly_v2_3_walk_1000hz_interp_padded.h5"
)

STEP_CHOICES = ("preprocess", "stac", "postprocess")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_dirs(base_dir: Path) -> list[Path]:
    """Sorted `Predictions_3D_*` dirs found recursively under `base_dir`.

    If `base_dir` itself matches that pattern (per-folder slurm jobs point
    directly at one dir, same convention as the three batch_*.py stage
    scripts), return `[base_dir]` rather than searching inside it."""
    if base_dir.is_dir() and base_dir.match("Predictions_3D_*"):
        return [base_dir]
    return sorted(p for p in base_dir.rglob("Predictions_3D_*") if p.is_dir())


# ---------------------------------------------------------------------------
# Pure sbatch-script builders (unit-tested; no SLURM/filesystem access)
# ---------------------------------------------------------------------------

def _array_spec(n_dirs: int, max_concurrent: int | None = None) -> str:
    """SLURM `--array=` value for `n_dirs` contiguous 0-based task ids, e.g.
    23 -> "0-22", with `%<max_concurrent>` appended when set, e.g. "0-22%8"."""
    spec = f"0-{n_dirs - 1}"
    if max_concurrent:
        spec += f"%{max_concurrent}"
    return spec


def build_stage_array_script(
    *,
    job_name: str,
    partition: str,
    account: str,
    cpus: int,
    mem: int,
    time_limit: str,
    requeue: bool,
    conda_env: str,
    n_dirs: int,
    max_concurrent: int | None,
    out_dir: str,
    manifest_path: str,
    dataset: str,
    anatomy: str,
    paths: str,
    postprocessing: str,
    force: bool,
    steps: list[str],
) -> str:
    """One array task per discovered `Predictions_3D_*` dir: reads its own
    line from `manifest_path` and runs, for that dir only, in order and
    aborting on the first failure (`set -e`):
      batch_process_predictions.py -> batch_run_stac.py ->
      batch_postprocess_predictions.py --postprocessing <group>
    `steps` (subset of preprocess/stac/postprocess) restricts which of the
    three actually run, for a partial re-run."""
    requeue_line = "#SBATCH --requeue" if requeue else ""
    force_flag = " --force" if force else ""

    step_cmds = []
    if "preprocess" in steps:
        step_cmds.append(
            f'echo "[preprocess] $DIR"\n'
            f'python -u scripts/batch_process_predictions.py '
            f'--dataset {dataset} --anatomy {anatomy} --paths {paths} '
            f'--base-dir "$DIR"{force_flag}'
        )
    if "stac" in steps:
        step_cmds.append(
            f'echo "[stac] $DIR"\n'
            f'python -u scripts/batch_run_stac.py '
            f'--dataset {dataset} --anatomy {anatomy} --paths {paths} '
            f'--base-dir "$DIR"{force_flag}'
        )
    if "postprocess" in steps:
        step_cmds.append(
            f'echo "[postprocess] $DIR"\n'
            f'python -u scripts/batch_postprocess_predictions.py '
            f'--dataset {dataset} --anatomy {anatomy} --paths {paths} '
            f'--base-dir "$DIR" --postprocessing {postprocessing}{force_flag}'
        )
    commands = "\n".join(step_cmds)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus=1
#SBATCH --mem={mem}G
#SBATCH --array={_array_spec(n_dirs, max_concurrent)}
#SBATCH --open-mode=append
#SBATCH -o {out_dir}/slurm-stage-%A_%a.out
{requeue_line}
set -e
set -x
module load cuda/12.9.1
source ~/.bashrc
micromamba activate {conda_env}
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR=/gscratch/portia/eabe/data/Johnson_lab/torch_cache/jax_cache_v2_3
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID  task: $SLURM_ARRAY_TASK_ID"
cd {PROJECT_DIR}
DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "{manifest_path}")
if [ -z "$DIR" ]; then
    echo "Error: no manifest line for task $SLURM_ARRAY_TASK_ID in {manifest_path}" >&2
    exit 1
fi
echo "Dir: $DIR"
{commands}
"""


def build_finalize_script(
    *,
    job_name: str,
    partition: str,
    account: str,
    cpus: int,
    mem: int,
    time_limit: str,
    requeue: bool,
    conda_env: str,
    out_dir: str,
    dataset: str,
    anatomy: str,
    paths: str,
    base_dir: str,
    combined_h5: str,
    out_h5: str,
    dependency: str,
) -> str:
    """Dependent (afterok on the stage array) session-level finalize:
      combine_data.py (+base_dir=<root>, needs the leading `+`) ->
      pack_reference_clips.py (packs the combined interpolated h5) ->
      audit_reference_clips.py (QC gate -- exits non-zero on any frozen-leg
      clip, so a bad rebuild fails this job instead of silently publishing).
    Needs a GPU (combine/pack use JAX) but only one task."""
    requeue_line = "#SBATCH --requeue" if requeue else ""
    dependency_line = f"#SBATCH --dependency={dependency}" if dependency else ""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus=1
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o {out_dir}/slurm-finalize-%A.out
{dependency_line}
{requeue_line}
set -e
set -x
module load cuda/12.9.1
source ~/.bashrc
micromamba activate {conda_env}
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR=/gscratch/portia/eabe/data/Johnson_lab/torch_cache/jax_cache_v2_3
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID"
cd {PROJECT_DIR}
python -u scripts/combine_data.py paths={paths} dataset={dataset} anatomy={anatomy} +base_dir={base_dir}
python -u scripts/export/pack_reference_clips.py --input {combined_h5} --output {out_h5} --anatomy {anatomy}
python -u scripts/qc/audit_reference_clips.py --h5 {out_h5}
"""


# ---------------------------------------------------------------------------
# Submission plumbing
# ---------------------------------------------------------------------------

def slurm_submit(script: str, *, dependency: str | None = None) -> str:
    """Submit a job script via stdin (sbatch --parsable) and return its job id."""
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    try:
        out = subprocess.check_output(cmd, input=script, universal_newlines=True)
        return out.strip().split(";")[0].split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-dir', type=Path, required=True,
                   help='Root to search for Predictions_3D_* dirs (or a single '
                        'such dir); also where dir_manifest.txt and per-task '
                        'logs are written')
    p.add_argument('--dataset', default='free_running', help='Dataset name')
    p.add_argument('--anatomy', default='v2_3', help='Anatomy version')
    p.add_argument('--paths', default='hyak', help='Hydra paths config group')
    p.add_argument('--postprocessing', default='v2_3',
                   help='Postprocessing config group, forwarded as '
                        '--postprocessing to batch_postprocess_predictions.py')
    p.add_argument('--out-h5', type=Path, default=DEFAULT_OUT_H5,
                   help='Packed reference-clip dataset path (finalize output)')
    p.add_argument('--partition', default='gpu-l40s', help='SLURM partition')
    p.add_argument('--cpus', type=int, default=8, help='CPUs per task')
    p.add_argument('--mem', type=int, default=64, help='Memory per task, in GB')
    p.add_argument('--time', default='12:00:00',
                   help='Per-task time limit for the stage array')
    p.add_argument('--finalize-time', default='03:00:00',
                   help='Time limit for the finalize job')
    p.add_argument('--conda-env', default='3d_tracking', help='Conda/micromamba env')
    p.add_argument('--account', default='portia', help='SLURM account')
    p.add_argument('--max-concurrent', type=int, default=None,
                   help='Cap on concurrently running array tasks (--array=...%%N)')
    p.add_argument('--force', action='store_true',
                   help='Pass --force through to each stage script')
    p.add_argument('--steps', default='preprocess,stac,postprocess',
                   help='Comma list of stages to run per dir '
                        '(subset of preprocess,stac,postprocess)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print both scripts + planned dependency; create no '
                        'directories, write no manifest, submit nothing')
    p.add_argument('--requeue', dest='requeue', action='store_true', default=None,
                   help='Add #SBATCH --requeue. Default: on for ckpt* partitions')
    p.add_argument('--no-requeue', dest='requeue', action='store_false',
                   help='Disable automatic requeue on preemption')
    args = p.parse_args()

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    bad = [s for s in steps if s not in STEP_CHOICES]
    if bad:
        print(f"Error: unknown --steps value(s) {bad}; choose from {STEP_CHOICES}",
              file=sys.stderr)
        sys.exit(1)

    base_dir = args.base_dir
    if not base_dir.exists():
        print(f"Error: base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    dirs = discover_dirs(base_dir)
    n_dirs = len(dirs)
    if n_dirs == 0:
        print(f"Error: no Predictions_3D_* dirs found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {n_dirs} Predictions_3D_* dir(s) under {base_dir}:")
    for d in dirs:
        print(f"  {d}")

    requeue = args.requeue
    if requeue is None:
        requeue = args.partition.startswith('ckpt')

    out_dir = str(base_dir)
    manifest_path = str(base_dir / "dir_manifest.txt")
    combined_h5 = str(base_dir / args.anatomy /
                      f"ik_output_combined_{args.anatomy}_{args.dataset}_interpolated.h5")

    stage_job_name = f"dirarr_{args.dataset}_{args.anatomy}"[:60]
    stage_script = build_stage_array_script(
        job_name=stage_job_name, partition=args.partition, account=args.account,
        cpus=args.cpus, mem=args.mem, time_limit=args.time, requeue=requeue,
        conda_env=args.conda_env, n_dirs=n_dirs, max_concurrent=args.max_concurrent,
        out_dir=out_dir, manifest_path=manifest_path, dataset=args.dataset,
        anatomy=args.anatomy, paths=args.paths, postprocessing=args.postprocessing,
        force=args.force, steps=steps,
    )

    print(f"\n--- stage array script{' (dry-run)' if args.dry_run else ''} ---")
    print(stage_script)

    if args.dry_run:
        array_job_id = "<ARRAY_JOBID>"
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
        manifest_text = "\n".join(str(d) for d in dirs) + "\n"
        Path(manifest_path).write_text(manifest_text)
        print(f"Wrote manifest: {manifest_path} ({n_dirs} lines)")
        array_job_id = slurm_submit(stage_script)
        print(f"Submitted stage array: {array_job_id}  "
              f"(--array={_array_spec(n_dirs, args.max_concurrent)})")

    dependency = f"afterok:{array_job_id}"
    finalize_job_name = f"dirfin_{args.dataset}_{args.anatomy}"[:60]
    finalize_script = build_finalize_script(
        job_name=finalize_job_name, partition=args.partition, account=args.account,
        cpus=args.cpus, mem=args.mem, time_limit=args.finalize_time, requeue=requeue,
        conda_env=args.conda_env, out_dir=out_dir, dataset=args.dataset,
        anatomy=args.anatomy, paths=args.paths, base_dir=str(base_dir),
        combined_h5=combined_h5, out_h5=str(args.out_h5), dependency=dependency,
    )

    print(f"\n--- finalize script{' (dry-run)' if args.dry_run else ''} "
          f"(dependency={dependency}) ---")
    print(finalize_script)

    if args.dry_run:
        print(f"\n[dry-run] would submit stage array (--array="
              f"{_array_spec(n_dirs, args.max_concurrent)}), then finalize with "
              f"--dependency={dependency}. Submitting nothing.")
        return

    finalize_job_id = slurm_submit(finalize_script, dependency=dependency)
    print(f"Submitted finalize: {finalize_job_id}  (dependency={dependency})")
    print(f"\nDependency chain: stage_array({array_job_id}) -> finalize({finalize_job_id})")
    print(f"Logs   : {out_dir}/slurm-stage-{array_job_id}_*.out , "
          f"{out_dir}/slurm-finalize-{finalize_job_id}.out")
    print(f"Monitor: squeue -u $USER")


if __name__ == "__main__":
    main()
