#!/usr/bin/env python3
"""
Submit parallel SLURM jobs to run the full 3D tracking pipeline on every
Predictions_3D_* folder under a dataset directory.

Strategy: the pipeline (`run_full_pipeline.py`) only uses 1 GPU, so instead of
running it sequentially over every folder we submit one sbatch job *per folder*
that contains a `<dataset>_bouts_summary.csv` file. Each job runs the
preprocess -> stac -> postprocess steps on a single folder. An optional final
`combine` job is submitted with an `afterok` dependency on all per-folder jobs.

Usage:
    python scripts/slurm_run.py --dataset free_walking --anatomy v1
    python scripts/slurm_run.py --dataset free_walking --anatomy v1 --dry-run
    python scripts/slurm_run.py --dataset courtship --anatomy v1 --no-combine
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_ROOT = Path('/gscratch/portia/eabe/data/Johnson_lab')

# Partition -> nodelist (from sinfo); auto-selected based on --partition.
GPU_NODELISTS = {
    'gpu-a40':  'g[3040-3047,3050-3057,3060-3067,3070-3077]',
    'gpu-a100': 'g[3080-3087]',
    'gpu-l40':  'g[3090-3099,3115-3119]',
    'gpu-l40s': 'g[3100-3114,3120-3124,3133-3137]',
    'gpu-h200': 'g[3125-3132]',
    'ckpt-g2':  'g[3090-3137]',
}


def slurm_submit(script: str) -> str:
    """Submit a job script via stdin and return its job id."""
    try:
        out = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        return out.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def find_folders(base_dir: Path, dataset: str) -> list[Path]:
    """Find every Predictions_3D_* folder containing a <dataset>_bouts*summary.csv.

    Matches both single-animal (`<dataset>_bouts_summary.csv`) and multi-animal
    per-fly (`<dataset>_bouts_fly0_summary.csv`, `_fly1_summary.csv`, ...) layouts.
    Folders containing multiple per-fly summaries are de-duplicated.
    """
    pattern = f"{dataset}_bouts*summary.csv"
    folders = sorted({p.parent for p in base_dir.rglob(pattern)
                      if p.parent.match("Predictions_3D_*")})
    return folders


def build_script(
    *,
    job_name: str,
    partition: str,
    nodelist_line: str,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    project_dir: Path,
    folder: Path,
    dataset: str,
    anatomy: str,
    paths: str,
    steps: str,
    extra: str,
    requeue_line: str = "",
    dependency_line: str = "",
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus=1
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
{requeue_line}
{dependency_line}
module load cuda/12.9.1
set -x
source ~/.bashrc
nvidia-smi
conda activate {conda_env}
unset LD_LIBRARY_PATH
echo "Node: $SLURMD_NODENAME"
echo "Folder: {folder}"
cd {project_dir}
python -u scripts/run_full_pipeline.py \\
    --dataset {dataset} \\
    --anatomy {anatomy} \\
    --paths {paths} \\
    --base-dir {folder} \\
    --steps {steps} \\
    --yes {extra}
"""


def build_combine_script(
    *,
    job_name: str,
    partition: str,
    nodelist_line: str,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    project_dir: Path,
    dataset: str,
    anatomy: str,
    paths: str,
    base_dir: Path,
    requeue_line: str = "",
    dependency_line: str = "",
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus=1
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
{requeue_line}
{dependency_line}
module load cuda/12.9.1
set -x
source ~/.bashrc
nvidia-smi
conda activate {conda_env}
unset LD_LIBRARY_PATH
cd {project_dir}
python -u scripts/run_full_pipeline.py \\
    --dataset {dataset} \\
    --anatomy {anatomy} \\
    --paths {paths} \\
    --base-dir {base_dir} \\
    --steps combine \\
    --yes
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='free_walking',
                   help='Dataset name; expects <dataset>_bouts_summary.csv per folder')
    p.add_argument('--anatomy', default='v1', help='Anatomy version (v1, v2_muscles, ...)')
    p.add_argument('--paths', default='hyak', help='Paths config (hyak, workstation, ...)')
    p.add_argument('--base-dir', type=Path, default=None,
                   help='Dataset root to search (default: DATA_ROOT/<dataset>)')
    p.add_argument('--steps', default='preprocess,split_valid,stac,postprocess',
                   help='Per-folder steps to run (combine is submitted as a separate job)')
    p.add_argument('--no-combine', action='store_true',
                   help='Do not submit a final combine job after the per-folder jobs')
    p.add_argument('--conda-env', default='3d_tracking', help='Conda environment to use')
    p.add_argument('--partition', default='gpu-l40s', help='SLURM partition to submit jobs to')
    p.add_argument('--cpus', type=int, default=8, help='Number of CPUs per job')
    p.add_argument('--mem', type=int, default=128, help='Memory per job in GB')
    p.add_argument('--time', default='24:00:00', help='Per-folder job time limit')
    p.add_argument('--combine-time', default='03:00:00', help='Time limit for combine job')
    p.add_argument('--extra', default='',
                   help='Extra args appended to run_full_pipeline.py (e.g. --force)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print scripts and folder list without submitting')
    p.add_argument('--requeue', dest='requeue', action='store_true', default=None,
                   help='Add #SBATCH --requeue so preempted jobs are automatically '
                        'requeued. Default: on for ckpt* partitions, off otherwise.')
    p.add_argument('--no-requeue', dest='requeue', action='store_false',
                   help='Disable automatic requeue on preemption.')
    args = p.parse_args()

    # Default: requeue on for preemptible ckpt* partitions.
    if args.requeue is None:
        args.requeue = args.partition.startswith('ckpt')
    requeue_line = "#SBATCH --requeue" if args.requeue else ""

    base_dir = args.base_dir if args.base_dir is not None else DATA_ROOT / args.dataset
    if not base_dir.exists():
        print(f"Error: base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    folders = find_folders(base_dir, args.dataset)
    if not folders:
        print(f"No Predictions_3D_* folders with {args.dataset}_bouts*summary.csv "
              f"under {base_dir} — nothing to submit, skipping.")
        return

    print(f"Found {len(folders)} folder(s) under {base_dir}:")
    for f in folders:
        print(f"  {f}")

    nodelist_line = (f"#SBATCH --nodelist={GPU_NODELISTS[args.partition]}"
                     if args.partition in GPU_NODELISTS else "")
    project_dir = Path(__file__).resolve().parent.parent
    (project_dir / "OutFiles").mkdir(exist_ok=True)

    job_ids: list[str] = []
    for folder in folders:
        job_name = f"pipe_{args.dataset}_{folder.name[-15:]}"
        script = build_script(
            job_name=job_name,
            partition=args.partition,
            nodelist_line=nodelist_line,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            conda_env=args.conda_env,
            project_dir=project_dir,
            folder=folder,
            dataset=args.dataset,
            anatomy=args.anatomy,
            paths=args.paths,
            steps=args.steps,
            extra=args.extra,
            requeue_line=requeue_line,
        )
        if args.dry_run:
            print("\n--- SCRIPT (dry-run) ---")
            print(script)
            continue
        jid = slurm_submit(script)
        print(f"Submitted {job_name}: {jid}")
        job_ids.append(jid)

    if args.no_combine or args.dry_run or not job_ids:
        if args.dry_run and not args.no_combine:
            print("\n[dry-run] would submit combine job with afterok dependency on all above")
        return

    dep = f"#SBATCH --dependency=afterok:{':'.join(job_ids)}"
    combine_script = build_combine_script(
        job_name=f"combine_{args.dataset}",
        partition=args.partition,
        nodelist_line=nodelist_line,
        cpus=args.cpus,
        mem=args.mem,
        time_limit=args.combine_time,
        conda_env=args.conda_env,
        project_dir=project_dir,
        dataset=args.dataset,
        anatomy=args.anatomy,
        paths=args.paths,
        base_dir=base_dir,
        requeue_line=requeue_line,
        dependency_line=dep,
    )
    cjid = slurm_submit(combine_script)
    print(f"Submitted combine job: {cjid} (afterok on {len(job_ids)} jobs)")


if __name__ == "__main__":
    main()


'''

squeue -u $USER -h -o "%i %j" | awk '/pipe_courtship|combine_courtship/ {print $1}' | xargs -r scancel

python ./scripts/slurm_run.py --dataset free_walking --anatomy v1 --paths hyak --base-dir /gscratch/portia/eabe/data/Johnson_lab/free_walking/session11
python ./scripts/slurm_run.py --dataset courtship --anatomy v1 --paths hyak --base-dir /gscratch/portia/eabe/data/Johnson_lab/courtship/04092026_bouts --extra=--force --partition ckpt-g2 --dry-run



'''