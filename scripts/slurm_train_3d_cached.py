#!/usr/bin/env python3
"""
Submit a SLURM job to train the JAX v2vNet 3D pose model
(third_party/jarvis_jax) using cached volume heatmaps precomputed from ViTPose keypoints.

Companion to scripts/slurm_train_hybridnet.py (which trains the full 3D pose model
with ViTPose frozen). This one trains a lighter v2vNet: single-host data-parallel
across N GPUs on one node, frozen ViTPose (inference baked into cache), 3D heatmap
regression with optional graph-Laplacian bone prior, 3D soft-argmax.

The job body has TWO steps:
  1. Precompute-if-absent: if the cache-dir lacks train_meta.json or val_meta.json,
     run precompute_repro_cache.py to build volume heatmaps. This is cached across
     runs (shared cache dir), so only runs once.
  2. Train: run the cached v2vNet trainer, resuming from ckpt if present.

Preemption resume
-----------------
The job is single-host (JAX sees only the local node's GPUs), so --gpus all land
on ONE node. ckpt* partitions are preemptible, so the job is submitted with
--requeue and a FIXED run directory: --ckpt-dir is a literal baked into the
script at submit time (NOT derived from $SLURM_JOB_ID or date inside the job), so
a preempt+requeue reruns the same script and the trainer's restore_latest()
picks up from the last checkpoint. Re-submitting with the same --run-name resumes
that run; a fresh --run-name (the default, date-stamped) starts a new run.

Usage:
    # Train on ckpt-g2 with 8 GPUs (preemptible, auto-resume), a fresh run:
    python scripts/slurm_train_3d_cached.py

    # Resume / continue a specific run (same ckpt dir):
    python scripts/slurm_train_3d_cached.py --run-name cached3d_v3_8gpu_20260620

    # Non-preemptible L40S node, 4 GPUs, custom hyperparams:
    python scripts/slurm_train_3d_cached.py --partition gpu-l40s --gpus 4 --batch 128 --lr 5e-4

    # See the script without submitting:
    python scripts/slurm_train_3d_cached.py --dry-run
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"

# Persistent defaults (outside git; large artifacts / run outputs).
DEFAULT_DATA_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
DEFAULT_VITPOSE_CKPT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_8gpu_20260620/final"
DEFAULT_CACHE_DIR = "/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v3"
DEFAULT_RUNS_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs"

# Partition -> nodelist (from sinfo). Only partitions whose nodes have enough
# GPUs-per-node for --gpus are valid for this single-host job.
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


def build_script(
    *,
    job_name: str,
    partition: str,
    nodelist_line: str,
    requeue_line: str,
    gpus: int,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    pkg_dir: Path,
    run_dir: str,
    data_root: str,
    vitpose_ckpt: str,
    cache_dir: str,
    steps: int,
    batch: int,
    lr: str,
    laplacian_weight: float,
    save_every: int,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o {run_dir}/slurm-%j.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
{requeue_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
unset LD_LIBRARY_PATH                       # let JAX use its bundled CUDA wheels
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID"
nvidia-smi -L
cd {pkg_dir}
python -u -c "import jax; print('jax devices:', jax.device_count())"
# Precompute cache if absent (train split)
[ -f {cache_dir}/train_meta.json ] || python -u scripts/precompute_repro_cache.py --root {data_root} --vitpose-ckpt {vitpose_ckpt} --cache-dir {cache_dir} --split train
# Precompute cache if absent (val split)
[ -f {cache_dir}/val_meta.json ] || python -u scripts/precompute_repro_cache.py --root {data_root} --vitpose-ckpt {vitpose_ckpt} --cache-dir {cache_dir} --split val
python -u -m jarvis_jax.train.train_3d_cached \\
    --cache-dir {cache_dir} \\
    --out {run_dir}/final \\
    --ckpt-dir {run_dir}/ckpt \\
    --steps {steps} \\
    --batch {batch} \\
    --lr {lr} \\
    --laplacian-weight {laplacian_weight} \\
    --save-every {save_every}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', default=None,
                   help='Run directory name under the runs root. Resuming uses the '
                        'same name (its ckpt-dir). Default: date-stamped fresh run.')
    p.add_argument('--runs-root', default=DEFAULT_RUNS_ROOT,
                   help=f'Parent dir for run outputs (default: {DEFAULT_RUNS_ROOT})')
    p.add_argument('--root', default=DEFAULT_DATA_ROOT,
                   help='red_data_unified_V3 root with 3D framesets (for precompute)')
    p.add_argument('--vitpose-ckpt', default=DEFAULT_VITPOSE_CKPT,
                   help=f'Trained 2D ViTPose checkpoint for frozen backbone '
                        f'(default: {DEFAULT_VITPOSE_CKPT})')
    p.add_argument('--cache-dir', default=DEFAULT_CACHE_DIR,
                   help=f'Shared cache dir for precomputed volume heatmaps '
                        f'(default: {DEFAULT_CACHE_DIR})')
    p.add_argument('--gpus', type=int, default=8,
                   help='GPUs (all on ONE node; batch must be divisible by this). '
                        'Default: 8')
    p.add_argument('--batch', type=int, default=256,
                   help='Global batch size (split across --gpus). Cached steps are '
                        'tiny, so larger batches are fine. Must be divisible by '
                        '--gpus. Default: 256 (32/GPU)')
    p.add_argument('--steps', type=int, default=20000, help='Total optimizer steps')
    p.add_argument('--lr', type=float, default=3e-4,
                   help='BASE learning rate, calibrated at --base-batch. The '
                        'actual LR passed to the trainer is this scaled to --batch '
                        'by --lr-scaling (default base: 3e-4 @ batch 256)')
    p.add_argument('--base-batch', type=int, default=256,
                   help='Batch size at which --lr is calibrated (default: 256)')
    p.add_argument('--lr-scaling', choices=['none', 'linear', 'sqrt'], default='sqrt',
                   help="How to scale --lr from --base-batch to --batch: 'sqrt' "
                        "(recommended for AdamW), 'linear' (lr*ratio), or 'none'. "
                        "Default: sqrt")
    p.add_argument('--laplacian-weight', type=float, default=0.0,
                   help='Weight for graph-Laplacian bone prior loss (default: 0.0, off)')
    p.add_argument('--save-every', type=int, default=1000,
                   help='Checkpoint cadence in steps (default: 1000)')
    p.add_argument('--conda-env', default='3d_tracking',
                   help='micromamba/conda env to activate (default: 3d_tracking)')
    p.add_argument('--partition', default='ckpt-g2',
                   help='SLURM partition (default: ckpt-g2, preemptible)')
    p.add_argument('--cpus', type=int, default=32, help='CPUs per job (default: 32)')
    p.add_argument('--mem', type=int, default=128, help='Memory GB (default: 128)')
    p.add_argument('--time', default='3-00:00:00', help='Per-allocation time limit')
    p.add_argument('--requeue', dest='requeue', action='store_true', default=None,
                   help='Add #SBATCH --requeue (default: on for ckpt* partitions)')
    p.add_argument('--no-requeue', dest='requeue', action='store_false',
                   help='Disable automatic requeue on preemption')
    p.add_argument('--dry-run', action='store_true',
                   help='Print the script without submitting')
    args = p.parse_args()

    if args.batch % args.gpus != 0:
        print(f"Error: --batch ({args.batch}) must be divisible by --gpus "
              f"({args.gpus}) for data-parallel sharding.", file=sys.stderr)
        sys.exit(1)

    if not PKG_DIR.is_dir():
        print(f"Error: jarvis_jax package not found at {PKG_DIR}", file=sys.stderr)
        sys.exit(1)

    # Scale the base LR (calibrated at --base-batch) to the actual --batch.
    # sqrt is the safe default for AdamW; linear over-scales adaptive optimizers.
    ratio = args.batch / args.base_batch
    factor = {'none': 1.0, 'linear': ratio, 'sqrt': math.sqrt(ratio)}[args.lr_scaling]
    eff_lr = args.lr * factor
    lr_str = f"{eff_lr:.3e}"

    # Resolve the run dir ONCE here, so it is a fixed literal in the script and
    # survives requeue (date computed at submit time, not inside the job).
    run_name = args.run_name or f"cached3d_v3_{args.gpus}gpu_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = str(Path(args.runs_root) / run_name)

    # Default: requeue on for preemptible ckpt* partitions.
    requeue = args.requeue
    if requeue is None:
        requeue = args.partition.startswith('ckpt')
    requeue_line = "#SBATCH --requeue" if requeue else ""

    nodelist_line = (f"#SBATCH --nodelist={GPU_NODELISTS[args.partition]}"
                     if args.partition in GPU_NODELISTS else "")

    resuming = (Path(run_dir) / "ckpt").is_dir()

    job_name = f"cached3d_{run_name}"[:60]
    script = build_script(
        job_name=job_name,
        partition=args.partition,
        nodelist_line=nodelist_line,
        requeue_line=requeue_line,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        time_limit=args.time,
        conda_env=args.conda_env,
        pkg_dir=PKG_DIR,
        run_dir=run_dir,
        data_root=args.root,
        vitpose_ckpt=args.vitpose_ckpt,
        cache_dir=args.cache_dir,
        steps=args.steps,
        batch=args.batch,
        lr=lr_str,
        laplacian_weight=args.laplacian_weight,
        save_every=args.save_every,
    )

    print(f"Run dir : {run_dir}"
          f"{'  (RESUMING — ckpt/ exists)' if resuming else '  (fresh)'}")
    print(f"Compute : {args.partition}, {args.gpus} GPU(s)/1 node, "
          f"batch {args.batch} ({args.batch // args.gpus}/GPU), "
          f"{'requeue on' if requeue else 'requeue off'}")
    if args.lr_scaling == 'none' or args.batch == args.base_batch:
        lr_note = f"lr {lr_str}"
    else:
        lr_note = (f"lr {lr_str} ({args.lr_scaling}-scaled from {args.lr:g} "
                   f"@ base-batch {args.base_batch} -> batch {args.batch})")
    print(f"Train   : {args.steps} steps, {lr_note}, "
          f"laplacian-weight {args.laplacian_weight}, save every {args.save_every}")
    print(f"Cache   : {args.cache_dir} (precompute-if-absent)")

    if args.dry_run:
        print("\n--- script (dry-run) ---")
        print(script)
        return

    Path(run_dir).mkdir(parents=True, exist_ok=True)  # for the -o log path
    jid = slurm_submit(script)
    print(f"\nSubmitted {job_name}: {jid}")
    print(f"Monitor : squeue -j {jid}")
    print(f"Log     : {run_dir}/slurm-{jid}.out")


if __name__ == "__main__":
    main()
