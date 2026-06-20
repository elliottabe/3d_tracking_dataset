#!/usr/bin/env python3
"""
Submit a SLURM job to train the JAX ViTPose KeypointDetect model
(third_party/jarvis_jax) on red_data_unified_V3.

Companion to scripts/slurm_train.py (which trains the PyTorch JARVIS stack).
This one trains the JAX/Flax ViTPose reimplementation: single-host data-parallel
across N GPUs on one node, MAE-pretrained backbone, foreground-weighted heatmap
loss, 2-group (backbone/head) LR fine-tune.

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
    python scripts/slurm_train_vit.py

    # Resume / continue a specific run (same ckpt dir):
    python scripts/slurm_train_vit.py --run-name v3_8gpu_20260620

    # Non-preemptible L40S node, 4 GPUs, custom hyperparams:
    python scripts/slurm_train_vit.py --partition gpu-l40s --gpus 4 --batch 32 --lr 1e-3

    # See the script without submitting:
    python scripts/slurm_train_vit.py --dry-run
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
DEFAULT_MAE_NPZ = "/gscratch/portia/eabe/data/Johnson_lab/mae_vitb.npz"
DEFAULT_RUNS_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs"

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
    mae_npz: str,
    steps: int,
    batch: int,
    lr: str,
    backbone_lr_mult: str,
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
python -u -m jarvis_jax.scripts.train_keypoints \\
    --root {data_root} \\
    --out {run_dir}/final \\
    --ckpt-dir {run_dir}/ckpt \\
    --mae-npz {mae_npz} \\
    --steps {steps} \\
    --batch {batch} \\
    --lr {lr} \\
    --backbone-lr-mult {backbone_lr_mult} \\
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
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help='red_data_unified_V3 root')
    p.add_argument('--mae-npz', default=DEFAULT_MAE_NPZ,
                   help='MAE-pretrained ViT-B npz for backbone init')
    p.add_argument('--gpus', type=int, default=8,
                   help='GPUs (all on ONE node; batch must be divisible by this). '
                        'Default: 8')
    p.add_argument('--batch', type=int, default=128,
                   help='Global batch size (split across --gpus). Default: 64 (8/GPU)')
    p.add_argument('--steps', type=int, default=20000, help='Total optimizer steps')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='BASE head learning rate, calibrated at --base-batch. The '
                        'actual LR passed to the trainer is this scaled to --batch '
                        'by --lr-scaling (default base: 1e-3 @ batch 8)')
    p.add_argument('--base-batch', type=int, default=8,
                   help='Batch size at which --lr is calibrated (default: 8)')
    p.add_argument('--lr-scaling', choices=['none', 'linear', 'sqrt'], default='sqrt',
                   help="How to scale --lr from --base-batch to --batch: 'sqrt' "
                        "(recommended for AdamW), 'linear' (lr*ratio), or 'none'. "
                        "Default: sqrt")
    p.add_argument('--backbone-lr-mult', default='0.1',
                   help='Backbone LR = lr * this (default: 0.1)')
    p.add_argument('--save-every', type=int, default=500,
                   help='Checkpoint cadence in steps (default: 500)')
    p.add_argument('--conda-env', default='3d_tracking',
                   help='micromamba/conda env to activate (default: 3d_tracking)')
    p.add_argument('--partition', default='ckpt-g2',
                   help='SLURM partition (default: ckpt-g2, preemptible)')
    p.add_argument('--cpus', type=int, default=32, help='CPUs per job (default: 32)')
    p.add_argument('--mem', type=int, default=128, help='Memory GB (default: 128)')
    p.add_argument('--time', default='8:00:00', help='Per-allocation time limit')
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
    run_name = args.run_name or f"v3_{args.gpus}gpu_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = str(Path(args.runs_root) / run_name)

    # Default: requeue on for preemptible ckpt* partitions.
    requeue = args.requeue
    if requeue is None:
        requeue = args.partition.startswith('ckpt')
    requeue_line = "#SBATCH --requeue" if requeue else ""

    nodelist_line = (f"#SBATCH --nodelist={GPU_NODELISTS[args.partition]}"
                     if args.partition in GPU_NODELISTS else "")

    resuming = (Path(run_dir) / "ckpt").is_dir()

    job_name = f"jaxvit_{run_name}"[:60]
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
        data_root=args.data_root,
        mae_npz=args.mae_npz,
        steps=args.steps,
        batch=args.batch,
        lr=lr_str,
        backbone_lr_mult=args.backbone_lr_mult,
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
    print(f"Train   : {args.steps} steps, {lr_note}, backbone x{args.backbone_lr_mult}, "
          f"save every {args.save_every}")

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
