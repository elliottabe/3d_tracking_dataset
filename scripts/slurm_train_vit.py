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
--requeue and a FIXED run directory: run_id is a literal baked into the
script at submit time (NOT derived from $SLURM_JOB_ID or date inside the job), so
a preempt+requeue reruns the same script and the trainer's restore_latest()
picks up from the last checkpoint. Re-submitting with the same --run-name resumes
that run; a fresh --run-name (the default, date-stamped) starts a new run.

Usage:
    # Train on ckpt-g2 with 8 GPUs (preemptible, auto-resume), a fresh run:
    python scripts/slurm_train_vit.py --run-name myrun

    # Resume / continue a specific run (same ckpt dir):
    python scripts/slurm_train_vit.py --run-name v3_8gpu_20260620

    # Pass Hydra overrides:
    python scripts/slurm_train_vit.py --run-name myrun train.total_steps=30000

    # See the script without submitting:
    python scripts/slurm_train_vit.py --run-name t --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"


def compose_cfg(paths: str, slurm: str, run_name: str, passthrough: list[str]):
    """Compose a Hydra config at submit time to read slurm/paths values."""
    sys.path.insert(0, str(PKG_DIR))
    from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
    from hydra import initialize_config_dir, compose
    register_resolvers()
    # Override runs_root to the vit-specific runs root
    overrides = [f"paths={paths}", f"slurm={slurm}", f"run_id={run_name}",
                 "train=vit2d", "model=vitpose",
                 "paths.runs_root=${paths.vit_runs_root}"] + passthrough
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


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
    account: str,
    nodelist_line: str,
    requeue_line: str,
    gpus: int,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    mail_user: str,
    exclude_line: str,
    pkg_dir: Path,
    run_dir: str,
    run_name: str,
    paths: str,
    overrides_str: str,
) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}G
#SBATCH --open-mode=append
#SBATCH -o {run_dir}/slurm-%j.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user={mail_user}
{nodelist_line}
{exclude_line}
{requeue_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
unset LD_LIBRARY_PATH                       # let JAX use its bundled CUDA wheels
unset JAX_PLATFORMS                         # NEVER inherit JAX_PLATFORMS=cpu from the submit env
                                            # (sbatch --export=ALL) -- that silently trains on CPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID"
nvidia-smi -L
cd {pkg_dir}
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.scripts.train_keypoints \\
    run_id={run_name} model=vitpose train=vit2d paths={paths} \\
    'paths.runs_root=${{paths.vit_runs_root}}'{overrides_str}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', default=None,
                   help='Run directory name (run_id). Resuming uses the same name. '
                        'Default: date-stamped fresh run.')
    p.add_argument('--paths', default='hyak',
                   help='Hydra paths config group (default: hyak)')
    p.add_argument('--slurm', default='ckpt_g2',
                   help='Hydra slurm config group (default: ckpt_g2)')
    p.add_argument('--dry-run', action='store_true',
                   help='Print the script without submitting')
    args, passthrough = p.parse_known_args()

    if not PKG_DIR.is_dir():
        print(f"Error: jarvis_jax package not found at {PKG_DIR}", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_name or f"v3_{time.strftime('%Y%m%d_%H%M%S')}"

    # Compose config at submit time to read slurm/paths values.
    cfg = compose_cfg(args.paths, args.slurm, run_name, passthrough)
    sl = cfg.slurm
    pt = cfg.paths

    run_dir = str(Path(pt.runs_root) / run_name)

    requeue_line = "#SBATCH --requeue" if sl.requeue else ""
    nodelist_line = (f"#SBATCH --nodelist={sl.nodelist}"
                     if getattr(sl, 'nodelist', None) else "")
    exclude_line = (f"#SBATCH --exclude={sl.exclude}"
                    if getattr(sl, 'exclude', None) else "")

    # Build the Hydra overrides string for the job body command
    overrides_str = (" \\\n    " + " \\\n    ".join(passthrough)) if passthrough else ""

    resuming = (Path(run_dir) / "ckpt").is_dir()
    job_name = f"jaxvit_{run_name}"[:60]

    script = build_script(
        job_name=job_name,
        partition=sl.partition,
        account=sl.account,
        nodelist_line=nodelist_line,
        requeue_line=requeue_line,
        gpus=sl.gpus,
        cpus=sl.cpus,
        mem=sl.mem,
        time_limit=sl.time,
        conda_env=sl.conda_env,
        mail_user=sl.mail_user,
        exclude_line=exclude_line,
        pkg_dir=PKG_DIR,
        run_dir=run_dir,
        run_name=run_name,
        paths=args.paths,
        overrides_str=overrides_str,
    )

    print(f"Run dir : {run_dir}"
          f"{'  (RESUMING — ckpt/ exists)' if resuming else '  (fresh)'}")
    print(f"Compute : {sl.partition}, {sl.gpus} GPU(s)/1 node, "
          f"{'requeue on' if sl.requeue else 'requeue off'}")
    if passthrough:
        print(f"Overrides: {' '.join(passthrough)}")

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
