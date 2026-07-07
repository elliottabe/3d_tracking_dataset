#!/usr/bin/env python3
"""
Submit ONE SLURM job that runs the full JAX courtship-prediction pipeline for a
session: D2 SAM3 masks (all bouts) -> D3 JAX 3D prediction (all bouts), producing
per-fly data3D_fly{0,1}.csv.

Uses the standard SLURM launcher pattern. The job body has TWO sequential stages
on one multi-GPU node:
  1. SAM3 masks  (scripts/sam3_masks.py) — PyTorch; reuse_masks=true skips bouts
     whose sam3_masks.npz already exists, so a requeue resumes cheaply.
  2. JAX predict (scripts/predict_session.py) — reads stage-1 masks, runs the
     v2vNet+ViTPose 3D pipeline, writes Predictions_3D-style CSVs.

--run-name selects the TRAINED v2vNet run to predict with (run_id): the predict
entrypoint loads ${paths.runs_root}/${run_id}/final. Output + masks live under
--out (default ${paths.processed_root}/<dataset>/SessionN/<rec>); masks go in
<out>/sam3_masks and predictions go in <out>/predictions; both stages share the
same root so stage 2 reads stage 1's output.

Per-stage CUDA libs: SAM3 (PyTorch) needs the cu13 wheels on LD_LIBRARY_PATH;
JAX needs its own bundled CUDA, so LD_LIBRARY_PATH is unset before stage 2.

Preemption resume
-----------------
Single-host job (JAX sees one node's GPUs). ckpt* partitions are preemptible, so
the job is submitted --requeue with a FIXED out dir baked in at submit time. On
preempt+requeue the same script reruns: stage 1 skips finished bouts
(reuse_masks), stage 2 re-runs idempotently (overwrites CSVs). Re-submitting with
the same --run-name/--out resumes; a fresh --out starts clean.

Usage:
    # Predict all of session 0 with the run4 v2vNet on ckpt-g2 (8 GPU, requeue):
    python scripts/slurm_predict_session.py --run-name run4

    # A different session / output dir, pass Hydra overrides to both stages:
    python scripts/slurm_predict_session.py --run-name run4 \\
        --session-dir /path/to/SessionX/vid --out /path/to/out \\
        predict_session.batch=8

    # See the script without submitting:
    python scripts/slurm_predict_session.py --run-name run4 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"


def build_script(
    *,
    job_name: str,
    partition: str,
    account: str,
    nodelist_line: str,
    exclude_line: str,
    requeue_line: str,
    gpus: int,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    mail_user: str,
    pkg_dir: Path,
    log_dir: str,
    run_name: str,
    paths: str,
    session_dir: str,
    masks_dir: str,
    out: str,
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
#SBATCH -o {log_dir}/slurm-%j.out
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user={mail_user}
{nodelist_line}
{exclude_line}
{requeue_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"   # cv2 (both stages)
echo "Node: $SLURMD_NODENAME  job: $SLURM_JOB_ID"
nvidia-smi -L
cd {pkg_dir}
set -e   # fail-fast: if Stage 1 (SAM3) errors, do NOT run Stage 2 on partial masks
# --- Stage 1: SAM3 masks (PyTorch; needs cu13 CUDA libs) ---
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
python -u scripts/sam3_masks.py paths={paths} sam3=default \\
    sam3.session_dir={session_dir} sam3.out={masks_dir} sam3.sam3_compile=false{overrides_str}
# --- Stage 2: JAX 3D predict (needs JAX's bundled CUDA -> unset LD_LIBRARY_PATH) ---
module load cuda/12.9.1                      # batch nodes don't expose libcuda by default -> JAX falls back to CPU without this
unset LD_LIBRARY_PATH                        # let JAX use its bundled CUDA wheels
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python -u scripts/predict_session.py paths={paths} model=hybridnet predict_session=default \\
    run_id={run_name} predict_session.session_dir={session_dir} \\
    predict_session.masks_dir={masks_dir} predict_session.out={out}{overrides_str}
"""


def compose_cfg(paths: str, slurm: str, run_name: str, passthrough: list[str]):
    """Compose a Hydra config at submit time to read slurm/paths values."""
    sys.path.insert(0, str(PKG_DIR))
    from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
    from hydra import initialize_config_dir, compose
    register_resolvers()
    overrides = [f"paths={paths}", f"slurm={slurm}", f"run_id={run_name}",
                 "model=hybridnet", "predict_session=default", "sam3=default"] + passthrough
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-name', required=True,
                   help='Trained v2vNet run to predict with (run_id); the predict '
                        'entrypoint loads ${paths.runs_root}/${run_id}/final. e.g. run4')
    p.add_argument('--session-dir', default=None,
                   help='Session video dir. Default: the sam3/predict_session config default (Session0).')
    p.add_argument('--dataset', default=None,
                   help='Dataset name for the processed path (default: derive from --session-dir)')
    p.add_argument('--out', default=None,
                   help='Output root. Default: ${paths.processed_root}/<dataset>/SessionN/<rec>. '
                        'Masks go in <out>/sam3_masks; predictions in <out>/predictions (resume-safe).')
    p.add_argument('--paths', default='hyak', help='Hydra paths config group (default: hyak)')
    p.add_argument('--slurm', default='ckpt_g2', help='Hydra slurm config group (default: ckpt_g2)')
    p.add_argument('--dry-run', action='store_true', help='Print the script without submitting')
    args, passthrough = p.parse_known_args()

    if not PKG_DIR.is_dir():
        print(f"Error: jarvis_jax package not found at {PKG_DIR}", file=sys.stderr)
        sys.exit(1)

    cfg = compose_cfg(args.paths, args.slurm, args.run_name, passthrough)
    sl = cfg.slurm

    from jarvis_jax.predict.paths_util import processed_dir_for
    session_dir = args.session_dir or cfg.predict_session.session_dir
    out_root = args.out or processed_dir_for(
        cfg.paths.processed_root, session_dir, dataset=args.dataset)
    masks_dir = f"{out_root}/sam3_masks"
    out = f"{out_root}/predictions"
    log_dir = out_root

    requeue_line = "#SBATCH --requeue" if sl.requeue else ""
    nodelist_line = (f"#SBATCH --nodelist={sl.nodelist}"
                     if getattr(sl, 'nodelist', None) else "")
    exclude_line = (f"#SBATCH --exclude={sl.exclude}"
                    if getattr(sl, 'exclude', None) else "")
    # Passthrough is appended to BOTH stage commands. This is safe: config.yaml
    # composes all groups (sam3, predict_session, ...), so a key meant for one
    # stage (e.g. predict_session.batch) exists in the other's merged config and
    # is accepted-but-ignored rather than failing Hydra's struct check.
    overrides_str = (" " + " ".join(passthrough)) if passthrough else ""
    job_name = f"predsess_{args.run_name}"[:60]

    script = build_script(
        job_name=job_name,
        partition=sl.partition, account=sl.account,
        nodelist_line=nodelist_line, exclude_line=exclude_line, requeue_line=requeue_line,
        gpus=sl.gpus, cpus=sl.cpus, mem=sl.mem, time_limit=sl.time,
        conda_env=sl.conda_env, mail_user=sl.mail_user,
        pkg_dir=PKG_DIR, log_dir=log_dir,
        run_name=args.run_name, paths=args.paths,
        session_dir=session_dir, masks_dir=masks_dir, out=out,
        overrides_str=overrides_str,
    )

    print(f"Model   : run_id={args.run_name} -> {cfg.paths.runs_root}/{args.run_name}/final")
    print(f"Session : {session_dir}")
    print(f"Out     : {out_root}  (masks -> {masks_dir}, predictions -> {out})")
    print(f"Compute : {sl.partition}, {sl.gpus} GPU(s)/1 node, "
          f"{'requeue on' if sl.requeue else 'requeue off'}")
    if passthrough:
        print(f"Overrides: {' '.join(passthrough)}")

    if args.dry_run:
        print("\n--- script (dry-run) ---")
        print(script)
        return

    Path(out_root).mkdir(parents=True, exist_ok=True)  # for the -o log path
    jid = slurm_submit(script)
    print(f"\nSubmitted {job_name}: {jid}")
    print(f"Monitor : squeue -j {jid}")
    print(f"Log     : {out_root}/slurm-{jid}.out")


if __name__ == "__main__":
    main()
