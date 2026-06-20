#!/usr/bin/env python3
"""
Submit SLURM jobs to train the JARVIS-HybridNet network stack
(CenterDetect / KeypointDetect / HybridNet) for a given project.

Companion to scripts/slurm_run.py (which submits the *prediction* pipeline).
This one submits *training* jobs by invoking `jarvis-local train <stage>` inside
the bundled third_party/JARVIS-HybridNet repo.

One job per stage. Stages run in the order given, each with an `afterok`
dependency on the previous one, so a full `centerDetect,keypointDetect,hybridNet`
chain trains end-to-end and HybridNet only starts once KeypointDetect's weights
exist (it loads them via --weights_keypoint_detect latest).

Defaults reproduce the verified runs:
  * KeypointDetect: `jarvis-local train keypointDetect --weights_path latest <project>`
    with MASK_CONTAINMENT_WEIGHT set in the project config.
  * HybridNet:      `jarvis-local train hybridNet --weights_keypoint_detect latest <project>`
    with HYBRIDNET.LAPLACIAN_WEIGHT (graph-Laplacian shape prior) set in config.

Usage:
    # Just train keypoints (default):
    python scripts/slurm_train.py --project unified_V3_masked

    # Full stack, chained:
    python scripts/slurm_train.py --project unified_V3_masked \
        --stages centerDetect,keypointDetect,hybridNet

    # Train keypoints from a pretrain instead of resuming latest:
    python scripts/slurm_train.py --project unified_V3_masked --kp-weights None --pretrain EcoSet

    # See the scripts without submitting:
    python scripts/slurm_train.py --project unified_V3_masked --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

# JARVIS-HybridNet lives under this repo's third_party/. Training is launched
# from there (jarvis-local writes models to projects/<project>/models, relative
# to that cwd) and slurm logs go to its OutFiles/.
PROJECT_DIR = Path(__file__).resolve().parent.parent
JARVIS_ROOT = PROJECT_DIR / "third_party" / "JARVIS-HybridNet"

# Partition -> nodelist (from sinfo); auto-selected based on --partition.
GPU_NODELISTS = {
    'gpu-a40':  'g[3040-3047,3050-3057,3060-3067,3070-3077]',
    'gpu-a100': 'g[3080-3087]',
    'gpu-l40':  'g[3090-3099,3115-3119]',
    'gpu-l40s': 'g[3100-3114,3120-3124,3133-3137]',
    'gpu-h200': 'g[3125-3132]',
    'ckpt-g2':  'g[3090-3137]',
}

# Accept stage names case-insensitively and map to the jarvis-local subcommand.
STAGE_ALIASES = {
    'centerdetect':   'centerDetect',
    'center':         'centerDetect',
    'keypointdetect': 'keypointDetect',
    'keypoint':       'keypointDetect',
    'kp':             'keypointDetect',
    'hybridnet':      'hybridNet',
    'hybrid':         'hybridNet',
}


def slurm_submit(script: str) -> str:
    """Submit a job script via stdin and return its job id."""
    try:
        out = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        return out.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)


def build_train_cmd(stage: str, project: str, *, epochs, kp_weights, pretrain,
                    hyb_kp_weights, hyb_weights, hyb_mode) -> str:
    """Construct the `jarvis-local train ...` command for one stage."""
    ep = f" --num_epochs {epochs}" if epochs else ""
    if stage == 'centerDetect':
        w = (f" --pretrained_weights {pretrain}"
             if pretrain and pretrain != 'None' else "")
        return f"jarvis-local train centerDetect{ep}{w} {project}"
    if stage == 'keypointDetect':
        if kp_weights and kp_weights != 'None':
            w = f" --weights_path {kp_weights}"
        elif pretrain and pretrain != 'None':
            w = f" --pretrained_weights {pretrain}"
        else:
            w = ""
        return f"jarvis-local train keypointDetect{ep}{w} {project}"
    if stage == 'hybridNet':
        w = f" --weights_keypoint_detect {hyb_kp_weights}"
        if hyb_weights and hyb_weights != 'None':
            w += f" --weights_hybridnet {hyb_weights}"
        w += f" --mode {hyb_mode}"
        return f"jarvis-local train hybridNet{ep}{w} {project}"
    raise ValueError(f"unknown stage {stage!r}")


def build_script(
    *,
    job_name: str,
    partition: str,
    nodelist_line: str,
    cpus: int,
    mem: int,
    time_limit: str,
    conda_env: str,
    jarvis_root: Path,
    train_cmd: str,
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
#SBATCH -o ./OutFiles/slurm-train-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
{nodelist_line}
#SBATCH --exclude=g[3107,3115,3109]
{requeue_line}
{dependency_line}
set -x
source ~/.bashrc
micromamba activate {conda_env}
unset LD_LIBRARY_PATH
echo "Node: $SLURMD_NODENAME"
nvidia-smi
cd {jarvis_root}
{train_cmd}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True,
                   help='JARVIS project name (e.g. unified_V3_masked)')
    p.add_argument('--stages', default='keypointDetect',
                   help='Comma-separated stages to train, in order. Each gets an '
                        'afterok dependency on the previous one. Options: '
                        'centerDetect, keypointDetect, hybridNet '
                        '(default: keypointDetect)')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override NUM_EPOCHS for every submitted stage '
                        '(default: use the value in the project config)')
    p.add_argument('--kp-weights', default='latest',
                   help="KeypointDetect: weights to load before training. "
                        "'latest' resumes the most recent run, a .pth path loads "
                        "that file, 'None' starts from --pretrain (default: latest)")
    p.add_argument('--pretrain', default='None',
                   help="Pretrained weights for centerDetect/keypointDetect when "
                        "not resuming (e.g. EcoSet, MonkeyHand). Default: None")
    p.add_argument('--hyb-kp-weights', default='latest',
                   help='HybridNet: KeypointDetect weights to load '
                        '(default: latest)')
    p.add_argument('--hyb-weights', default='None',
                   help="HybridNet: HybridNet weights to load. 'None' = random "
                        'init of the 3D part (default: None)')
    p.add_argument('--hyb-mode', default='3D_only',
                   choices=['3D_only', 'last_layers', 'bifpn', 'all'],
                   help='HybridNet: which part of the net to train (default: 3D_only)')
    p.add_argument('--conda-env', default='3d_tracking',
                   help='micromamba/conda environment to activate (default: 3d_tracking)')
    p.add_argument('--partition', default='gpu-l40s',
                   help='SLURM partition to submit jobs to (default: gpu-l40s)')
    p.add_argument('--cpus', type=int, default=16, help='CPUs per job (default: 16)')
    p.add_argument('--mem', type=int, default=64, help='Memory per job in GB (default: 64)')
    p.add_argument('--time', default='1-00:00:00', help='Per-stage job time limit')
    p.add_argument('--no-chain', action='store_true',
                   help='Submit each stage independently (no afterok dependencies). '
                        'Only safe when stages do not depend on each other; HybridNet '
                        'needs KeypointDetect, so do not use this for that chain.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print scripts without submitting')
    p.add_argument('--requeue', dest='requeue', action='store_true', default=None,
                   help='Add #SBATCH --requeue so preempted jobs are requeued. '
                        'Default: on for ckpt* partitions, off otherwise.')
    p.add_argument('--no-requeue', dest='requeue', action='store_false',
                   help='Disable automatic requeue on preemption.')
    args = p.parse_args()

    # Normalise / validate the requested stages.
    stages = []
    for raw in args.stages.split(','):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in STAGE_ALIASES:
            print(f"Error: unknown stage {raw!r}. Choose from "
                  f"centerDetect, keypointDetect, hybridNet.", file=sys.stderr)
            sys.exit(1)
        stages.append(STAGE_ALIASES[key])
    if not stages:
        print("Error: no stages requested.", file=sys.stderr)
        sys.exit(1)

    if not JARVIS_ROOT.is_dir():
        print(f"Error: JARVIS-HybridNet not found at {JARVIS_ROOT}", file=sys.stderr)
        sys.exit(1)

    # Default: requeue on for preemptible ckpt* partitions.
    if args.requeue is None:
        args.requeue = args.partition.startswith('ckpt')
    requeue_line = "#SBATCH --requeue" if args.requeue else ""

    nodelist_line = (f"#SBATCH --nodelist={GPU_NODELISTS[args.partition]}"
                     if args.partition in GPU_NODELISTS else "")
    (JARVIS_ROOT / "OutFiles").mkdir(exist_ok=True)

    print(f"Project: {args.project}")
    print(f"Stages : {' -> '.join(stages)}"
          f"{'' if args.no_chain else '  (afterok-chained)'}")

    prev_jid = None
    for stage in stages:
        train_cmd = build_train_cmd(
            stage, args.project,
            epochs=args.epochs,
            kp_weights=args.kp_weights,
            pretrain=args.pretrain,
            hyb_kp_weights=args.hyb_kp_weights,
            hyb_weights=args.hyb_weights,
            hyb_mode=args.hyb_mode,
        )
        dependency_line = (f"#SBATCH --dependency=afterok:{prev_jid}"
                           if (prev_jid and not args.no_chain) else "")
        job_name = f"train_{stage}_{args.project}"[:60]
        script = build_script(
            job_name=job_name,
            partition=args.partition,
            nodelist_line=nodelist_line,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            conda_env=args.conda_env,
            jarvis_root=JARVIS_ROOT,
            train_cmd=train_cmd,
            requeue_line=requeue_line,
            dependency_line=dependency_line,
        )
        if args.dry_run:
            print(f"\n--- {stage} (dry-run) ---")
            print(script)
            continue
        jid = slurm_submit(script)
        dep = f"  (afterok:{prev_jid})" if (prev_jid and not args.no_chain) else ""
        print(f"Submitted {job_name}: {jid}{dep}")
        prev_jid = jid

    if not args.dry_run:
        print("\nMonitor with: squeue -u $USER")
        print(f"Logs:        {JARVIS_ROOT}/OutFiles/slurm-train-<jobid>.out")


if __name__ == "__main__":
    main()


'''
squeue -u $USER -h -o "%i %j" | awk '/train_/ {print $1}' | xargs -r scancel

# Keypoints only (resume latest), the verified V3 setup:
python scripts/slurm_train.py --project unified_V3_masked

# Keypoints then HybridNet (graph-Laplacian), chained:
python scripts/slurm_train.py --project unified_V3_masked --stages keypointDetect,hybridNet

# Full stack from scratch:
python scripts/slurm_train.py --project unified_V3_masked \
    --stages centerDetect,keypointDetect,hybridNet --kp-weights None --pretrain EcoSet
'''
