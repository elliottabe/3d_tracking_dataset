#!/bin/bash
# Submit a one-off GPU task to ckpt-all instead of running it on the node you
# are sitting on.
#
# The interactive allocation is often ALSO running a training job; a JAX/MuJoCo
# task started there preallocates most of a GPU and competes with it (measured:
# a solver A/B held 22.8 GB while training held 21 GB on a 46 GB card, taking
# GPU 0 to 44.5/46). Anything that does not need the local session should go to
# the queue.
#
# Usage:
#   scripts/slurm/submit_task.sh <name> '<command line>'
#   scripts/slurm/submit_task.sh --time 2:00:00 --mem 32 ab18 'python /tmp/x.py'
#
# Logs land in slurm_logs/<name>-<jobid>.out.
set -euo pipefail
TIME=04:00:00; MEM=48; CPUS=8; GPUS=1
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --time) TIME="$2"; shift 2;;
    --mem)  MEM="$2";  shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    *) echo "unknown flag $1" >&2; exit 2;;
  esac
done
NAME="${1:?usage: submit_task.sh [flags] <name> '<command>'}"
CMD="${2:?usage: submit_task.sh [flags] <name> '<command>'}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO/slurm_logs"
SB=$(mktemp /tmp/submit_task.XXXXXX.sh)
cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=ckpt-all
#SBATCH --account=portia
#SBATCH --constraint=h200|a100|l40s|l40|a40
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --gpus=$GPUS
#SBATCH --mem=${MEM}G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $REPO/slurm_logs/$NAME-%j.out
set -x
source ~/.bashrc
micromamba activate 3d_tracking
module load cuda/12.9.1
export LD_PRELOAD="\$CONDA_PREFIX/lib/libstdc++.so.6"
unset LD_LIBRARY_PATH        # let JAX use its bundled CUDA wheels
unset JAX_PLATFORMS          # never inherit JAX_PLATFORMS=cpu from the submit env
export MUJOCO_GL=egl
# GPU memory grows with demand instead of a fixed preallocated fraction (user
# decision 2026-09-04): a queued task owns its --gpus, so a cap only causes
# avoidable OOMs (a 0.6 fraction OOMed ViTPose training at batch 32 on
# 2026-08-31 while the card had room). Override per job by prefixing the
# command, e.g. 'export XLA_PYTHON_CLIENT_PREALLOCATE=true && python ...'.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_PYTHON_CLIENT_MEM_FRACTION
cd $REPO
$CMD
EOF
JID=$(sbatch --parsable "$SB")
echo "submitted $NAME as $JID -> $REPO/slurm_logs/$NAME-$JID.out"
rm -f "$SB"
