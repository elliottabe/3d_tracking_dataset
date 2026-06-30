#!/bin/bash
#SBATCH --job-name=cse_v2v250
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#
# C3 V2VNet-250 training on the PREBUILT 250-channel cache, 8 GPUs.
# The 250-ch cache is ~122 GB; to_device_sharded splits axis 0 across 8 devices
# (~15 GB/GPU), fixing the 1-GPU OOM. Cache already exists -> no rebuild.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
CACHE=$WORK/cache_250
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_v2v250

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.cse.train_v2v_cse \
  --cache-dir "$CACHE" --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
  --steps 20000 --batch 32 --laplacian-weight 0.05 --sharpen 3.0
echo "V2V DONE"
