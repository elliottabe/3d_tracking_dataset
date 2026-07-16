#!/bin/bash
#SBATCH --job-name=cse_v2v350
#SBATCH --partition=ckpt-g2
#SBATCH --account=portia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=224G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodelist=g[3090-3137]
#SBATCH --exclude=g[3107,3115,3109]
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=eabe@uw.edu
#
# Wings v1 / stage 2: train V2VNet-350 on the prebuilt 350-channel cache, 8 GPUs.
# train_v2v_cse infers J=350 from volumes.shape[1].  The 350-ch cache is ~170 GB;
# to_device_sharded splits axis 0 across 8 devices (~21 GB/GPU), as for the 250 run.
# V2VNet channels scale with J, so this trains from scratch (no warm-start path
# from the 250 head).  ckpt-dir + restore_latest -> auto-resume on preemption.
set -x
source ~/.bashrc
micromamba activate 3d_tracking
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L

PKG=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
WORK=/gscratch/portia/eabe/data/Johnson_lab/cse_work
CACHE=$WORK/cache_350
RUN=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/cse_v2v350

cd "$PKG"
python -u -c "import jax; print('jax devices:', jax.device_count())"
python -u -m jarvis_jax.densepose.train_v2v_cse \
  --cache-dir "$CACHE" --out "$RUN/final" --ckpt-dir "$RUN/ckpt" \
  --steps 20000 --batch 32 --laplacian-weight 0.05 --sharpen 3.0
echo "V2V350 DONE"
