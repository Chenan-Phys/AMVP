#!/bin/bash
#SBATCH --job-name=fullnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-4
#SBATCH --time=48:00:00
#SBATCH --output=/home/bingxing2/home/scx7901/DeepHall_deploy/logs/fullnet%A_%a.out
set -eo pipefail
BASE=/home/bingxing2/home/scx7901/DeepHall_deploy; CUDA=/home/bingxing2/apps/compilers/cuda/cuda-12.2
cd $BASE/DeepHall; source $BASE/venv_py311_clean/bin/activate
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/targets/sbsa-linux/lib:$CUDA/lib64:$BASE/venv_py311_clean/lib/python3.11/site-packages/nvidia/cudnn/lib
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.46
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA --xla_gpu_force_compilation_parallelism=1"
export DEEPHALL_MIX_LAMBDA=adaptive DEEPHALL_LAMBDA_SUBSAMPLE=64 DEEPHALL_GDIR_SUBSAMPLE=280
SEEDS=(60110237 60220471 60330619 60440853 60551097 60661231 60771483 60881627 60991861 61101205)
S1=${SEEDS[$((SLURM_ARRAY_TASK_ID*2))]}; S2=${SEEDS[$((SLURM_ARRAY_TASK_ID*2+1))]}
echo "[fullnet] GPU task $SLURM_ARRAY_TASK_ID -> seeds $S1 $S2 (full net heads_dim=64, g_dir=280, MEM0.46)"
for S in $S1 $S2; do
  deephall system.nspins="[6,0]" system.flux=15 seed=$S batch_size=1120 optim.iterations=100000 \
    log.save_path=$BASE/results_fullnet/N6_seed${S}_mix > $BASE/logs/fullnet${S}.log 2>&1 &
done
wait
