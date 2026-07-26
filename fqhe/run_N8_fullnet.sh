#!/bin/bash
#SBATCH --job-name=dhN8mstd10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --array=0-14
#SBATCH --exclude=paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-25
#SBATCH --output=/home/bingxing2/home/scx7901/DeepHall_deploy/logs/dhN8mstd10_%A_%a.out

set -eo pipefail
# Reported N8 (nu=1/3): heads_dim=64, batch 1120, gdir subsample 280, 200000 iterations.
# The production run reached 200k in stages (100k then continued) due to walltime; a single
# 200k run with these settings reproduces the same configuration. mix = AMVP, std = standard.

BASE=/home/bingxing2/home/scx7901/DeepHall_deploy
CUDA=/home/bingxing2/apps/compilers/cuda/cuda-12.2
cd "$BASE/DeepHall"
source "$BASE/venv_py311_clean/bin/activate"

export PATH="$CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA/targets/sbsa-linux/lib:$CUDA/lib64:$BASE/venv_py311_clean/lib/python3.11/site-packages/nvidia/cudnn/lib"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA --xla_gpu_force_compilation_parallelism=1"
unset DEEPHALL_PRINT_LAMSTAR

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 525 ]; then
  echo "bad driver $DRV -> requeue"
  scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true
  sleep 30
  exit 0
fi

OUT="$BASE/results_nu13_N8_10seed_heads64_b1120_gdir280"
mkdir -p "$OUT" "$BASE/logs"

SEEDS=(20866935 24768934 25010759 31732867 61731900 62068300 85416708 94380549 98674887 99179825)
ARGS_BASE=(
  "system.nspins=[8,0]"
  "system.flux=21"
  "network.psiformer.heads_dim=64"
  "batch_size=1120"
  "optim.iterations=200000"
)

run_mix() {
  local seed="$1"
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
  echo "[dhN8mstd10] MIX seed=$seed N=8 nu=1/3 flux=21 heads64 batch1120 iters200k gdir_subsample=280 host=$(hostname)"
  DEEPHALL_MIX_LAMBDA=adaptive DEEPHALL_LAMBDA_SUBSAMPLE=64 DEEPHALL_GDIR_SUBSAMPLE=280 \
    deephall "${ARGS_BASE[@]}" "seed=$seed" "log.save_path=$OUT/N8_seed${seed}_mix" \
    > "$BASE/logs/dhN8mstd10_N8_seed${seed}_mix_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
}

run_std() {
  local seed="$1"
  env -u DEEPHALL_MIX_LAMBDA -u DEEPHALL_LAMBDA_SUBSAMPLE -u DEEPHALL_GDIR_SUBSAMPLE \
    deephall "${ARGS_BASE[@]}" "seed=$seed" "log.save_path=$OUT/N8_seed${seed}_std" \
    > "$BASE/logs/dhN8mstd10_N8_seed${seed}_std_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
}

idx="$SLURM_ARRAY_TASK_ID"
if [ "$idx" -lt 10 ]; then
  run_mix "${SEEDS[$idx]}"
else
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
  pair=$((idx - 10))
  s1="${SEEDS[$((2 * pair))]}"
  s2="${SEEDS[$((2 * pair + 1))]}"
  echo "[dhN8mstd10] STD paired seeds=$s1,$s2 N=8 nu=1/3 flux=21 heads64 batch1120 iters200k host=$(hostname)"
  run_std "$s1" &
  p1=$!
  sleep 8
  run_std "$s2" &
  p2=$!
  status=0
  wait "$p1" || status=1
  wait "$p2" || status=1
  exit "$status"
fi
