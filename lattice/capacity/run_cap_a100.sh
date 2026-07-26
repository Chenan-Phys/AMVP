#!/bin/bash
#SBATCH --job-name=capsds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/home/bingxing2/home/scx7901/lambda_exp_gateoff_a100/logs/capsds_%A_%a.out
#SBATCH --exclude=paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-25
set -euo pipefail
ROOT=/home/bingxing2/home/scx7901/lambda_exp_gateoff_a100
DEV=/home/bingxing2/apps/anaconda/2021.11/envs/dev251205
PY=/home/bingxing2/home/scx7901/.conda/envs/myenv/bin/python
export PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$(ls -d $DEV/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export JAX_PLATFORMS=cuda
cd "$ROOT"; mkdir -p logs results_capacity_samesds
PERGPU=${PERGPU:-5}
MEMFRAC=$(awk "BEGIN{printf \"%.3f\", 0.90/$PERGPU}")
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/bingxing2/apps/compilers/cuda/cuda-12.8 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer="
E0=-43.3029671617
FLUX=0.3
STEPS=${STEPS:-50000}
OUTDIR=$ROOT/results_capacity_samesds
SEEDS=(26517 34682 37281 48261 53829 57093 71046 82935 84390 85302)
JOBS=()
for cfg in "256:2:w256d2"; do
  IFS=':' read -r W D TAG <<< "$cfg"
  for s in "${SEEDS[@]}"; do
    for m in std dir combined; do
      lab=$m; [ "$m" = "combined" ] && lab=mix
      JOBS+=("$W:$D:$TAG:$s:$m:$lab")
    done
  done
done
NJOB=${#JOBS[@]}
t=$SLURM_ARRAY_TASK_ID
i0=$((t*PERGPU))
echo "[task $t] jobs $i0..$((i0+PERGPU-1)) of $NJOB MEMFRAC=$MEMFRAC host=$(hostname) $(date)"
for ((j=i0; j<i0+PERGPU && j<NJOB; j++)); do
  IFS=':' read -r W D TAG seed mode lab <<< "${JOBS[$j]}"
  out=$OUTDIR/${TAG}_${lab}_s${seed}.pkl
  echo "  [spawn] $TAG $lab w=$W d=$D seed=$seed"
  "$PY" run_cap.py --flux "$FLUX" --mode "$mode" --seed "$seed" --width "$W" --depth "$D" --steps $STEPS --e0 "$E0" --out "$out" &
  sleep 8
done
wait
echo "[task $t] done $(date)"
