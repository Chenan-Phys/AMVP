#!/bin/bash
#SBATCH --job-name=f05V2s75
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=36:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/bingxing2/home/scx7901/lambda_exp_a100/logs/fermion_f05V2_a100_100k_s75_%A_%a.out

set -eo pipefail

DEV=/home/bingxing2/apps/anaconda/2021.11/envs/dev251205
PY=/home/bingxing2/home/scx7901/.conda/envs/myenv/bin/python
ROOT=/home/bingxing2/home/scx7901/lambda_exp_a100
OUTDIR="$ROOT/results_fermion_f05V2_10seed_100k_start75k_a100"

export PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$(ls -d $DEV/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/bingxing2/apps/compilers/cuda/cuda-12.8 --xla_gpu_force_compilation_parallelism=1"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
export DRIFT_TOL_DECAY_START_STEP=75000
export DRIFT_TOL_DECAY_END_STEP=100000

cd "$ROOT"
mkdir -p logs "$OUTDIR"

MODES=(std dir combined)
SEED_SET=(20317 21129 22483 23671 24196 25733 26841 27409 28652 29731)

idx=$SLURM_ARRAY_TASK_ID
mode_idx=$((idx / 5))
chunk_idx=$((idx % 5))

MODE=${MODES[$mode_idx]}
seed_i=$((chunk_idx * 2))
SEEDS=("${SEED_SET[$seed_i]}" "${SEED_SET[$((seed_i + 1))]}")

echo "[fermion f05 V2 A100] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID mode=$MODE seeds=${SEEDS[*]}"
echo "[fermion f05 V2 A100] settings L=50 sites=100 half-filled flux=0.5pi V=2.0 w128 d2 batch=1024 nsweeps=48 local_frac=0.0 lr=0.03 diag=0.02 diagmin=0.005 steps=100000 rollback_drift_start=75000 decay_to=100000 concurrency=${#SEEDS[@]}"

pids=()
for seed in "${SEEDS[@]}"; do
  log="$ROOT/logs/a100_fermion_f05V2_${MODE}_seed${seed}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_100k_s75.log"
  out="$OUTDIR/fermion_f05V2_${MODE}_seed${seed}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pkl"
  "$PY" "$ROOT/run_fermion_lambda.py" \
    --mode "$MODE" \
    --seed "$seed" \
    --steps 100000 \
    --L 50 \
    --flux 0.5 \
    --V 2.0 \
    --width 128 \
    --depth 2 \
    --batch 1024 \
    --nsweeps 48 \
    --local-frac 0.0 \
    --lr 0.03 \
    --diag 0.02 \
    --diagmin 0.005 \
    --lrdecay 0.3 \
    --warmup 100 \
    --out "$out" >"$log" 2>&1 &
  pids+=("$!")
  sleep 8
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
