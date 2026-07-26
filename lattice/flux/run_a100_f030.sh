#!/bin/bash
# SLURM launcher for the 0.3pi flux-ladder runs: sweeps seeds x methods
# (std / dir / combined) via run_lambda_plr.py. The job array is split into
# chunks so several runs share one GPU, with the XLA memory fraction scaled
# per chunk. E0 = -43.3030.
#SBATCH --job-name=ladA030
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/bingxing2/home/scx7901/lambda_exp_gateoff_a100/logs/a030_%A_%a.out
set -euo pipefail

ROOT=/home/bingxing2/home/scx7901/lambda_exp_gateoff_a100
DEV=/home/bingxing2/apps/anaconda/2021.11/envs/dev251205
PY=/home/bingxing2/home/scx7901/.conda/envs/myenv/bin/python
export PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$(ls -d $DEV/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')
export JAX_PLATFORMS=cuda
OUTDIR=$ROOT/results
cd "$ROOT"; mkdir -p logs results

STAGGER=${STAGGER:-8}
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/bingxing2/apps/compilers/cuda/cuda-12.8 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer="
E0=-43.3029671617
FLUX=0.3

SEEDS=(26517 34682 37281 48261 53829 57093 71046 82935 84390 85302)
METHODS=("std:1:std1" "std:2:std2" "std:3:std3" "std:4:std4" "std:5:std5" "dir:1:dir" "combined:1:mix")
JOBS=()
for m in "${METHODS[@]}"; do
  IFS=':' read -r mode plr label <<< "$m"
  for s in "${SEEDS[@]}"; do JOBS+=("$mode:$plr:$label:$s"); done
done
NJOB=${#JOBS[@]}                                     # 70

# packing: 3 tasks of 6 + 2 tasks of 5 = 28
CHUNKS=(6 6 6 5 5)
t=$SLURM_ARRAY_TASK_ID
i0=0; for ((k=0;k<t;k++)); do i0=$((i0+${CHUNKS[$k]})); done
CNT=${CHUNKS[$t]}
MEMFRAC=$(awk "BEGIN{printf \"%.3f\", 0.90/$CNT}")
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC

if [ "$i0" -ge "$NJOB" ]; then echo "[task $t] no work"; exit 0; fi
echo "[task $t] CNT=$CNT MEMFRAC=$MEMFRAC f030 jobs $i0..$((i0+CNT-1)) of $NJOB host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

pids=()
for ((j=i0; j<i0+CNT && j<NJOB; j++)); do
  IFS=':' read -r mode plr label seed <<< "${JOBS[$j]}"
  out=$OUTDIR/f030_${label}_s${seed}.pkl
  log=$ROOT/logs/f030_${label}_s${seed}.log
  echo "  [spawn] $label mode=$mode plr=$plr seed=$seed"
  "$PY" run_lambda_plr.py --flux "$FLUX" --mode "$mode" --phase-lr "$plr" --seed "$seed" \
       --steps 50000 --e0 "$E0" --local-frac 0.0 --out "$out" > "$log" 2>&1 &
  pids+=("$!")
  sleep "$STAGGER"
done
st=0
for p in "${pids[@]}"; do wait "$p" || st=1; done
echo "[task $t] done st=$st $(date)"
exit "$st"
