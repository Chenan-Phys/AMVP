#!/bin/bash
#SBATCH --job-name=chiCONT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --exclude=paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-12,paraai-n32-h-01-agent-25
#SBATCH --output=/home/bingxing2/home/scx7901/chiral_single/logs/chi_%A_%a.out
# Chiral XXX L=50, alpha=1, shared-trunk two-head w512 d2: one 50000-step training run per (seed,mode),
# learning rate linearly annealed 0.03 -> 0.003 over steps [30000,45000]. Modes: std/dir/couple(=mix).
set -eo pipefail
H=/home/bingxing2/home/scx7901; DEV=/home/bingxing2/apps/anaconda/2021.11/envs/dev251205
PY=$H/.conda/envs/myenv/bin/python
export PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$(ls -d $DEV/lib/python3.11/site-packages/nvidia/*/lib|tr "\n" ":")
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bingxing2/apps/compilers/cuda/cuda-12.8
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export DRIFT_TOL_DECAY_START_STEP=30000 DRIFT_TOL_DECAY_END_STEP=45000
ROOT=$H/chiral_single; cd "$ROOT"; mkdir -p results_w512
S=(8067695 7846816 8708686 8576532 9717296 8640888 9316904 5616437 5195522 8657445)
TASKS=(); for s in "${S[@]}"; do for m in std dir couple; do TASKS+=("$m $s"); done; done   # 30 (seeds x std/dir/couple)
NPACK=30
for ((i=SLURM_ARRAY_TASK_ID; i<${#TASKS[@]}; i+=NPACK)); do
  read mode s <<< "${TASKS[$i]}"
  $PY run_chiral_single.py --L 50 --alpha 1.0 --mode $mode --seed $s \
    --steps 50000 --width 512 --depth 2 --e0 -23.748041 \
    --lr_end 0.003 --lr_lin_start 30000 --lr_lin_end 45000 \
    --out "$ROOT/results_w512/chiCS512_L50_a1.0_${mode}_s${s}.csv" > "$ROOT/logs/chi_${mode}_s${s}.log" 2>&1 &
done
wait; echo "chiCONT pack $SLURM_ARRAY_TASK_ID done"
