#!/bin/bash
#SBATCH -p gpu --gres=gpu:1 --cpus-per-task=8 --time=10:00:00
#SBATCH --exclude=paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-11,paraai-n32-h-01-agent-12,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27
#SBATCH -o /home/bingxing2/home/scx7901/torus_flux025/logs/%x_%A.out
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1|cut -d. -f1)
if [ "$DRV" -lt 525 ]; then scontrol requeue $SLURM_JOB_ID; sleep 60; exit 0; fi
H=/home/bingxing2/home/scx7901; DEV=/home/bingxing2/apps/anaconda/2021.11/envs/dev251205
PY=$H/.conda/envs/myenv/bin/python
export PATH=/home/bingxing2/apps/compilers/cuda/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$(ls -d $DEV/lib/python3.11/site-packages/nvidia/*/lib|tr "\n" ":")
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/bingxing2/apps/compilers/cuda/cuda-12.8
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.22}
cd $H/torus_flux025
echo "PACK TASKS=$TASKS STEPS=${STEPS:-50000} DRIFT=[$DRIFT_TOL_DECAY_START_STEP,$DRIFT_TOL_DECAY_END_STEP]"
for t in $TASKS; do
  seed=${t%:*}; mode=${t#*:}
  $PY run_square_torus.py --L 8 --flux 0.7853981633974483 --jz 0.0 --mode $mode --seed $seed \
     --steps ${STEPS:-50000} --width 256 --batch 1024 --out ${RESDIR:-results50k}/${mode}_s${seed}.csv \
     > logs/${mode}_s${seed}.log 2>&1 &
done
wait; echo ALL_DONE
