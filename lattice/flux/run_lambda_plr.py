import argparse, sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))
import jax
if tuple(int(x) for x in jax.__version__.split('.')[:2]) >= (0, 7):
    jax.config.update('jax_default_matmul_precision', 'highest')
from alt_nnvmc.lattice import flux_ladder
from sr_lambda import train_lambda
p = argparse.ArgumentParser()
p.add_argument('--flux', type=float, required=True)
p.add_argument('--mode', required=True, choices=['std','dir','combined'])
p.add_argument('--seed', type=int, required=True)
p.add_argument('--steps', type=int, default=50000)
p.add_argument('--e0', type=float, default=None)
p.add_argument('--local-frac', dest='local_frac', type=float, default=0.0)
p.add_argument('--phase-lr', dest='phase_lr', type=float, default=1.0)
p.add_argument('--out', required=True)
a = p.parse_args()
model = flux_ladder(length=50, flux=a.flux * np.pi)
train_lambda(model, mode=a.mode, seed=a.seed, steps=a.steps, e0=a.e0, out=a.out,
             L=None, ansatz='mlp', width=128, depth=2, batch=1024, n_sweeps=48,
             lr=0.03, diag_shift=0.02, diag_min=0.005, lr_decay=0.3,
             phase_lr=a.phase_lr, local_frac=a.local_frac, warmup=100)
