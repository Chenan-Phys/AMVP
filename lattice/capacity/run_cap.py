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
p.add_argument('--width', type=int, default=128)
p.add_argument('--depth', type=int, default=2)
p.add_argument('--batch', type=int, default=1024)
p.add_argument('--diag-shift', dest='diag_shift', type=float, default=0.02)
p.add_argument('--diag-min', dest='diag_min', type=float, default=0.005)
p.add_argument('--lr', type=float, default=0.03)
p.add_argument('--phase-lr', dest='phase_lr', type=float, default=1.0)
p.add_argument('--out', required=True)
a = p.parse_args()
model = flux_ladder(length=50, flux=a.flux * np.pi)
train_lambda(model, mode=a.mode, seed=a.seed, steps=a.steps, e0=a.e0, out=a.out,
             L=None, ansatz='mlp', width=a.width, depth=a.depth, batch=a.batch, n_sweeps=48,
             lr=a.lr, diag_shift=a.diag_shift, diag_min=a.diag_min, lr_decay=0.3,
             phase_lr=a.phase_lr, local_frac=0.0, warmup=100)
