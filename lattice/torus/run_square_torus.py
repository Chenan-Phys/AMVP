"""Driver: square_flux TORUS (both PBC), shared-trunk single net, mix vs std."""
from __future__ import annotations
import argparse
from math import pi
from alt_nnvmc.lattice import square_flux_torus
from train_single import train_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--flux", type=float, default=0.25 * pi, help="plaquette flux (rad)")
    ap.add_argument("--jz", type=float, default=0.0)
    ap.add_argument("--mode", choices=["std", "dir", "combined"], default="std")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--diag_min", type=float, default=0.005)
    ap.add_argument("--e0", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--continue-from", type=str, default=None)
    args = ap.parse_args()

    model = square_flux_torus(L=args.L, flux=args.flux, j1=1.0, jz=args.jz)
    print(f"[run_torus] model={model.name} n_sites={model.n_sites} n_up={model.n_up} "
          f"flux={args.flux:.6f}({args.flux/pi:g}pi) jz={args.jz} mode={args.mode} "
          f"seed={args.seed} steps={args.steps} width={args.width} batch={args.batch}", flush=True)

    train_single(model, mode=args.mode, seed=args.seed, steps=args.steps,
                 batch=args.batch, width=args.width, depth=2, n_sweeps=48,
                 lr=args.lr, diag_shift=0.02, diag_min=args.diag_min, lr_decay=0.3,
                 warmup=100, e0=args.e0, out=args.out, continue_from=args.continue_from)


if __name__ == "__main__":
    main()
