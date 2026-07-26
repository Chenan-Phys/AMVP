"""Driver: chiral XXX chain (alpha), coupled shared-trunk single net, 4-mode compare.

ONE shared-trunk network (w512 d2, U/Phi heads on one theta) trained with SR on the
chiral chain (chiral_chain + the chiral triple term in engine_chiral_single._eloc_core).

Modes: std / dir / nocouple (lam_naive, A_s=A_d=0) / couple (lam_opt, A_s,A_d included).
Default hyperparameters: batch 1024, n_sweeps 64, lr 0.03, diag_shift 0.02,
diag_min 0.005, lr_decay 1.0; 300-window best-mean rollback, 50000 steps.
"""
from __future__ import annotations
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
from alt_nnvmc.lattice import chiral_chain
from train_chiral_single import train_chiral_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--mode", required=True,
                    choices=["std", "dir", "nocouple", "couple", "combined"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--e0", type=float, default=None)
    ap.add_argument("--lr_decay", type=float, default=1.0)   # <1 anneals lr: 0.03 -> 0.03*lr_decay
    ap.add_argument("--diag_min", type=float, default=0.005)
    ap.add_argument("--lr_end", type=float, default=None)    # if set: LINEAR lr 0.03->lr_end over [lr_lin_start,lr_lin_end], then const
    ap.add_argument("--lr_lin_start", type=int, default=30000)
    ap.add_argument("--lr_lin_end", type=int, default=45000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--continue-from", type=str, default=None)
    args = ap.parse_args()

    model = chiral_chain(n_sites=args.L, J=1.0, Delta=1.0, alpha=args.alpha)
    print(f"[chiral-single] {model.name} L={args.L} a={args.alpha} mode={args.mode} "
          f"seed={args.seed} steps={args.steps} w={args.width} d={args.depth} "
          f"n_up={model.n_up} n_triples={len(model.triples)} e0={args.e0}", flush=True)

    train_chiral_single(model, mode=args.mode, seed=args.seed, steps=args.steps,
                        batch=1024, width=args.width, depth=args.depth, n_sweeps=64,
                        lr=0.03, diag_shift=0.02, diag_min=args.diag_min, lr_decay=args.lr_decay,
                        log_every=200, cov_every=200, warmup=100,
                        e0=args.e0, out=args.out, continue_from=args.continue_from,
                        lr_end=args.lr_end, lr_lin_start=args.lr_lin_start, lr_lin_end=args.lr_lin_end)


if __name__ == "__main__":
    main()
