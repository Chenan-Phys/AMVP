#!/usr/bin/env python3
"""Run lambda-mixing optimization for the spinless fermion flux ladder.

This wrapper is intentionally small: it delegates the optimizer to sr_lambda.py
and only constructs the fermion ladder while checking that the Jordan-Wigner
strings required for leg hopping are present.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from alt_nnvmc.lattice import fermion_ladder  # noqa: E402
from sr_lambda import train_lambda  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("std", "dir", "combined"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--nsweeps", type=int, default=48)
    p.add_argument("--local-frac", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--flux", type=float, default=0.5, help="flux in units of pi")
    p.add_argument("--V", type=float, default=2.0)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--diag", type=float, default=0.02)
    p.add_argument("--diagmin", type=float, default=0.005)
    p.add_argument("--lrdecay", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

    model = fermion_ladder(L=args.L, flux=args.flux * np.pi, V=args.V)
    jw_edges = [edge for edge in model.edges if getattr(edge, "jw_string", ())]
    expected_jw = 2 * (args.L - 1)
    print(
        "[fermion-check] "
        f"model={model.name} sites={model.n_sites} n_up={model.n_up} "
        f"edges={len(model.edges)} jw_edges={len(jw_edges)} expected_jw={expected_jw}",
        flush=True,
    )
    print(
        "[fermion-check] sample_jw="
        + repr([(e.i, e.j, round(float(e.phase), 6), tuple(e.jw_string)) for e in jw_edges[:6]]),
        flush=True,
    )
    if len(jw_edges) != expected_jw:
        raise RuntimeError(
            f"fermion_ladder JW check failed: got {len(jw_edges)} JW edges, "
            f"expected {expected_jw}"
        )
    if args.check_only:
        return

    train_lambda(
        model,
        mode=args.mode,
        seed=args.seed,
        steps=args.steps,
        batch=args.batch,
        ansatz="mlp",
        width=args.width,
        depth=args.depth,
        L=None,
        n_sweeps=args.nsweeps,
        lr=args.lr,
        diag_shift=args.diag,
        diag_min=args.diagmin,
        lr_decay=args.lrdecay,
        e0=None,
        local_frac=args.local_frac,
        warmup=args.warmup,
        out=args.out,
    )


if __name__ == "__main__":
    main()
