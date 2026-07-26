# FQHE (ν = 1/3) training

Built on the external **DeepHall** continuum quantum-Hall framework. These launchers only supply the
run configuration; install DeepHall separately.

- `run_N6_fullnet.sh` — N = 6, 2Q = 15, heads_dim = 64, batch 1120, 100k iterations
- `run_N8_fullnet.sh` — N = 8, 2Q = 21, heads_dim = 64, batch 1120, 200k iterations
  (the production N8 run reached 200k in stages due to walltime; a single 200k run reproduces the setting)

AMVP estimator via env vars: `DEEPHALL_MIX_LAMBDA=adaptive`, `DEEPHALL_LAMBDA_SUBSAMPLE`, `DEEPHALL_GDIR_SUBSAMPLE`.

Both the mix (AMVP) and std (standard) estimators are trained for the figure comparison; the open
state archive ships the mix (AMVP) checkpoints.

**Frozen / reported energy:** continue an additional **3000 steps at lr = 0** from the final
checkpoint; report the MAD-clipped robust mean of those 3000 steps (per-particle E_c/N).
