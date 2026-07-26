# NNvMC training scripts — adaptive minimum-variance phase (AMVP) estimator

Training code for the models in the paper. The phase-gradient estimator is set per run:
`std` (standard / score), `dir` (direct), or `couple`/`mix` (the adaptive minimum-variance mixture, AMVP).

## `lattice/` — spin & interacting-fermion lattice models (JAX)
| folder | model | net |
|---|---|---|
| `flux/` | two-leg flux (Meissner) ladder, Φ=0.3π, J_leg=1 J_rung=0.8 J_z=0.5 | separated amplitude/phase |
| `chiral/` | chiral XXX spin chain L=50, α=1, PBC | shared-trunk two-head |
| `torus/` | 8×8 flux torus, Φ=0.25π, J_z=0 | shared-trunk two-head |
| `capacity/` | flux-ladder MLP width/depth ablations | MLP |
| `fermion/` | 100-site interacting spinless-fermion flux ladder, Φ=0.5π | separated |

Each folder has a `run_*.sh` launcher plus its training entry (`train_*.py` / `run_*.py`) and engine.
Optimisation is minSR (Adam variants included). Pass the estimator via the `--mode`/`mode=` argument.

## `fqhe/` — fractional quantum Hall, ν = 1/3
FQHE runs are built on the external **DeepHall** framework (FermiNet-style continuum QH ansatz);
only the run launchers are included here. They give the exact configuration:

```
deephall system.nspins=[N,0] system.flux=2Q seed=S batch_size=1120 optim.iterations=100000 ...
```
with N=6 (2Q=15) and N=8 (2Q=21). The AMVP estimator is enabled through environment variables
`DEEPHALL_MIX_LAMBDA=adaptive` (with `DEEPHALL_LAMBDA_SUBSAMPLE`, `DEEPHALL_GDIR_SUBSAMPLE`).

- `run_N6_fullnet.sh` — N = 6
- `run_N8_fullnet.sh` — N = 8

**Frozen (reported) energy.** After the main optimisation run finishes, continue an **additional
3000 steps with the learning rate set to 0** (parameters θ evolve no further; only the walkers keep
sampling), starting from the final checkpoint. The reported converged energy is the robust mean
(MAD-clipped) over those 3000 lr = 0 steps.

## Dependencies & notes

- The lattice model / Hamiltonian factories (flux ladder, chiral chain, flux torus, fermion ladder)
  and the shared MCMC + minSR engine are imported from the `alt_nnvmc` package, installed separately.
  The Hamiltonian couplings and the Peierls gauge are exactly as listed in the table above and in the
  paper; the JAX drivers here call those factories with those parameters.
- The ResNet capacity control (Fig. 5c,d) replaces the two MLPs with residual convolutional networks
  (stem 2×3 kernels, 24 channels, six residual blocks, spatial sum-pooling head, ≈4.2×10⁴ parameters),
  enabled via `ansatz='resnet'` in the flux/capacity engine — a configuration of the same driver,
  not a separate launcher.
- The capacity depth sweep (`capacity/run_cap_bsweep.sh`) reproduces the reported width-128 depth-3
  standard-estimator point; the depth-2 width-128 point reuses the flux-ladder run.

> Cluster-specific paths / SBATCH headers in the launchers are illustrative — adapt them to your setup.
