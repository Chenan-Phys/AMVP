# Neural quantum state checkpoints (10 seeds per system)

Each run provides two saved network states: `rolling_eval.pkl` (a fixed-parameter evaluation state
taken at a pre-specified checkpoint of the rolling-energy trace; the same rule is applied to all
estimators and seeds) and `converged.pkl` (the final training step). FQHE provides the final
converged checkpoint (`converged.npz`) for the mix (AMVP) estimator. Modes: mix (adaptive
minimum-variance, AMVP), dir (direct), std / std1-5 (standard). Systems: flux, chiral, torus,
capacity (MLP width/depth), fermion (interacting spinless fermions), fqhe (N=6, N=8, nu=1/3).
