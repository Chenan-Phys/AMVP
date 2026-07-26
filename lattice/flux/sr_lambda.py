"""Combined (control-variate) phase-gradient estimator with adaptive rollback.

  g_phase(lam) = (1-lam) g_std + lam g_dir       [grads_mix]
  closed-form minimum-variance  lam_hat = (Var_std - Cov)/(Var_std + Var_dir - 2 Cov).
Modes: std(lam=0)  dir(lam=1)  combined(lam=lam_hat).

The training loop guards against basin escape with an adaptive rollback using 4
conditions (nonfinite / energy_jump / slow_collapse / drift_from_best with a
30-step rolling mean + decaying tolerance), reason-dependent rollback
targets, and lr_scale = max(0.03, x0.5) on each rollback.
"""
from __future__ import annotations
import math, os, pickle, time
from collections import deque
import jax, jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
from alt_nnvmc.nnmcmc import make_engine, init_walkers


def train_lambda(model, *, mode, seed, steps, batch=1024, ansatz="mlp", width=128, depth=2,
                 L=None, n_sweeps=48, lr=0.03, diag_shift=0.02, diag_min=0.005, lr_decay=0.3,
                 phase_lr=1.0, log_every=250, cov_every=250, e0=None, local_frac=0.0,
                 warmup=100, out=None):
    eng = make_engine(model, ansatz=ansatz, width=width, depth=depth, L=L,
                      n_sweeps=n_sweeps, local_frac=local_frac)
    Umod, Pmod = eng["Umod"], eng["Pmod"]
    N, n_up = eng["N"], eng["n_up"]
    key = jax.random.PRNGKey(seed)
    key, ka, kb, kw = jax.random.split(key, 4)
    a = Umod.init(ka, jnp.zeros((1, N), jnp.int32))
    b = Pmod.init(kb, jnp.zeros((1, N), jnp.int32))
    _, unravel_a = ravel_pytree(a); _, unravel_b = ravel_pytree(b)
    Npa = ravel_pytree(a)[0].size; Npb = ravel_pytree(b)[0].size
    walkers = init_walkers(kw, batch, N, n_up)
    sample_jit = jax.jit(eng["sample"])

    def logderivs(a, b, configs):
        af = ravel_pytree(a)[0]; bf = ravel_pytree(b)[0]

        Us = lambda af1, c1: Umod.apply(unravel_a(af1), c1[None])[0]
        Ps = lambda bf1, c1: Pmod.apply(unravel_b(bf1), c1[None])[0]
        OUa = jax.vmap(lambda c1: jax.grad(Us)(af, c1))(configs)
        OPb = jax.vmap(lambda c1: jax.grad(Ps)(bf, c1))(configs)
        return OUa - OUa.mean(0), OPb - OPb.mean(0)

    @jax.jit
    def sr_step(a, b, walkers, key, diag, lr_t, lam):
        walkers, acc = eng["sample"](a, walkers, key)
        g_amp, g_ph, Em, var = eng["grads_mix"](a, b, walkers, lam)   # combined: (1-lam)std + lam dir
        F = jnp.concatenate([ravel_pytree(g_amp)[0], ravel_pytree(g_ph)[0]])
        OUa, OPb = logderivs(a, b, walkers)
        B = walkers.shape[0]
        ReO = jnp.concatenate([OUa, jnp.zeros((B, Npb))], axis=1)
        ImO = jnp.concatenate([jnp.zeros((B, Npa)), OPb], axis=1)
        A = jnp.concatenate([ReO, ImO], axis=0)
        M = A / jnp.sqrt(B)
        G = M @ M.T + diag * jnp.eye(2 * B, dtype=M.dtype)
        alpha = jnp.linalg.solve(G, M @ F)
        dtheta = (F - M.T @ alpha) / diag
        dtheta = jnp.nan_to_num(dtheta, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = jnp.linalg.norm(dtheta)
        dtheta = jnp.where(nrm > 5.0, dtheta * (5.0 / (nrm + 1e-12)), dtheta)
        a = unravel_a(ravel_pytree(a)[0] - lr_t * dtheta[:Npa])
        b = unravel_b(ravel_pytree(b)[0] - phase_lr * lr_t * dtheta[Npa:])
        return a, b, walkers, Em, var, acc

    gcov = jax.jit(eng["grad_variance"])   # -> (Var_std, Var_dir, Cov)

    for _ in range(max(0, int(warmup))):
        key, k = jax.random.split(key)
        walkers, _ = sample_jit(a, walkers, k)
    print(f"  [warmup] sweeps={int(warmup)}", flush=True)

    t0 = time.time()
    log_d0, log_dmin = jnp.log(diag_shift), jnp.log(diag_min)
    log_l0, log_lmin = jnp.log(lr), jnp.log(lr * lr_decay)
    E_hist = deque(maxlen=15)
    lr_scale = 1.0
    n_rollback = 0
    n_slow_bad = 0
    STATE_BUF = 5
    state_buf = deque(maxlen=STATE_BUF)
    ROLL_STEP_WINDOW = 30
    DRIFT_TOL_START_FRAC = 0.080
    DRIFT_TOL_END_FRAC = 0.0025
    DRIFT_TOL_START_ABS = 3.00
    DRIFT_TOL_END_ABS = 0.08
    DRIFT_TOL_POWER = 1.0
    DRIFT_TOL_DECAY_START_STEP = 30000
    DRIFT_TOL_DECAY_END_STEP = 45000
    roll_E_hist = deque(maxlen=ROLL_STEP_WINDOW)
    best_E = float("inf"); best_state = None
    best_roll_E = float("inf"); best_roll_inst_E = float("nan")
    best_roll_mad = float("nan"); best_roll_state = None; best_roll_step = -1

    # ---- estimator state ----
    lam_fixed = {"std": 0.0, "dir": 1.0}.get(mode, None)   # None => combined
    lam_hat = 0.0 if mode == "std" else 1.0
    vps = vpd = vcov = float("nan"); lam_opt = lam_hat
    rows = []

    checkpoint_dir = None
    if out:
        checkpoint_dir = os.path.splitext(os.path.abspath(out))[0] + "_states"
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_checkpoint(name, step, a_state, b_state, walkers_state, extra=None):
        if checkpoint_dir is None:
            return
        payload = dict(
            step=int(step), mode=mode, seed=int(seed), steps=int(steps),
            amp_params=jax.device_get(a_state), phase_params=jax.device_get(b_state),
            walkers=jax.device_get(walkers_state), key=jax.device_get(key),
            lr_scale=float(lr_scale), n_rollback=int(n_rollback),
            E_hist=list(E_hist), roll_E_hist=list(roll_E_hist),
            best_E=float(best_E), best_rolling_E=float(best_roll_E),
            best_rolling_step=int(best_roll_step), e0=e0,
        )
        if extra:
            payload.update(extra)
        path = os.path.join(checkpoint_dir, f"{name}.pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh)
        os.replace(tmp, path)

    for t in range(0, steps + 1):
        key, k = jax.random.split(key)
        frac = t / max(steps, 1)
        diag = jnp.exp(log_d0 + (log_dmin - log_d0) * frac)
        lr_t = jnp.exp(log_l0 + (log_lmin - log_l0) * frac) * lr_scale
        is_log_step = (t % log_every == 0 or t == steps)

        # --- variance/cov + closed-form lam_hat (refresh at log steps) ---
        if (t % cov_every == 0) or (t == steps) or is_log_step:
            vps, vpd, vcov = [float(x) for x in gcov(a, b, walkers)]
            denom = vps + vpd - 2.0 * vcov
            lam_opt = float(np.clip((vps - vcov) / denom if abs(denom) > 1e-12 else 0.0,
                                    0.0, 1.0))
            if mode == "combined":
                lam_hat = lam_opt
        lam_t = lam_fixed if lam_fixed is not None else lam_hat

        prev_a, prev_b, prev_walkers = a, b, walkers
        prop_a, prop_b, prop_walkers, Em, var, acc = sr_step(
            a, b, walkers, k, diag, lr_t, jnp.float32(lam_t))
        E = float(Em)
        measured_state = (prev_a, prev_b, prop_walkers)
        roll_E = float("nan"); drift_tol = float("nan")
        rollback_to_E = float("nan"); rollback_target = None

        # ====================== ROLLBACK ======================
        bad = False
        bad_reason = ""

        # 1. 数值炸了
        if not math.isfinite(E):
            bad = True
            bad_reason = "nonfinite_E"

        # 2. 突然大跳（单步）
        if not bad and len(E_hist) >= 10:
            arr = np.asarray(E_hist)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-9
            if abs(E - med) > max(12.0 * mad, 0.05 * abs(med)):
                bad = True
                bad_reason = "energy_jump"

        # 3. 慢慢变差：连续 3 次才触发
        if not bad and len(E_hist) >= 10:
            arr = np.asarray(E_hist)
            med = float(np.median(arr))
            if med < 0 and E > med + 0.20 * abs(med):
                n_slow_bad += 1
            else:
                n_slow_bad = 0
            if n_slow_bad >= 3:
                bad = True
                bad_reason = "slow_collapse"
                n_slow_bad = 0
        else:
            n_slow_bad = 0

        # 4. 30-SR-step rolling mean drift: catches slow basin escape that never
        # spikes. This is independent of log_every; log rows only display it.
        roll_improved = False
        if not bad:
            trial_roll = list(roll_E_hist) + [E]
            if len(trial_roll) >= ROLL_STEP_WINDOW:
                arr = np.asarray(trial_roll[-ROLL_STEP_WINDOW:], dtype=float)
                roll_E = float(np.mean(arr))
                roll_med = float(np.median(arr))
                roll_mad = float(np.median(np.abs(arr - roll_med))) + 1e-9
                if (best_roll_state is not None and math.isfinite(best_roll_E)
                        and len(trial_roll) >= ROLL_STEP_WINDOW
                        and t >= DRIFT_TOL_DECAY_START_STEP):
                    ref_mad = best_roll_mad if math.isfinite(best_roll_mad) else roll_mad
                    # Be permissive during early basin exploration, then tighten
                    # the guard so late-stage phase drift still rolls back.
                    decay_span = max(1, DRIFT_TOL_DECAY_END_STEP - DRIFT_TOL_DECAY_START_STEP)
                    progress = min(1.0, max(0.0, (t - DRIFT_TOL_DECAY_START_STEP)
                                             / decay_span))
                    decay = (1.0 - progress) ** DRIFT_TOL_POWER
                    sched_frac = DRIFT_TOL_END_FRAC + (
                        DRIFT_TOL_START_FRAC - DRIFT_TOL_END_FRAC) * decay
                    sched_abs = DRIFT_TOL_END_ABS + (
                        DRIFT_TOL_START_ABS - DRIFT_TOL_END_ABS) * decay
                    drift_tol = max(4.0 * ref_mad, sched_frac * abs(best_roll_E),
                                    sched_abs)
                    if roll_E > best_roll_E + drift_tol:
                        bad = True
                        bad_reason = "drift_from_best"
                        rollback_target = best_roll_state
                        rollback_to_E = best_roll_E
                if not bad and roll_E < best_roll_E:
                    roll_improved = True

        if bad:
            if rollback_target is None:
                if bad_reason == "slow_collapse" and len(state_buf) >= STATE_BUF:
                    rollback_target = state_buf[0][:3]
                    rollback_to_E = float(state_buf[0][3])
                elif len(state_buf) > 0:
                    rollback_target = state_buf[-1][:3]
                    rollback_to_E = float(state_buf[-1][3])
                else:
                    rollback_target = (prev_a, prev_b, prev_walkers)
            a, b, walkers = rollback_target
            lr_scale = max(0.03, lr_scale * 0.5)
            n_rollback += 1
            # clear history so next steps don't immediately re-trigger
            E_hist.clear()
            roll_E_hist.clear()
            n_slow_bad = 0
            state_buf.clear()
        else:
            a, b, walkers = prop_a, prop_b, prop_walkers
            state_buf.append((*measured_state, E))
            E_hist.append(E)
            roll_E_hist.append(E)
            lr_scale = min(1.0, lr_scale + 2e-3)
            if E < best_E:
                best_E = E; best_state = measured_state
            if roll_improved:
                best_roll_E = roll_E
                best_roll_inst_E = E
                best_roll_mad = roll_mad
                best_roll_state = measured_state
                best_roll_step = t
                ba, bb, bw = best_roll_state
                _save_checkpoint("rolling_eval", t, ba, bb, bw,
                                 extra=dict(kind="rolling_eval", energy=float(best_roll_E),
                                            inst_energy=float(best_roll_inst_E)))
        # ==================== END ROLLBACK ====================

        if is_log_step or bad:
            lu = float(lam_t)
            vcomb = (1 - lu) ** 2 * vps + lu ** 2 * vpd + 2 * lu * (1 - lu) * vcov
            row = dict(step=t, energy=E, variance=float(var), acceptance=float(acc),
                       wall_s=time.time() - t0, n_rollback=n_rollback,
                       energy_error=(E - e0) if e0 is not None else float("nan"),
                       rollback=bool(bad), bad_reason=bad_reason,
                       lr_scale=float(lr_scale), rolling_E=roll_E,
                       best_rolling_E=best_roll_E, best_rolling_step=best_roll_step,
                       gvar_std=vps, gvar_dir=vpd, gcov=vcov,
                       lam_opt=lam_opt, lam_used=lu, gvar_comb=vcomb)
            rows.append(row)
            if bad:
                print(f"  step {t} [{mode}]: E={E:.4f} ROLLBACK={bad_reason} "
                      f"to={rollback_to_E:.4f} best_roll={best_roll_E:.4f} "
                      f"lr_scale={lr_scale:.3f}", flush=True)
            elif is_log_step:
                print(f"  step {t} [{mode}]: E={E:.4f} err={row['energy_error']:.4f} "
                      f"lam_opt={lam_opt:+.3f} Vstd={vps:.3e} Vdir={vpd:.3e} "
                      f"Vcomb={vcomb:.3e} nrb={n_rollback}", flush=True)

        if checkpoint_dir is not None and (t % 300 == 0 or t == steps):
            _save_checkpoint(f"step_{t:06d}", t, a, b, walkers,
                             extra=dict(kind="periodic", energy=float(E),
                                        lam_opt=float(lam_opt), lam_used=float(lam_t)))

    # ---- reported state: 30-step rolling-energy minimum ----
    eval_E = best_roll_E if best_roll_state is not None else best_E
    eval_kind = "best_rolling" if best_roll_state is not None else "best_energy"
    eval_step = best_roll_step if best_roll_state is not None else -1
    if not math.isfinite(eval_E):
        eval_E = rows[-1]["energy"] if rows else float("nan")
        eval_kind = "latest"; eval_step = steps
    print(f"  [final] {eval_kind} E={float(eval_E):.6f} "
          f"step={int(eval_step)}", flush=True)
    if rows:
        rows.append(dict(step=steps, energy=float(eval_E), final_row=True,
                         eval_kind=eval_kind, eval_step=int(eval_step),
                         n_rollback=n_rollback,
                         energy_error=(float(eval_E) - e0) if e0 is not None else float("nan")))
    if out:
        with open(out, "wb") as fh:
            pickle.dump(rows, fh)
    return rows
