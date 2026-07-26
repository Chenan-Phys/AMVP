"""Single-theta SR + rollback trainer for the shared-network flux-torus study.

ONE parameter tree `theta`:
  - logderivs_single -> centered (OU, OP), both over the FULL theta (no block pad).
  - SR: F = ravel(g_amp + g_phase_mix);  A = concat([OU, OP], axis=0) -> (2B, Ntheta);
        M = A/sqrt(B); G = M M^T + diag I_{2B}; alpha = solve(G, M F);
        dtheta = (F - M^T alpha)/diag; nan_to_num; norm-clip at 5; theta -= lr_t dtheta.
        SINGLE update (no phase_lr split).
Includes lr/diag annealing, closed-form lam_opt at cov steps, rollback, the final
rolling-energy-minimum checkpointing and CSV logging.
"""
from __future__ import annotations
import csv as _csv
import os as _os
_AMPCV = float(_os.environ.get("AMP_CV", "0") or 0)
_GCLIP = float(_os.environ.get("AMP_CV_GAMMA_CLIP", "10") or 0)
import math, os, pickle, time
from collections import deque
import jax, jax.numpy as jnp, numpy as np
from jax.flatten_util import ravel_pytree
from engine_single import make_engine, init_walkers


def _write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def train_single(model, *, mode, seed, steps, batch=1024, width=128, depth=2,
                 n_sweeps=48, lr=0.03, diag_shift=0.02, diag_min=0.005, lr_decay=0.3,
                 log_every=250, cov_every=250, e0=None, local_frac=0.0,
                 warmup=100, out=None, continue_from=None):
    eng = make_engine(model, width=width, depth=depth, n_sweeps=n_sweeps, local_frac=local_frac)
    net = eng["net"]
    N, n_up = eng["N"], eng["n_up"]
    U = eng["U"]; Phi = eng["Phi"]
    key = jax.random.PRNGKey(seed)
    key, kt, kw = jax.random.split(key, 3)
    theta = net.init(kt, jnp.zeros((1, N), jnp.int32))
    _, unravel = ravel_pytree(theta)
    Ntheta = ravel_pytree(theta)[0].size
    walkers = init_walkers(kw, batch, N, n_up)
    sample_jit = jax.jit(eng["sample"])

    def logderivs_single(theta, configs):
        thf = ravel_pytree(theta)[0]
        Us = lambda tf1, c1: U(unravel(tf1), c1[None])[0]
        Ps = lambda tf1, c1: Phi(unravel(tf1), c1[None])[0]
        OU = jax.vmap(lambda c1: jax.grad(Us)(thf, c1))(configs)
        OP = jax.vmap(lambda c1: jax.grad(Ps)(thf, c1))(configs)
        return OU - OU.mean(0), OP - OP.mean(0)

    @jax.jit
    def sr_step(theta, walkers, key, diag, lr_t, lam, gamma):
        key, rkey = jax.random.split(key)
        walkers, acc = eng["sample"](theta, walkers, key)
        g_amp, g_ph, Em, var = eng["grads_mix"](theta, walkers, lam, gamma, rkey)
        # amp and phase share theta -> ADD their full-theta gradients
        g_tot = jax.tree_util.tree_map(lambda ga, gp: ga + gp, g_amp, g_ph)
        F = ravel_pytree(g_tot)[0]
        OU, OP = logderivs_single(theta, walkers)
        B = walkers.shape[0]
        A = jnp.concatenate([OU, OP], axis=0)        # (2B, Ntheta), both over full theta
        M = A / jnp.sqrt(B)
        G = M @ M.T + diag * jnp.eye(2 * B, dtype=M.dtype)
        alpha = jnp.linalg.solve(G, M @ F)
        dtheta = (F - M.T @ alpha) / diag
        dtheta = jnp.nan_to_num(dtheta, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = jnp.linalg.norm(dtheta)
        dtheta = jnp.where(nrm > 5.0, dtheta * (5.0 / (nrm + 1e-12)), dtheta)
        theta = unravel(ravel_pytree(theta)[0] - lr_t * dtheta)
        return theta, walkers, Em, var, acc

    gcov = jax.jit(eng["grad_variance"])   # -> (Var_std, Var_dir, Cov)

    # --- continue from checkpoint: load payload up front to skip warmup ---
    _ckpt = None
    if continue_from:
        with open(continue_from, "rb") as _fh:
            _ckpt = pickle.load(_fh)
        print(f"  [continue] loading {continue_from} (saved step={int(_ckpt['step'])})",
              flush=True)

    if _ckpt is None:
        for _ in range(max(0, int(warmup))):
            key, k = jax.random.split(key)
            walkers, _ = sample_jit(theta, walkers, k)
        print(f"  [warmup] sweeps={int(warmup)}", flush=True)
    else:
        print(f"  [continue] skipping warmup (continuing prior trajectory)", flush=True)

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
    DRIFT_TOL_START_FRAC = 0.040
    DRIFT_TOL_END_FRAC = 0.0025
    DRIFT_TOL_START_ABS = 1.50
    DRIFT_TOL_END_ABS = 0.08
    DRIFT_TOL_POWER = 1.0
    DRIFT_TOL_DECAY_START_STEP = int(os.environ.get("DRIFT_TOL_DECAY_START_STEP", "10000"))
    DRIFT_TOL_DECAY_END_STEP = int(os.environ.get("DRIFT_TOL_DECAY_END_STEP", str(steps)))
    roll_E_hist = deque(maxlen=ROLL_STEP_WINDOW)
    best_E = float("inf"); best_state = None
    best_roll_E = float("inf"); best_roll_inst_E = float("nan")
    best_roll_mad = float("nan"); best_roll_state = None; best_roll_step = -1

    # ---- estimator state ----
    lam_fixed = {"std": 0.0, "dir": 1.0}.get(mode, None)   # None => combined
    lam_hat = 0.0 if mode == "std" else 1.0
    vps = vpd = vcov = float("nan"); lam_opt = lam_hat
    va = vda = cad = cpd = cdd = float("nan"); gamma_opt = gamma_hat = 0.0
    rho2 = rho_perp2 = vfull_std = vfull_cv = float("nan")
    a_s = a_d = lam_naive = float("nan")
    rows = []

    # ---- continue: restore theta + walkers + key + loop bookkeeping ----
    start_step = 0
    if _ckpt is not None:
        theta = jax.tree_util.tree_map(jnp.asarray, _ckpt["theta"])
        walkers = jnp.asarray(_ckpt["walkers"])
        key = jnp.asarray(_ckpt["key"])
        start_step = int(_ckpt["step"])
        lr_scale = float(_ckpt.get("lr_scale", lr_scale))
        n_rollback = int(_ckpt.get("n_rollback", 0))
        E_hist = deque(_ckpt.get("E_hist", []), maxlen=15)
        roll_E_hist = deque(_ckpt.get("roll_E_hist", []), maxlen=ROLL_STEP_WINDOW)
        best_E = float(_ckpt.get("best_E", best_E))
        best_roll_E = float(_ckpt.get("best_rolling_E", best_roll_E))
        best_roll_step = int(_ckpt.get("best_rolling_step", best_roll_step))
        # best_roll_state / best_roll_mad are NOT in the periodic step_*.pkl, but the
        # sibling rolling_eval.pkl carries the rolling-energy theta+walkers+mad.  Prefer it.
        _bw_path = os.path.join(os.path.dirname(os.path.abspath(continue_from)),
                                "rolling_eval.pkl")
        if os.path.exists(_bw_path):
            try:
                with open(_bw_path, "rb") as _bfh:
                    _bw = pickle.load(_bfh)
                _bw_theta = jax.tree_util.tree_map(jnp.asarray, _bw["theta"])
                _bw_walkers = jnp.asarray(_bw["walkers"])
                best_roll_state = (_bw_theta, _bw_walkers)
                if math.isfinite(float(_bw.get("best_rolling_E", float("nan")))):
                    best_roll_E = float(_bw["best_rolling_E"])
                if "best_rolling_step" in _bw:
                    best_roll_step = int(_bw["best_rolling_step"])
                if math.isfinite(float(_bw.get("inst_energy", float("nan")))):
                    best_roll_inst_E = float(_bw["inst_energy"])
                print(f"  [continue] restored rolling_eval from {_bw_path} "
                      f"(best_roll_E={best_roll_E:.6f} step={best_roll_step})",
                      flush=True)
            except Exception as _e:
                print(f"  [continue] WARN could not load rolling_eval.pkl: {_e}",
                      flush=True)
        if best_roll_state is None and math.isfinite(best_roll_E):
            # fall back: seed history with current theta so the reported state still
            # considers the recovered best_roll_E.
            best_roll_state = (theta, walkers)
            print(f"  [continue] rolling_eval.pkl absent; seeding best_roll_state=theta "
                  f"(best_roll_E={best_roll_E:.6f})", flush=True)
        print(f"  [continue] start_step={start_step} lr_scale={lr_scale} "
              f"n_rollback={n_rollback} best_E={best_E:.6f} "
              f"E_hist={len(E_hist)} roll_E_hist={len(roll_E_hist)}", flush=True)

    checkpoint_dir = None
    if out:
        checkpoint_dir = os.path.splitext(os.path.abspath(out))[0] + "_states"
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_checkpoint(name, step, theta_state, walkers_state, extra=None):
        if checkpoint_dir is None:
            return
        payload = dict(
            step=int(step), mode=mode, seed=int(seed), steps=int(steps),
            theta=jax.device_get(theta_state),
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

    for t in range(start_step, steps + 1):
        key, k = jax.random.split(key)
        frac = t / max(steps, 1)
        diag = jnp.exp(log_d0 + (log_dmin - log_d0) * frac)
        lr_t = jnp.exp(log_l0 + (log_lmin - log_l0) * frac) * lr_scale
        is_log_step = (t % log_every == 0 or t == steps)

        # --- variance/cov + closed-form lam_hat (refresh at log steps) ---
        if (t % cov_every == 0) or (t == steps) or is_log_step:
            (vps, vpd, vcov, a_s, a_d, va, vda, cad, cpd, cdd) = [float(x) for x in gcov(theta, walkers)]
            denom = vps + vpd - 2.0 * vcov
            # naive (A_s=A_d) kept only for the diagnostic comparison panel
            lam_naive = float(np.clip((vps - vcov) / denom if abs(denom) > 1e-12 else 0.0,
                                      0.0, 1.0))
            # COUPLED lam* (Eq. 51, lam=1->dir convention): (Vs - C + A_s - A_d)/denom
            lam_opt = float(np.clip((vps - vcov + a_s - a_d) / denom if abs(denom) > 1e-12 else 0.0,
                                    0.0, 1.0))
            if mode == "combined":
                lam_hat = lam_opt
            # joint-optimal amplitude-CV weight over g_amp + (1-lam)g_ps + lam g_pd + gamma D_a
            if _AMPCV > 0 and abs(vda) > 1e-12:
                _lg = lam_opt if mode == "combined" else lam_hat
                gamma_opt = float(np.clip(
                    -(cad + (1.0 - _lg) * cpd + _lg * cdd) / (vda + 1e-12), -_GCLIP, _GCLIP))
                gamma_hat = gamma_opt
                # diagnostics: raw rho^2 and phase-orthogonalized rho_perp^2
                rho2 = (cad * cad) / (va * vda + 1e-12)
                _det = vps * vpd - vcov * vcov + 1e-12
                _al = (cpd * vpd - cdd * vcov) / _det
                _be = (cdd * vps - cpd * vcov) / _det
                _vdap = vda - _al * cpd - _be * cdd
                _cadp = cad - _al * a_s - _be * a_d
                rho_perp2 = (_cadp * _cadp) / (va * _vdap + 1e-12)
                vfull_std = va + vps + 2.0 * a_s
                _om = 1.0 - _lg
                vfull_cv = (va + _om*_om*vps + _lg*_lg*vpd + 2*_om*_lg*vcov
                            + 2*_om*a_s + 2*_lg*a_d
                            + gamma_hat*gamma_hat*vda + 2*gamma_hat*cad
                            + 2*gamma_hat*(_om*cpd + _lg*cdd))
        lam_t = lam_fixed if lam_fixed is not None else lam_hat
        gamma_t = gamma_hat if _AMPCV > 0 else 0.0

        prev_theta, prev_walkers = theta, walkers
        prop_theta, prop_walkers, Em, var, acc = sr_step(
            theta, walkers, k, diag, lr_t, jnp.float32(lam_t), jnp.float32(gamma_t))
        E = float(Em)
        measured_state = (prev_theta, prop_walkers)
        roll_E = float("nan"); drift_tol = float("nan")
        rollback_to_E = float("nan"); rollback_target = None

        # ====================== ROLLBACK ======================
        bad = False
        bad_reason = ""

        # 1. nonfinite
        if not math.isfinite(E):
            bad = True
            bad_reason = "nonfinite_E"

        # 2. single-step jump
        if not bad and len(E_hist) >= 10:
            arr = np.asarray(E_hist)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) + 1e-9
            if abs(E - med) > max(12.0 * mad, 0.05 * abs(med)):
                bad = True
                bad_reason = "energy_jump"

        # 3. slow collapse: 3 in a row
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

        # 4. 30-step rolling-mean drift from best
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
                    rollback_target = state_buf[0][:2]
                    rollback_to_E = float(state_buf[0][2])
                elif len(state_buf) > 0:
                    rollback_target = state_buf[-1][:2]
                    rollback_to_E = float(state_buf[-1][2])
                else:
                    rollback_target = (prev_theta, prev_walkers)
            theta, walkers = rollback_target
            lr_scale = max(0.03, lr_scale * 0.5)
            n_rollback += 1
            E_hist.clear()
            roll_E_hist.clear()
            n_slow_bad = 0
            state_buf.clear()
        else:
            theta, walkers = prop_theta, prop_walkers
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
                bt, bw = best_roll_state
                _save_checkpoint("rolling_eval", t, bt, bw,
                                 extra=dict(kind="rolling_eval", energy=float(best_roll_E),
                                            inst_energy=float(best_roll_inst_E)))
        # ==================== END ROLLBACK ====================

        if is_log_step or bad:
            lu = float(lam_t)
            vcomb = (1 - lu) ** 2 * vps + lu ** 2 * vpd + 2 * lu * (1 - lu) * vcov
            row = dict(step=t, energy=E, variance=float(var), acceptance=float(acc),
                       wall_s=time.time() - t0, n_rollback=n_rollback,
                       energy_error=(100.0 * abs(E - e0) / abs(e0)) if e0 else float("nan"),
                       rollback=bool(bad), bad_reason=bad_reason,
                       lr_scale=float(lr_scale), rolling_E=roll_E,
                       best_rolling_E=best_roll_E, best_rolling_step=best_roll_step,
                       gvar_std=vps, gvar_dir=vpd, gcov=vcov,
                       A_s=a_s, A_d=a_d, lam_opt=lam_opt, lam_naive=lam_naive,
                       lam_used=lu, gvar_comb=vcomb,
                       ampcv_gamma=gamma_hat, ampcv_Va=va, ampcv_Vda=vda,
                       ampcv_Cad=cad, ampcv_rho2=rho2, ampcv_rho_perp2=rho_perp2,
                       ampcv_Vfull_std=vfull_std, ampcv_Vfull_cv=vfull_cv)
            rows.append(row)
            if bad:
                print(f"  step {t} [{mode}]: E={E:.4f} ROLLBACK={bad_reason} "
                      f"to={rollback_to_E:.4f} best_roll={best_roll_E:.4f} "
                      f"lr_scale={lr_scale:.3f}", flush=True)
            elif is_log_step:
                print(f"  step {t} [{mode}]: E={E:.4f} err={row['energy_error']:.4f} "
                      f"lam_opt={lam_opt:+.3f} Vstd={vps:.3e} Vdir={vpd:.3e} "
                      f"Vcomb={vcomb:.3e} nrb={n_rollback}", flush=True)

        if checkpoint_dir is not None and (t % 1000 == 0 or t == steps):
            _save_checkpoint(f"step_{t:06d}", t, theta, walkers,
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
                         energy_error=(100.0 * abs(float(eval_E) - e0) / abs(e0)) if e0 else float("nan")))
    if out:
        # save both a pickle (full rows) and a CSV (flat)
        with open(os.path.splitext(out)[0] + ".pkl", "wb") as fh:
            pickle.dump(rows, fh)
        _write_csv(out if out.endswith(".csv") else out + ".csv", rows)
    return rows
