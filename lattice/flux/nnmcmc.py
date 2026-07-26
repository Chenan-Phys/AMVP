"""Real neural-network ansatz + MCMC VMC on the lattice spin models (JAX/flax).

Ansatz: log psi(s) = U_alpha(s) + i Phi_beta(s), two separate real MLPs over a
fixed-magnetization spin configuration s in {0,1}^N. Sampling by Metropolis
exchange moves (swap one up and one down spin -> stays in the Sz sector).

Phase channel can be trained with the standard score estimator, the direct
local-energy-derivative estimator g_dir, or a convex mix:
    g_phase(lambda) = (1-lambda) * g_phase_std + lambda * g_phase_dir
Amplitude always uses the standard estimator. This is the real-NN analogue of
the table-ansatz study.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax.flatten_util import ravel_pytree


class MLP(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, s):
        x = 2.0 * s.astype(jnp.float32) - 1.0
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.width)(x))
        return nn.Dense(1)(x)[..., 0]


class CNN(nn.Module):
    """2D conv (SAME/zero pad) feature extractor + flatten + dense head over a rows x cols lattice.
    Keeps conv LOCALITY but NOT forced translation invariance (flatten retains position) -> suits
    open-boundary flux ladders. 2-leg ladder: rows=2, cols=L. Square: rows=cols=L."""
    rows: int
    cols: int
    channels: int
    depth: int

    @nn.compact
    def __call__(self, s):
        x = (2.0 * s.astype(jnp.float32) - 1.0).reshape(s.shape[0], self.rows, self.cols, 1)
        kh = min(2, self.rows)
        for _ in range(self.depth):
            x = jnp.tanh(nn.Conv(self.channels, (kh, 3), padding="SAME")(x))
        x = x.reshape(x.shape[0], -1)  # flatten -> keep per-site/position info (not translation-invariant)
        x = jnp.tanh(nn.Dense(2 * self.channels)(x))
        return nn.Dense(1)(x)[..., 0]


class ResNet(nn.Module):
    """Deep residual CNN (Chen-Heyl, Nat. Phys. 2024 style). Stem conv + `depth` residual
    blocks, each = (LayerNorm -> ReLU -> Conv) x2 with a skip connection, then a
    size-extensive sum-pool head. 2*depth conv layers total -> goes deep without vanishing
    gradients. rows x cols lattice (ladder: rows=2; chain: rows=1)."""
    rows: int
    cols: int
    channels: int
    depth: int  # number of residual blocks (2 conv layers each)

    @nn.compact
    def __call__(self, s):
        x = (2.0 * s.astype(jnp.float32) - 1.0).reshape(s.shape[0], self.rows, self.cols, 1)
        kh = min(2, self.rows)
        x = nn.Conv(self.channels, (kh, 3), padding="SAME")(x)  # stem
        for _ in range(self.depth):
            h = nn.relu(nn.LayerNorm()(x))
            h = nn.Conv(self.channels, (kh, 3), padding="SAME")(h)
            h = nn.relu(nn.LayerNorm()(h))
            h = nn.Conv(self.channels, (kh, 3), padding="SAME")(h)
            x = x + h  # residual skip
        x = nn.relu(nn.LayerNorm()(x))
        x = jnp.sum(x, axis=(1, 2))  # (B, channels) size-extensive (scales to large lattices)
        return nn.Dense(1)(x)[..., 0]


class Transformer(nn.Module):
    """Multi-head self-attention ansatz over the spin sequence (size-extensive
    sum-pool head). Used for both the log-amplitude and the phase network."""
    N: int
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    mlp_dim: int = 64

    @nn.compact
    def __call__(self, s):
        B, N = s.shape
        spin = (2.0 * s.astype(jnp.float32) - 1.0)[..., None]        # (B,N,1)
        # Real-space Fourier coordinate encoding (per-site position -> sin/cos features),
        # following the transformer/NNvMC real-space mapping. Makes the net position-aware
        # (crucial for position-dependent flux phase). Static (not learned).
        coord = jnp.arange(N, dtype=jnp.float32) / N                 # (N,) normalized site coord
        freqs = jnp.array([1., 2., 3., 4., 6., 8.], jnp.float32)
        pe = jnp.concatenate(
            [jnp.sin(2 * jnp.pi * freqs[None, :] * coord[:, None]),
             jnp.cos(2 * jnp.pi * freqs[None, :] * coord[:, None])], axis=-1)  # (N,2F)
        pe = jnp.broadcast_to(pe[None], (B, N, pe.shape[-1]))
        tok = jnp.concatenate([spin, pe], axis=-1)                   # (B,N,1+2F)
        x = nn.Dense(self.d_model)(tok)                              # (B,N,d)
        for _ in range(self.n_layers):
            h = nn.LayerNorm()(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads, qkv_features=self.d_model, deterministic=True
            )(h, h)
            x = x + h
            h = nn.LayerNorm()(x)
            h = nn.Dense(self.mlp_dim)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.d_model)(h)
            x = x + h
        x = jnp.sum(x, axis=1)                                       # (B,d) size-extensive
        return nn.Dense(1)(x)[..., 0]                                # (B,)


def init_walkers(key, B, N, n_up):
    base = jnp.concatenate([jnp.ones((n_up,)), jnp.zeros((N - n_up,))])
    return jax.vmap(lambda k: jax.random.permutation(k, base))(
        jax.random.split(key, B)
    ).astype(jnp.int32)


def make_engine(model, ansatz="mlp", width=64, depth=2, channels=16, L=None,
                d_model=32, n_heads=4, n_layers=2, mlp_dim=64, n_sweeps=16, real_only=False,
                cnn_rows=2, local_frac=0.0):
    edges = [(e.i, e.j, float(e.jxy), float(e.jz), float(e.phase)) for e in model.edges]
    N = model.n_sites
    if ansatz == "cnn":
        rows = cnn_rows
        cols = N // rows
        Umod = CNN(rows, cols, channels, depth)
        Pmod = CNN(rows, cols, channels, depth)
    elif ansatz == "resnet":
        rows = cnn_rows
        cols = N // rows
        Umod = ResNet(rows, cols, channels, depth)
        Pmod = ResNet(rows, cols, channels, depth)
    elif ansatz == "transformer":
        Umod = Transformer(N, d_model, n_heads, n_layers, mlp_dim)
        Pmod = Transformer(N, d_model, n_heads, n_layers, mlp_dim)
    else:
        Umod = MLP(width, depth)
        Pmod = MLP(width, depth)

    def U(a, c):
        return Umod.apply(a, c)

    def P(b, c):
        if real_only:  # force Phi=0 -> real wavefunction (measures phase's energy contribution)
            return jnp.zeros((c.shape[0],), dtype=jnp.float32)
        return Pmod.apply(b, c)

    ei = jnp.array([e[0] for e in edges], dtype=jnp.int32)
    ej = jnp.array([e[1] for e in edges], dtype=jnp.int32)
    ejxy = jnp.array([e[2] for e in edges], dtype=jnp.float32)
    ejz = jnp.array([e[3] for e in edges], dtype=jnp.float32)
    ephase = jnp.array([e[4] for e in edges], dtype=jnp.float32)
    # per-edge Jordan-Wigner string site (-1 = none); ladder fermion hops carry 1 site
    estr = jnp.array([e.jw_string[0] if e.jw_string else -1 for e in model.edges],
                     dtype=jnp.int32)
    Ne = len(edges)

    # chiral term: each triple (i,j,k) -> 3 cyclic (a,b,c); chiral hop swaps a<->b
    _cyc = []
    for _t in model.triples:
        _cyc += [(_t[0], _t[1], _t[2]), (_t[1], _t[2], _t[0]), (_t[2], _t[0], _t[1])]
    Nc = len(_cyc)
    chiral_alpha = float(model.chiral_alpha)
    if Nc:
        ca = jnp.array([h[0] for h in _cyc], dtype=jnp.int32)
        cb = jnp.array([h[1] for h in _cyc], dtype=jnp.int32)
        cc = jnp.array([h[2] for h in _cyc], dtype=jnp.int32)

    def _swap_all(configs):
        # (Ne, B, N): configs with the two sites of each edge exchanged
        def one(i, j):
            ci = configs[:, i]; cj = configs[:, j]
            return configs.at[:, i].set(cj).at[:, j].set(ci)
        return jax.vmap(one)(ei, ej)

    def eloc(a, b, configs):
        B, Nn = configs.shape
        U0 = U(a, configs)
        P0 = P(b, configs)
        spm = 2.0 * configs.astype(jnp.float32) - 1.0
        diag = jnp.sum(ejz[None, :] * 0.25 * spm[:, ei] * spm[:, ej], axis=1)  # (B,)
        sw = _swap_all(configs).reshape(Ne * B, Nn)
        Usw = U(a, sw).reshape(Ne, B)
        Psw = P(b, sw).reshape(Ne, B)
        si = configs[:, ei].T  # (Ne, B)
        sj = configs[:, ej].T
        active = si != sj
        sign = jnp.where((si == 0) & (sj == 1), 1.0, -1.0)
        coeff = 0.5 * ejxy[:, None] * jnp.exp(1j * sign * ephase[:, None])  # (Ne, B)
        jw = jnp.where(estr[:, None] >= 0,
                       1.0 - 2.0 * configs[:, jnp.maximum(estr, 0)].T.astype(jnp.float32),
                       1.0)
        coeff = coeff * jw   # Jordan-Wigner sign (1-2 n_k) for fermionic hops
        ratio = jnp.exp((Usw - U0[None, :]) + 1j * (Psw - P0[None, :]))     # (Ne, B)
        off = jnp.sum(jnp.where(active, coeff * ratio, jnp.zeros((), jnp.complex64)), axis=0)
        if Nc and chiral_alpha != 0.0:
            swc = jax.vmap(
                lambda a_, b_: configs.at[:, a_].set(configs[:, b_]).at[:, b_].set(configs[:, a_])
            )(ca, cb).reshape(Nc * B, Nn)
            Uswc = U(a, swc).reshape(Nc, B)
            Pswc = P(b, swc).reshape(Nc, B)
            sca = configs[:, ca].T; scb = configs[:, cb].T; scc = configs[:, cc].T  # (Nc, B)
            activec = sca != scb
            szc = jnp.where(scc == 1, 1.0, -1.0)    # S^z_c (the 3rd spin)
            signc = jnp.where(sca == 0, 1.0, -1.0)  # +1 if a was down (S+_a S-_b)
            coeffc = chiral_alpha * 0.5j * 0.5 * szc * signc          # (Nc, B) = alpha*(i/2)*(s_c/2)
            ratioc = jnp.exp((Uswc - U0[None, :]) + 1j * (Pswc - P0[None, :]))
            off = off + jnp.sum(jnp.where(activec, coeffc * ratioc, jnp.zeros((), jnp.complex64)), axis=0)
        return diag.astype(jnp.complex64) + off

    def _global_move(a, configs, key):
        B = configs.shape[0]
        ku, kd, ka = jax.random.split(key, 3)
        ui = jax.random.categorical(ku, jnp.where(configs == 1, 0.0, -1e9), axis=1)
        di = jax.random.categorical(kd, jnp.where(configs == 0, 0.0, -1e9), axis=1)
        bidx = jnp.arange(B)
        prop = configs.at[bidx, ui].set(0).at[bidx, di].set(1)
        logr = 2.0 * (U(a, prop) - U(a, configs))
        acc = jnp.log(jax.random.uniform(ka, (B,))) < logr
        configs = jnp.where(acc[:, None], prop, configs)
        return configs, jnp.mean(acc.astype(jnp.float32))

    def _local_move(a, configs, key):
        B = configs.shape[0]
        ke, ka = jax.random.split(key, 2)
        eidx = jax.random.randint(ke, (B,), 0, Ne)
        si = ei[eidx]; sj = ej[eidx]
        bidx = jnp.arange(B)
        ci = configs[bidx, si]; cj = configs[bidx, sj]
        prop = configs.at[bidx, si].set(cj).at[bidx, sj].set(ci)
        logr = 2.0 * (U(a, prop) - U(a, configs))
        acc = (jnp.log(jax.random.uniform(ka, (B,))) < logr) & (ci != cj)
        configs = jnp.where(acc[:, None], prop, configs)
        return configs, jnp.mean(acc.astype(jnp.float32))

    def _mcmc_step(carry, key):
        a, configs = carry
        kc, kstep = jax.random.split(key, 2)
        do_local = jax.random.uniform(kc) < local_frac
        configs, acc = jax.lax.cond(
            do_local,
            lambda: _local_move(a, configs, kstep),
            lambda: _global_move(a, configs, kstep),
        )
        return (a, configs), acc

    def sample(a, configs, key):
        (a, configs), accs = jax.lax.scan(_mcmc_step, (a, configs), jax.random.split(key, n_sweeps))
        return configs, jnp.mean(accs)

    def _clip_eloc(e, C=5.0):
        # clip per-sample local-energy outliers (spikes from configs with Psi~0 in the
        # denominator) to [median +- C*MAD]. Robust -> kills the -1e9 energy blowups.
        med = jax.lax.stop_gradient(jnp.median(e))
        mad = jax.lax.stop_gradient(jnp.median(jnp.abs(e - med)) + 1e-8)
        return jnp.clip(e, med - C * mad, med + C * mad)

    def _amp_phase_std(a, b, configs):
        el = eloc(a, b, configs)
        eR = _clip_eloc(el.real); eI = _clip_eloc(el.imag)   # clip eloc outliers
        Em = jnp.mean(eR)
        cR = jax.lax.stop_gradient(eR - Em)
        cI = jax.lax.stop_gradient(eI - jnp.mean(eI))
        g_amp = jax.grad(lambda aa: 2.0 * jnp.mean(cR * U(aa, configs)))(a)
        g_ps = jax.grad(lambda bb: 2.0 * jnp.mean(cI * P(bb, configs)))(b)
        return g_amp, g_ps, Em, jnp.var(eR)

    def grads_std(a, b, configs):
        # standard estimator only (no derivative-of-E_loc) -> cheaper per step
        g_amp, g_ps, Em, v = _amp_phase_std(a, b, configs)
        return g_amp, g_ps, Em, v

    def grads_mix(a, b, configs, lam):
        # direct phase estimator (g_pd) merged with el via value_and_grad (one eloc backward)
        def loss_el(bb):
            el = eloc(a, bb, configs)
            return jnp.mean(el.real), el   # NO clip inside value_and_grad: the median (sort) blows up
            # the XLA compile of the ResNet eloc-gradient (g_dir hangs in compile). g_pd outliers are
            # handled downstream by the dtheta norm-clip + the energy rollback instead.
        (_, el), g_pd = jax.value_and_grad(loss_el, has_aux=True)(b)
        eR = _clip_eloc(el.real); eI = _clip_eloc(el.imag)   # clip for reported energy + g_ps
        Em = jnp.mean(eR)
        cR = jax.lax.stop_gradient(eR - Em)
        cI = jax.lax.stop_gradient(eI - jnp.mean(eI))
        g_amp = jax.grad(lambda aa: 2.0 * jnp.mean(cR * U(aa, configs)))(a)
        g_ps = jax.grad(lambda bb: 2.0 * jnp.mean(cI * P(bb, configs)))(b)
        g_ph = jax.tree_util.tree_map(lambda s, d: (1.0 - lam) * s + lam * d, g_ps, g_pd)
        return g_amp, g_ph, Em, jnp.var(eR)

    def grad_variance(a, b, configs):
        """Per-sample variance of the PHASE gradient estimators (theory smoking-gun):
        std/score  g_ps,k = 2 (E_I,k - <E_I>) d_beta Phi(R_k)   (Lemma 1: stays > 0)
        direct     g_pd,k = d_beta E_R(R_k)                      (Thm 3/5: diagonal-var annihilated)
        Returns (Var(g_ps), Var(g_pd)) = trace of the batch covariance of each estimator."""
        el = eloc(a, b, configs)
        eI = _clip_eloc(el.imag)          # match actual g_ps: clip eloc.imag
        cI = eI - jnp.mean(eI)
        bf, unrav = ravel_pytree(b)
        Pf = lambda bb, c: P(unrav(bb), c[None])[0]
        OPb = jax.vmap(lambda c: jax.grad(Pf)(bf, c))(configs)           # (K, Nb) d_beta Phi
        g_ps_k = 2.0 * cI[:, None] * OPb                                 # (K, Nb) score per-sample
        eRf = lambda bb, c: eloc(a, unrav(bb), c[None])[0].real
        g_pd_k = jax.vmap(lambda c: jax.grad(eRf)(bf, c))(configs)       # (K, Nb) direct per-sample
        vof = lambda gk: jnp.mean(jnp.sum(gk ** 2, axis=1)) - jnp.sum(jnp.mean(gk, axis=0) ** 2)
        cof = lambda gk, hk: jnp.mean(jnp.sum(gk * hk, axis=1)) - jnp.sum(jnp.mean(gk, axis=0) * jnp.mean(hk, axis=0))
        return vof(g_ps_k), vof(g_pd_k), cof(g_ps_k, g_pd_k)

    return dict(Umod=Umod, Pmod=Pmod, eloc=eloc, sample=sample,
                grads_std=grads_std, grads_mix=grads_mix, grad_variance=grad_variance,
                N=N, n_up=model.n_up)


def train(model, *, lam, seed, steps, batch=512, lr_u=3e-3, lr_phi=3e-3,
          ansatz="mlp", width=64, depth=2, channels=16, L=None,
          d_model=48, n_heads=4, n_layers=3, mlp_dim=96,
          n_sweeps=16, log_every=100, e0=None, grad_clip=None):
    eng = make_engine(model, ansatz=ansatz, width=width, depth=depth,
                      channels=channels, L=L, d_model=d_model, n_heads=n_heads,
                      n_layers=n_layers, mlp_dim=mlp_dim, n_sweeps=n_sweeps)
    N, n_up = eng["N"], eng["n_up"]
    key = jax.random.PRNGKey(seed)
    key, ka, kb, kw = jax.random.split(key, 4)
    a = eng["Umod"].init(ka, jnp.zeros((1, N), jnp.int32))
    b = eng["Pmod"].init(kb, jnp.zeros((1, N), jnp.int32))
    if grad_clip is not None:
        opt_a = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_u))
        opt_b = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr_phi))
    else:
        opt_a = optax.adam(lr_u)
        opt_b = optax.adam(lr_phi)
    osa = opt_a.init(a)
    osb = opt_b.init(b)
    walkers = init_walkers(kw, batch, N, n_up)

    use_dir = (lam != 0.0)  # std runs skip the (expensive) g_pd -> honest per-step cost

    @jax.jit
    def step_fn(a, b, osa, osb, walkers, key):
        walkers, acc = eng["sample"](a, walkers, key)
        if use_dir:
            g_amp, g_ph, Em, var = eng["grads_mix"](a, b, walkers, lam)
        else:
            g_amp, g_ph, Em, var = eng["grads_std"](a, b, walkers)
        ua, osa = opt_a.update(g_amp, osa)
        a = optax.apply_updates(a, ua)
        ub, osb = opt_b.update(g_ph, osb)
        b = optax.apply_updates(b, ub)
        return a, b, osa, osb, walkers, Em, var, acc

    # burn-in
    for _ in range(50):
        key, k = jax.random.split(key)
        walkers, _ = jax.jit(eng["sample"])(a, walkers, k)

    rows = []
    t0 = time.time()
    for t in range(steps + 1):
        key, k = jax.random.split(key)
        a, b, osa, osb, walkers, Em, var, acc = step_fn(a, b, osa, osb, walkers, k)
        if t % log_every == 0 or t == steps:
            E = float(Em)
            rows.append(dict(step=t, energy=E, variance=float(var),
                             acceptance=float(acc), wall_s=time.time() - t0,
                             energy_error=(E - e0) if e0 is not None else float("nan")))
    return rows
