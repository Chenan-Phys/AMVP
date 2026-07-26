"""Single shared-parameter neural-network VMC engine for the chiral XXX chain.

ONE flax module `SharedMLP` = a shared tanh trunk (depth=2, width=128) feeding TWO
separate linear heads -> (U, Phi) per config, where
    log psi(s) = U(theta, s) + i Phi(theta, s).
Both heads share ONE parameter pytree `theta`: the amplitude/phase split lives in
the heads, not in two disjoint networks.

Per-channel gradients are taken w.r.t. the single theta and ADDED (not
concatenated), because amplitude and phase share theta:
    g(lam) = g_amp + (1-lam) g_pstd + lam g_pdir.
"""
from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.flatten_util import ravel_pytree


class SharedMLP(nn.Module):
    """Shared tanh trunk -> two linear heads. Returns (U, Phi) per config.

    U   = log-amplitude head value (B,)
    Phi = phase head value         (B,)
    One parameter pytree `theta` covers trunk + both heads.
    """
    width: int = 128
    depth: int = 2

    @nn.compact
    def __call__(self, s):
        x = 2.0 * s.astype(jnp.float32) - 1.0
        for _ in range(self.depth):
            x = jnp.tanh(nn.Dense(self.width)(x))
        U = nn.Dense(1, name="head_U")(x)[..., 0]
        Phi = nn.Dense(1, name="head_Phi")(x)[..., 0]
        return U, Phi


def init_walkers(key, B, N, n_up):
    base = jnp.concatenate([jnp.ones((n_up,)), jnp.zeros((N - n_up,))])
    return jax.vmap(lambda k: jax.random.permutation(k, base))(
        jax.random.split(key, B)
    ).astype(jnp.int32)


def make_engine(model, width=128, depth=2, n_sweeps=48, local_frac=0.0):
    edges = [(e.i, e.j, float(e.jxy), float(e.jz), float(e.phase)) for e in model.edges]
    N = model.n_sites
    net = SharedMLP(width, depth)
    _use_ampcv = float(os.environ.get("AMP_CV", "0") or 0) > 0   # gate the amplitude control variate

    def U(theta, c):
        return net.apply(theta, c)[0]

    def Phi(theta, c):
        return net.apply(theta, c)[1]

    def UPhi(theta, c):
        return net.apply(theta, c)

    ei = jnp.array([e[0] for e in edges], dtype=jnp.int32)
    ej = jnp.array([e[1] for e in edges], dtype=jnp.int32)
    ejxy = jnp.array([e[2] for e in edges], dtype=jnp.float32)
    ejz = jnp.array([e[3] for e in edges], dtype=jnp.float32)
    ephase = jnp.array([e[4] for e in edges], dtype=jnp.float32)
    # per-edge Jordan-Wigner string site (-1 = none); ladder fermion hops carry 1 site
    estr = jnp.array([e.jw_string[0] if e.jw_string else -1 for e in model.edges],
                     dtype=jnp.int32)
    Ne = len(edges)

    # ---- chiral term: each triple (i,j,k) -> 3 cyclic (a,b,c); chiral hop swaps a<->b.
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

    def _eloc_core(theta, configs, detach_U, detach_Phi=False):
        """Local energy. If detach_U, every U(...) is wrapped in stop_gradient (U0,
        Usw all frozen) so only the Phi path carries gradient -> isolates g_pdir."""
        B, Nn = configs.shape
        U0, P0 = UPhi(theta, configs)
        if detach_U:
            U0 = jax.lax.stop_gradient(U0)
        if detach_Phi:
            P0 = jax.lax.stop_gradient(P0)
        spm = 2.0 * configs.astype(jnp.float32) - 1.0
        diag = jnp.sum(ejz[None, :] * 0.25 * spm[:, ei] * spm[:, ej], axis=1)  # (B,)
        sw = _swap_all(configs).reshape(Ne * B, Nn)
        Usw, Psw = UPhi(theta, sw)
        Usw = Usw.reshape(Ne, B); Psw = Psw.reshape(Ne, B)
        if detach_U:
            Usw = jax.lax.stop_gradient(Usw)
        if detach_Phi:
            Psw = jax.lax.stop_gradient(Psw)
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
        # ---- chiral triple term (shared-trunk UPhi + detach) ----
        if Nc and chiral_alpha != 0.0:
            swc = jax.vmap(
                lambda a_, b_: configs.at[:, a_].set(configs[:, b_]).at[:, b_].set(configs[:, a_])
            )(ca, cb).reshape(Nc * B, Nn)
            Uswc, Pswc = UPhi(theta, swc)
            Uswc = Uswc.reshape(Nc, B); Pswc = Pswc.reshape(Nc, B)
            if detach_U:
                Uswc = jax.lax.stop_gradient(Uswc)
            if detach_Phi:
                Pswc = jax.lax.stop_gradient(Pswc)
            sca = configs[:, ca].T; scb = configs[:, cb].T; scc = configs[:, cc].T  # (Nc, B)
            activec = sca != scb
            szc = jnp.where(scc == 1, 1.0, -1.0)    # S^z_c (the 3rd spin)
            signc = jnp.where(sca == 0, 1.0, -1.0)  # +1 if a was down (S+_a S-_b)
            coeffc = chiral_alpha * 0.5j * 0.5 * szc * signc          # (Nc, B) = alpha*(i/2)*(s_c/2)
            ratioc = jnp.exp((Uswc - U0[None, :]) + 1j * (Pswc - P0[None, :]))
            off = off + jnp.sum(jnp.where(activec, coeffc * ratioc, jnp.zeros((), jnp.complex64)), axis=0)
        return diag.astype(jnp.complex64) + off

    def eloc(theta, configs):
        return _eloc_core(theta, configs, detach_U=False)

    def eloc_Udetached(theta, configs):
        return _eloc_core(theta, configs, detach_U=True)

    def eloc_Phidetached(theta, configs):
        return _eloc_core(theta, configs, detach_U=False, detach_Phi=True)

    def _global_move(theta, configs, key):
        B = configs.shape[0]
        ku, kd, ka = jax.random.split(key, 3)
        ui = jax.random.categorical(ku, jnp.where(configs == 1, 0.0, -1e9), axis=1)
        di = jax.random.categorical(kd, jnp.where(configs == 0, 0.0, -1e9), axis=1)
        bidx = jnp.arange(B)
        prop = configs.at[bidx, ui].set(0).at[bidx, di].set(1)
        logr = 2.0 * (U(theta, prop) - U(theta, configs))
        acc = jnp.log(jax.random.uniform(ka, (B,))) < logr
        configs = jnp.where(acc[:, None], prop, configs)
        return configs, jnp.mean(acc.astype(jnp.float32))

    def _local_move(theta, configs, key):
        B = configs.shape[0]
        ke, ka = jax.random.split(key, 2)
        eidx = jax.random.randint(ke, (B,), 0, Ne)
        si = ei[eidx]; sj = ej[eidx]
        bidx = jnp.arange(B)
        ci = configs[bidx, si]; cj = configs[bidx, sj]
        prop = configs.at[bidx, si].set(cj).at[bidx, sj].set(ci)
        logr = 2.0 * (U(theta, prop) - U(theta, configs))
        acc = (jnp.log(jax.random.uniform(ka, (B,))) < logr) & (ci != cj)
        configs = jnp.where(acc[:, None], prop, configs)
        return configs, jnp.mean(acc.astype(jnp.float32))

    def _mcmc_step(carry, key):
        theta, configs = carry
        kc, kstep = jax.random.split(key, 2)
        do_local = jax.random.uniform(kc) < local_frac
        configs, acc = jax.lax.cond(
            do_local,
            lambda: _local_move(theta, configs, kstep),
            lambda: _global_move(theta, configs, kstep),
        )
        return (theta, configs), acc

    def sample(theta, configs, key):
        (theta, configs), accs = jax.lax.scan(
            _mcmc_step, (theta, configs), jax.random.split(key, n_sweeps))
        return configs, jnp.mean(accs)

    def _clip_eloc(e, C=5.0):
        # clip per-sample local-energy outliers to [median +- C*MAD]. Robust.
        med = jax.lax.stop_gradient(jnp.median(e))
        mad = jax.lax.stop_gradient(jnp.median(jnp.abs(e - med)) + 1e-8)
        return jnp.clip(e, med - C * mad, med + C * mad)

    # ----- per-channel gradients w.r.t. the single theta (ADDED, not concatenated) -----

    def grads_std(theta, configs):
        """Amplitude + standard phase score, both w.r.t. theta."""
        el = eloc(theta, configs)
        eR = _clip_eloc(el.real); eI = _clip_eloc(el.imag)
        Em = jnp.mean(eR)
        cR = jax.lax.stop_gradient(eR - Em)
        cI = jax.lax.stop_gradient(eI - jnp.mean(eI))
        g_amp = jax.grad(lambda th: 2.0 * jnp.mean(cR * U(th, configs)))(theta)
        g_ps = jax.grad(lambda th: 2.0 * jnp.mean(cI * Phi(th, configs)))(theta)
        return g_amp, g_ps, Em, jnp.var(eR)

    def grads_mix(theta, configs, lam, gamma=0.0, rkey=None):
        """g_amp + (1-lam) g_pstd + lam g_pdir, all full-theta pytrees, ADDED.

        g_pdir = grad_theta mean(Re eloc_Udetached): U frozen, only Phi path live
        -> the measure-free direct phase estimator.
        """
        def loss_el(th):
            el = eloc_Udetached(th, configs)
            return jnp.mean(el.real), el   # NO clip inside value_and_grad (median sort blows up compile)
        (_, el_det), g_pd = jax.value_and_grad(loss_el, has_aux=True)(theta)
        # reported energy + g_ps come from the NON-detached eloc (same configs)
        el = eloc(theta, configs)
        eR = _clip_eloc(el.real); eI = _clip_eloc(el.imag)
        Em = jnp.mean(eR)
        cR = jax.lax.stop_gradient(eR - Em)
        cI = jax.lax.stop_gradient(eI - jnp.mean(eI))
        g_amp = jax.grad(lambda th: 2.0 * jnp.mean(cR * U(th, configs)))(theta)
        g_ps = jax.grad(lambda th: 2.0 * jnp.mean(cI * Phi(th, configs)))(theta)
        g_ph = jax.tree_util.tree_map(lambda s, d: (1.0 - lam) * s + lam * d, g_ps, g_pd)
        # amplitude control variate D_a = grad_theta Re(mean eloc_Phidetached), <D_a>=0.
        # Gated on AMP_CV>0 (default off): when off, skip entirely (gamma is 0, so
        # g_amp += gamma*D_a is a no-op) and avoid a redundant phase backprop per step.
        if _use_ampcv:
            def _loss_amp(th):
                return jnp.mean(eloc_Phidetached(th, configs).real)
            D_a = jax.grad(_loss_amp)(theta)
            if os.environ.get('AMP_CV_RAND') and rkey is not None:
                leaves, tdef = jax.tree_util.tree_flatten(D_a)
                keys = jax.random.split(rkey, len(leaves))
                def _randlike(d, k):
                    nrm = jnp.sqrt(jnp.sum(d*d) + 1e-30)
                    z = jax.random.normal(k, d.shape, d.dtype)
                    return z * (nrm / jnp.sqrt(jnp.sum(z*z) + 1e-30))
                D_a = jax.tree_util.tree_unflatten(tdef, [_randlike(d, k) for d, k in zip(leaves, keys)])
            g_amp = jax.tree_util.tree_map(lambda ga, da: ga + gamma * da, g_amp, D_a)
        return g_amp, g_ph, Em, jnp.var(eR)

    def grad_variance(theta, configs):
        """Per-sample trace variance of the PHASE estimators (theory smoking-gun):
        std/score  g_ps,k = 2 (E_I,k - <E_I>) d_theta Phi(R_k)
        direct     g_pd,k = d_theta Re eloc_Udetached(R_k)
        Returns (Var(g_ps), Var(g_pd), Cov(g_ps,g_pd), A_s, A_d) over the full theta
        vector, where A_s=tr Cov[g_amp,g_ps] and A_d=tr Cov[g_amp,g_pd] are the
        amplitude<->phase cross-covariances of Eq. (51) (coupled lambda*).
        """
        el = eloc(theta, configs)
        eI = _clip_eloc(el.imag)          # match actual g_ps
        cI = eI - jnp.mean(eI)
        eR = _clip_eloc(el.real)          # match actual g_amp
        cR = eR - jnp.mean(eR)
        thf, unrav = ravel_pytree(theta)
        Phif = lambda tf, c: Phi(unrav(tf), c[None])[0]
        OPt = jax.vmap(lambda c: jax.grad(Phif)(thf, c))(configs)        # (K, Ntheta) d_theta Phi
        g_ps_k = 2.0 * cI[:, None] * OPt                                 # (K, Ntheta) score per-sample
        eRf = lambda tf, c: eloc_Udetached(unrav(tf), c[None])[0].real
        g_pd_k = jax.vmap(lambda c: jax.grad(eRf)(thf, c))(configs)      # (K, Ntheta) direct per-sample
        Uf = lambda tf, c: U(unrav(tf), c[None])[0]
        OUt = jax.vmap(lambda c: jax.grad(Uf)(thf, c))(configs)          # (K, Ntheta) d_theta U
        g_amp_k = 2.0 * cR[:, None] * OUt                                # (K, Ntheta) amplitude per-sample
        vof = lambda gk: jnp.mean(jnp.sum(gk ** 2, axis=1)) - jnp.sum(jnp.mean(gk, axis=0) ** 2)
        cof = lambda gk, hk: jnp.mean(jnp.sum(gk * hk, axis=1)) - jnp.sum(jnp.mean(gk, axis=0) * jnp.mean(hk, axis=0))
        eRf_amp = lambda tf, c: eloc_Phidetached(unrav(tf), c[None])[0].real
        g_da_k = jax.vmap(lambda c: jax.grad(eRf_amp)(thf, c))(configs)   # (K, Ntheta) amp direct D_a
        return (vof(g_ps_k), vof(g_pd_k), cof(g_ps_k, g_pd_k),
                cof(g_amp_k, g_ps_k), cof(g_amp_k, g_pd_k),
                vof(g_amp_k), vof(g_da_k), cof(g_amp_k, g_da_k),
                cof(g_ps_k, g_da_k), cof(g_pd_k, g_da_k))   # +Va,Vda,Cad,Cpd,Cdd

    def chirality_profile(theta, configs):
        """Per-ORIGINAL-triple scalar chirality chi_i = <S_i.(S_{i+1} x S_{i+2})>.
        Exactly the chiral operator used in _eloc_core WITHOUT the alpha factor,
        grouped back into the len(model.triples) triangles (3 cyclic terms each).
        Returns real (n_triples,) averaged over the config batch."""
        if not Nc:
            return jnp.zeros((0,))
        B, Nn = configs.shape
        U0, P0 = UPhi(theta, configs)
        swc = jax.vmap(
            lambda a_, b_: configs.at[:, a_].set(configs[:, b_]).at[:, b_].set(configs[:, a_])
        )(ca, cb).reshape(Nc * B, Nn)
        Uswc, Pswc = UPhi(theta, swc)
        Uswc = Uswc.reshape(Nc, B); Pswc = Pswc.reshape(Nc, B)
        sca = configs[:, ca].T; scb = configs[:, cb].T; scc = configs[:, cc].T
        activec = sca != scb
        szc = jnp.where(scc == 1, 1.0, -1.0)
        signc = jnp.where(sca == 0, 1.0, -1.0)
        coeffc = 0.5j * 0.5 * szc * signc          # chirality operator (alpha factored out)
        ratioc = jnp.exp((Uswc - U0[None, :]) + 1j * (Pswc - P0[None, :]))
        per_cyc = jnp.where(activec, coeffc * ratioc, jnp.zeros((), jnp.complex64))  # (Nc, B)
        per_triple = per_cyc.reshape(len(model.triples), 3, B).sum(axis=1)           # (Ntri, B)
        return jnp.real(jnp.mean(per_triple, axis=1))                                # (Ntri,)

    return dict(net=net, U=U, Phi=Phi, UPhi=UPhi, eloc=eloc,
                eloc_Udetached=eloc_Udetached, sample=sample,
                grads_std=grads_std, grads_mix=grads_mix, grad_variance=grad_variance,
                chirality_profile=chirality_profile,
                N=N, n_up=model.n_up)
