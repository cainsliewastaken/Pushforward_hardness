"""Pre-flight hyperparameter selection from the initial state (2026-10-13; the side chat's plan). Three measurements on one
batch at step 0, after one statistics refresh, set the TARGET-dependent choices before training:
  1. regime: Lanczos on the true (whole-network) kernel; fraction of the residual's energy on its top-32 directions
     (the Krylov floor at time zero): large -> smooth, small -> rough;
  2. block size: for blocks of 128 / 256 / 512 / full, the fraction of the true per-layer top-32 directions' energy captured
     by the top-256 Kronecker modes (tracking.py part (b)); smooth -> smallest block with fidelity >= threshold, rough ->
     the memory choice (256);
  3. output-side statistic and budget: probe Gauss-Newton factor if rough, empirical gradient covariance if smooth; one
     trial step on a fresh batch gives the held-out alignment c, and the trust-region constant is set to 0.8 x 2c.
Everything trajectory-dependent (lr, refresh period, statistics horizon, decay) stays with the online rules."""
import math, copy, torch
from torch.func import jvp, vjp
from mechanism import lanczos_topk, fwd_lins

def _true_kernel_fastfrac(plist, xb, yb, act, L, dev, k=32, m=48):
    with torch.enable_grad():
        out, vjp_fn = vjp(lambda *q: fwd_lins(q, xb, act, L), *plist)
    E = (out - yb).detach().reshape(-1); N = E.numel()
    def matvec(v):
        g = vjp_fn(v.reshape(out.shape))
        with torch.enable_grad():
            _, Kv = jvp(lambda *q: fwd_lins(q, xb, act, L), tuple(plist), g)
        return Kv.reshape(-1)
    ev, ritz = lanczos_topk(matvec, N, dev, m=m, k=k)
    coef = ritz.T @ E
    return float((coef * coef).sum() / (E @ E + 1e-30)), float(ev[0]), float(ev[-1])

def _block_fidelity(KQ, xb, k=32, Ktop=256):
    """mean over hidden layers and over the true top-k per-layer directions of the energy fraction on the top-Ktop
    Kronecker modes of this KQ's (block) bases."""
    vals = []
    for l in range(KQ.L - 1):
        if KQ.S[l].e is None or KQ.A[l].e is None: continue
        W0 = KQ.Wt[l].data; lam = (KQ.S[l].e[:, None] * KQ.A[l].e[None, :]).flatten(); order = torch.argsort(lam, descending=True)
        f_l = lambda Wl: KQ.forward_with(xb, l, Wl)
        with torch.enable_grad():
            out, vjp_fn = vjp(f_l, W0)
        N = out.numel()
        def matvec(v):
            g = vjp_fn(v.reshape(out.shape))[0]
            with torch.enable_grad():
                _, Kv = jvp(f_l, (W0,), (g,))
            return Kv.reshape(-1)
        ev, ritz = lanczos_topk(matvec, N, W0.device, m=min(2 * k, 64), k=k)
        fr = 0.0
        for i in range(ritz.shape[1]):
            u = vjp_fn(ritz[:, i].reshape(out.shape))[0].flatten(); c2 = u * u
            fr += float(c2[order[:Ktop]].sum() / (c2.sum() + 1e-30))
        vals.append(fr / max(1, ritz.shape[1]))
    return sum(vals) / max(1, len(vals))

def preflight(a, lins, make_KQ, forward_with_KQ, draw, gtrain, xb, yb, mse, DEV, emit, smooth_thresh=0.01, fid_thresh=0.9):
    # smooth_thresh: calibrated on two targets only (width 4096 at init: rough 0.004, smooth 0.026)
    """returns dict(block, S_mode, rem, regime, fastfrac, c, fidelities). lins are not modified."""
    L = len(lins); act = torch.nn.functional.gelu
    # 1. regime from the true kernel at init
    plist = [t.detach().clone() for lin in lins for t in (lin.weight, lin.bias)]
    fastfrac, lam1, lamk = _true_kernel_fastfrac(plist, xb, yb, act, L, DEV)
    regime = 'smooth' if fastfrac >= smooth_thresh else 'rough'
    emit('pf_fastfrac', fastfrac, 0); emit('pf_lam1', lam1, 0); emit('pf_lam32', lamk, 0)
    # 2. block fidelity after one statistics refresh (zero-lr step) for each block size
    fids = {}
    lins_copy = copy.deepcopy(lins)
    for b in (128, 256, 512, 0):
        a_b = copy.copy(a); a_b.block = b; a_b.rem = 0.0; a_b.hyper = 0; a_b.fused = 0; a_b.eig_iters = 0   # exact eigh once (subspace iteration fails on 4096-wide blocks)
        KQ_b = make_KQ(lins_copy, a_b)
        with torch.enable_grad():
            out = KQ_b.forward(xb, keep=True); loss = mse(out, yb); loss.backward(retain_graph=True)
        KQ_b.step(1, 1e-9, out, xb, yb, mse, None, None, None)             # statistics + first refresh; lr ~ 0 (exactly 0 makes the cap 0 x inf = NaN)
        fids[b if b else 'full'] = _block_fidelity(KQ_b, xb)
        emit(f'pf_fidelity_b{b if b else "full"}', fids[b if b else 'full'], 0)
        del KQ_b; torch.cuda.empty_cache()
    if regime == 'smooth':
        block = next((b for b in (128, 256, 512) if fids[b] >= fid_thresh), 512)   # none passes -> the largest affordable, not full
    else: block = 256
    S_mode = 'emp' if regime == 'smooth' else 'mc'
    # 3. one trial step at the base lr on the chosen configuration; held-out alignment on a fresh batch
    a_t = copy.copy(a); a_t.block = block; a_t.S_mode = S_mode; a_t.rem = 0.0; a_t.fused = 0
    lins_t = copy.deepcopy(lins); KQ_t = make_KQ(lins_t, a_t)
    xv, yv = draw(a.batch, gtrain)
    with torch.no_grad(): f0v = KQ_t.forward(xv)
    with torch.enable_grad():
        out = KQ_t.forward(xb, keep=True); loss = mse(out, yb); loss.backward(retain_graph=True)
    KQ_t.step(1, 1e-9, out, xb, yb, mse, None, None, None)                  # statistics + refresh at lr ~ 0
    with torch.enable_grad():
        out = KQ_t.forward(xb, keep=True); loss = mse(out, yb); loss.backward(retain_graph=True)
    KQ_t.step(2, a.lr, out, xb, yb, mse, None, None, None)                  # the trial step
    with torch.no_grad():
        f1v = KQ_t.forward(xv); Ev = f0v - yv; dfv = f1v - f0v
        c = float(-(Ev * dfv).sum() / (Ev.norm() * dfv.norm() + 1e-30)); rho_nat = float(dfv.norm() / (Ev.norm() + 1e-30))
    rem = max(0.05, min(0.5, 0.8 * 2.0 * c))
    emit('pf_heldout_cos', c, 0); emit('pf_natural_step', rho_nat, 0); emit('pf_rem', rem, 0); emit('pf_block', float(block), 0); emit('pf_S_emp', 1.0 if S_mode == 'emp' else 0.0, 0)
    del KQ_t; torch.cuda.empty_cache()
    return dict(block=block, S_mode=S_mode, rem=rem, regime=regime, fastfrac=fastfrac, c=c, fidelities=fids)
