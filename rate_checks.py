"""Checks of multistep_best_window.tex (Sections 7, 8, 9, 10) on trained checkpoints. Pi = I throughout.

rates     (Theorem "rates per unit motion", Section 10): at T, partition the held-out starts by their free deviation over
          the first T steps (inside: max_{q<=T} |e_k(t_q)| <= rho r0); a = grad of the free T-step error on the inside
          starts; g_1 (one-step loss on the true states of the first T steps), g_T (free window-T loss, all starts),
          gt_T (nudged window-T loss at r0, all starts). Rates per unit motion R_X = <a, g_X>/|g_X|, the exact
          delta = gt_T - c a and g_out = g_T - c a, the lower bound on the nudged rate and the upper bound on the free rate.
surrogate (Proposition "the correction is trained only inside the radius", Section 9): V_T (free) vs the nudged loss over
          T, and the bound from the propagated nudging corrections (Lipschitz constants from the 3x3 Jacobians at the two
          trajectories' states, a proxy for the maximum over the segment).
curv      (Lemma "exact properties of the nudged rollout", Section 8): largest Gauss-Newton eigenvalue of the window loss
          against the window, free and nudged.
screen    (Theorem "online rule", Section 7): on rollouts to N, a_q = <g_V, g^(q)> for every horizon from ONE backward pass
          (g_V) and ONE forward-mode pass (Psi g_V); the running mean and the first-term signs a_{p+1} - abar_p; the exact
          per-horizon gradients (batched backward) as a check; and the exact decrease of the N-step loss by the step -alpha g_p
          for every window p (one rollout each), against the first-order ranking.

  python rate_checks.py --runs dt0.08_onestep ... [--parts rates surrogate curv screen] [--r0 0.01 0.03 0.1 0.3 1]
Writes <out_root>/rate_checks.pt: {(run, cls, seed): dict}.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.func import jvp

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, nsteps, rollout_nudged, sq
from msr.curvature import gn_lam_max


def load_run(cfg, run, dev, horizon):
    ck = torch.load(os.path.join(cfg["out_root"], "runs", run, "ckpts.pt"), map_location=dev, weights_only=False)
    data = make_data(cfg, ck["dt"], dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()}, horizon=horizon)
    return ck, data, Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])


def window_loss(m, th, uT, p, r0, mask=None):
    """(1/p) sum_{q<=p} mean_{k in mask} (1/2)|e_k(t_q)|^2 of the (nudged) rollout from uT[0]; mask None: all starts."""
    seq = rollout_nudged(m.arch, th, m.c, uT, p, r0)
    per = 0.5 * ((uT[1:p + 1] - seq[1:]) ** 2).sum(-1)          # (p, n)
    if mask is not None:
        per = per[:, mask]
    return per.mean()


def grad_of(f, th):
    t = th.detach().clone().requires_grad_(True)
    return torch.autograd.grad(f(t), t)[0].detach()


def inner(a, b):
    return float(a @ b)


def nrm(a):
    return float(torch.linalg.norm(a))


def part_rates(m, th, uT, pT, r0s, rho, dev):
    n = uT.shape[1]
    with torch.no_grad():
        free = rollout_nudged(m.arch, th, m.c, uT, pT, None)
        devT = (uT[1:pT + 1] - free[1:]).norm(dim=-1).max(0).values        # free deviation over the first T steps
    X, Y = uT[:pT].reshape(-1, 3), uT[1:pT + 1].reshape(-1, 3)
    g_1 = grad_of(lambda t: 0.5 * ((Y - m.Gt(t, X)) ** 2).sum(-1).mean(), th)
    g_T = grad_of(lambda t: window_loss(m, t, uT, pT, None), th)
    out = dict(devT=devT.cpu().numpy(), g1_norm=nrm(g_1), gT_norm=nrm(g_T), by_r0={})
    for r0 in r0s:
        inside = devT <= rho * r0
        c = float(inside.double().mean())
        if inside.sum() == 0 or inside.sum() == n:
            out["by_r0"][r0] = dict(c=c, note="no partition")
            continue
        a = grad_of(lambda t: window_loss(m, t, uT, pT, None, inside), th)
        gt_T = grad_of(lambda t: window_loss(m, t, uT, pT, r0), th)
        at = grad_of(lambda t: window_loss(m, t, uT, pT, r0, inside), th)
        gt_out = grad_of(lambda t: window_loss(m, t, uT, pT, r0, ~inside), th)
        an = nrm(a)
        delta = gt_T - c * a
        g_out = g_T - c * a
        dn, gon = nrm(delta), nrm(g_out)
        R = lambda g: inner(a, g) / nrm(g)
        lower = an * (1 - 2 * dn / (c * an + dn)) if dn < c * an else float("nan")
        cos_out = inner(a, g_out) / (an * gon)
        denom = abs(gon - c * an)
        upper_free = an * (c * an + gon * max(0.0, cos_out)) / denom if denom > 0 else float("inf")
        r = dict(c=c, n_in=int(inside.sum()), a_norm=an, R_nudged=R(gt_T), R_onestep=R(g_1), R_free=R(g_T),
                 delta_norm=dn, delta_over_ca=dn / (c * an), lower_nudged=lower, cos_a_g1=inner(a, g_1) / (an * nrm(g_1)),
                 gout_norm=gon, cos_a_gout=cos_out, upper_free=upper_free,
                 in_departure=nrm(at - a) / an, gt_out_norm=(1 - c) * nrm(gt_out),
                 cert_vs_onestep=bool(lower > R(g_1)) if not math.isnan(lower) else False,
                 cert_vs_free=bool(lower > upper_free) if not math.isnan(lower) else False,
                 gtT_norm=nrm(gt_T))
        out["by_r0"][r0] = r
        print(f"    rates r0={r0:g}: c={c:.3f} |a|={an:.3g} R: nudged {r['R_nudged']:.3g} (lower {lower:.3g}) "
              f"onestep {r['R_onestep']:.3g} free {r['R_free']:.3g} (upper {upper_free:.3g}) | delta/(c|a|) {r['delta_over_ca']:.3g} "
              f"in-dep {r['in_departure']:.2e} |g_out|/(c|a|) {gon / (c * an):.3g} cos(a,g_out) {cos_out:+.3f} "
              f"cert>1step {r['cert_vs_onestep']} cert>free {r['cert_vs_free']}", flush=True)
    return out


def part_surrogate(m, th, uT, pT, r0s, dev):
    n = uT.shape[1]
    with torch.no_grad():
        free = rollout_nudged(m.arch, th, m.c, uT, pT, None)
        e = uT[:pT + 1] - free                                         # (pT+1, n, 3), e_0 = 0
        VT = float((0.5 * (e[1:] ** 2).sum(-1)).mean())
        Jf = torch.linalg.matrix_norm(m.Jstate(th, free[:pT].reshape(-1, 3)), ord=2).reshape(pT, n)
    out = dict(V_T=VT, by_r0={})
    for r0 in r0s:
        with torch.no_grad():
            seq = rollout_nudged(m.arch, th, m.c, uT, pT, r0)
            et = uT[:pT + 1] - seq
            VTt = float((0.5 * (et[1:] ** 2).sum(-1)).mean())
            relaxed = uT[:pT] - r0 * et[:pT] / torch.sqrt(r0 * r0 + (et[:pT] ** 2).sum(-1, keepdim=True))
            Jn = torch.linalg.matrix_norm(m.Jstate(th, relaxed.reshape(-1, 3)), ord=2).reshape(pT, n)
            ell = torch.maximum(Jf, Jn)                                 # proxy for the segment maximum
            b = torch.zeros(n, device=dev)
            bound = 0.0
            for q in range(pT):
                b = ell[q] * (b + et[q].norm(dim=-1) ** 3 / (2 * r0 * r0))
                bound += float((0.5 * b * (e[q + 1].norm(dim=-1) + et[q + 1].norm(dim=-1))).mean())
            bound /= pT
            gap_meas = float((0.5 * ((e[1:] - et[1:]) ** 2).sum(-1)).mean()) ** 0.5
        out["by_r0"][r0] = dict(V_T_nudged=VTt, gap=abs(VT - VTt), bound=bound, rms_err_diff=gap_meas,
                                max_dev_nudged=float(et.norm(dim=-1).max()))
        print(f"    surrogate r0={r0:g}: V_T {VT:.4g} nudged {VTt:.4g} gap {abs(VT - VTt):.3g} bound {bound:.3g} "
              f"max|e~| {out['by_r0'][r0]['max_dev_nudged']:.3g}", flush=True)
    return out


def part_curv(m, th, uT, dt, windows, r0s, dev):
    out = {}
    for T in windows:
        p = nsteps(T, dt)
        for r0 in [None] + list(r0s):
            t0 = time.time()
            lam = gn_lam_max(m, th, uT, p, None, tol=1e-3, r0=r0)
            out[(T, r0)] = lam
            print(f"    curv window {T:g} r0={r0}: lam_max {lam:.4g} ({time.time() - t0:.0f}s)", flush=True)
    return out


def per_horizon_grads(m, th, uT, pN, r0, chunk):
    """grad of (1/2)|e(t_q)|^2 (mean over starts) for q = 1..pN: one graph, batched backward passes. (pN, P)"""
    t = th.detach().clone().requires_grad_(True)
    seq = rollout_nudged(m.arch, t, m.c, uT, pN, r0)
    L = torch.stack([0.5 * sq(uT[q] - seq[q]) for q in range(1, pN + 1)])
    eye = torch.eye(pN, device=uT.device)
    G = [torch.autograd.grad(L, t, grad_outputs=eye[s:s + chunk], is_grads_batched=True, retain_graph=True)[0]
         for s in range(0, pN, chunk)]
    return torch.cat(G).detach()


def part_screen(m, th, uT, pN, r0s, lrs, chunk, dev):
    n = uT.shape[1]
    out = {}
    for r0 in [None] + list(r0s):
        gV = grad_of(lambda t: window_loss(m, t, uT, pN, r0), th)
        with torch.no_grad():
            seq = rollout_nudged(m.arch, th, m.c, uT, pN, r0)
            e = uT[:pN + 1] - seq
        _, tan = jvp(lambda t: rollout_nudged(m.arch, t, m.c, uT, pN, r0), (th,), (gV,))   # Psi g_V at every horizon
        a_q = -(tan[1:] * e[1:]).sum(-1).mean(-1)                      # <g_V, g^(q)> = -(1/n) sum_k <Psi_kq g_V, e_kq>
        G = per_horizon_grads(m, th, uT, pN, r0, chunk)                 # exact g^(q)
        a_q_exact = G @ gV
        abar = torch.cumsum(a_q, 0) / torch.arange(1, pN + 1, device=dev)
        first_term = a_q[1:] - abar[:-1]                               # sign of the first term for p -> p+1
        gp = torch.cumsum(G, 0) / torch.arange(1, pN + 1, device=dev)[:, None]
        gp_norm = torch.linalg.norm(gp, dim=1)
        V0 = float(window_loss(m, th, uT, pN, r0))
        rows = []
        for lr in lrs:
            s = lr * math.sqrt(m.P)
            alpha = s / float(gp_norm[-1])                             # fixed step size: the window-N step has length s
            with torch.no_grad():
                dec = torch.tensor([V0 - float(window_loss(m, th - alpha * gp[p], uT, pN, r0)) for p in range(pN)])
            fo = alpha * abar.cpu()                                     # first-order decrease alpha <g_V, g_p>
            ft_sign = (first_term > 0).cpu().numpy()
            greedy = int(np.argmax(~ft_sign)) + 1 if (~ft_sign).any() else pN
            rows.append(dict(lr=lr, s=s, alpha=alpha, dec=dec.numpy(), first_order=fo.numpy(), argmax_exact=int(dec.argmax()) + 1,
                             argmax_first_order=int(fo.argmax()) + 1, greedy_stop=greedy,
                             best_dec=float(dec.max()), dec_at_N=float(dec[-1]), dec_at_greedy=float(dec[greedy - 1])))
            print(f"    screen r0={r0} lr {lr:.0e}: exact argmax p={rows[-1]['argmax_exact']} (dec {rows[-1]['best_dec']:.3e}, "
                  f"at N {rows[-1]['dec_at_N']:.3e}) first-order argmax {rows[-1]['argmax_first_order']} greedy stop {greedy} "
                  f"(dec {rows[-1]['dec_at_greedy']:.3e})", flush=True)
        out[r0] = dict(a_q=a_q.cpu().numpy(), a_q_exact=a_q_exact.cpu().numpy(), abar=abar.cpu().numpy(),
                       first_term=first_term.cpu().numpy(), gp_norm=gp_norm.cpu().numpy(), V0=V0, steps=rows,
                       jvp_check=float((a_q - a_q_exact).abs().max() / a_q.abs().max()))
        print(f"    screen r0={r0}: JVP vs exact a_q max rel err {out[r0]['jvp_check']:.1e}; signs of a_(p+1)-abar_p: "
              + "".join("+" if x > 0 else "-" for x in first_term.cpu().numpy()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--parts", nargs="*", default=["rates", "surrogate", "curv", "screen"])
    ap.add_argument("--classes", nargs="*", default=["euler"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--T", type=float, default=1.28)
    ap.add_argument("--N", type=float, default=2.56)
    ap.add_argument("--r0", nargs="*", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--rho", type=float, default=0.5, help="inside: free deviation over [0,T] at most rho r0")
    ap.add_argument("--windows", nargs="*", type=float, default=[0.32, 0.64, 1.28, 2.56])
    ap.add_argument("--curv-r0", nargs="*", type=float, default=[0.03, 0.3])
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-curv", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outf = a.out or os.path.join(cfg["out_root"], "rate_checks.pt")
    res = torch.load(outf, weights_only=False) if os.path.exists(outf) else {}
    hz = max([a.T, a.N] + list(a.windows))
    for run in a.runs:
        ck, data, arch = load_run(cfg, run, dev, hz)
        dt = ck["dt"]
        uT = data["uT_test"][:, :a.n]
        final = max(ck["theta"])
        for j, (cls, seed) in enumerate(ck["names"]):
            if seed not in a.seeds or cls not in a.classes:
                continue
            m = Model(arch, cls, dt, data["stats"])
            th = ck["theta"][final][j].detach()
            key = (run, cls, seed)
            r = res.get(key, {})
            r.update(dt=dt, T=a.T, N=a.N, rho=a.rho, n=a.n)
            print(f"{run} {cls} s{seed}:", flush=True)
            t0 = time.time()
            if "rates" in a.parts:
                r["rates"] = part_rates(m, th, uT, nsteps(a.T, dt), a.r0, a.rho, dev)
            if "surrogate" in a.parts:
                r["surrogate"] = part_surrogate(m, th, uT, nsteps(a.T, dt), a.r0, dev)
            if "curv" in a.parts:
                r["curv"] = part_curv(m, th, data["uT_test"][:, :a.n_curv], dt, a.windows, a.curv_r0, dev)
            if "screen" in a.parts:
                r["screen"] = part_screen(m, th, uT, nsteps(a.N, dt), a.r0, a.lrs, a.chunk, dev)
            res[key] = r
            torch.save(res, outf)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {outf}")


if __name__ == "__main__":
    main()
