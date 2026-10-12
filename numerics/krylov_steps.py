"""Section 'Krylov series' (numerics): the one-step results on single training steps, for the sine
networks of sine_deep.py (no training; uses the saved snapshots of parameters and Adam moments).

At each snapshot step k the Adam step k -> k+1 is reproduced exactly (gradient at theta_k, moments m, v,
learning rate of the cosine schedule). Pi is the preconditioner of that step, 1/(sqrt(v_hat_{k+1}) + eps),
held fixed; the rescaled step is g = Pi^(-1/2) (theta_{k+1} - theta_k), its length s = |g|, and c_p = 1.
For each run:
  F(s)        exact smallest error after a step of length s with the features at theta_k (K formed from
              n matrix-free products and diagonalized);
  Fbar(s)     the same with the features averaged along the step taken, Psi_bar = int_0^1 Psi(theta_k +
              tau dtheta) dtau (4-point Gauss-Legendre); with these, Delta E = Psi_bar g exactly, so
              ||E_new|| >= Fbar(s) holds exactly (Section 'Scope and limitations', reading (i));
  achieved    (||E|| - ||E_new||) / (||E|| - F(s)): the decrease the step achieved as a fraction of the
              largest decrease that a step of its length can achieve with the features at theta_k;
  Krylov      F_M(s') for s' = s, 10 s, 100 s (single steps of those lengths), M <= MMAX, against the exact
              F(s'): monotonicity, F_M <= F, and M_conv = smallest M with F(s') - F_M(s') <= 1e-6 (||E|| - F(s'));
  q_E         distinct eigenvalues of K (including 0) carrying a component of E above 1e-12 ||E||
              (eigenvalues distinct when they differ by more than 1e-9 relatively).
Check of the reproduction: 10 steps replayed from the first snapshot against the saved snapshot 10 later.
Output: <dir>/krylov_steps.csv."""
import argparse
import os
import sys
import time
import numpy as np
import torch
from torch.func import jvp, vjp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import krylov_mf as KM
import sine_deep as SD

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
MMAX = 200
MULTS = [1, 10, 100]
NQ = 4
B1, B2, EPS = 0.9, 0.999, 1e-8


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def adam_step(th, m, v, k, X, Y, cfg):
    """Step k of sine_deep.train (Adam branch): returns theta_{k+1}, m_{k+1}, v_{k+1}."""
    ps = [p.clone().requires_grad_(True) for p in th]
    E = SD.forward(ps, X) - Y
    gr = torch.autograd.grad(0.5 * E.pow(2).mean(1).sum(), ps)
    a = cfg["lr"] * 0.5 * (1 + np.cos(np.pi * k / max(cfg["ckpts"])))
    m2 = [B1 * mi + (1 - B1) * g for mi, g in zip(m, gr)]
    v2 = [B2 * vi + (1 - B2) * g * g for vi, g in zip(v, gr)]
    th2 = [p.detach() - a * (mi / (1 - B1 ** (k + 1))) / ((vi / (1 - B2 ** (k + 1))).sqrt() + EPS)
           for p, mi, vi in zip(th, m2, v2)]
    return th2, m2, v2


def kmat(Kop, B, n, dev):
    eye = torch.eye(n, dtype=torch.float64, device=dev)
    K = torch.zeros(B, n, n, dtype=torch.float64, device=dev)
    for i in range(n):
        K[:, :, i] = Kop(eye[i].expand(B, n).contiguous())
    return 0.5 * (K + K.transpose(1, 2))


def exact_reduced(Kr, u):
    """Eigenpairs of one kernel matrix (eigenvalues clipped at 0) and the coefficients of u."""
    lam, V = torch.linalg.eigh(Kr)
    lam = torch.clamp(lam.flip(-1), min=0.0).cpu().numpy()
    e = (V.flip(-1).T @ u).cpu().numpy()
    return np.append(lam, 0.0), np.append(e, 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--snaps", required=True)
    a = ap.parse_args()
    D = os.path.join(HERE, a.dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(a.snaps, map_location=dev)
    cfg, snaps = blob["cfg"], blob["snaps"]
    X, Y, _, runs, _ = SD.setup(cfg, dev)
    n, B = cfg["n"], len(runs)
    ks = sorted(snaps)
    # reproduction check: replay 10 steps from the first snapshot
    k0 = ks[0]
    if k0 + 10 in snaps:
        th, m, v = snaps[k0]["params"], snaps[k0]["m"], snaps[k0]["v"]
        for kk in range(k0, k0 + 10):
            th, m, v = adam_step(th, m, v, kk, X, Y, cfg)
        ref = snaps[k0 + 10]["params"]
        dif = max(float((p - q).abs().max() / q.abs().max()) for p, q in zip(th, ref))
        log(f"reproduction check: 10 replayed steps from {k0} vs saved {k0 + 10}: max relative difference {dif:.2e}")
    gl_t, gl_w = C.gauss_legendre_01(NQ)
    rows = []
    for k in ks:
        t0 = time.time()
        th, m, v = snaps[k]["params"], snaps[k]["m"], snaps[k]["v"]
        th2, _, v2 = adam_step(th, m, v, k, X, Y, cfg)
        pis = [1 / ((vi / (1 - B2 ** (k + 1))).sqrt() + EPS) for vi in v2]
        dth = [q - p for p, q in zip(th, th2)]
        g2 = sum((d * d / w).reshape(B, -1).sum(1) for d, w in zip(dth, pis))
        s = g2.sqrt().cpu().numpy()
        net = SD.Net(th, X, pis)
        Es = (net.out().detach() - Y) / np.sqrt(n)
        En = (SD.forward(th2, X).detach() - Y) / np.sqrt(n)
        K = kmat(net.K, B, n, dev)
        G, aa, orth, _ = KM.lanczos(net.K, Es, MMAX)

        # averaged features: Psi_bar^T w = sum_q w_q J_q^T w Pi^(1/2), Psi_bar h = sum_q w_q J_q Pi^(1/2) h
        nets_q = [SD.Net([p + t * d for p, d in zip(th, dth)], X, None) for t in gl_t]
        sq = [w.sqrt() for w in pis]

        def Kbar(w):
            gs = [tuple(gi * si for gi, si in zip(nq._vjp(w), sq)) for nq in nets_q]
            gsum = tuple(sum(wq * gq[i] for wq, gq in zip(gl_w, gs)) for i in range(len(sq)))
            out = 0
            for wq, nq in zip(gl_w, nets_q):
                out = out + wq * jvp(nq.f, tuple(nq.params), tuple(gi * si for gi, si in zip(gsum, sq)))[1]
            return out / n

        # Delta E = Psi_bar g with g = Pi^(-1/2) dtheta: check the quadrature
        dE_q = 0
        for wq, nq in zip(gl_w, nets_q):
            dE_q = dE_q + wq * jvp(nq.f, tuple(nq.params), tuple(dth))[1]
        quad_err = ((dE_q / np.sqrt(n) - (En - Es)).norm(dim=1) / (En - Es).norm(dim=1).clamp(min=1e-300)).cpu().numpy()
        Kb = kmat(Kbar, B, n, dev)
        log(f"step {k}: kernels formed in {time.time() - t0:.0f}s; median step length {np.median(s):.3g}")
        for r, (seed, name) in enumerate(runs):
            lam_ext, e_ext = exact_reduced(K[r], Es[r])
            lamb, eb = exact_reduced(Kb[r], Es[r])
            nE, nEn = float(Es[r].norm()), float(En[r].norm())
            F = C.F_closed(lam_ext, e_ext, float(s[r]))[0]
            Fb = C.F_closed(lamb, eb, float(s[r]))[0]
            row = dict(seed=seed, target=name, step=k, s=float(s[r]), err=nE, err_new=nEn, F=F, Fbar=Fb,
                       certified_decrease=nE - F, achieved=(nE - nEn) / (nE - F) if nE > F else np.nan,
                       new_minus_Fbar=(nEn - Fb) / nE, quad_err=float(quad_err[r]), orth=float(orth[r]),
                       q_E=int(C.distinct_count(lam_ext, e_ext, comp_rtol=1e-12, eig_rtol=1e-9)))
            for mult in MULTS:
                sm = mult * float(s[r])
                Fm = C.F_closed(lam_ext, e_ext, sm)[0]
                FM = np.array(KM.krylov_F(G[r], aa[r], sm, list(range(1, MMAX + 1))))
                dec = nE - Fm
                conv = np.nonzero(Fm - FM <= 1e-6 * dec)[0]
                row.update({f"F_x{mult}": Fm, f"Mconv_x{mult}": int(conv[0] + 1) if len(conv) else -1,
                            f"maxover_x{mult}": float(np.max(FM - Fm) / nE),
                            f"maxdrop_x{mult}": float(max(0.0, -np.min(np.diff(FM))) / nE),
                            **{f"F_M{M}_x{mult}": float(FM[M - 1]) for M in (1, 2, 5, 10, 20, 50, 100, 200)}})
            rows.append(row)
        C.write_rows(os.path.join(D, "krylov_steps.csv"), rows)
        q = [x for x in rows if x["step"] == k]
        ach = np.array([x["achieved"] for x in q])
        log(f"step {k}: achieved median {np.nanmedian(ach):.4f} [{np.nanmin(ach):.4f}, {np.nanmax(ach):.4f}]; "
            f"min (||E_new|| - Fbar)/||E|| {min(x['new_minus_Fbar'] for x in q):.2e}; max quadrature error {max(x['quad_err'] for x in q):.1e}; "
            + "; ".join(f"x{mu}: Mconv median {np.median([x[f'Mconv_x{mu}'] for x in q]):.0f} max {max(x[f'Mconv_x{mu}'] for x in q)}, "
                        f"max over {max(x[f'maxover_x{mu}'] for x in q):.1e}" for mu in MULTS))
    log("ok")
