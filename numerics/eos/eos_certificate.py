"""Certificate that lambda_1 rises because the output falls short of the target, and the progressive-sharpening
coefficient of Damian et al.: python eos/eos_certificate.py <run dir> [...] --device cuda.

For a network with a linear output layer theta_L (tanh runs: the last two parameter tensors), Theorem 9.1 gives
exactly, under gradient flow,
    d lambda_1/dt = main + T_L,   main = 2 (lambda_1 - k_L)/|theta_L|^2 <Yhat, Y - Yhat>,   T_L = -2 <Psi h_perp, E>,
with h = S(v_1) g_{v_1}, h_perp = h - c_L theta_L. By Cauchy-Schwarz |T_L| <= 2 ||Psi h_perp|| ||E||, so
    main > 2 ||Psi h_perp|| ||E||   certifies   d lambda_1/dt > 0,
whatever the alignment of Psi h_perp with E. The script evaluates both sides at every saved checkpoint.

It also computes the progressive-sharpening coefficient alpha = -grad L . grad S of Damian et al. (their Definition 2)
in two versions: for the kernel (Gauss-Newton) sharpness lambda_1(K), alpha_GN = 2 h . g_E = main + T_L, and for the
Hessian sharpness S_H = lambda_max(Hessian of the loss), alpha_H = -grad L . grad S_H with grad S_H = D^3 L[u, u]
(u the top Hessian eigenvector), together with their Assumption 4 ratio alpha / (|grad L| |grad S|).
Output: <run dir>/certificate.csv."""
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch
from torch.func import grad, jvp
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import eos_sine as ES
from eos_decompose import jac_scaled, S_times, unflat

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-hessian", action="store_true", help="skip the Hessian part; only the certificates")
    a = ap.parse_args()
    torch.set_default_device(a.device)
    for d in a.runs:
        meta = json.load(open(os.path.join(d, "eos_meta.json")))
        ES.set_arch(meta.get("arch", "tanh"))
        runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
        spec = np.load(os.path.join(d, "eos_spectra.npz"))
        X = torch.tensor(spec["x"])[:, :, None]
        Yall = torch.tensor(spec["y"])
        B, n = Yall.shape
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "snaps", "params_*.npz"))):
            k = int(os.path.basename(f)[7:14])
            if k == 0:
                continue
            z = np.load(f)
            P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
            for b in range(B):
                ps = [p[b:b + 1] for p in P]
                X1, y = X[b:b + 1], Yall[b]
                J = jac_scaled(ps, X1).cpu().numpy()
                lam, U = np.linalg.eigh(J @ J.T / n)
                lam1, v1 = lam[-1], np.sqrt(n) * U[:, -1]
                Yh = ES.forward(ps, X1)[0].detach().cpu().numpy()
                Y = y.cpu().numpy()
                E = Yh - Y
                g_v1 = J.T @ v1 / n
                h = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).cpu().numpy()
                sizes = [p.numel() for p in ps]
                off = np.concatenate([[0], np.cumsum(sizes)])
                Lidx = np.arange(off[-3], off[-1])               # output layer: last weight and bias
                th = np.concatenate([p.reshape(-1).cpu().numpy() for p in ps])
                thL = th[Lidx]
                kL = float(np.sum((J[:, Lidx].T @ v1 / n) ** 2))
                cL = float(thL @ h[Lidx]) / float(thL @ thL)
                hp = h.copy()
                hp[Lidx] -= cL * thL
                Jhp = J @ hp
                nrm = lambda u: np.linalg.norm(u) / np.sqrt(n)
                main_t = 2 * (lam1 - kL) / float(thL @ thL) * float(Yh @ (Y - Yh)) / n
                T_L = -2 * float(Jhp @ E) / n
                rhs = 2 * nrm(Jhp) * nrm(E)
                gE = J.T @ E / n
                alpha_gn = -2 * float(h @ gE)                    # = -grad L . grad lambda_1(K) = main + T_L
                # weighted certificate: |<a, E>| <= ||(K+g)^(-1/2) a|| ||(K+g)^(1/2) E|| for every g > 0,
                # and |<Psi h_perp, E>| = |h_perp . g_E| <= |h_perp| ||E||_K (the limit g -> 0)
                ak, ek = U.T @ Jhp / np.sqrt(n), U.T @ E / np.sqrt(n)
                lamc = np.maximum(lam, 0.0)
                gams = lam1 * 10.0 ** np.linspace(-10, 2, 121)
                fg = [np.sqrt(np.sum(ak ** 2 / (lamc + g)) * np.sum((lamc + g) * ek ** 2)) for g in gams]
                i_best = int(np.argmin(fg))
                b0 = float(np.linalg.norm(hp) * np.linalg.norm(gE))
                bound_w = 2 * min(fg[i_best], b0)
                wrow = dict(bound_w=bound_w, certified_w=bool(main_t > bound_w),
                            ratio_w=main_t / bound_w if bound_w > 0 else np.nan,
                            gamma_best_rel=(gams[i_best] / lam1) if fg[i_best] < b0 else 0.0,
                            rho=float(np.linalg.norm(gE) ** 2 / nrm(E) ** 2))
                if a.no_hessian:
                    rows.append(dict(seed=runs[b][0], target=runs[b][1], lr_rel=runs[b][2], step=k,
                                     eta_lam1=eta[b] * lam1, main=main_t, T_L=T_L, rhs=rhs, **wrow))
                    continue
                # Hessian sharpness and its gradient D^3 L[u, u]
                def loss(*q):
                    return 0.5 * (ES.forward(list(q), X1)[0] - y).pow(2).mean()
                gL = grad(loss, argnums=tuple(range(len(ps))))
                hvp = lambda vv: jvp(lambda *q: gL(*q), tuple(ps), unflat(vv, ps))[1]
                def mv(v):
                    hv = hvp(torch.tensor(np.asarray(v, np.float64).ravel()))
                    return torch.cat([x.reshape(-1) for x in hv]).cpu().numpy()
                op = LinearOperator((len(th), len(th)), matvec=mv, dtype=np.float64)
                vals, vecs = eigsh(op, k=1, which="LA", tol=1e-8, ncv=30)
                SH, u = float(vals[0]), torch.tensor(vecs[:, 0])
                uu = unflat(u, ps)
                def curv(*q):                                    # u^T Hessian(q) u
                    gq = lambda *r: jvp(lambda *s: gL(*s), tuple(r), uu)[1]
                    return sum((x.reshape(-1) * w.reshape(-1)).sum() for x, w in zip(gq(*q), uu))
                gS = torch.cat([x.reshape(-1) for x in grad(curv, argnums=tuple(range(len(ps))))(*ps)]).cpu().numpy()
                alpha_h = -float(gS @ gE)
                rows.append(dict(seed=runs[b][0], target=runs[b][1], lr_rel=runs[b][2], step=k,
                                 eta_lam1=eta[b] * lam1, eta_SH=eta[b] * SH, err=nrm(E),
                                 main=main_t, T_L=T_L, rhs=rhs, certified=bool(main_t > rhs),
                                 ratio=main_t / rhs if rhs > 0 else np.nan,
                                 alpha_gn=alpha_gn, alpha_gn_check=(main_t + T_L - alpha_gn) / abs(alpha_gn),
                                 alpha_h=alpha_h,
                                 a4_gn=alpha_gn / (np.linalg.norm(gE) * 2 * np.linalg.norm(h)), cos_TL=T_L / rhs if rhs > 0 else np.nan,
                                 a4_h=alpha_h / (np.linalg.norm(gE) * np.linalg.norm(gS)), **wrow))
            print(f"{d} step {k} done", flush=True)
        C.write_rows(os.path.join(d, "certificate_w.csv" if a.no_hessian else "certificate.csv"), rows)


if __name__ == "__main__":
    main()
