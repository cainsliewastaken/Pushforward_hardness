"""Which eigendirections of K move lambda_1: python eos/eos_coupling.py <out dir of eos_sine.py>.

Under gradient flow, d lambda_1/dt = -2 <q, E> with q = J h, h = S(v1) g_{v1}: g_{v1} = (1/n) J^T v_1,
S(v1) the Hessian of <v_1, N> in the parameters (h by one gradient + one jvp), and q the output change of
the parameter direction h (one more product with J). In the eigenbasis of K,
    d lambda_1/dt = -2 sum_k q_k e_k,   q_k = <v_k, q>,  e_k = <v_k, E>,
so each eigendirection's contribution is exact and needs no division by lambda_k.
For each snapshot and network it reports (per GD step, times eta):
  * the total and its parts from eigenvalue bands lambda_k / lambda_1 in [1e-1, 1], [1e-2, 1e-1], ...;
  * on the slow directions (lambda_k < slow_rel * lambda_1): the cosine between q and -E, and between q and
    the output Y_hat, restricted to those directions;
  * the median |q_k| and |q_k| / sqrt(lambda_k) per band.
Output: <out>/coupling.csv (one row per network and snapshot), coupling_bands.csv."""
import argparse
import glob
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import sine_deep as SD
from eos_decompose import jac_scaled, S_times
import eos_sine as ES

torch.set_default_dtype(torch.float64)
BANDS = [(1e-1, 1.01), (1e-2, 1e-1), (1e-3, 1e-2), (1e-4, 1e-3), (1e-6, 1e-4), (0.0, 1e-6)]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--slow-rel", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    meta = json.load(open(os.path.join(a.out, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
    spec = np.load(os.path.join(a.out, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    Y = torch.tensor(spec["y"])
    B, n = Y.shape
    rows, brows = [], []
    for f in sorted(glob.glob(os.path.join(a.out, "snaps", "params_*.npz"))):
        k = int(os.path.basename(f)[7:14])
        if k == 0:
            continue  # output layer zero: first-order term vanishes identically
        z = np.load(f)
        P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
        for b in range(B):
            ps = [p[b:b + 1] for p in P]
            X1 = X[b:b + 1]
            J = jac_scaled(ps, X1).numpy()
            lam, U = np.linalg.eigh(J @ J.T / n)
            lam, U = lam[::-1].copy(), U[:, ::-1].copy()
            Yh = ES.forward(ps, X1)[0].detach().numpy()
            E = Yh - Y[b].numpy()
            v1 = np.sqrt(n) * U[:, 0]
            g_v1 = J.T @ v1 / n
            h = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).numpy()
            q = J @ h
            qk, ek, yk = U.T @ q / np.sqrt(n), U.T @ E / np.sqrt(n), U.T @ Yh / np.sqrt(n)
            contrib = -2 * qk * ek * eta[b]                  # per GD step
            rel = lam / lam[0]
            slow = rel < a.slow_rel
            r = dict(target=runs[b][1], omega=float(runs[b][1][3:]), lr_rel=runs[b][2], step=k,
                     err=np.linalg.norm(E) / np.sqrt(n), eta_lam1=eta[b] * lam[0],
                     dlam1_step=float(contrib.sum()), from_slow=float(contrib[slow].sum()),
                     share_slow=float(contrib[slow].sum() / contrib.sum()) if contrib.sum() != 0 else np.nan,
                     n_slow=int(slow.sum()), err_frac_slow=float(np.sum(ek[slow] ** 2) / np.sum(ek ** 2)),
                     cos_q_minusE_slow=cos(qk[slow], -ek[slow]), cos_q_Yhat_slow=cos(qk[slow], yk[slow]),
                     cos_q_minusE_all=cos(qk, -ek), cos_q_Yhat_all=cos(qk, yk),
                     frac_up_slow=float(np.sum(np.abs(contrib[slow])[contrib[slow] > 0]) / np.sum(np.abs(contrib[slow])))
                     if slow.any() else np.nan)
            if meta.get("arch") == "relu_h":
                # homogeneous of degree L: J theta = L Yhat (Euler) and theta . h = (L - 1) lambda_1, so the radial
                # part of h gives q_rad = ((L - 1) lambda_1 / |theta|^2) J theta = (L (L - 1) lambda_1 / |theta|^2) Yhat
                L = meta["depth"] + 1
                th = np.concatenate([p.reshape(-1).numpy() for p in ps])
                Jth = J @ th
                rad_coef = float(th @ h) / float(th @ th)
                q_rad = rad_coef * Jth
                r.update(euler_rel_err=float(np.linalg.norm(Jth - L * Yh) / np.linalg.norm(L * Yh)),
                         homog_rel_err=float((th @ h - (L - 1) * lam[0]) / ((L - 1) * lam[0])),
                         alpha_rad=L * (L - 1) * lam[0] / float(th @ th), theta_norm=float(np.sqrt(th @ th)),
                         dlam1_radial=float(-2 * (q_rad @ E) / n * eta[b]),
                         undershoot=float(Yh @ (Y[b].numpy() - Yh) / n))
                r["dlam1_tangential"] = r["dlam1_step"] - r["dlam1_radial"]
                r["share_radial"] = r["dlam1_radial"] / r["dlam1_step"] if r["dlam1_step"] != 0 else np.nan
                # each layer is homogeneous of degree 1 on its own: J_l theta_l = Yhat and
                # theta_l . h_l = lambda_1 - <v1, K_l v1>; the projection of h on the layer directions theta_l gives
                # q_layers = alpha_layers Yhat with alpha_layers = sum_l (lambda_1 - <v1, K_l v1>) / |theta_l|^2
                i0, alpha_pred, coef_sum, eul, hom = 0, 0.0, 0.0, 0.0, 0.0
                for p in ps:
                    m = p.numel()
                    thl, hl, Jl = th[i0:i0 + m], h[i0:i0 + m], J[:, i0:i0 + m]
                    i0 += m
                    nl2 = float(thl @ thl)
                    if nl2 == 0:
                        continue
                    kl = float(np.sum((Jl.T @ v1 / n) ** 2))          # <v1, K_l v1>
                    alpha_pred += (lam[0] - kl) / nl2
                    coef_sum += float(thl @ hl) / nl2
                    eul = max(eul, float(np.linalg.norm(Jl @ thl - Yh) / np.linalg.norm(Yh)))
                    hom = max(hom, abs(float(thl @ hl) - (lam[0] - kl)) / lam[0])
                q_lay = coef_sum * Yh
                r.update(alpha_layers=alpha_pred, layer_euler_err=eul, layer_homog_err=hom,
                         dlam1_layers=float(-2 * (q_lay @ E) / n * eta[b]),
                         dlam1_layers_formula=float(2 * alpha_pred * (Yh @ (Y[b].numpy() - Yh)) / n * eta[b]))
                r["share_layers"] = r["dlam1_layers"] / r["dlam1_step"] if r["dlam1_step"] != 0 else np.nan
            rows.append(r)
            for lo, hi in BANDS:
                m = (rel >= lo) & (rel < hi)
                if m.any():
                    brows.append(dict(target=r["target"], lr_rel=r["lr_rel"], step=k, band_lo=lo, band_hi=hi,
                                      count=int(m.sum()), contrib=float(contrib[m].sum()),
                                      med_abs_q=float(np.median(np.abs(qk[m]))),
                                      med_abs_c=float(np.median(np.abs(qk[m]) / np.sqrt(np.maximum(lam[m], 1e-300)))),
                                      med_abs_e=float(np.median(np.abs(ek[m])))))
            if k % 10000 == 0:
                print(f"{r['target']:6s} lr {r['lr_rel']} step {k:6d} eta*lam1 {r['eta_lam1']:.2f} err {r['err']:.3g} | "
                      f"dlam1/step {r['dlam1_step']:+.2e} slow share {r['share_slow']:.2f} | "
                      f"slow: err frac {r['err_frac_slow']:.2f} cos(q,-E) {r['cos_q_minusE_slow']:+.2f} "
                      f"cos(q,Yhat) {r['cos_q_Yhat_slow']:+.2f} up-frac {r['frac_up_slow']:.2f} | "
                      f"all: cos(q,-E) {r['cos_q_minusE_all']:+.2f} cos(q,Yhat) {r['cos_q_Yhat_all']:+.2f}"
                      + (f" | radial share {r['share_radial']:.2f} layer share {r['share_layers']:.2f} "
                         f"euler err {r['layer_euler_err']:.1e} homog err {r['layer_homog_err']:.1e}"
                         if "share_radial" in r else ""), flush=True)
    C.write_rows(os.path.join(a.out, "coupling.csv"), rows)
    C.write_rows(os.path.join(a.out, "coupling_bands.csv"), brows)


if __name__ == "__main__":
    main()
