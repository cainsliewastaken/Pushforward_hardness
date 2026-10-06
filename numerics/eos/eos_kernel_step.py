"""One gradient-descent step and the full kernel: python eos/eos_kernel_step.py <out dir of eos_sine.py> --device cuda.

At every saved snapshot, for each network, take the actual step theta' = theta - eta g_E and, for the eigenvectors v_k
of K(theta) (k in KS, ||v_k|| = 1), compare (all exact except the column marked first order):
  q_k  = <v_k, K v_k> = lambda_k,  q_k' = <v_k, K' v_k> = |g_k'|^2 with g_k' = (1/n) J(theta')^T v_k;
  rel_k     = (q_k' - lambda_k) / lambda_k                     actual relative change of the quadratic form;
  rel1_k    = -2 eta g_k . S(v_k) g_E / lambda_k                first-order term of (F1);
  rem_k     = rel_k - rel1_k                                   all higher-order terms of (F1);
  sq_k      = |sqrt(q_k') - sqrt(lambda_k)| / sqrt(lambda_k)    actual relative change of sqrt;
  bnd_k     = |g_k' - g_k| / sqrt(lambda_k) = eta |Sbar(v_k) g_E| / sqrt(lambda_k)   the bound (F3), exact numerator;
  a_k       = |S(v_k) g_E|, its first-order numerator, and slow_k = |S(v_k) g_{E_gamma}| / a_k, gamma = 1/(eta*gamma_delta);
  lamk_next = lambda_k(theta'), the k-th eigenvalue after the step.
Output: <out>/kernel_step.csv."""
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
import eos_sine as ES
from eos_decompose import jac_scaled, S_times, unflat

torch.set_default_dtype(torch.float64)
KS = [1, 2, 5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gamma-delta", type=float, default=1000)
    a = ap.parse_args()
    torch.set_default_device(a.device)
    meta = json.load(open(os.path.join(a.out, "eos_meta.json")))
    ES.set_arch(meta.get("arch", "tanh"))
    runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
    spec = np.load(os.path.join(a.out, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    Y = torch.tensor(spec["y"])
    B, n = Y.shape
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "snaps", "params_*.npz"))):
        step = int(os.path.basename(f)[7:14])
        z = np.load(f)
        P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
        for b in range(B):
            ps = [p[b:b + 1] for p in P]
            X1 = X[b:b + 1]
            J = jac_scaled(ps, X1)
            lam, U = np.linalg.eigh((J @ J.T / n).cpu().numpy())
            lam, U = lam[::-1].copy(), U[:, ::-1].copy()
            E = (ES.forward(ps, X1)[0].detach() - Y[b]).cpu().numpy()
            Jt = J.cpu().numpy().T
            gE = Jt @ E / n
            gam = 1.0 / (eta[b] * a.gamma_delta)
            Eg = U @ (gam / (lam + gam) * (U.T @ E))
            gEg = Jt @ Eg / n
            ps1 = [p - eta[b] * s for p, s in zip(ps, unflat(torch.tensor(gE), ps))]
            J1 = jac_scaled(ps1, X1)
            Jt1 = J1.cpu().numpy().T
            lam_next = np.linalg.eigvalsh((J1 @ J1.T / n).cpu().numpy())[::-1]
            row = dict(seed=runs[b][0], target=runs[b][1], lr_rel=runs[b][2], step=step, eta_lam1=eta[b] * lam[0],
                       err=np.linalg.norm(E) / np.sqrt(n))
            for k in KS:
                if lam[k - 1] < 1e-10 * lam[0]:
                    continue
                v = np.sqrt(n) * U[:, k - 1]
                g, g1 = Jt @ v / n, Jt1 @ v / n
                Sg = S_times(ps, X1, torch.tensor(v), torch.tensor(gE)).cpu().numpy()
                Sgs = S_times(ps, X1, torch.tensor(v), torch.tensor(gEg)).cpu().numpy()
                lk, qk1 = lam[k - 1], float(g1 @ g1)
                row.update({f"lam{k}": lk, f"lam{k}_next": lam_next[k - 1], f"rel{k}": (qk1 - lk) / lk,
                            f"rel1_{k}": -2 * eta[b] * float(g @ Sg) / lk,
                            f"sq{k}": abs(np.sqrt(qk1) - np.sqrt(lk)) / np.sqrt(lk),
                            f"bnd{k}": float(np.linalg.norm(g1 - g)) / np.sqrt(lk),
                            f"a{k}": float(np.linalg.norm(Sg)),
                            f"slow{k}": float(np.linalg.norm(Sgs)) / max(float(np.linalg.norm(Sg)), 1e-300)})
                row[f"rem{k}"] = row[f"rel{k}"] - row[f"rel1_{k}"]
            rows.append(row)
            s = " ".join(f"k{k}: rel {row[f'rel{k}']:+.1e} 1st {row[f'rel1_{k}']:+.1e} sq/bnd {row[f'sq{k}'] / row[f'bnd{k}']:.2f} "
                         f"a/a1 {row[f'a{k}'] / row['a1']:.2f} sqrt(l/l1) {np.sqrt(row[f'lam{k}'] / lam[0]):.3f}"
                         for k in (1, 10, 20) if f"rel{k}" in row)
            print(f"{row['target']:6s} lr {row['lr_rel']} step {step:6d} eta*lam1 {row['eta_lam1']:.2f} | {s}", flush=True)
    C.write_rows(os.path.join(a.out, "kernel_step.csv"), rows)
    os._exit(0)


if __name__ == "__main__":
    main()
