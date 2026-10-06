"""Layer-scaling decomposition of the change of lambda_1 for any smooth network (tanh with biases):
python eos/eos_general.py <out dir of an eos_sine.py run with --arch tanh>.

Blocks are layers, theta_l = (W_l, b_l). With u_l = Psi_l theta_l (the output change from scaling layer l),
delta_l = u_l - Yhat (zero for the output layer), lambda_1 = top eigenvalue of K, v_1 its eigenvector,
k_l = <v1, K_l v1>, h = S(v1) g_{v1}, q = Psi h:
  theta_l . h_l = lambda_1 - k_l + eps_l                       (defines eps_l)
  q = alpha Yhat + sum_l c_l delta_l + (sum_l eps_l/|theta_l|^2) Yhat + Psi h_perp,
  alpha = sum_l (lambda_1 - k_l)/|theta_l|^2,  c_l = theta_l . h_l / |theta_l|^2,
so the first-order change of lambda_1 per GD step, -2 eta <q, E>, splits exactly into
  main   = 2 eta alpha <Yhat, Y - Yhat>
  defect = -2 eta <sum_l c_l u_l, E> - main     (from delta_l and eps_l; zero for ReLU without later biases)
  T      = -2 eta <q, E> - (main + defect)       (parameter moves that rescale no layer).
delta_l splits exactly into an activation part and a bias part:
  delta_l = A_l - B_l,  B_l = sum_{m > l} Psi_{b_m} b_m  (later biases),  A_l = delta_l + B_l,
where A_l = sum_{m >= l, hidden} sum_j dN/da_{m,j} psi(z_{m,j}), psi(z) = z sigma'(z) - sigma(z).
Output: <out>/general.csv."""
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
from eos_decompose import jac_scaled, S_times

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--print-steps", type=int, nargs="+", default=[4000, 10000, 20000])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_device(a.device)
    meta = json.load(open(os.path.join(a.out, "eos_meta.json")))
    assert meta.get("arch", "tanh") == "tanh", "this script is for the tanh network with biases"
    ES.set_arch("tanh")
    runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
    spec = np.load(os.path.join(a.out, "eos_spectra.npz"))
    X = torch.tensor(spec["x"])[:, :, None]
    Y = torch.tensor(spec["y"])
    B, n = Y.shape
    rows = []
    for f in sorted(glob.glob(os.path.join(a.out, "snaps", "params_*.npz"))):
        k = int(os.path.basename(f)[7:14])
        if k == 0:
            continue
        z = np.load(f)
        P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
        for b in range(B):
            ps = [p[b:b + 1] for p in P]
            X1 = X[b:b + 1]
            J = jac_scaled(ps, X1).cpu().numpy()
            sizes = [p.numel() for p in ps]
            off = np.concatenate([[0], np.cumsum(sizes)])
            cols = [np.arange(off[i], off[i + 1]) for i in range(len(ps))]
            th = np.concatenate([p.reshape(-1).cpu().numpy() for p in ps])
            lam, U = np.linalg.eigh(J @ J.T / n)
            lam1, u1 = lam[-1], U[:, -1]
            v1 = np.sqrt(n) * u1
            Yh = ES.forward(ps, X1)[0].detach().cpu().numpy()
            E = Yh - Y[b].cpu().numpy()
            res = -E                                    # Y - Yhat
            g_v1 = J.T @ v1 / n
            h = S_times(ps, X1, torch.tensor(v1), torch.tensor(g_v1)).cpu().numpy()
            q = J @ h
            total = float(-2 * (q @ E) / n * eta[b])
            nlayers = len(ps) // 2                      # params = [W1, b1, W2, b2, ..., Wo, bo]
            blocks = [np.concatenate([cols[2 * l], cols[2 * l + 1]]) for l in range(nlayers)]
            bias_cols = [cols[2 * l + 1] for l in range(nlayers)]
            alpha, q_layers = 0.0, np.zeros(n)
            r = dict(seed=runs[b][0], target=runs[b][1], omega=float(runs[b][1][3:]), lr_rel=runs[b][2], step=k,
                     err=np.linalg.norm(E) / np.sqrt(n), eta_lam1=eta[b] * lam1,
                     yhat_norm=np.linalg.norm(Yh) / np.sqrt(n), residual_along_output=float(Yh @ res / n))
            for l, bc in enumerate(blocks):
                thl, hl, Jl = th[bc], h[bc], J[:, bc]
                nl2 = float(thl @ thl)
                ul = Jl @ thl
                dl = ul - Yh
                Bl = sum((J[:, bias_cols[m]] @ th[bias_cols[m]] for m in range(l + 1, nlayers)), np.zeros(n))
                Al = dl + Bl
                kl = float(np.sum((Jl.T @ v1 / n) ** 2))
                thh = float(thl @ hl)
                eps = thh - (lam1 - kl)
                alpha += (lam1 - kl) / nl2
                q_layers += (thh / nl2) * ul
                yn = np.linalg.norm(Yh)
                r.update({f"L{l}_theta2": nl2, f"L{l}_delta_rel": np.linalg.norm(dl) / yn,
                          f"L{l}_act_rel": np.linalg.norm(Al) / yn, f"L{l}_bias_rel": np.linalg.norm(Bl) / yn,
                          f"L{l}_eps_rel": eps / lam1, f"L{l}_k_rel": kl / lam1})
            layer_part = float(-2 * (q_layers @ E) / n * eta[b])
            main_t = float(2 * alpha * (Yh @ res) / n * eta[b])
            r.update(alpha=alpha, total=total, main=main_t, defect=layer_part - main_t, T=total - layer_part,
                     share_main=main_t / total if total else np.nan,
                     share_defect=(layer_part - main_t) / total if total else np.nan,
                     share_T=(total - layer_part) / total if total else np.nan)
            rows.append(r)
            if k in a.print_steps:
                lay = " ".join(f"L{l}: delta {r[f'L{l}_delta_rel']:.2f} (act {r[f'L{l}_act_rel']:.2f}, "
                               f"bias {r[f'L{l}_bias_rel']:.2f}) eps {r[f'L{l}_eps_rel']:+.2f}" for l in range(nlayers))
                print(f"{r['target']:6s} lr {r['lr_rel']} step {k:6d} eta*lam1 {r['eta_lam1']:.2f} | dlam1/step "
                      f"{total:+.2e} = main {r['share_main']:+.2f} + defect {r['share_defect']:+.2f} + T "
                      f"{r['share_T']:+.2f} | {lay}", flush=True)
    C.write_rows(os.path.join(a.out, "general.csv"), rows)


if __name__ == "__main__":
    main()
