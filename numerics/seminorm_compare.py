"""Step bounds of the roughness functions of Section 'Examples' against the Krylov series, for the
sine runs of sine_deep.py (no training; uses the saved checkpoint parameters).

For each run, checkpoint and gamma, at the step length s_gamma of the run (|c_p| s = Upsilon_E(gamma)^(1/2),
from sine_rows.csv), the step bound of Theorem 'Supremum over roughness functions'(a),
    b_R(s) = (R(E) - |c_p| s sigma_R) / C_R  <=  F(s) = ||E_gamma||,
for
  cluster    R(u) = ||P u||, P removes a degree-p least-squares polynomial on each cluster of k
             consecutive inputs; C_R = 1, sigma_R^2 = lambda_max(P K P); best (k, p) per run over
             k in {5, 10, 20, 50}, p in {0, 1, 2};
  high-pass  manifold high-pass R_m (k_g = 8); C_R = 1, sigma_R^2 = lambda_max(P_m K P_m); best m per run
             over m in {2, 4, 8, 16, 24, 48, 96};
  Lipschitz  R(u) = max_{i != j} |u_i - u_j| / |x_i - x_j| (adjacent pairs suffice in 1-D);
             C_R = sqrt(2n)/d_min, sigma_R = max_adjacent |Psi_i - Psi_{i+1}| / d.
K is formed column by column from matrix-free products (n products per network); the Krylov values
F_M(s_gamma) come from sine_rows.csv.
Output: <dir>/seminorm_rows.csv."""
import argparse
import csv
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import sine_deep as SD

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
HP_MS = [2, 4, 8, 16, 24, 48, 96]
CL = [(k, p) for k in (5, 10, 20, 50) for p in (0, 1, 2)]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def cluster_basis(x, k, p):
    """Orthonormal basis of piecewise polynomials of degree p on clusters of k consecutive sorted inputs."""
    n = len(x)
    cols = []
    for s in range(0, n, k):
        idx = np.arange(s, min(s + k, n))
        V = np.vander(x[idx] - x[idx].mean(), min(p + 1, len(idx)), increasing=True)
        Q, _ = np.linalg.qr(V)
        for j in range(Q.shape[1]):
            c = np.zeros(n)
            c[idx] = Q[:, j]
            cols.append(c)
    return np.array(cols).T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results folder of sine_deep.py under numerics/")
    ap.add_argument("--ckpts-file", required=True)
    ap.add_argument("--steps", default=None, help="checkpoints to analyse (default: all)")
    a = ap.parse_args()
    D = os.path.join(HERE, a.dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(a.ckpts_file, map_location=dev)
    cfg = blob["cfg"]
    X, Y, _, runs, _ = SD.setup(cfg, dev)
    n, B = cfg["n"], len(runs)
    rows_in = list(csv.DictReader(open(os.path.join(D, "sine_rows.csv"))))
    look = {(int(r["seed"]), r["target"], int(float(r["step"])), float(r["gamma_rel"])): r for r in rows_in}
    gammas = sorted({float(r["gamma_rel"]) for r in rows_in})
    steps = sorted(blob["ckpts"]) if not a.steps else [int(s) for s in a.steps.split(",")]
    xs = {s: X[i, :, 0].cpu().numpy() for i, (s, _) in enumerate(runs)}
    bases = {}
    for s, x in xs.items():
        Bhp, _ = C.graph_highpass_basis(x[:, None], k=8, m=max(HP_MS), n_out=1)
        bases[s] = {**{("hp", m): Bhp[:, :m] for m in HP_MS}, **{("cl", k, p): cluster_basis(x, k, p) for k, p in CL}}
    out = []
    for step in steps:
        t0 = time.time()
        params, pis = blob["ckpts"][step]
        params = [p.to(dev) for p in params]
        pis = None if pis is None else [q.to(dev) for q in pis]
        net = SD.Net(params, X, pis)
        E = (net.out().detach() - Y)                                     # values (not scaled)
        Kmat = torch.zeros(B, n, n, dtype=torch.float64, device=dev)
        eye = torch.eye(n, dtype=torch.float64, device=dev)
        for j in range(n):
            Kmat[:, :, j] = net.K(eye[j].expand(B, n).contiguous())
        Kmat = 0.5 * (Kmat + Kmat.transpose(1, 2))
        log(f"step {step}: kernel matrices formed in {time.time() - t0:.0f}s")
        for r, (seed, name) in enumerate(runs):
            x = xs[seed]
            K = Kmat[r].cpu().numpy()
            e = E[r].cpu().numpy()
            es = e / np.sqrt(n)
            res = {}
            for key, Bm in bases[seed].items():
                P = np.eye(n) - Bm @ Bm.T
                R = float(np.linalg.norm(P @ es))
                sig = float(np.sqrt(max(np.linalg.eigvalsh(P @ K @ P)[-1], 0)))
                res[key] = (R, sig, 1.0)
            d = np.diff(x)
            Rl = float(np.max(np.abs(np.diff(e)) / d))
            kd = np.diag(K)
            sl = float(np.sqrt(n * np.max(np.clip(kd[:-1] + kd[1:] - 2 * np.diag(K, 1), 0, None) / d ** 2)))
            res[("lip",)] = (Rl, sl, np.sqrt(2 * n) / d.min())
            for g in gammas:
                row = look[(seed, name, step, g)]
                cs, F = float(row["cs"]), float(row["E_gamma"])
                b = {key: (R - cs * sig) / Cr for key, (R, sig, Cr) in res.items()}
                bh = max((v, k) for k, v in b.items() if k[0] == "hp")
                bc = max((v, k) for k, v in b.items() if k[0] == "cl")
                out.append(dict(seed=seed, target=name, step=step, gamma_rel=g, err=float(row["err"]), F=F, cs=cs,
                                b_highpass=bh[0], best_m=bh[1][1], b_highpass_m24=b[("hp", 24)],
                                b_cluster=bc[0], best_k=bc[1][1], best_p=bc[1][2], b_lipschitz=b[("lip",)],
                                C_lipschitz=res[("lip",)][2],
                                **{f"F_M{M}": float(row[f"F_M{M}"]) for M in SD.MS}))
        C.write_rows(os.path.join(D, "seminorm_rows.csv"), out)
        log(f"step {step}: done in {time.time() - t0:.0f}s")
    bad = [q for q in out if max(q["b_highpass"], q["b_cluster"], q["b_lipschitz"]) > q["F"] * (1 + 1e-6) + 1e-12]
    log(f"check b_R <= F(s): {len(out) - len(bad)}/{len(out)} ok")
    for g in (1e-4, 1e-2, 0.3):
        q = [r for r in out if r["step"] == steps[-1] and r["gamma_rel"] == g and r["F"] > 0]
        med = lambda k: np.median([max(r[k], 0) / r["F"] for r in q])
        log(f"step {steps[-1]}, gamma {g:g}: median bound / F(s): Krylov F_1 {med('F_M1'):.3f}, F_100 {med('F_M100'):.3f}, "
            f"cluster {med('b_cluster'):.3f}, high-pass {med('b_highpass'):.3f}, Lipschitz {med('b_lipschitz'):.3g}")
    log("ok")
