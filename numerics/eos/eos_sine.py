"""Edge of stability on sine targets: full-batch GD, kernel and Hessian spectra along training.

Same data, network and initialization as sine_deep.py (inputs uniform on [-1, 1], first-layer weights
N(0, 1) times 3 and biases N(0, 1), hidden weights N(0, 1/width), output layer zero, so the output is
zero at initialization). Loss (1/2)||N - y||^2 with <a, b> = a.b / n, c_p = 1, Pi = I, so the stability
limit of GD is eta * lambda_1(K) < 2 for the Gauss-Newton part of the Hessian.

For each target sin(omega x) and each step size eta = lr_rel / lambda_1(K at initialization), trains
with full-batch GD and records the loss at every step and, every --log-every steps:
  * the full spectrum of K = (1/n) J J^T (explicit n x n matrix);
  * lambda_1 of the full Hessian of the loss (Lanczos on Hessian-vector products), every --hess-every logs;
  * the removal rate rho = <E, K E> / ||E||^2 and the target rate <Y, K Y> / ||Y||^2;
  * the band eigenvalue lambda_max(P K P), P = projection off the Fourier modes cos/sin(pi k x) with
    pi k < omega / 2 (and the constant), and ||P Y||, ||P E||;
  * the overlaps of the top eigenvector of K with Y and with E;
  * for each Delta in --deltas: the slow part ||E_gamma|| at gamma = 1 / (eta Delta), the fixed-kernel
    forecast ||(I - eta K)^Delta E|| (all directions, and only those with eta lambda < 2);
  * the displacement D = |theta - theta_0| and the path length sum |theta_{k+1} - theta_k|.
Every --chord-every logs (and at the last step) it evaluates the chord bound, which holds for any
training path because Y_hat_0 = 0:
  int_0^1 sqrt(lambda_max(P K(theta_0 + tau (theta - theta_0)) P)) dtau  >=  (||P Y|| - ||P E||) / D,
with --chord-nodes Gauss-Legendre nodes, plus the largest lambda_1(K) at those nodes.

Output: <out>/eos_log.csv (one row per network and log), eos_chord.csv, eos_loss.npy (steps x networks),
eos_spectra.npz (K spectrum and E at every log), eos_meta.json; parameter snapshots in <out>/snaps/."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.func import jacrev, grad, jvp
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import core as C
import sine_deep as SD

T0 = time.time()
ARCH = "tanh"


def set_arch(arch):
    global ARCH
    ARCH = arch


def forward(params, X):
    """tanh: the network of sine_deep.py (biases in every layer). relu_h: ReLU network that is positively
    homogeneous of degree L = number of weight layers in its parameters: input (x, 1), no biases after the
    first layer (the first-layer bias is the weight on the constant input), no output bias."""
    if ARCH == "tanh":
        return SD.forward(params, X)
    Xt = torch.cat([X, torch.ones_like(X)], -1)
    h = torch.relu(torch.einsum("bnd,bhd->bnh", Xt, params[0]))
    for W in params[1:-1]:
        h = torch.relu(torch.einsum("bnk,bhk->bnh", h, W))
    return torch.einsum("bnh,bkh->bnk", h, params[-1])[..., 0]


def relu_params(P, depth):
    """relu_h parameters from the sine_deep draws: first layer [w, b] as weights on (x, 1), hidden weights
    scaled by sqrt(2) (He initialization), hidden and output biases dropped, output layer zero."""
    W1 = torch.stack([P[0][:, :, 0], P[1]], -1)
    hid = [P[2 + 2 * i] * np.sqrt(2.0) for i in range(depth - 1)]
    return [W1] + hid + [P[-2]]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def one(params, b):
    return [p[b:b + 1].detach() for p in params]


def kernel_matrix(ps, X1):
    """K = A A^T with A = J / sqrt(n) for a single network (ps sliced to batch 1); returns (K, n)."""
    n = X1.shape[1]
    f1 = lambda *q: forward(list(q), X1)[0]
    Js = jacrev(f1, argnums=tuple(range(len(ps))))(*ps)
    A = torch.cat([j.reshape(n, -1) for j in Js], 1) / np.sqrt(n)
    return A @ A.T


def hess_top(ps, X1, y1):
    """Largest algebraic eigenvalue of the Hessian of (1/2) mean (N - y)^2 for one network."""
    shapes, sizes = [p.shape for p in ps], [p.numel() for p in ps]
    d = sum(sizes)

    def loss(*q):
        E = forward(list(q), X1)[0] - y1
        return 0.5 * E.pow(2).mean()

    g = grad(loss, argnums=tuple(range(len(ps))))

    def mv(v):
        v = torch.from_numpy(np.asarray(v, np.float64).ravel()).to(ps[0].device)
        vs, i = [], 0
        for s, m in zip(shapes, sizes):
            vs.append(v[i:i + m].reshape(s))
            i += m
        hv = jvp(lambda *q: g(*q), tuple(ps), tuple(vs))[1]
        return torch.cat([h.reshape(-1) for h in hv]).cpu().numpy()

    op = LinearOperator((d, d), matvec=mv, dtype=np.float64)
    return float(eigsh(op, k=1, which="LA", tol=1e-6, ncv=24, return_eigenvectors=False)[0])


def band_basis(x, cutoff):
    """Euclidean-orthonormal basis of {1, cos(pi k x), sin(pi k x) : pi k < cutoff} on the points x."""
    cols, k = [np.ones_like(x)], 1
    while np.pi * k < cutoff:
        cols += [np.cos(np.pi * k * x), np.sin(np.pi * k * x)]
        k += 1
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    return Q


def band_eig(K, Q):
    Pm = np.eye(K.shape[0]) - Q @ Q.T
    return float(np.linalg.eigvalsh(Pm @ K @ Pm)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omegas", type=float, nargs="+", default=[1, 2, 5, 10, 20, 40])
    ap.add_argument("--lr-rel", type=float, nargs="+", default=[0.5, 1.0],
                    help="eta = lr_rel / lambda_1(K at init); the edge is at lambda_1 = (2 / lr_rel) lambda_1(0)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--arch", choices=["tanh", "relu_h"], default="tanh",
                    help="relu_h: positively homogeneous ReLU network (see forward)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--hess-every", type=int, default=5, help="in logs")
    ap.add_argument("--chord-every", type=int, default=20, help="in logs")
    ap.add_argument("--chord-nodes", type=int, default=4)
    ap.add_argument("--deltas", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--snap-every", type=int, default=20, help="in logs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or cuda; cuda avoids the CPU math library (crashes under srun)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    dev = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "snaps"), exist_ok=True)

    omegas = [int(w) if float(w).is_integer() else w for w in a.omegas]
    cfg = dict(omegas=omegas, random_labels=False, width=a.width, n=a.n, depth=a.depth, seeds=a.seeds)
    X0, Y0, P0, runs0, _ = SD.setup(cfg, dev)
    set_arch(a.arch)
    if a.arch == "relu_h":
        P0 = relu_params(P0, a.depth)
    L = len(a.lr_rel)
    X = X0.repeat(L, 1, 1)
    Y = Y0.repeat(L, 1)
    params0 = [p.repeat(L, *([1] * (p.dim() - 1))) for p in P0]
    runs = [(seed, name, lr) for lr in a.lr_rel for (seed, name) in runs0]
    B, n = X.shape[0], X.shape[1]
    xs = X[:, :, 0].cpu().numpy()
    ys = Y.cpu().numpy()
    omega_of = [float(name[3:]) for (_, name, _) in runs]
    bases = [band_basis(xs[b], omega_of[b] / 2) for b in range(B)]

    # step sizes from lambda_1(K) at initialization (same for every target of a seed)
    lam1_0 = np.empty(B)
    for b in range(B):
        lam1_0[b] = float(torch.linalg.eigvalsh(kernel_matrix(one(params0, b), X[b:b + 1]))[-1])
    eta = np.array([lr for (_, _, lr) in runs]) / lam1_0
    log(f"B={B} networks, d={sum(p[0].numel() for p in params0)} parameters, n={n}; "
        f"lambda_1(0) median {np.median(lam1_0):.4g}; eta {sorted(set(np.round(eta, 6)))}")
    eta_t = [torch.tensor(eta, device=dev).reshape(B, *([1] * (p.dim() - 1))) for p in params0]

    ps = [p.clone().requires_grad_(True) for p in params0]
    flat0 = torch.cat([p.detach().reshape(B, -1) for p in params0], 1)
    loss_hist = np.empty((a.steps + 1, B))
    path = np.zeros(B)
    rows, chord_rows, spec_lam, spec_E, spec_steps = [], [], [], [], []
    li = 0
    for k in range(a.steps + 1):
        out = forward(ps, X)
        E = out - Y
        loss_hist[k] = 0.5 * E.detach().pow(2).mean(1).cpu().numpy()
        if not np.all(np.isfinite(loss_hist[k])):
            log(f"non-finite loss at step {k}; stopping")
            loss_hist = loss_hist[:k + 1]
            break
        last = k == a.steps
        if k % a.log_every == 0 or last:
            Ed = E.detach().cpu().numpy()
            flat = torch.cat([p.detach().reshape(B, -1) for p in ps], 1)
            D = (flat - flat0).norm(dim=1).cpu().numpy()
            lam_all, E_all = np.empty((B, n)), Ed.copy()
            do_h = li % a.hess_every == 0 or last
            do_c = li % a.chord_every == 0 or last
            for b in range(B):
                Km = kernel_matrix(one(ps, b), X[b:b + 1]).cpu().numpy()
                lam, U = np.linalg.eigh(Km)
                lam, U = lam[::-1].copy(), U[:, ::-1].copy()
                lam_all[b] = lam
                e, y = Ed[b], ys[b]
                c = U.T @ e
                en, yn = np.linalg.norm(e), np.linalg.norm(y)
                Q = bases[b]
                Pe, Py = e - Q @ (Q.T @ e), y - Q @ (Q.T @ y)
                r = dict(seed=runs[b][0], target=runs[b][1], omega=omega_of[b], lr_rel=runs[b][2],
                         eta=eta[b], step=k, loss=loss_hist[k, b], err=en / np.sqrt(n),
                         lam1=lam[0], eta_lam1=eta[b] * lam[0], lam2_lam1=lam[1] / lam[0],
                         lam5_lam1=lam[4] / lam[0], lam10_lam1=lam[9] / lam[0], trK_lam1=lam.sum() / lam[0],
                         rho=float(e @ Km @ e) / en ** 2 if en > 0 else np.nan,
                         target_rate=float(y @ Km @ y) / yn ** 2,
                         band_lam=band_eig(Km, Q), band_dim=Q.shape[1],
                         PY=np.linalg.norm(Py) / np.sqrt(n), PE=np.linalg.norm(Pe) / np.sqrt(n),
                         ov_v1_Y=abs(U[:, 0] @ y) / yn, ov_v1_E=abs(U[:, 0] @ e) / en if en > 0 else np.nan,
                         D=D[b], path=path[b])
                r["eta_rho"] = eta[b] * r["rho"]
                r["band_lam_lam1"] = r["band_lam"] / lam[0]
                stable = eta[b] * lam < 2
                for dl in a.deltas:
                    gam = 1.0 / (eta[b] * dl)
                    r[f"slow_{dl}"] = np.sqrt(np.sum((gam / (lam + gam)) ** 2 * c ** 2)) / np.sqrt(n)
                    with np.errstate(over="ignore"):
                        fac = np.exp(2 * dl * np.log(np.maximum(np.abs(1 - eta[b] * lam), 1e-300)))
                    r[f"fcast_{dl}"] = np.sqrt(np.sum(fac * c ** 2)) / np.sqrt(n)
                    r[f"fcast_stable_{dl}"] = np.sqrt(np.sum(fac[stable] * c[stable] ** 2)) / np.sqrt(n)
                if do_h:
                    r["hess_lam1"] = hess_top(one(ps, b), X[b:b + 1], Y[b])
                    r["eta_hess_lam1"] = eta[b] * r["hess_lam1"]
                rows.append(r)
                if do_c and D[b] > 0:
                    tau, w = C.gauss_legendre_01(a.chord_nodes)
                    sq, lmax = 0.0, 0.0
                    for t, wt in zip(tau, w):
                        pt = [(p0[b:b + 1] + t * (p.detach()[b:b + 1] - p0[b:b + 1])) for p0, p in zip(params0, ps)]
                        Kt = kernel_matrix(pt, X[b:b + 1]).cpu().numpy()
                        sq += wt * np.sqrt(max(band_eig(Kt, Q), 0.0))
                        lmax = max(lmax, float(np.linalg.eigvalsh(Kt)[-1]))
                    rhs = (r["PY"] - r["PE"]) / D[b]
                    chord_rows.append(dict(seed=runs[b][0], target=runs[b][1], omega=omega_of[b],
                                           lr_rel=runs[b][2], eta=eta[b], step=k, D=D[b],
                                           avg_sqrt_band=sq, rhs=rhs, margin=sq - rhs,
                                           chord_lam1_max=lmax, eta_chord_lam1_max=eta[b] * lmax,
                                           demand_band_lam=rhs ** 2, two_over_eta=2 / eta[b]))
            spec_lam.append(lam_all)
            spec_E.append(E_all)
            spec_steps.append(k)
            if li % a.snap_every == 0 or last:
                np.savez(os.path.join(a.out, "snaps", f"params_{k:07d}.npz"),
                         **{f"p{i}": p.detach().cpu().numpy() for i, p in enumerate(ps)})
            rr = [r for r in rows if r["step"] == k]
            log(f"step {k}: err " + " ".join(f"{r['err']:.3g}" for r in rr)
                + " | eta*lam1 " + " ".join(f"{r['eta_lam1']:.2f}" for r in rr))
            li += 1
            C.write_rows(os.path.join(a.out, "eos_log.csv"), rows)
            if chord_rows:
                C.write_rows(os.path.join(a.out, "eos_chord.csv"), chord_rows)
        if last:
            break
        loss = 0.5 * E.pow(2).mean(1).sum()
        gr = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            step_sq = torch.zeros(B, device=dev)
            for p, g_, et in zip(ps, gr, eta_t):
                dp = et * g_
                p -= dp
                step_sq += dp.reshape(B, -1).pow(2).sum(1)
            path += step_sq.sqrt().cpu().numpy()

    np.save(os.path.join(a.out, "eos_loss.npy"), loss_hist)
    np.savez(os.path.join(a.out, "eos_spectra.npz"), steps=np.array(spec_steps),
             lam=np.array(spec_lam), E=np.array(spec_E), x=xs, y=ys)
    meta = dict(vars(a), runs=runs, eta=eta.tolist(), lam1_0=lam1_0.tolist(), seconds=time.time() - T0)
    with open(os.path.join(a.out, "eos_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
