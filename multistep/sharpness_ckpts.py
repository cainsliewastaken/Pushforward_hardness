"""Sharpness and alignment of the nudged losses along a training run (checkpoints saved by train_kappa.py --save-every).

For each checkpoint and kappa, on one fixed batch of training starts:
  L_k          nudged window loss (kappa = 0: the objective V_T; kappa = 1: the one-step loss on the window's states)
  lam          top eigenvalue of Hess L_k (plain parameters), by power iteration with Hessian-vector products
  lam_pre      top eigenvalue of P^{-1/2} Hess L_k P^{-1/2}, P = sqrt(v_hat) + eps from the run's Adam state (Adam's
               preconditioned sharpness; Adam with momentum beta1 is stable below (2 + 2 beta1) / ((1 - beta1) lr))
  cos          cosine between grad L_k and grad V_T (plain), cos_pre the same in Adam's metric (P^{-1})
  sqrt(L/lam)  stable path length per sqrt(step) for gradient descent at eta = 1/lam (ell_N <= sqrt(2 N L / lam))
  python sharpness_ckpts.py --run <run dir> [--part i --nparts n] --out file.pt
"""
import argparse, glob, os, torch
from msr.config import load_cfg
from msr.core import Arch, Model, make_data
from train_kappa import loss_kappa

KAPPAS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def _power(mv, P, dev, iters):
    v = torch.randn(P, device=dev, generator=torch.Generator(device=dev).manual_seed(0)); v /= v.norm()
    lam = 0.0
    for _ in range(iters):
        w = mv(v); lam = float(v @ w); v = w / (w.norm() + 1e-300)
    return lam


def top_eig(mv, P, dev, iters):
    """Largest (most positive) eigenvalue of a symmetric operator: power iteration finds the eigenvalue of largest
    magnitude; if that is negative, a second pass on mv - lam1 I (positive semidefinite shift) gives the top one."""
    lam1 = _power(mv, P, dev, iters)
    if lam1 >= 0:
        return lam1
    return _power(lambda v: mv(v) - lam1 * v, P, dev, iters) + lam1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg("config.yaml", ["n_train=8192", "n_test=64"] + a.set)
    data = make_data(cfg, a.dt, dev, horizon=a.p * a.dt)
    arch = Arch(cfg["hidden"], cfg["act"])
    c = Model(arch, "euler", a.dt, data["stats"]).c
    p, B = a.p, a.batch
    U = data["uT_train"][:p + 1, :B]
    files = sorted(glob.glob(os.path.join(a.run, "ckpts", "step*.pt")))[a.part::a.nparts]
    rows = []
    for f in files:
        ck = torch.load(f, map_location=dev)
        th = ck["theta"].to(dev)
        Pinv = None
        if ck["exp_avg_sq"] is not None and ck["adam_step"] > 0:
            b2 = ck["betas"][1]
            vhat = ck["exp_avg_sq"].to(dev) / (1 - b2 ** ck["adam_step"])
            Pinv = 1.0 / (vhat.sqrt() + ck["eps"])
        t = th.clone().requires_grad_(True)
        gV = torch.autograd.grad(loss_kappa(arch, t, c, U, p, None), t)[0]
        for kap in KAPPAS:
            K = torch.full((p, B), kap, device=dev)
            f_ = lambda x: loss_kappa(arch, x, c, U, p, K)

            def hvp(v):
                x = th.clone().requires_grad_(True)
                gr = torch.autograd.grad(f_(x), x, create_graph=True)[0]
                return torch.autograd.grad(gr @ v, x)[0]
            t = th.clone().requires_grad_(True)
            Lk = f_(t)
            gk = torch.autograd.grad(Lk, t)[0]
            lam = top_eig(hvp, arch.P, dev, a.iters)
            row = dict(step=ck["step"], lr=ck["lr"], path_length=ck["path_length"], kappa=kap, L=float(Lk), lam=lam,
                       cos=float(gV @ gk / (gV.norm() * gk.norm() + 1e-300)))
            if Pinv is not None:
                s = Pinv.sqrt()
                row["lam_pre"] = top_eig(lambda v: s * hvp(s * v), arch.P, dev, a.iters)
                row["cos_pre"] = float((gV * Pinv * gk).sum() / ((gV * Pinv * gV).sum().sqrt() * (gk * Pinv * gk).sum().sqrt() + 1e-300))
            rows.append(row)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    torch.save(rows, a.out)


if __name__ == "__main__":
    main()
