"""Sine targets with deeper networks and/or Adam, bounds from the matrix-free Krylov series.

Same data and first layer as sine.py / large_sine.py (inputs uniform on [-1, 1], first-layer weights
N(0, 1) times IN_SCALE = 3, first-layer biases N(0, 1), same random draws per seed, output layer
zero). Options:
  --depth D     number of hidden tanh layers (1 = the network of the paper). Hidden-to-hidden weights
                are N(0, 1/width), hidden biases zero.
  --opt gd|adam full-batch gradient descent on (1/2)||N - y||^2 (c_p = 1, Pi = I), or Adam on the
                same loss with a cosine schedule from --lr to zero; then Pi is Adam's preconditioner
                1/(sqrt(v_hat) + eps) at the checkpoint and Psi = J Pi^(1/2).
  --lr-rel k    (gd) learning rate k / lambda_1(K at initialization) per network; --lr fixes it.
The kernel is applied through torch.func vector-Jacobian and Jacobian-vector products, so any depth
works; slow parts by conjugate gradients, Krylov values by Lanczos (krylov_mf). The manifold
high-pass constant sigma_R^2 = lambda_max(P K P) is the largest Ritz value of 60 Lanczos steps.
Exact checks (explicit Jacobian, eigendecomposition) for the runs of the first seed.

Output: <out>/sine_rows.csv, sine_curves.csv, sine_meta.json."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.func import jvp, vjp, vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import krylov_mf as KM

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
IN_SCALE = 3.0
MS = [1, 2, 5, 10, 20, 50, 100]
GAMMAS = [1e-4, 1e-2, 0.1, 0.3, 1.0]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def forward(params, X):
    """params: [W1 (B,H,1), b1 (B,H), (W (B,H,H), b (B,H))*, Wo (B,1,H), bo (B,1)]; X (B,n,1) -> (B,n)."""
    h = torch.tanh(torch.einsum("bnd,bhd->bnh", X, params[0]) + params[1][:, None, :])
    for W, b in zip(params[2:-2:2], params[3:-2:2]):
        h = torch.tanh(torch.einsum("bnk,bhk->bnh", h, W) + b[:, None, :])
    return torch.einsum("bnh,bkh->bnk", h, params[-2])[..., 0] + params[-1]


def setup(cfg, dev):
    names = [f"sin{w}" for w in cfg["omegas"]] + (["random"] if cfg["random_labels"] else [])
    H, n, D = cfg["width"], cfg["n"], cfg["depth"]
    X, Y, runs, P = [], [], [], [[] for _ in range(2 * D + 2)]
    for seed in cfg["seeds"]:
        g = np.random.default_rng(seed)                       # same draws as sine.setup
        x = np.sort(g.uniform(-1, 1, n))
        W1 = IN_SCALE * g.normal(size=H)
        b1 = g.normal(size=H)
        yr = g.normal(size=n)
        gh = np.random.default_rng(10000 + seed)              # deeper layers: separate stream
        hid = [(gh.normal(size=(H, H)) / np.sqrt(H), np.zeros(H)) for _ in range(D - 1)]
        for name in names:
            X.append(x)
            Y.append(yr if name == "random" else np.sin(float(name[3:]) * x))
            ps = [W1[:, None], b1] + [a for wb in hid for a in wb] + [np.zeros((1, H)), np.zeros(1)]
            for i, p in enumerate(ps):
                P[i].append(p)
            runs.append((seed, name))
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float64, device=dev)
    return t(X)[:, :, None], t(Y), [t(p) for p in P], runs, names


class Net:
    """Kernel K = (1/n) Psi Psi^T, Psi = J Pi^(1/2), for B networks at given parameters."""

    def __init__(self, params, X, pis=None):
        self.params, self.X, self.pis = [p.detach() for p in params], X, pis
        self.B, self.n = X.shape[0], X.shape[1]
        f = lambda *ps: forward(list(ps), X)
        self.f = f
        self.y, self._vjp = vjp(f, *self.params)

    def out(self):
        return self.y

    def K(self, v):
        g = self._vjp(v)
        if self.pis is not None:
            g = tuple(a * p for a, p in zip(g, self.pis))
        return jvp(self.f, tuple(self.params), g)[1] / self.n

    def psi_scaled(self, b, chunk=100):
        """Explicit Psi / sqrt(n) of network b, (n, d). Checks only."""
        ps = [p[b:b + 1] for p in self.params]
        f1 = lambda *q: forward(list(q), self.X[b:b + 1])
        _, vf = vjp(f1, *ps)
        eye = torch.eye(self.n, dtype=self.X.dtype, device=self.X.device)
        rows = []
        for i in range(0, self.n, chunk):
            gs = vmap(lambda c: vf(c[None]))(eye[i:i + chunk])
            rows.append(torch.cat([g.reshape(g.shape[0], -1) for g in gs], 1))
        J = torch.cat(rows, 0)
        if self.pis is not None:
            J = J * torch.cat([p[b].reshape(-1) for p in self.pis]).sqrt()[None, :]
        return J / np.sqrt(self.n)


def train(cfg, X, Y, P, dev):
    """Batched full-batch training; returns error curves and {step: (params, pis)} at checkpoints."""
    ps = [p.clone().requires_grad_(True) for p in P]
    B, n, steps = X.shape[0], X.shape[1], max(cfg["ckpts"])
    if cfg["opt"] == "gd":
        if cfg["lr"]:
            lr = torch.full((B,), cfg["lr"], dtype=torch.float64, device=dev)
        else:
            lam1 = KM.lam1_krylov(Net(P, X).K, B, n, torch.float64, dev)
            lr = torch.tensor(cfg["lr_rel"] / lam1, dtype=torch.float64, device=dev)
            log(f"gd: lambda_1 at init {np.median(lam1):.4g} (median), lr {cfg['lr_rel']}/lambda_1")
        cfg["lr_used_median"] = float(lr.median())
    m = [torch.zeros_like(p) for p in ps]
    v = [torch.zeros_like(p) for p in ps]
    b1c, b2c, eps = 0.9, 0.999, 1e-8
    curve, saved, snaps = [], {}, {}
    for k in range(steps + 1):
        out = forward(ps, X)
        E = out - Y
        if k % cfg["log_every"] == 0:
            curve.append((k, E.detach().pow(2).mean(1).sqrt().cpu().numpy()))
        if k in cfg["snap_steps"]:
            snaps[k] = dict(params=[p.detach().cpu().clone() for p in ps], m=[x.cpu().clone() for x in m],
                            v=[x.cpu().clone() for x in v])
        if k in cfg["ckpts"]:
            pis = None
            if cfg["opt"] == "adam" and k > 0:
                pis = [1 / ((vi / (1 - b2c ** k)).sqrt() + eps) for vi in v]
            saved[k] = ([p.detach().clone() for p in ps], pis)
        if k == steps:
            break
        loss = 0.5 * E.pow(2).mean(1).sum()
        gr = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            if cfg["opt"] == "gd":
                for p, g in zip(ps, gr):
                    p -= lr.view(-1, *([1] * (p.dim() - 1))) * g
            else:
                a = cfg["lr"] * 0.5 * (1 + np.cos(np.pi * k / steps))
                for i, (p, g) in enumerate(zip(ps, gr)):
                    m[i].mul_(b1c).add_(g, alpha=1 - b1c)
                    v[i].mul_(b2c).addcmul_(g, g, value=1 - b2c)
                    p -= a * (m[i] / (1 - b1c ** (k + 1))) / ((v[i] / (1 - b2c ** (k + 1))).sqrt() + eps)
    torch.save(dict(cfg=cfg, ckpts={k: ([p.cpu() for p in ps_], None if pi is None else [x.cpu() for x in pi])
                                    for k, (ps_, pi) in saved.items()}),
               cfg["snap_path"].replace(".pt", "_ckpts.pt"))
    log(f"saved checkpoint parameters and preconditioners to {cfg['snap_path'].replace('.pt', '_ckpts.pt')}")
    if snaps:
        torch.save(dict(cfg=cfg, snaps=snaps), cfg["snap_path"])
        log(f"saved {len(snaps)} snapshots (parameters, Adam m and v before the update of that step) to {cfg['snap_path']}")
    return curve, saved


def analyse(cfg, X, Y, runs, saved, dev, out, exact_seeds):
    n, B = cfg["n"], len(runs)
    hp = {}
    for r, (seed, _) in enumerate(runs):
        if seed not in hp:
            Bm, _ = C.graph_highpass_basis(X[r, :, 0].cpu().numpy()[:, None], k=cfg["knn"], m=cfg["m_hp"], n_out=1)
            hp[seed] = torch.tensor(Bm, dtype=torch.float64, device=dev)
    Bs = torch.stack([hp[s] for s, _ in runs])                               # (B, n, m)
    Pr = lambda u: u - torch.einsum("bnm,bm->bn", Bs, torch.einsum("bnm,bn->bm", Bs, u))
    rows = []
    for k in sorted(saved):
        t0 = time.time()
        params, pis = saved[k]
        net = Net(params, X, pis)
        o = net.out().detach()
        Es = (o - Y) / np.sqrt(n)
        lam1 = KM.lam1_krylov(net.K, B, n, Es.dtype, dev)
        sig2 = KM.lam1_krylov(lambda w: Pr(net.K(Pr(w))), B, n, Es.dtype, dev, M=60)
        G, a, orth, nre = KM.lanczos(net.K, Es, max(MS))
        cg = {g: KM.cg_slow(net.K, Es, torch.tensor(g * lam1, dtype=Es.dtype, device=dev)) for g in GAMMAS}
        # the target U and the output in the coordinates of the kernel (Section 'The dataset in the
        # coordinates of the kernel'): slow parts U_gamma, Yhat_gamma, and the Krylov values of U
        Us, Os = Y / np.sqrt(n), o / np.sqrt(n)
        GU, aU, orthU, _ = KM.lanczos(net.K, Us, max(MS))
        cgU = {g: KM.cg_slow(net.K, Us, torch.tensor(g * lam1, dtype=Es.dtype, device=dev)) for g in GAMMAS}
        cgO = {g: KM.cg_slow(net.K, Os, torch.tensor(g * lam1, dtype=Es.dtype, device=dev)) for g in GAMMAS}
        RU =Pr(Y / np.sqrt(n)).norm(dim=1).cpu().numpy()
        RN = Pr(o / np.sqrt(n)).norm(dim=1).cpu().numpy()
        RE = Pr(Es).norm(dim=1).cpu().numpy()
        for r, (seed, name) in enumerate(runs):
            ex = None
            if seed in exact_seeds:
                A = net.psi_scaled(r)
                ex = KM.Exact(A)
                e_ex = ex.reduced(Es[r])
                u_ex = ex.reduced(Us[r])
                sig_ex = float(torch.linalg.matrix_norm(A - hp[seed] @ (hp[seed].T @ A), ord=2))
            sigma = float(np.sqrt(max(sig2[r], 0)))
            err = float(Es[r].norm())
            for g in GAMMAS:
                gam = g * lam1[r]
                slow, ups, cres, cits = (x[r] for x in cg[g])
                cs = float(np.sqrt(ups))
                FM = KM.krylov_F(G[r], a[r], cs, MS)
                row = dict(seed=seed, target=name, omega=(0 if name == "random" else float(name[3:])), step=k,
                           gamma_rel=g, err=err, RU=RU[r], RN=RN[r], RE=RE[r], sigma=sigma, lam1=lam1[r], cs=cs,
                           E_gamma=float(slow), cg_res=float(cres), cg_its=int(cits), floor=RU[r] - RN[r] - sigma * cs,
                           b_R=RE[r] - sigma * cs, orth=orth[r], restarts=int(nre[r]),
                           **{f"F_M{M}": val for M, val in zip(MS, FM)})
                uslow, uups, ucres, _ = (x[r] for x in cgU[g])
                ucs = float(np.sqrt(uups))
                row.update(U_norm=float(Us[r].norm()), U_gamma=float(uslow), U_cs=ucs, U_cg_res=float(ucres),
                           Yhat_gamma=float(cgO[g][0][r]), U_orth=orthU[r],
                           **{f"U_F_M{M}": val for M, val in zip(MS, KM.krylov_F(GU[r], aU[r], ucs, MS))})
                if ex is not None:
                    row.update(lam1_exact=ex.lam_ext[0], sigma_exact=sig_ex, F_exact=C.F_closed(ex.lam_ext, e_ex, cs)[0],
                               E_gamma_exact=float(np.linalg.norm(gam * e_ex / (ex.lam_ext + gam))),
                               U_gamma_exact=float(np.linalg.norm(gam * u_ex / (ex.lam_ext + gam))))
                rows.append(row)
        C.write_rows(os.path.join(out, "sine_rows.csv"), rows)
        log(f"checkpoint {k}: analysed {B} runs in {time.time() - t0:.0f}s")
    return rows


def checks(rows):
    tol = lambda q: 1e-9 * max(q["err"], 1e-12)
    c = dict(n=len(rows),
             FM_monotone=sum(all(q[f"F_M{a}"] <= q[f"F_M{b}"] + tol(q) for a, b in zip(MS, MS[1:])) for q in rows),
             FM_le_Egamma=sum(q[f"F_M{MS[-1]}"] <= q["E_gamma"] * (1 + 1e-6) + tol(q) for q in rows),
             floor_le_bR=sum(q["floor"] <= q["b_R"] + tol(q) for q in rows),
             max_cg_res=max(q["cg_res"] for q in rows), max_orth=max(q["orth"] for q in rows),
             UFM_le_Ugamma=sum(q[f"U_F_M{MS[-1]}"] <= q["U_gamma"] * (1 + 1e-6) + 1e-9 * q["U_norm"] for q in rows),
             Eq35=sum(q["E_gamma"] >= q["U_gamma"] - q["Yhat_gamma"] - 1e-9 * q["U_norm"] for q in rows),
             max_U_cg_res=max(q["U_cg_res"] for q in rows))
    ex = [q for q in rows if "F_exact" in q]
    if ex:
        c.update(n_exact=len(ex),
                 FM_le_Fexact=sum(q[f"F_M{MS[-1]}"] <= q["F_exact"] + tol(q) for q in ex),
                 max_rel_cg_vs_exact=max(abs(q["E_gamma"] - q["E_gamma_exact"]) / max(q["err"], 1e-300) for q in ex),
                 max_rel_Ucg_vs_exact=max(abs(q["U_gamma"] - q["U_gamma_exact"]) / q["U_norm"] for q in ex),
                 max_rel_lam1=max(abs(q["lam1"] / q["lam1_exact"] - 1) for q in ex),
                 max_rel_sigma=max(abs(q["sigma"] / q["sigma_exact"] - 1) for q in ex if q["sigma_exact"] > 0))
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder name under numerics/")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--opt", choices=["gd", "adam"], default="gd")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-rel", type=float, default=None)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--ckpts", default=None, help="comma-separated; default steps/100, steps/10, steps")
    ap.add_argument("--omegas", default="1,2,5,10,15,20,30,40,60,80,100,120,160,200,250")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--exact-seeds", type=int, default=1, help="exact checks for the first k seeds")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--snap-offsets", default="", help="e.g. 10,100,1000: also save parameters this many steps "
                    "after every checkpoint except the last (for the step-length study of Section 6.2)")
    ap.add_argument("--snap-dir", default=os.path.join(os.environ.get("SCRATCH", HERE), "pf_snapshots"))
    a = ap.parse_args()
    if a.opt == "gd" and a.lr is None and a.lr_rel is None:
        ap.error("gd needs --lr or --lr-rel")
    if a.opt == "adam" and a.lr is None:
        a.lr = 1e-3
    S = a.steps
    cfg = dict(depth=a.depth, width=a.width, n=a.n, opt=a.opt, lr=a.lr, lr_rel=a.lr_rel, steps=S,
               ckpts=[int(c) for c in a.ckpts.split(",")] if a.ckpts else [S // 100, S // 10, S],
               omegas=[int(w) for w in a.omegas.split(",")], random_labels=not a.no_random,
               seeds=list(range(a.seeds)), knn=8, m_hp=24, in_scale=IN_SCALE, log_every=a.log_every)
    offs = [int(o) for o in a.snap_offsets.split(",") if o]
    cfg["snap_steps"] = sorted({k + o for k in cfg["ckpts"][:-1] for o in [0] + offs if k + o <= S}) if offs else []
    os.makedirs(a.snap_dir, exist_ok=True)
    cfg["snap_path"] = os.path.join(a.snap_dir, f"{a.out}.pt")
    out = os.path.join(HERE, a.out)
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {dev} ({torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}), out {out}")
    log("config", cfg)
    X, Y, P, runs, names = setup(cfg, dev)
    log(f"training {len(runs)} runs, d = {sum(p[0].numel() for p in P)} parameters each")
    t0 = time.time()
    curve, saved = train(cfg, X, Y, P, dev)
    log(f"training done in {time.time() - t0:.0f}s; nan runs: {int(np.isnan(curve[-1][1]).sum())}")
    C.write_rows(os.path.join(out, "sine_curves.csv"),
                 [dict(seed=s, target=nm, step=k, err=float(e[r])) for k, e in curve for r, (s, nm) in enumerate(runs)])
    rows = analyse(cfg, X, Y, runs, saved, dev, out, exact_seeds=set(cfg["seeds"][:a.exact_seeds]))
    ck = checks(rows)
    log("checks", ck)
    json.dump(dict(cfg=cfg, MS=MS, GAMMAS=GAMMAS, checks=ck, seconds=time.time() - T0),
              open(os.path.join(out, "sine_meta.json"), "w"), indent=1, default=float)
    last = max(cfg["ckpts"])
    log(f"error by checkpoint (mean over seeds) | last checkpoint at gamma = 0.3 lambda_1: slow part, F_1, F_100, floor")
    for nm in names:
        e = [np.mean([q["err"] for q in rows if q["target"] == nm and q["step"] == k and q["gamma_rel"] == 0.3])
             for k in sorted(cfg["ckpts"])]
        q = [v for v in rows if v["target"] == nm and v["step"] == last and v["gamma_rel"] == 0.3]
        m = lambda key: np.mean([v[key] for v in q])
        log(f"  {nm:8} " + " ".join(f"{x:6.3f}" for x in e) +
            f" | {m('E_gamma'):6.3f} {m('F_M1'):6.3f} {m('F_M100'):6.3f} {m('floor'):6.3f}")
    log("ok")
