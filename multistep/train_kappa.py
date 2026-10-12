"""One model (Euler class, one seed), data-parallel over the GPUs of one node, trained with one of three rollout losses
over a window of p = T/dt steps:

  onestep : kappa = 1  (every step starts from the true state; the one-step loss on the window's states)
  free    : kappa = 0  (naive multistep: the free rollout loss, which is the objective V_T itself)
  adaptive: per-start, per-step kappa_{k,i} chosen before every Adam step by projected ascent on the exact derivative
            dD/dkappa_{k,i} = < grad V_T(theta+), Pi d(grad L_kappa)/dkappa_{k,i} >,
            D = V_T(theta) - V_T(theta+), theta+ = theta - Pi grad L_kappa (lookahead with Adam's current preconditioner
            theta+ is exactly the Adam step the optimizer takes, with clipping and momentum; in the derivative the clip
            factor and Adam's second moment are held fixed, so d theta+ = -Pi_eff d grad L).

Nudged rollout: x_i = kappa_i uT_i + (1 - kappa_i) u_i, u_{i+1} = Gt(x_i); the gradient flows through (1 - kappa_i) u_i.
Objective V_T = (1/p) sum_{q=1..p} (1/2)|e(t_q)|^2 of the free rollout (average error over [0, T]).

  torchrun --standalone --nproc-per-node=4 train_kappa.py --mode adaptive --dt 0.02 --T 0.64
Writes <out_root>/kappa/<tag>/{log.csv, kappa_samples.pt, final.pt, meta.json}.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from msr.config import load_cfg
from msr.core import Arch, Model, gt, make_data, nsteps, sq


def lr_at(cfg, k):
    w, S, lr0, lr1 = cfg["warmup"], cfg["steps"], cfg["lr"], cfg["lr_end"]
    if k < w:
        return lr0 * (k + 1) / w
    x = (k - w) / max(S - w, 1)
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * x))


def rollout_kappa(arch, th, c, uT, p, kap):
    """Rows 1..p of the nudged rollout, shape (p, n, 3). kap: None (free) or (p, n) with kap[i] the weight at step i."""
    out, u = [], uT[0]
    for q in range(1, p + 1):
        if kap is not None and q > 1:
            k = kap[q - 1][:, None]
            u = k * uT[q - 1] + (1 - k) * u
        u = gt(arch, th, u, c)
        out.append(u)
    return torch.stack(out)


def loss_kappa(arch, th, c, uT, p, kap):
    seq = rollout_kappa(arch, th, c, uT, p, kap)
    return 0.5 * sq(uT[1:p + 1] - seq).mean()


def allmean(t, ws):
    if ws > 1:
        dist.all_reduce(t)
        t /= ws
    return t




def adam_lookahead(opt, th, lr, gbar, clip):
    """theta+ of the Adam step the optimizer will take from the global gradient gbar (with norm clipping and momentum),
    and the diagonal factor Pi_eff with d theta+ = -Pi_eff * d gbar when the clip factor and v are held fixed."""
    b1, b2 = opt.defaults["betas"]
    eps = opt.defaults["eps"]
    cf = min(1.0, clip / (float(torch.linalg.norm(gbar)) + 1e-300))
    gc = cf * gbar
    st = opt.state.get(th, {})
    n = int(st["step"]) + 1 if "step" in st else 1
    m = st["exp_avg"] if "exp_avg" in st else torch.zeros_like(gbar)
    v = st["exp_avg_sq"] if "exp_avg_sq" in st else torch.zeros_like(gbar)
    m1 = b1 * m + (1 - b1) * gc
    v1 = b2 * v + (1 - b2) * gc * gc
    den = (v1 / (1 - b2 ** n)).sqrt() + eps
    upd = lr * (m1 / (1 - b1 ** n)) / den
    return th.detach() - upd, lr * (1 - b1) * cf / ((1 - b1 ** n) * den)


def kappa_step(arch, th, c, uT, p, look, ws, iters, eta, kap0=None):
    """Projected ascent on dD/dkappa for this rank's starts; look(gbar) -> (theta+, Pi_eff) is the step the optimizer
    takes. Returns kappa (p, n) and the last hypergradient."""
    n = uT.shape[1]
    kap = torch.zeros(p, n, device=uT.device) if kap0 is None else kap0.clone()
    h = torch.zeros_like(kap)
    for _ in range(iters):
        kap.requires_grad_(True)
        L = loss_kappa(arch, th, c, uT, p, kap)
        g = torch.autograd.grad(L, th, create_graph=True)[0]
        gbar = allmean(g.detach().clone(), ws)                       # global training gradient (the step uses it)
        thp, Pi = look(gbar)
        thp = thp.requires_grad_(True)                               # lookahead theta+
        V = loss_kappa(arch, thp, c, uT, p, None)                    # objective on this rank's starts
        gV = allmean(torch.autograd.grad(V, thp)[0], ws)
        # dD/dkappa_local = < grad V(theta+), Pi d gbar / dkappa_local >, gbar = mean over ranks
        h = torch.autograd.grad((g * (Pi * gV)).sum() / ws, kap)[0]
        h[0] = 0.0                                                   # kappa_0 acts on a zero deviation
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)     # guard: a non-finite lookahead leaves kappa unchanged
        with torch.no_grad():
            m = h.abs().max()
            m = allmean(m.clone(), 1) if ws == 1 else _allmax(m)
            kap = (kap.detach() + eta * h / (m + 1e-300)).clamp_(0.0, 1.0)
    return kap.detach(), h.detach()


def _allmax(t):
    t = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


@torch.no_grad()
def eval_free(arch, th, c, uT, p):
    """Free rollout from uT[0] over uT.shape[0]-1 steps; returns per-start errors |e(t_q)|, shape (P, n)."""
    u, errs = uT[0], []
    for q in range(1, uT.shape[0]):
        u = gt(arch, th, u, c)
        errs.append(torch.sqrt(((uT[q] - u) ** 2).sum(-1)))
    return torch.stack(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--mode", required=True, choices=["onestep", "free", "adaptive", "schedule"])
    ap.add_argument("--kgrid", default="1,0.5,0.2,0.1,0.05,0", help="schedule: kappa levels, from one-step down to free")
    ap.add_argument("--check-every", type=int, default=250, help="schedule: steps between stability checks")
    ap.add_argument("--margin", type=float, default=0.8,
                    help="schedule: a level is stable if lr * (Adam-preconditioned sharpness) < margin * (2 + 2 b1)/(1 - b1)")
    ap.add_argument("--sharp-iters", type=int, default=20, help="schedule: power iterations per sharpness estimate")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--T", type=float, default=0.64, help="window = objective horizon")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kiters", type=int, default=2, help="adaptive: ascent iterations on kappa per training step")
    ap.add_argument("--keta", type=float, default=0.5, help="adaptive: largest kappa change per iteration")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop cleanly after this wall time (saves everything)")
    ap.add_argument("--save-every", type=int, default=0, help="save parameters + Adam state every this many steps (0: off)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if ws > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    dev = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        torch.cuda.set_device(dev)
    torch.set_num_threads(cfg["threads"])

    dt, p = a.dt, nsteps(a.T, a.dt)
    tag = a.tag or f"dt{dt}_T{a.T}_{a.mode}_s{a.seed}"
    out = os.path.join(cfg["out_root"], "kappa", tag)
    if rank == 0:
        os.makedirs(out, exist_ok=True)
    long_T = max(cfg.get("long_horizons", [2.56])[:4])
    data = make_data(cfg, dt, dev, horizon=max(long_T, a.T))
    arch = Arch(cfg["hidden"], cfg["act"])
    model = Model(arch, "euler", dt, data["stats"])
    c = model.c
    th = arch.init(a.seed, dev).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=cfg["lr"])
    uTr, uTe = data["uT_train"], data["uT_test"]
    ntr = uTr.shape[1]
    Bl = cfg["batch"] // ws
    gen = torch.Generator(device=dev).manual_seed(1000 * a.seed + 17 * rank + {"onestep": 1, "free": 2, "adaptive": 3, "schedule": 4}[a.mode])
    Uev = uTe[:p + 1, :cfg["n_grad"]]

    def log_eval(k):
        E = eval_free(arch, th.detach(), c, Uev, p)                  # (p, n)
        per_start = torch.sqrt((E ** 2).mean(0))                     # rms over [0, T] per start
        return [float(torch.sqrt((E ** 2).mean())), float(per_start.median())]

    t0, log, samples = time.time(), [], []
    stop_at = None if a.max_minutes is None else t0 + 60 * a.max_minutes
    k_done, plen = 0, 0.0

    def save_ckpt(step, lr):
        """Parameters and Adam state after `step` training steps (for the sharpness checks)."""
        st = opt.state.get(th, {})
        os.makedirs(os.path.join(out, "ckpts"), exist_ok=True)
        torch.save(dict(step=step, lr=lr, path_length=plen, theta=th.detach().cpu(),
                        exp_avg=st["exp_avg"].cpu() if "exp_avg" in st else None,
                        exp_avg_sq=st["exp_avg_sq"].cpu() if "exp_avg_sq" in st else None,
                        adam_step=int(st["step"]) if "step" in st else 0,
                        betas=opt.defaults["betas"], eps=opt.defaults["eps"]),
                   os.path.join(out, "ckpts", f"step{step:06d}.pt"))

    if rank == 0 and a.save_every:
        save_ckpt(0, lr_at(cfg, 0))

    kgrid = [float(x) for x in a.kgrid.split(",")]
    kcur = kgrid[0]
    b1 = opt.defaults["betas"][0]
    thr = a.margin * (2 + 2 * b1) / (1 - b1)

    def pre_sharp(U, kval):
        """Top eigenvalue of P^{-1/2} Hess L_kval P^{-1/2} on this batch, P = sqrt(v_hat) + eps (Adam's current state)."""
        st = opt.state[th]
        vhat = st["exp_avg_sq"] / (1 - opt.defaults["betas"][1] ** int(st["step"]))
        s = (1.0 / (vhat.sqrt() + opt.defaults["eps"])).sqrt()
        K = torch.full((p, U.shape[1]), kval, device=dev)

        def mv(v):
            x = th.detach().clone().requires_grad_(True)
            gr = torch.autograd.grad(loss_kappa(arch, x, c, U, p, K), x, create_graph=True)[0]
            return s * torch.autograd.grad(gr @ (s * v), x)[0]

        def power(f):
            v = torch.randn(th.numel(), device=dev, generator=torch.Generator(device=dev).manual_seed(0)); v /= v.norm()
            lam = 0.0
            for _ in range(a.sharp_iters):
                w = f(v); lam = float(v @ w); v = w / (w.norm() + 1e-300)
            return lam
        lam1 = power(mv)
        return lam1 if lam1 >= 0 else power(lambda v: mv(v) - lam1 * v) + lam1

    def schedule_check(kc, U, lr):
        """Move one level toward free if the next level is stable for Adam at this lr; one level back toward one-step if
        the current level is not. Decided on rank 0 (its batch) and broadcast."""
        i = kgrid.index(kc)
        new = torch.tensor([kc], device=dev)
        if rank == 0 and opt.state.get(th, {}).get("step", 0):
            if i > 0 and lr * pre_sharp(U, kgrid[i]) > thr:
                new[0] = kgrid[i - 1]
            elif i + 1 < len(kgrid) and lr * pre_sharp(U, kgrid[i + 1]) < thr:
                new[0] = kgrid[i + 1]
        if ws > 1:
            dist.broadcast(new, 0)
        return float(new.item())
    for k in range(cfg["steps"]):
        lr = lr_at(cfg, k)
        idx = torch.randint(ntr, (Bl,), generator=gen, device=dev)
        U = uTr[:p + 1, idx]
        if a.mode == "adaptive":
            kap, h = kappa_step(arch, th, c, U, p, lambda gb: adam_lookahead(opt, th, lr, gb, cfg["clip"]), ws, a.kiters, a.keta)
            if rank == 0 and k % cfg["log_every"] == 0:              # drift and one-step error at the current theta
                with torch.no_grad():
                    free = rollout_kappa(arch, th.detach(), c, U, p, None)
                    drift = torch.sqrt(((U[1:p] - free[:p - 1]) ** 2).sum(-1))                      # |e(t_i)|, i = 1..p-1
                    epsn = torch.sqrt(((U[2:p + 1] - gt(arch, th.detach(), U[1:p], c)) ** 2).sum(-1))  # |eps_i|
        elif a.mode == "onestep":
            kap = torch.ones(p, Bl, device=dev)
        elif a.mode == "schedule":
            if k > 0 and k % a.check_every == 0:
                kcur = schedule_check(kcur, U, lr)
            kap = torch.full((p, Bl), kcur, device=dev)
        else:
            kap = None
        L = loss_kappa(arch, th, c, U, p, kap)
        opt.zero_grad()
        L.backward()
        allmean(th.grad, ws)
        th.grad *= min(1.0, cfg["clip"] / (float(torch.linalg.norm(th.grad)) + 1e-300))
        for gr in opt.param_groups:
            gr["lr"] = lr
        prev = th.detach().clone()
        opt.step()
        plen += float(torch.linalg.norm(th.detach() - prev))           # Euclidean parameter path length
        k_done = k + 1
        if rank == 0 and a.save_every and k_done % a.save_every == 0:
            save_ckpt(k_done, lr)

        if rank == 0 and (k % cfg["log_every"] == 0 or k == cfg["steps"] - 1):
            row = [k, lr, float(L)] + log_eval(k)
            if a.mode == "schedule":
                row += [kcur]
            if a.mode == "adaptive":
                with torch.no_grad():
                    kk = kap[1:p]
                    row += [float(kk.mean()), float((kk > 0.5).double().mean())]
                    if k % (10 * cfg["log_every"]) == 0:
                        m = min(256, Bl)
                        samples.append(dict(step=k, kappa=kk[:, :m].cpu(), drift=drift[:, :m].cpu(), eps=epsn[:, :m].cpu(),
                                            h=h[1:p, :m].cpu()))
            log.append(row)
            if k % (20 * cfg["log_every"]) == 0:
                print(f"[{tag}] step {k} {time.time() - t0:.0f}s loss {float(L):.3e} heldout rms[0,T] {row[3]:.3e} "
                      f"median {row[4]:.3e}" + (f" kappa mean {row[5]:.3f}" if a.mode == "adaptive" else ""), flush=True)
        if stop_at is not None:
            flag = torch.tensor([1.0 if time.time() > stop_at else 0.0], device=dev)
            if ws > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                if rank == 0:
                    print(f"[{tag}] wall-time limit reached at step {k_done}", flush=True)
                break

    if rank == 0:
        thd = th.detach()
        Ufin = uTe[:nsteps(long_T, dt) + 1]
        E = eval_free(arch, thd, c, Ufin, p)                           # (P_long, n_test)
        per_start = torch.sqrt((E[:p] ** 2).mean(0))
        res = dict(rms_0T=float(torch.sqrt((E[:p] ** 2).mean())), median_0T=float(per_start.median()),
                   q90_0T=float(per_start.quantile(0.9)),
                   rms_at={f"{h:g}": float(torch.sqrt((E[nsteps(h, dt) - 1] ** 2).mean())) for h in cfg["long_horizons"] if h <= long_T},
                   median_at={f"{h:g}": float(E[nsteps(h, dt) - 1].median()) for h in cfg["long_horizons"] if h <= long_T})
        hdr = (["step", "lr", "loss", "heldout_rms_0T", "heldout_median_0T"] + (["kappa_mean", "kappa_frac_gt_half"] if a.mode == "adaptive" else [])
               + (["kappa"] if a.mode == "schedule" else []))
        np.savetxt(os.path.join(out, "log.csv"), np.array(log), delimiter=",", header=",".join(hdr), comments="")
        torch.save(dict(theta=thd.cpu(), stats={kk: v.cpu() for kk, v in data["stats"].items()}, cfg=cfg), os.path.join(out, "final.pt"))
        if samples:
            torch.save(samples, os.path.join(out, "kappa_samples.pt"))
        json.dump(dict(mode=a.mode, dt=dt, T=a.T, p=p, seed=a.seed, steps_done=k_done, steps_cfg=cfg["steps"], world=ws,
                       kiters=a.kiters, keta=a.keta, wall_s=time.time() - t0, final=res), open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"[{tag}] done {time.time() - t0:.0f}s steps {k_done} final {json.dumps(res)}", flush=True)
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
