"""Train one KS stepper (Euler-class MLP) with the one-step, free or kappa-nudged window loss; Adam, cosine lr.

  python ks/train.py --mode onestep --tag onestep_s0
  python ks/train.py --mode kappa --kappa 0.5 --p 20 --tag k0.5_s0
  python ks/train.py --mode free --p 20 --init <run>/final.pt      (fine-tune from a saved model)

Writes <out_root>/ks/<tag>/{log.csv, final.pt, meta.json}; final.pt = dict(theta, stats, cfg, args).
Held-out evaluation: free rollout rms and median over [0, T_eval] and at the long horizons.
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ks.config import load_cfg
from ks.core import Arch, Model, eval_free, loss_kappa, make_data


def lr_at(cfg, k, lr0):
    w, S, lr1 = cfg["warmup"], cfg["steps"], cfg["lr_end"]
    if k < w:
        return lr0 * (k + 1) / w
    x = (k - w) / max(S - w, 1)
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--mode", required=True, choices=["onestep", "free", "kappa", "online"])
    ap.add_argument("--kgrid", default="1,0.5,0.2,0.05,0")
    ap.add_argument("--window", type=int, default=250, help="online: steps between decisions")
    ap.add_argument("--margin", type=float, default=0.8)
    ap.add_argument("--sharp-iters", type=int, default=10)
    ap.add_argument("--n-rate", type=int, default=None, help="online: starts of the rate batch")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--p", type=int, default=None, help="window steps (default cfg p_train; onestep: 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--init", default=None, help="final.pt to start from (fine-tuning)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    if a.steps:
        cfg["steps"] = a.steps
    lr0 = a.lr or cfg["lr"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    out = os.path.join(cfg["out_root"], "ks", a.tag)
    os.makedirs(out, exist_ok=True)

    p = 1 if a.mode == "onestep" else (a.p or cfg["p_train"])
    online = a.mode == "online"
    kap = {"onestep": 1.0, "free": None, "kappa": a.kappa, "online": 1.0}[a.mode]
    if a.mode == "kappa":
        assert kap is not None
    t0 = time.time()
    data = make_data(cfg, dev, horizon=max(cfg["long_horizons"]))
    dt = data["dt"]
    arch = Arch(cfg["N"], cfg["hidden"], cfg["act"])
    model = Model(arch, dt, data["stats"])
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        th = ck["theta"].to(dev).clone().requires_grad_(True)
        model = Model(arch, dt, ck["stats"])
    else:
        th = arch.init(a.seed, dev).requires_grad_(True)
    opt = torch.optim.Adam([th], lr=lr0)
    uTr, uTe = data["uT_train"], data["uT_test"]
    ntr, ptr = uTr.shape[1], uTr.shape[0] - 1
    gen = torch.Generator(device=dev).manual_seed(1000 * a.seed + 7)
    p_eval = int(round(cfg["T_eval"] / dt))
    Uev = uTe[:p_eval + 1, :cfg["n_eval"]]
    print(f"data {tuple(uTr.shape)} test {tuple(uTe.shape)} P={arch.P} dev={dev} ({time.time() - t0:.0f}s); "
          f"mode={a.mode} p={p} kap={kap} lr={lr0} steps={cfg['steps']}", flush=True)

    ok = None
    if online:
        from online import OnlineKappa
        from ks.core import rollout_kappa
        n_rate = a.n_rate or 256
        U_rate = uTr[:p + 1, :n_rate].to(dev)
        U_hold = uTe[:p + 1, :cfg["n_eval"]].to(dev)
        ok = OnlineKappa(lambda t, U, pp, kk: loss_kappa(model, t, U, pp, kk), lambda t, U, pp, kk: rollout_kappa(model, t, U, pp, kk),
                         th, opt, p, [float(x) for x in a.kgrid.split(",")], a.margin, a.sharp_iters, cfg["clip"], os.path.join(out, "online.jsonl"), dev)
        if os.path.exists(os.path.join(out, "online.jsonl")):
            os.remove(os.path.join(out, "online.jsonl"))

    def log_eval():
        E = eval_free(model, th.detach(), Uev, p_eval)
        per = torch.sqrt((E ** 2).mean(0))
        return float(torch.sqrt((E ** 2).mean())), float(per.median())

    log = []
    for k in range(cfg["steps"]):
        lr = lr_at(cfg, k, lr0)
        for gr in opt.param_groups:
            gr["lr"] = lr
        idx = torch.randint(0, ntr, (cfg["batch"],), device=dev, generator=gen)
        if p == 1:
            q0 = torch.randint(0, ptr, (1,), device=dev, generator=gen).item()
            U = uTr[q0:q0 + 2, idx]
        else:
            q0 = torch.randint(0, ptr - p + 1, (1,), device=dev, generator=gen).item()
            U = uTr[q0:q0 + p + 1, idx]
        if online and k % a.window == 0:
            with torch.no_grad():
                Vh = sum(float(loss_kappa(model, th.detach(), U_hold[:, i:i + 32], p, 0.0)) for i in range(0, U_hold.shape[1], 32)) / max(1, (U_hold.shape[1] + 31) // 32)
            kap = ok.choose(U_rate, lr, Vh, k)
            kap = None if kap == 0.0 else kap
        L = loss_kappa(model, th, U, p, kap)
        opt.zero_grad(set_to_none=True)
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_([th], cfg["clip"])
        opt.step()
        if k % cfg["log_every"] == 0 or k == cfg["steps"] - 1:
            r, m = log_eval()
            log.append([k, lr, float(L.detach()), float(gn), r, m, (0.0 if kap is None else float(kap))])
            print(f"step {k:6d} lr {lr:.2e} loss {float(L):.3e} |g| {float(gn):.2e} free rms[0,T] {r:.4f} median {m:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if not math.isfinite(float(L)):
                print("non-finite loss, stopping", flush=True)
                break
    thd = th.detach()
    with open(os.path.join(out, "log.csv"), "w") as fh:
        fh.write("step,lr,loss,gradnorm,free_rms_0T,free_median_0T,kappa\n")
        for row in log:
            fh.write(",".join(f"{v:.6e}" for v in row) + "\n")
    torch.save(dict(theta=thd.cpu(), stats={k: v.cpu() for k, v in model.c.items()}, cfg=cfg, args=vars(a)),
               os.path.join(out, "final.pt"))
    final = {}
    with torch.no_grad():
        E = eval_free(model, thd, uTe[:, :cfg["n_eval"]], uTe.shape[0] - 1)       # (P, n)
        per = torch.sqrt((E[:p_eval] ** 2).mean(0))
        final["rms_0T"], final["median_0T"], final["q90_0T"] = float(torch.sqrt((E[:p_eval] ** 2).mean())), float(per.median()), float(per.quantile(0.9))
        final["rms_at"], final["median_at"] = {}, {}
        for T in cfg["long_horizons"]:
            q = int(round(T / dt)) - 1
            final["rms_at"][str(T)] = float(torch.sqrt((E[q] ** 2).mean()))
            final["median_at"][str(T)] = float(E[q].median())
    meta = dict(mode=a.mode, kappa=kap, p=p, seed=a.seed, steps=cfg["steps"], lr=lr0, init=a.init, wall_s=time.time() - t0, final=final)
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(final, indent=1), flush=True)


if __name__ == "__main__":
    main()
