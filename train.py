"""Train the models of one (dt, protocol, init): both classes x all seeds at once (vmap over models).

  python train.py --dt 0.01 --proto onestep
  python train.py --dt 0.01 --proto unroll064 --init scratch|ft    (ft: start from the onestep run)

Adam with linear warmup and cosine decay, per-model gradient-norm clipping (each model is
optimized independently; Adam is elementwise so stacking does not couple them). Minibatches of
`batch` starts are drawn independently per model from the n_train training starts.
Writes <out_root>/runs/<tag>/{ckpts.pt, log.csv, meta.json}.
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch
from torch.func import jvp, vmap

from msr.config import load_cfg, run_dir
from msr.curvature import gn_lam_max
from msr.core import Arch, Model, loss_onestep, loss_unrolled, loss_weights, make_data, nsteps, stack_consts, sq, train_steps


def ckpt_steps(cfg):
    S = cfg["steps"]
    pts = np.unique(np.round(np.logspace(1, math.log10(S), cfg["n_ckpt_log"] + 1)[:-1]).astype(int))
    return sorted(set(pts.tolist()) | {S})


def lr_at(cfg, k):
    w, S, lr0, lr1 = cfg["warmup"], cfg["steps"], cfg["lr"], cfg["lr_end"]
    if k < w:
        return lr0 * (k + 1) / w
    x = (k - w) / max(S - w, 1)
    return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--proto", required=True)
    ap.add_argument("--init", default="scratch", choices=["scratch", "ft"])
    ap.add_argument("--suffix", default="", help="appended to the run directory (variants of a protocol)")
    ap.add_argument("--from-run", default=None, help="fine-tune from this run directory (default: the onestep run)")
    ap.add_argument("--measure-curv", action="store_true", help="record each model's loss curvature at the start (meta.json)")
    ap.add_argument("--lr-rule", default="none", choices=["none", "sqrt", "inv", "line"],
                    help="sqrt/inv: per-model peak lr = lr * min(1, (lam_anchor / lam_start)^alpha), alpha = 1/2 or 1; "
                         "line: Adam direction u, step length min(lr schedule, line_c g.u / |Psi u|_w^2) (no anchor)")
    ap.add_argument("--lr-anchor-run", default=None, help="run whose meta.json lam_start is the anchor (paired with lr)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=yaml_value")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    if a.init == "ft":                       # fine-tuning: shorter schedule, lower peak lr
        cfg["steps"], cfg["lr"] = cfg["steps_ft"], cfg["lr_ft"]
    dt, pr = a.dt, cfg["protocols"][a.proto]
    p = train_steps(pr, dt)
    unrolled = p is not None
    assert unrolled or a.init == "scratch"
    out = run_dir(cfg, dt, a.proto, a.init) + a.suffix
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    data = make_data(cfg, dt, dev)
    arch = Arch(cfg["hidden"], cfg["act"])
    names = [(c, s) for c in cfg["classes"] for s in cfg["seeds"]]
    models = [Model(arch, c, dt, data["stats"]) for c, _ in names]
    C = stack_consts([mm.c for mm in models])
    if a.init == "scratch":
        TH = torch.stack([arch.init(s, dev) for _, s in names])
    else:
        srcdir = os.path.join(cfg["out_root"], "runs", a.from_run) if a.from_run else run_dir(cfg, dt, "onestep", "scratch")
        src = torch.load(os.path.join(srcdir, "ckpts.pt"), map_location=dev)
        assert abs(src["dt"] - dt) < 1e-12
        assert src["names"] == names
        for k, v in src["stats"].items():         # same normalization as the model being fine-tuned
            assert torch.allclose(v.cpu(), data["stats"][k].cpu(), rtol=1e-12, atol=0), k
        TH = src["theta"][max(src["theta"])].clone()
    M, B = len(names), cfg["batch"]
    uTr, uTe = data["uT_train"], data["uT_test"]
    ntr = uTr.shape[1]
    pwin = nsteps(cfg["onestep_window"], dt)
    pev = nsteps(cfg.get("eval_T", 0.64), dt)
    gen = torch.Generator(device=dev).manual_seed(int(round(1e6 * dt)) + zlib.crc32((a.proto + a.init + a.suffix).encode()))

    f_one = vmap(lambda th, c, X, Y: loss_onestep(arch, th, c, X, Y))
    w = loss_weights(pr.get("gamma", 0.0), dt, p) if unrolled else None
    f_unr = vmap(lambda th, c, U: loss_unrolled(arch, th, c, U, p, w))

    # optional: curvature of this run's own loss at its start, and per-model learning rates scaled by it
    lam0, scale = None, np.ones(M)
    if unrolled and (a.measure_curv or a.lr_rule != "none"):
        uc = uTe[:, :cfg.get("n_curv", 512)]
        lam0 = [gn_lam_max(models[j], TH[j], uc, p, w) for j in range(M)]
        print(f"[{a.proto}/{a.init} dt={dt}] curvature at start: " + " ".join(f"{x:.3g}" for x in lam0), flush=True)
    if a.lr_rule in ("sqrt", "inv"):
        anc = json.load(open(os.path.join(cfg["out_root"], "runs", a.lr_anchor_run, "meta.json")))
        assert anc["names"] == [list(nm) for nm in names] and anc.get("lam_start")
        alpha = dict(sqrt=0.5, inv=1.0)[a.lr_rule]
        scale = np.minimum(1.0, (np.array(anc["lam_start"]) / np.array(lam0)) ** alpha)   # never above the anchor lr
        print(f"[{a.proto}/{a.init} dt={dt}] lr rule {a.lr_rule}: peak lr per model "
              + " ".join(f"{cfg['lr'] * s:.3g}" for s in scale), flush=True)
    line = a.lr_rule == "line"
    per_model = lam0 is not None and not line
    if line:
        # Adam's direction, step length from the local quadratic model of the run's own loss along it:
        #   alpha = min(lr schedule, line_c * g.u / u^T G u),  u^T G u = |Psi u|_w^2 (one JVP of the weighted rollout)
        assert unrolled, "the line rule is implemented for unrolled losses"
        TH.requires_grad_(True)
        theta = lambda: TH
        mom, vel = torch.zeros_like(TH), torch.zeros_like(TH)
        b1, b2, eps = 0.9, 0.999, 1e-8                       # torch.optim.Adam defaults
        Wq = torch.ones(p, device=dev) if w is None else torch.tensor(w, device=dev)
        swq = torch.sqrt(Wq / Wq.sum())

        def uGu(th, c, U, u):
            f = lambda t: (swq[:, None, None] * _rollseq(arch, t, c, U[0], p)).reshape(-1) / math.sqrt(U.shape[1])
            return (jvp(f, (th,), (u,))[1] ** 2).sum()

        curv_dir = vmap(uGu)
        ratio = torch.zeros(M, device=dev)                   # line_c g.u / u^T G u, refreshed every line_every steps
        alpha = torch.zeros(M, device=dev)
    elif per_model:        # one Adam parameter group per model (Adam is elementwise: same math as stacked)
        params = [TH[j].detach().clone().requires_grad_(True) for j in range(M)]
        opt = torch.optim.Adam([{"params": [q], "lr": cfg["lr"] * scale[j]} for j, q in enumerate(params)], lr=cfg["lr"])
        theta = lambda: torch.stack(params)
    else:
        TH.requires_grad_(True)
        opt = torch.optim.Adam([TH], lr=cfg["lr"])
        theta = lambda: TH

    @torch.no_grad()
    def evaluate():
        """Held-out one-step loss over the training window and rollout error |e(T)| at the
        longest training horizon, per model (first n_grad held-out starts); for unrolled runs
        also the run's own loss on those starts."""
        TH_ = theta()
        U = uTe[:, :cfg["n_grad"]]
        X, Y = U[:pwin].reshape(-1, 3), U[1:pwin + 1].reshape(-1, 3)
        one = vmap(lambda th, c: loss_onestep(arch, th, c, X, Y))(TH_, C)
        e = vmap(lambda th, c: _roll(arch, th, c, U[0], pev))(TH_, C)
        own = vmap(lambda th, c: loss_unrolled(arch, th, c, U[:p + 1], p, w))(TH_, C) if unrolled else None
        return one, torch.sqrt(sq(U[pev][None] - e)), own

    cks = set(ckpt_steps(cfg))
    saved, log = {0: theta().detach().clone()}, []

    def save():
        torch.save(dict(names=names, theta=saved, stats={k: v.cpu() for k, v in data["stats"].items()},
                        cfg=cfg, dt=dt, proto=a.proto, init=a.init), os.path.join(out, "ckpts.pt"))

    for k in range(cfg["steps"] + 1):
        if k in cks:
            saved[k] = theta().detach().clone()
            save()
        if k % cfg["log_every"] == 0 or k == cfg["steps"]:
            one, eT, own = evaluate()
            log.append([k, lr_at(cfg, min(k, cfg["steps"] - 1))] + one.tolist() + eT.tolist()
                       + ([] if own is None else own.tolist()) + (alpha.tolist() if line else []))
            if k % (20 * cfg["log_every"]) == 0:
                print(f"[{a.proto}/{a.init} dt={dt}] step {k} {time.time() - t0:.0f}s  heldout one-step "
                      + " ".join(f"{x:.2e}" for x in one.tolist()), flush=True)
        if k == cfg["steps"]:
            break
        idx = torch.randint(ntr, (M, B), generator=gen, device=dev)
        if unrolled:
            U = uTr[:p + 1, idx].permute(1, 0, 2, 3)            # (M, p+1, B, 3)
            losses = f_unr(theta(), C, U)
        else:
            i = torch.randint(pwin, (M, B), generator=gen, device=dev)
            X, Y = uTr[i, idx], uTr[i + 1, idx]
            losses = f_one(theta(), C, X, Y)
        if line:
            TH.grad = None
            losses.sum().backward()
            g = TH.grad.detach()
            gc = g * torch.clamp(cfg["clip"] / (torch.linalg.norm(g, dim=1) + 1e-300), max=1.0)[:, None]
            n_t = k + 1
            mom.mul_(b1).add_(gc, alpha=1 - b1)
            vel.mul_(b2).addcmul_(gc, gc, value=1 - b2)
            u = (mom / (1 - b1 ** n_t)) / (vel.sqrt() / math.sqrt(1 - b2 ** n_t) + eps)
            if k % cfg.get("line_every", 1) == 0:
                quad = curv_dir(TH.detach(), C, U, u).clamp_min(1e-300)
                gu = (g * u).sum(1)
                ratio = torch.where(gu > 0, cfg.get("line_c", 0.25) * gu / quad, ratio)   # keep the last one if g.u <= 0
            alpha = torch.clamp(ratio, max=lr_at(cfg, k))
            with torch.no_grad():
                TH -= alpha[:, None] * u
            continue
        for j, g in enumerate(opt.param_groups):
            g["lr"] = lr_at(cfg, k) * (scale[j] if per_model else 1.0)
        opt.zero_grad()
        losses.sum().backward()
        if per_model:
            for q in params:
                q.grad *= min(1.0, cfg["clip"] / (float(torch.linalg.norm(q.grad)) + 1e-300))
        else:
            gn = torch.linalg.norm(TH.grad, dim=1)
            TH.grad *= torch.clamp(cfg["clip"] / (gn + 1e-300), max=1.0)[:, None]
        opt.step()

    save()
    hdr = (["step", "lr"] + [f"onestep_{c}{s}" for c, s in names] + [f"eT_{c}{s}" for c, s in names]
           + ([f"own_{c}{s}" for c, s in names] if unrolled else [])
           + ([f"alpha_{c}{s}" for c, s in names] if line else []))
    np.savetxt(os.path.join(out, "log.csv"), np.array(log), delimiter=",", header=",".join(hdr), comments="")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(dict(dt=dt, proto=a.proto, init=a.init, suffix=a.suffix, from_run=a.from_run, gamma=pr.get("gamma", 0.0), lr=cfg["lr"],
                       lam_start=lam0, lr_rule=a.lr_rule, lr_anchor_run=a.lr_anchor_run,
                       line_c=cfg.get("line_c", 0.25) if line else None, line_every=cfg.get("line_every", 1) if line else None,
                       lr_models=[float(cfg["lr"] * s) for s in scale], p=p, names=names, wall_s=time.time() - t0,
                       device=dev, P=arch.P, ckpts=sorted(saved)), fh, indent=1)
    print(f"[{a.proto}/{a.init} dt={dt}] done {time.time() - t0:.0f}s", flush=True)


def _rollseq(arch, th, c, u0, p):
    """Predicted states u_p(t_1..t_p), shape (p, n, 3)."""
    from msr.core import gt
    out, u = [], u0
    for _ in range(p):
        u = gt(arch, th, u, c)
        out.append(u)
    return torch.stack(out)


def _roll(arch, th, c, u0, p):
    from msr.core import gt
    u = u0
    for _ in range(p):
        u = gt(arch, th, u, c)
    return u


if __name__ == "__main__":
    main()
