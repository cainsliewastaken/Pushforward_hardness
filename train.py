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
from torch.func import vmap

from msr.config import load_cfg, run_dir
from msr.core import Arch, Model, loss_onestep, loss_unrolled, make_data, nsteps, stack_consts, sq, train_steps


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
    out = run_dir(cfg, dt, a.proto, a.init)
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
        src = torch.load(os.path.join(run_dir(cfg, dt, "onestep", "scratch"), "ckpts.pt"), map_location=dev)
        assert src["names"] == names
        for k, v in src["stats"].items():         # same normalization as the model being fine-tuned
            assert torch.allclose(v.cpu(), data["stats"][k].cpu(), rtol=1e-12, atol=0), k
        TH = src["theta"][max(src["theta"])].clone()
    TH.requires_grad_(True)
    opt = torch.optim.Adam([TH], lr=cfg["lr"])
    M, B = len(names), cfg["batch"]
    uTr, uTe = data["uT_train"], data["uT_test"]
    ntr = uTr.shape[1]
    pwin = nsteps(cfg["onestep_window"], dt)
    pev = nsteps(cfg.get("eval_T", 0.64), dt)
    gen = torch.Generator(device=dev).manual_seed(int(round(1e6 * dt)) + zlib.crc32((a.proto + a.init).encode()))

    f_one = vmap(lambda th, c, X, Y: loss_onestep(arch, th, c, X, Y))
    f_unr = vmap(lambda th, c, U: loss_unrolled(arch, th, c, U, p))

    @torch.no_grad()
    def evaluate():
        """Held-out one-step loss over the training window and rollout error |e(T)| at the
        longest training horizon, per model (first n_grad held-out starts)."""
        U = uTe[:, :cfg["n_grad"]]
        X, Y = U[:pwin].reshape(-1, 3), U[1:pwin + 1].reshape(-1, 3)
        one = vmap(lambda th, c: loss_onestep(arch, th, c, X, Y))(TH, C)
        e = vmap(lambda th, c: _roll(arch, th, c, U[0], pev))(TH, C)
        return one, torch.sqrt(sq(U[pev][None] - e))

    cks = set(ckpt_steps(cfg))
    saved, log = {0: TH.detach().clone()}, []

    def save():
        torch.save(dict(names=names, theta=saved, stats={k: v.cpu() for k, v in data["stats"].items()},
                        cfg=cfg, dt=dt, proto=a.proto, init=a.init), os.path.join(out, "ckpts.pt"))

    for k in range(cfg["steps"] + 1):
        if k in cks:
            saved[k] = TH.detach().clone()
            save()
        if k % cfg["log_every"] == 0 or k == cfg["steps"]:
            one, eT = evaluate()
            log.append([k, lr_at(cfg, min(k, cfg["steps"] - 1))] + one.tolist() + eT.tolist())
            if k % (20 * cfg["log_every"]) == 0:
                print(f"[{a.proto}/{a.init} dt={dt}] step {k} {time.time() - t0:.0f}s  heldout one-step "
                      + " ".join(f"{x:.2e}" for x in one.tolist()), flush=True)
        if k == cfg["steps"]:
            break
        idx = torch.randint(ntr, (M, B), generator=gen, device=dev)
        if unrolled:
            U = uTr[:p + 1, idx].permute(1, 0, 2, 3)            # (M, p+1, B, 3)
            losses = f_unr(TH, C, U)
        else:
            i = torch.randint(pwin, (M, B), generator=gen, device=dev)
            X, Y = uTr[i, idx], uTr[i + 1, idx]
            losses = f_one(TH, C, X, Y)
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, k)
        opt.zero_grad()
        losses.sum().backward()
        gn = torch.linalg.norm(TH.grad, dim=1)
        TH.grad *= torch.clamp(cfg["clip"] / (gn + 1e-300), max=1.0)[:, None]
        opt.step()

    save()
    hdr = ["step", "lr"] + [f"onestep_{c}{s}" for c, s in names] + [f"eT_{c}{s}" for c, s in names]
    np.savetxt(os.path.join(out, "log.csv"), np.array(log), delimiter=",", header=",".join(hdr), comments="")
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(dict(dt=dt, proto=a.proto, init=a.init, p=p, names=names, wall_s=time.time() - t0,
                       device=dev, P=arch.P, ckpts=sorted(saved)), fh, indent=1)
    print(f"[{a.proto}/{a.init} dt={dt}] done {time.time() - t0:.0f}s", flush=True)


def _roll(arch, th, c, u0, p):
    from msr.core import gt
    u = u0
    for _ in range(p):
        u = gt(arch, th, u, c)
    return u


if __name__ == "__main__":
    main()
