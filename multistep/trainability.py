"""Trainability check: does the curvature of each unrolled run's own loss, at the start of training,
predict whether training descended?

For every unrolled run and model: its training loss L (same window p and weights w_q as in training)
on held-out starts at every checkpoint; the largest eigenvalue of the loss's Gauss-Newton matrix
G = sum_q (w_q / sum w) (1/n) Psi_q^T Psi_q at the first and last checkpoint (Lanczos, matrix-free:
G v via one JVP and one VJP of the stacked, weighted rollout); and the step-size numbers
  lr * lam_max                       (gradient-descent stability: unstable above ~2)
  S = lam_max (lr sqrt(P))^2 / (2 L0)  (loss increase from one Adam-sized step along the top direction,
                                        relative to the starting loss)
with lr the run's peak learning rate.

  python trainability.py [--runs ...] [--n 512]
Writes <out_root>/trainability.pt: {(run, cls, seed): dict}.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.func import jvp, vjp

from msr.config import load_cfg
from msr.core import Arch, Model, loss_weights, make_data, sq
from msr.measure import _top_eig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=512, help="held-out starts")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(cfg["out_root"], "runs")
    if a.runs is None:
        a.runs = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{root}/*/train.ok")
                        if "_onestep" not in f)
    outf = a.out or os.path.join(cfg["out_root"], "trainability.pt")
    res = {}
    for run in a.runs:
        meta = json.load(open(os.path.join(root, run, "meta.json")))
        dt, p = meta["dt"], meta["p"]
        gamma = meta.get("gamma", 0.0)
        lr = meta.get("lr", cfg["lr_ft"] if meta["init"] == "ft" else cfg["lr"])   # sweep runs predate the lr field
        w = loss_weights(gamma, dt, p)
        W = torch.ones(p, device=dev) if w is None else torch.tensor(w, device=dev)
        ck = torch.load(os.path.join(root, run, "ckpts.pt"), map_location=dev, weights_only=False)
        data = make_data(cfg, dt, dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()}, horizon=p * dt)
        uT = data["uT_test"][:, :a.n]
        arch = Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])
        steps = sorted(ck["theta"])
        sw = torch.sqrt(W / W.sum())
        for j, (cls, seed) in enumerate(ck["names"]):
            m = Model(arch, cls, dt, data["stats"])

            def loss(th):
                with torch.no_grad():
                    e = uT - m.rollout(th, uT[0], p)
                return float((W * 0.5 * sq(e[1:])).sum() / W.sum())

            def lam(th):
                f = lambda t: (sw[:, None, None] * m.rollout(t, uT[0], p)[1:]).reshape(-1) / np.sqrt(a.n)
                _, vjp_fn = vjp(f, th)
                Gv = lambda x: vjp_fn(jvp(f, (th,), (x,))[1])[0]
                return _top_eig(Gv, m.P, dev, tol=1e-4)

            L = np.array([loss(ck["theta"][s][j]) for s in steps])
            th0, th1 = ck["theta"][steps[0]][j], ck["theta"][steps[-1]][j]
            l0, l1 = lam(th0), lam(th1)
            res[(run, cls, seed)] = dict(dt=dt, p=p, T_win=p * dt, gamma=gamma, lr=lr, init=meta["init"],
                                         from_run=meta.get("from_run"), steps=np.array(steps), L=L,
                                         end_over_start=L[-1] / L[0], max_spike=L.max() / L[0],
                                         lam_start=l0, lam_end=l1, lr_lam=lr * l0,
                                         S=l0 * (lr * np.sqrt(m.P)) ** 2 / (2 * L[0]))
            r = res[(run, cls, seed)]
            print(f"{run:28s} {cls:6s} s{seed}: lam0 {l0:.3g} lam1 {l1:.3g} lr*lam {r['lr_lam']:.3g} S {r['S']:.3g} "
                  f"end/start {r['end_over_start']:.3g} spike x{r['max_spike']:.3g}", flush=True)
        torch.save(res, outf)
    print(f"wrote {outf}")


if __name__ == "__main__":
    main()
