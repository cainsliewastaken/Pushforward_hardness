"""Measure every model of one training run at every checkpoint (M1-M7 on held-out starts).
Identity checks (FD of Psi and of L_p) are added at the final checkpoint.

  python measure_run.py --dt 0.01 --proto unroll064 --init ft
Writes <run dir>/meas.pt: {(cls, seed, step): dict of arrays}.
"""
import argparse
import os
import time

import torch

from msr.config import load_cfg, run_dir
from msr.core import Arch, Model, make_data
from msr.measure import measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--proto", required=True)
    ap.add_argument("--init", default="scratch")
    ap.add_argument("--steps", nargs="*", type=int, default=None, help="subset of checkpoint steps")
    ap.add_argument("--models", type=int, default=None, help="measure only the first N models (tests)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    rd = run_dir(cfg, a.dt, a.proto, a.init)
    ck = torch.load(os.path.join(rd, "ckpts.pt"), map_location=dev)
    tcfg = ck["cfg"]
    data = make_data(cfg, a.dt, dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()})
    arch = Arch(tcfg["hidden"], tcfg["act"])
    steps = sorted(ck["theta"]) if a.steps is None else a.steps
    final = max(ck["theta"])
    out, t0 = {}, time.time()
    for j, (cls, seed) in enumerate(ck["names"][:a.models]):
        m = Model(arch, cls, a.dt, data["stats"])
        for s in steps:
            th = ck["theta"][s][j]
            if not torch.isfinite(th).all():
                out[(cls, seed, s)] = dict(diverged=True)
                continue
            out[(cls, seed, s)] = measure(m, th, data, cfg, checks=(s == final), full=(s == final))
        print(f"[{rd}] {cls} seed {seed} measured ({time.time() - t0:.0f}s)", flush=True)
    torch.save(out, os.path.join(rd, "meas.pt"))


if __name__ == "__main__":
    main()
