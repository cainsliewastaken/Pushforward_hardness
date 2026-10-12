"""Recompute the quantities that depend on the mean-value Jacobian Bbar (rhobar, sum ln|rhobar|,
sum P q, |sum P q|/|e|, reconstruction error) at every run's final checkpoint with more
Gauss-Legendre points. 10 points (the sweep) is not enough when |e| is attractor-sized.

  python recompute_gl.py [--points 20 40] [--runs ...]
Writes <run dir>/meas_gl.pt: {(cls, seed, step): {n_points: dict}}; aggregate.py uses the largest n.
"""
import argparse
import glob
import os

import numpy as np
import torch

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, norm, nsteps
from msr.measure import forward_quantities


def bbar_quantities(m, th, data, cfg, n_gl):
    uT = data["uT_test"]
    ps = [nsteps(T, m.dt) for T in cfg["horizons"]]
    F = forward_quantities(m, th, uT, data["truth"], n_gl)
    e, rhob = F["e"], F["rhob"][1:]
    win = lambda a, p: float(a[:p - 1].mean()) if p > 1 else np.nan
    lnr = torch.log(rhob.abs())
    eN = np.array([norm(e[p]) for p in ps])
    S_q = np.array([norm(F["Q"][p]) for p in ps])
    return dict(rhob_m1=np.array([win(rhob - 1, p) for p in ps]),
                sum_ln_rhob=np.array([float(lnr[:p - 1].sum(0).mean()) if p > 1 else 0.0 for p in ps]),
                frac_rhob_nonpos=np.array([win((rhob <= 0).double(), p) for p in ps]),
                S_q=S_q, ratio_q=S_q / eN,
                recon_relerr=np.array([norm(F["S"][p] + F["Q"][p] - e[p]) / norm(e[p]) for p in ps]),
                recon_relerr_max=max(norm(F["S"][i] + F["Q"][i] - e[i]) / norm(e[i]) for i in range(1, uT.shape[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--points", nargs="*", type=int, default=[20, 40])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(cfg["out_root"], "runs")
    runs = a.runs or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{root}/*/measure.ok"))
    for run in runs:
        ck = torch.load(os.path.join(root, run, "ckpts.pt"), map_location=dev, weights_only=False)
        dt = ck["dt"]
        data = make_data(cfg, dt, dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()})
        arch = Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])
        final = max(ck["theta"])
        out = {}
        for j, (cls, seed) in enumerate(ck["names"]):
            m = Model(arch, cls, dt, data["stats"])
            out[(cls, seed, final)] = {g: bbar_quantities(m, ck["theta"][final][j], data, cfg, g) for g in a.points}
        torch.save(out, os.path.join(root, run, "meas_gl.pt"))
        worst = max(v[max(a.points)]["recon_relerr_max"] for v in out.values())
        print(f"{run}: max reconstruction error with {max(a.points)} points {worst:.2e}", flush=True)


if __name__ == "__main__":
    main()
