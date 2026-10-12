"""Exact identity checks, run before the sweep, at nontrivial parameters (the seed's initialization
with the zero output layer replaced by random weights), for both classes and every dt:
  0. analytic-free state Jacobian (forward mode) vs autograd;
  1. Psi^(p) g vs central finite differences of the rollout (random unit g);
  2. g^(Sigma)(p) vs finite differences of L_p (directional);
  3. e(t_p) = sum_{i<p} P(t_p,t_i)(eps_i + q_i), q_i from the 5-point Gauss-Legendre Bbar_i.
Writes <out_root>/checks.json and prints the max relative errors.

  python checks.py [--set n_test=256 ...]
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.func import jacrev

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, norm, nsteps
from msr.measure import forward_quantities, identity_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cfg["threads"])
    arch = Arch(cfg["hidden"], cfg["act"])
    res = {}
    for dt in cfg["dts"]:
        data = make_data(cfg, dt, dev, train=False)
        uT = data["uT_test"][:, :cfg["n_kernel"]]
        for cls in cfg["classes"]:
            m = Model(arch, cls, dt, data["stats"])
            th = arch.init(0, dev)
            nlast = 3 * cfg["hidden"][-1] + 3
            th[-nlast:] = 0.3 * torch.randn(nlast, generator=torch.Generator().manual_seed(1)).to(dev) / np.sqrt(cfg["hidden"][-1])
            u = uT[3, :8]
            Ja = torch.stack([jacrev(lambda x: m.Gt(th, x))(u[k]) for k in range(u.shape[0])])
            c0 = float(torch.linalg.norm(m.Jstate(th, u) - Ja) / torch.linalg.norm(Ja))
            r = identity_checks(m, th, uT, cfg)
            F = forward_quantities(m, th, uT, data["truth"], cfg["gl_points"])
            e = F["e"]
            rec = max(norm(F["S"][p] + F["Q"][p] - e[p]) / norm(e[p]) for p in range(1, data["pmax"] + 1))
            rec_by_T = {T: norm(F["S"][nsteps(T, dt)] + F["Q"][nsteps(T, dt)] - e[nsteps(T, dt)]) /
                        norm(e[nsteps(T, dt)]) for T in cfg["horizons"]}
            res[f"{cls}/dt{dt:g}"] = dict(chk_Jstate=c0, recon_max=rec, recon_by_T=rec_by_T,
                                          e_norm_Tmax=norm(e[-1]), **r)
            print(f"{cls:6s} dt={dt:<5g} Jstate {c0:.1e}  Psi-FD {r['chk_Psi_fd']:.1e}  "
                  f"gSigma-FD {r['chk_gSigma_fd']:.1e}  recon(max over p) {rec:.1e}  |e(Tmax)| {norm(e[-1]):.2e}",
                  flush=True)
    out = a.out or os.path.join(cfg["out_root"], "checks.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    for key in ("chk_Jstate", "chk_Psi_fd", "chk_gSigma_fd", "recon_max"):
        print(f"MAX {key}: {max(v[key] for v in res.values()):.2e}")


if __name__ == "__main__":
    main()
