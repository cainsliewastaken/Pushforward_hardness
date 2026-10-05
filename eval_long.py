"""Long-horizon rollout error of every run's final models: |e(T)|, I(T), R(T) for T in long_horizons
(default up to 5.12 = ~4.6 Lyapunov times), on all n_test held-out starts. Forward passes only.

  python eval_long.py --runs dt0.01_onestep dt0.08_unrollp32_ft ...   (default: every run with ckpts.pt)
Writes <out_root>/long_eval.pt: {(run, cls, seed): dict(T, e_norm, I, R, e_norm_t)} and also the
reference scale of a saturated error (RMS distance between independent attractor states).
"""
import argparse
import glob
import os

import numpy as np
import torch

from msr.config import load_cfg
from msr.core import Arch, Model, make_data, norm, nsteps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = os.path.join(cfg["out_root"], "runs")
    runs = a.runs or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{root}/*/train.ok"))
    Ts = cfg["long_horizons"]
    outf = os.path.join(cfg["out_root"], "long_eval.pt")
    res = torch.load(outf, weights_only=False) if os.path.exists(outf) else {}
    cache = {}
    for run in runs:
        ck = torch.load(os.path.join(root, run, "ckpts.pt"), map_location=dev, weights_only=False)
        dt = ck["dt"]
        if dt not in cache:
            cache[dt] = make_data(cfg, dt, dev, train=False, stats={k: v.to(dev) for k, v in ck["stats"].items()},
                                  horizon=max(Ts))
        data = cache[dt]
        for k in ("mu", "sd", "sf"):
            assert torch.allclose(data["stats"][k].cpu(), ck["stats"][k]), (run, k)
        uT = data["uT_test"]
        arch = Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])
        final = max(ck["theta"])
        for j, (cls, seed) in enumerate(ck["names"]):
            m = Model(arch, cls, dt, data["stats"])
            with torch.no_grad():
                th = ck["theta"][final][j]
                up = m.rollout(th, uT[0], uT.shape[0] - 1)
                e = uT - up
                eps = uT[1:] - m.Gt(th, uT[:-1])
                cum = torch.cumsum(eps, 0)
            ps = [nsteps(T, dt) for T in Ts]
            eN = np.array([norm(e[p]) for p in ps])
            I = np.array([norm(cum[p - 1]) for p in ps])
            res[(run, cls, seed)] = dict(T=np.array(Ts), dt=dt, step=final, e_norm=eN, I=I, R=eN - I,
                                         e_norm_t=np.array([norm(e[i]) for i in range(e.shape[0])]))
        print(f"{run}: done", flush=True)
    # saturation scale: RMS distance between independent states on the attractor
    u = cache[min(cache)]["uT_test"]
    res["saturation_rms"] = norm(u[-1] - u[-1].roll(1, 0))
    torch.save(res, outf)
    print(f"saturated-error scale {res['saturation_rms']:.3g}; wrote {outf}")


if __name__ == "__main__":
    main()
