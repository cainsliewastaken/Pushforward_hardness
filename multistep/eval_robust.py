"""Robust long-horizon evaluation: for the final models of every run (all roots), the distribution of the per-start
rollout error |e_k(T)| over the n_test held-out starts at the long_horizons: root-mean-square (what eval_long.py
reports as e_norm), median, 90th and 99th percentile. Forward passes only.
Writes <out_root>/robust_eval.pt: {(run, cls, seed): dict(T, rms, med, p90, p99)} and prints, per root, dt and class,
the run with the lowest error at each horizon by rms and by median."""
import argparse, os, sys, numpy as np, torch
from msr.config import load_cfg
from msr.core import Arch, Model, make_data, nsteps
from train import _roll

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", required=True, help="width:out_root pairs, e.g. 256:$R 128:$R/pilot/w128")
ap.add_argument("--match", nargs="*", default=None, help="only runs whose name contains one of these")
ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
for spec in a.roots:
    W, root = spec.split(":", 1); W = int(W)
    cfg = load_cfg("config.yaml", [f"hidden=[{W},{W},{W}]", f"out_root={root}"])
    Ts = cfg["long_horizons"]
    runs = sorted(r for r in os.listdir(os.path.join(root, "runs")) if os.path.exists(os.path.join(root, "runs", r, "ckpts.pt")))
    if a.match: runs = [r for r in runs if any(x in r for x in a.match)]
    runs = runs[a.part::a.nparts]
    res, datas = {}, {}
    for run in runs:
        ck = torch.load(os.path.join(root, "runs", run, "ckpts.pt"), map_location=dev, weights_only=False)
        dt = float(ck["dt"])
        if dt not in datas:
            datas[dt] = make_data(cfg, dt, dev, train=False, stats=ck["stats"], horizon=max(Ts))
        d = datas[dt]; uT = d["uT_test"]
        arch = Arch(ck["cfg"]["hidden"], ck["cfg"]["act"]); fin = max(ck["theta"])
        for j, (cls, seed) in enumerate(ck["names"]):
            m = Model(arch, cls, dt, d["stats"]); th = ck["theta"][fin][j]
            st = dict(T=np.array(Ts), rms=[], med=[], p90=[], p99=[])
            with torch.no_grad():
                u, q0 = uT[0], 0
                for T in Ts:
                    q = nsteps(T, dt); u = _roll(arch, th, m.c, u, q - q0); q0 = q
                    x = (uT[q] - u).norm(dim=-1)
                    st["rms"].append(float(torch.sqrt((x ** 2).mean()))); st["med"].append(float(x.median()))
                    st["p90"].append(float(torch.quantile(x, 0.9))); st["p99"].append(float(torch.quantile(x, 0.99)))
            res[(run, cls, seed)] = {k: np.array(v) for k, v in st.items()}
        print(f"W{W} {run}", flush=True)
    torch.save(res, os.path.join(root, f"robust_eval{'_' + a.match[0] if a.match else ''}_part{a.part}.pt"))
