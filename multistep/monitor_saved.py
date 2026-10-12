"""Run rollout_monitor on saved Lorenz-63 Euler models (multistep_roughness), dt = 0.08.

Monitored starts and target starts are disjoint halves of the held-out set; the target is the held-out window error
over N physical steps (default N = 32, physical time 2.56)."""
import argparse, os, sys, time, torch, numpy as np
from msr.config import load_cfg
from msr.core import Arch, Model, make_data
from rollout_monitor import RolloutMonitor, report

ap = argparse.ArgumentParser()
ap.add_argument("--widths", nargs="*", type=int, default=[128, 512])
ap.add_argument("--runs", nargs="*", default=["dt0.08_onestep", "dt0.08_unroll032_ft_s4k", "dt0.08_unroll128_ft_s4k_linec025"])
ap.add_argument("--N", type=int, default=32)
ap.add_argument("--n_starts", type=int, default=512)
ap.add_argument("--order", type=int, default=4)
ap.add_argument("--grid", nargs="*", type=int, default=[2, 4, 8, 16, 24, 32])
ap.add_argument("--krylov", action="store_true")
ap.add_argument("--higher_order", action="store_true")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", default=os.path.expandvars("$SCRATCH/multistep_roughness/monitor_saved.pt"))
a = ap.parse_args()
torch.set_num_threads(a.threads)
R = os.path.expandvars("$SCRATCH/multistep_roughness")
roots = {64: f"{R}/pilot/w64", 128: f"{R}/pilot/w128", 256: R, 512: f"{R}/pilot/w512"}
dt = 0.08
cfg = load_cfg("config.yaml", [f"n_test={2 * a.n_starts}"])
results, data = {}, None
for W in a.widths:
    for run in a.runs:
        f = f"{roots[W]}/runs/{run}/ckpts.pt"
        if not os.path.exists(f):
            print("missing", f); continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if data is None:
            data = make_data(cfg, dt, "cpu", train=False, stats=ck["stats"], horizon=a.N * dt)
        U = data["uT_test"]                                   # [N+1, 2*n_starts, 3]
        mon_U, tgt_U = U[:, :a.n_starts], U[:, a.n_starts:]
        arch = Arch(ck["cfg"]["hidden"], ck["cfg"]["act"])
        names = [tuple(x) for x in ck["names"]]
        j = names.index(("euler", 0))
        m = Model(arch, "euler", dt, data["stats"])
        th = ck["theta"][max(ck["theta"])][j].detach()
        mon = RolloutMonitor(lambda p, u: m.Gt(p, u), th)
        t0 = time.time()
        out = mon.profile(mon_U, a.N, target_u_true=tgt_U, order=a.order, grid=a.grid,
                          krylov=a.krylov, higher_order=a.higher_order)
        results[(W, run)] = out
        print(f"\n===== width {W}, {run} (Euler, seed 0)  [{time.time() - t0:.0f}s]")
        print(report(out, dt=dt, every=2), flush=True)
torch.save(results, a.out)
