"""Run the whole sweep on one node: a small scheduler that keeps `per_gpu` processes on each GPU.

Tasks: train onestep (per dt), train unrolled from scratch (per dt, protocol), train unrolled
fine-tuned from onestep (after that dt's onestep run), and measure every run after it trains.
Costliest tasks start first. Each task logs to <out_root>/logs/<tag>.log; a finished task writes
<run dir>/<kind>.ok, so rerunning the script skips work already done.

  python sweep.py --gpus 4 --per-gpu 4 [--set key=value ...]
Ends by writing <out_root>/SWEEP.done with the exit codes.
"""
import argparse
import os
import subprocess
import sys
import time

from msr.config import load_cfg, run_dir, run_tag
from msr.core import nsteps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="hide GPUs (tests)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    logs = os.path.join(cfg["out_root"], "logs")
    os.makedirs(logs, exist_ok=True)
    extra = ["--cfg", a.cfg] + (["--set"] + a.set if a.set else [])

    tasks = {}   # name -> dict(cmd, deps, cost, ok)

    def add(kind, dt, proto, init, deps, cost):
        tag = run_tag(dt, proto, init)
        script = "train.py" if kind == "train" else "measure_run.py"
        tasks[f"{kind}:{tag}"] = dict(
            cmd=[sys.executable, "-u", script, "--dt", str(dt), "--proto", proto, "--init", init] + extra,
            deps=deps, cost=cost, ok=os.path.join(run_dir(cfg, dt, proto, init), f"{kind}.ok"),
            log=os.path.join(logs, f"{kind}_{tag}.log"))

    for dt in cfg["dts"]:
        for proto, pr in cfg["protocols"].items():
            p = nsteps(pr["T_train"], dt) if "T_train" in pr else 1
            inits = cfg["inits"] if "T_train" in pr else ["scratch"]
            for init in inits:
                tag = run_tag(dt, proto, init)
                deps = [f"train:{run_tag(dt, 'onestep', 'scratch')}"] if init == "ft" else []
                add("train", dt, proto, init, deps, cost=p + (1000 if proto == "onestep" else 0))
                add("measure", dt, proto, init, [f"train:{tag}"], cost=nsteps(max(cfg["horizons"]), dt))

    done = {n for n, t in tasks.items() if os.path.exists(t["ok"])}
    failed, running = {}, {}
    slots = [g for g in range(a.gpus) for _ in range(a.per_gpu)]
    T0 = time.time()
    while True:
        for n, (proc, g, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[n]
            slots.append(g)
            if rc == 0:
                open(tasks[n]["ok"], "w").close()
                done.add(n)
            else:
                failed[n] = rc
            print(f"[{time.time() - T0:7.0f}s] {n} exit={rc} ({time.time() - t0:.0f}s)", flush=True)
        ready = [n for n, t in tasks.items() if n not in done and n not in failed and n not in running
                 and all(d in done for d in t["deps"])]
        blocked = [n for n, t in tasks.items() if n not in done and n not in failed and n not in running
                   and any(d in failed for d in t["deps"])]
        for n in blocked:
            failed[n] = "dep"
        ready.sort(key=lambda n: (-tasks[n]["cost"]))
        while ready and slots:
            n = ready.pop(0)
            g = slots.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if a.cpu else str(g))
            os.makedirs(os.path.dirname(tasks[n]["ok"]), exist_ok=True)
            fh = open(tasks[n]["log"], "w")
            running[n] = (subprocess.Popen(tasks[n]["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env), g, time.time())
            print(f"[{time.time() - T0:7.0f}s] start {n} on GPU {g}", flush=True)
        if not running and not ready:
            break
        time.sleep(5)
    with open(os.path.join(cfg["out_root"], "SWEEP.done"), "w") as fh:
        fh.write(f"done={len(done)} failed={len(failed)} wall={time.time() - T0:.0f}s\n")
        for n, rc in failed.items():
            fh.write(f"FAILED {n} {rc}\n")
    print(f"sweep finished: {len(done)} ok, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
