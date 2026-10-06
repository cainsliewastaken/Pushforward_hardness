"""Section 9.4 claims over several initializations: python eos/eos_aggregate.py <run dir> [<run dir> ...].

Each run dir is the output of eos_sine.py (one seed) after eos_decompose.py and eos_general.py. Prints, per
claim, the value for each run and the range over runs:
  1. sharpening by target: max of eta*lambda_1 and its median over the second half of the run;
  2. Hessian against kernel: largest |eta*lambda_max(Hessian) - eta*lambda_1| over the second half;
  3. first-order against exact one-step change of lambda_1 away from the edge (median relative remainder);
  4. share of the first-order rate carried by the slow part E_gamma, gamma = 1/(1000 eta), eta*lambda_1 < 1.9;
     median from step 10000 on and over all checkpoints;
  5. Theorem split away from the edge: medians of the shares of the main term, D and T;
  6. kernel growth for the targets that reach eta*lambda_1 >= 1.9: end/start ratios of lambda_1, lambda_50 and
     the median eigenvalue."""
import csv
import json
import os
import sys
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if k != "target":
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = np.nan
    return rows


runs = sys.argv[1:]
res = {}
jobs = []
for d in runs:
    meta = json.load(open(os.path.join(d, "eos_meta.json")))
    for sd in meta["seeds"]:
        jobs.append((d, meta, sd))
for d, meta, seed in jobs:
    only = lambda rows: [r for r in rows if r.get("seed", meta["seeds"][0]) == seed]   # old files have no seed column
    log = only(load(os.path.join(d, "eos_log.csv")))
    dec = only(load(os.path.join(d, "decompose.csv")))
    gen = only(load(os.path.join(d, "general.csv")))
    spec = np.load(os.path.join(d, "eos_spectra.npz"))
    T = max(r["step"] for r in log)
    out = {}
    for lr in sorted({r["lr_rel"] for r in log}):
        for tg in [t for t in dict.fromkeys(r["target"] for r in log)]:
            rr = [r for r in log if r["lr_rel"] == lr and r["target"] == tg]
            el = np.array([r["eta_lam1"] for r in rr])
            st = np.array([r["step"] for r in rr])
            eh = np.array([r.get("eta_hess_lam1", np.nan) for r in rr])
            half = st >= T / 2
            m = half & np.isfinite(eh)
            out[(lr, tg)] = dict(max=np.nanmax(el), med2=np.nanmedian(el[half]),
                                 hess_gap=np.nanmax(np.abs(eh[m] - el[m])) if m.any() else np.nan)
    off = [r for r in dec if r["step"] > 0 and r["eta_lam1"] < 1.9]
    rem = np.median([abs(r["remainder"]) / abs(r["dlam1_exact_step"]) for r in off if r["dlam1_exact_step"] != 0])
    sl_all = np.median([r["dlam1_from_slow"] / r["dlam1_pred"] for r in off])
    sl_late = np.median([r["dlam1_from_slow"] / r["dlam1_pred"] for r in off if r["step"] >= 10000])
    pos = np.mean([r["dlam1_pred"] > 0 for r in off])
    goff = [r for r in gen if r["eta_lam1"] < 1.9]
    split = {k: np.median([r[k] for r in goff]) for k in ("share_main", "share_defect", "share_T")}
    lam = spec["lam"]                                    # (logs, B, n)
    rl = [tuple(x) for x in meta["runs"]]
    grow = []
    for b, (sb, tg, lr) in enumerate(rl):
        if sb == seed and out[(float(lr), tg)]["max"] >= 1.9:
            l0, l1 = lam[0, b], lam[-1, b]
            grow.append((l1[0] / l0[0], l1[49] / l0[49], np.median(l1) / np.median(l0)))
    grow = np.array(grow)
    res[seed] = dict(targets=out, rem=rem, sl_all=sl_all, sl_late=sl_late, pos=pos, split=split, grow=grow)

seeds = sorted(res)
print("initializations:", seeds)
keys = list(res[seeds[0]]["targets"])
print("\n1-2. eta*lambda_1: max / median over 2nd half (per initialization); largest |Hessian - kernel| in eta*lambda units")
for k in keys:
    print(f"  lr_rel {k[0]} {k[1]:6s}: " + " | ".join(
        f"{res[s]['targets'][k]['max']:.2f}/{res[s]['targets'][k]['med2']:.2f} (gap {res[s]['targets'][k]['hess_gap']:.3f})"
        for s in seeds))
print("\n3. median |exact - first order| / |exact| away from the edge: " + ", ".join(f"{res[s]['rem']:.1e}" for s in seeds))
print("4. slow-part share of the first-order rate: from step 10000 " + ", ".join(f"{res[s]['sl_late']:.2f}" for s in seeds)
      + "; all checkpoints " + ", ".join(f"{res[s]['sl_all']:.2f}" for s in seeds)
      + "; fraction positive " + ", ".join(f"{res[s]['pos']:.2f}" for s in seeds))
print("5. Theorem split medians (main / D / T): " + " | ".join(
    f"{res[s]['split']['share_main']:+.2f} / {res[s]['split']['share_defect']:+.2f} / {res[s]['split']['share_T']:+.2f}" for s in seeds))
print("6. kernel growth for targets reaching the edge (end/start): lambda_1, lambda_50, median eigenvalue")
for s in seeds:
    g = res[s]["grow"]
    if len(g):
        print(f"  init {s}: lambda_1 {g[:,0].min():.1f}-{g[:,0].max():.1f}, lambda_50 {g[:,1].min():.1e}-{g[:,1].max():.1e}, "
              f"median {g[:,2].min():.0f}-{g[:,2].max():.0f}  ({len(g)} runs)")
