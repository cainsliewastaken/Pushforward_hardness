"""Side-by-side comparison of every difficulty measure for the direct and Euler targets on Lorenz-63,
in absolute state units and relative to the size of each target. Reads saved CSVs only.
  high-pass : manifold high-pass roughness, m = 300 (results_rsk/C_targets.csv, fig_stepper (c))
  slow      : part of the target on eigenvalues <= 1e-4 lambda_1 of the trained kernel, width 32
              (results_rsk/C_table_steppers.csv, fig_stepper (b))
  unbuild   : part of the target outside the range of the kernel at initialization, centred
              (results/ab_lorenz.csv)
  val error : validation error of the trained steppers, width 32 (fig_stepper (a))"""
import csv
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: list(csv.DictReader(open(os.path.join(HERE, p))))
tg = {float(r["h"]): r for r in rd("results_rsk/C_targets.csv")}
st = {float(r["h"]): r for r in rd("results_rsk/C_table_steppers.csv") if r["width"] == "32"}
ab = collections.defaultdict(list)
for r in rd("results/ab_lorenz.csv"):
    if r["centred"] == "True":
        ab[(float(r["h"]), r["target"])].append(float(r["norm"]) * float(r["rel_floor"]))
mean = lambda v: sum(v) / len(v)
print(f"{'h':5} {'measure':10} | {'abs D':>8} {'abs E':>8} {'D/E':>6} | {'rel D':>7} {'rel E':>7} {'E/D':>6}")
for h in sorted(tg):
    sD, sE = float(tg[h]["size_direct"]), float(tg[h]["size_euler"])
    items = [("high-pass", float(tg[h]["W_direct"]), float(tg[h]["W_euler"]))]
    if h in st:
        items.append(("slow", float(st[h]["slow_direct"]), float(st[h]["slow_euler"])))
    if (h, "direct") in ab:
        items.append(("unbuild", mean(ab[(h, "direct")]), mean(ab[(h, "euler")])))
    if h in st:
        items.append(("val error", float(st[h]["val_direct"]), float(st[h]["val_euler"])))
    for name, aD, aE in items:
        print(f"{h:<5} {name:10} | {aD:8.4f} {aE:8.4f} {aD / aE:6.2f} | {100 * aD / sD:6.2f}% {100 * aE / sE:6.2f}% "
              f"{(aE / sE) / (aD / sD):6.2f}")
    print(f"{'':5} {'size':10} | {sD:8.3f} {sE:8.3f} {sD / sE:6.2f} |")
