"""Summary of results/ab_lorenz.csv (centred targets): per h, means over seeds of the size of each
target, its relative unbuildable part (outside the range of K), and the absolute floor = size x part."""
import csv
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "ab_lorenz.csv"))) if r["centred"] == "True"]
g = collections.defaultdict(list)
for r in rows:
    g[(float(r["h"]), r["target"])].append(r)
m = lambda rs, k: sum(float(r[k]) for r in rs) / len(rs)
print("h      | size D   size E  ratio | rel.floor D  rel.floor E  ratio | abs.floor D  abs.floor E | rate D/E | s(.5) D  E")
for h in sorted({k[0] for k in g}):
    D, E = g[(h, "direct")], g[(h, "euler")]
    sD, sE, fD, fE = m(D, "norm"), m(E, "norm"), m(D, "rel_floor"), m(E, "rel_floor")
    aD = sum(float(r["norm"]) * float(r["rel_floor"]) for r in D) / len(D)
    aE = sum(float(r["norm"]) * float(r["rel_floor"]) for r in E) / len(E)
    print(f"{h:<6} | {sD:6.2f}  {sE:6.2f}  {sD / sE:5.1f} | {fD:10.2e}  {fE:10.2e}  {fE / fD:5.1f} | "
          f"{aD:10.4f}  {aE:10.4f} | {m(D, 'rho_over_lam1') / m(E, 'rho_over_lam1'):7.1f} | "
          f"{m(D, 's_to_0.5'):5.2f} {m(E, 's_to_0.5'):5.2f}")
