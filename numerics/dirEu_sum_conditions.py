"""Reads results/dirEu_sum.csv and reports, per (h, seed, convention), the largest s up to which
  truth: F_D(s) > F_E(s)
  A: F_id(s) > ||Phi_h - id|| + F_E(s)        (current theorem, with Euler's actual error)
  B: F_id((1 + c_E) s) > 2 F_E(s)             (new bound)
hold on the whole prefix of the s grid."""
import csv
import collections
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "dirEu_sum.csv"))))
g = collections.defaultdict(list)
for r in rows:
    g[(float(r["h"]), int(r["seed"]), r["conv"])].append(r)


def upto(rs, cond):
    last = 0.0
    for r in rs:
        if not cond(r):
            break
        last = float(r["s"])
    return last


x = lambda r, k: float(r[k])
truth = lambda r: x(r, "F_D") > x(r, "F_E")
A = lambda r: x(r, "F_id_old") > x(r, "two_dphi") / 2 + x(r, "F_E")
B = lambda r: x(r, "F_id_new") > 2 * x(r, "F_E")
AB = lambda r: A(r) or B(r)
frac = lambda rs, c: sum(map(c, rs)) / len(rs)
print(f"{'h':6}{'seed':5}{'conv':6}{'truth':>9}{'A':>9}{'B':>9}{'A or B':>9}   "
      f"{'frac truth':>10}{'frac AorB':>10}  B at s=1000  F_id(inf)  F_E(1000)")
for k, rs in sorted(g.items()):
    print(f"{k[0]:<6}{k[1]:<5}{k[2]:6}{upto(rs, truth):9.2f}{upto(rs, A):9.2f}{upto(rs, B):9.2f}"
          f"{upto(rs, AB):9.2f}   {frac(rs, truth):10.2f}{frac(rs, AB):10.2f}  {str(B(rs[-1])):11}"
          f"{x(rs[-1], 'F_id_new'):10.4f}  {x(rs[-1], 'F_E'):9.4f}")
