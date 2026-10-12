"""Reads results/soft_lorenz.csv. For each h and checkpoint (width 32 and 128): median error of the
direct and Euler networks over the 5 initializations, whether the two ranges are disjoint, and the
two-sided Mann-Whitney p value (smallest possible with 5 vs 5 is 0.008)."""
import csv
import os
import numpy as np
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
L = list(csv.DictReader(open(os.path.join(HERE, "results", "soft_lorenz.csv"))))
hs = sorted({r["h"] for r in L}, key=float)
steps = sorted({r["step"] for r in L}, key=int)


def vals(key, **sel):
    return np.array([float(r[key]) for r in L if all(r[a] == b for a, b in sel.items())])


def verdict(d, e):
    p = mannwhitneyu(d, e, alternative="two-sided").pvalue
    if d.min() > e.max():
        w = "D>E"
    elif e.min() > d.max():
        w = "E>D"
    else:
        w = " ~ "
    return f"{np.median(d) / np.median(e):4.2f} {w} p={p:.3f}"


for W in ("32", "128"):
    for key in ("err_train", "err_val"):
        print(f"\nwidth {W}, {key}: median ratio direct/Euler, D>E or E>D if the ranges over 5 inits are disjoint (~ if they overlap), Mann-Whitney p")
        print("h     " + " | ".join(f"step {k:>5}       " for k in steps))
        for h in hs:
            print(f"{h:5} " + " | ".join(verdict(vals(key, h=h, width=W, stepper="direct", step=k),
                                                  vals(key, h=h, width=W, stepper="euler", step=k)) for k in steps))
