"""Reads results/soft_lorenz.csv and results/soft_sine.csv (no computation)."""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: list(csv.DictReader(open(os.path.join(HERE, "results", p))))
L = rd("soft_lorenz.csv")
f = lambda r, k: float(r[k])


def med(key, **sel):
    v = [f(r, key) for r in L if all(str(r[a]) == str(b) for a, b in sel.items())]
    return float(np.median(v))


hs = sorted({r["h"] for r in L}, key=float)
MS = [1, 2, 5, 10, 20, 50]
for W in ("32", "128"):
    print(f"\nwidth {W}, final step: ratio direct/Euler")
    print("h      ||V_g|| 1e-4  ||V_g|| 1e-2  train  val   | Krylov F_M / ||E_g|| for the ERROR at gamma=1e-4 lam1: direct M=1,5,50 | euler M=1,5,50")
    for h in hs:
        s = dict(h=h, width=W, step="6000")
        rat = lambda k: med(k, stepper="direct", **s) / med(k, stepper="euler", **s)
        conv = lambda st: " ".join(f"{med(f'E_F_M{M}', stepper=st, **s) / med('E_res_0.0001', stepper=st, **s):5.2f}" for M in (1, 5, 50))
        print(f"{h:6} {rat('V_res_0.0001'):10.2f} {rat('V_res_0.01'):12.2f} {rat('err_train'):7.2f} {rat('err_val'):5.2f} | {conv('direct')} | {conv('euler')}")
print("\nwidth 32, final step: Krylov F_M / ||V_g|| for the TARGET at gamma = 1e-4 lam1 and 1e-2 lam1 (direct | euler), M = 1,2,5,10,20,50")
for h in hs:
    s = dict(h=h, width="32", step="6000")
    c = lambda st, tag, res: " ".join(f"{med(f'V_{tag}_M{M}', stepper=st, **s) / med(res, stepper=st, **s):4.2f}" for M in MS)
    print(f"{h:6} 1e-4: {c('direct', 'F', 'V_res_0.0001')} | {c('euler', 'F', 'V_res_0.0001')}   1e-2: {c('direct', 'Fg2', 'V_res_0.01')} | {c('euler', 'Fg2', 'V_res_0.01')}")
print("\nwidth 32: at which checkpoints is the direct error larger than the Euler error, and the residual ||E_g|| (1e-4)?  entries: err D/E, residual D/E")
steps = sorted({r["step"] for r in L}, key=int)
print("h      " + "  ".join(f"step {k:>5}" for k in steps))
for h in hs:
    out = []
    for k in steps:
        s = dict(h=h, width="32", step=k)
        out.append(f"{med('err_train', stepper='direct', **s) / med('err_train', stepper='euler', **s):4.2f},{med('E_res_0.0001', stepper='direct', **s) / med('E_res_0.0001', stepper='euler', **s):4.2f}")
    print(f"{h:6} " + "  ".join(f"{o:>10}" for o in out))

S = rd("soft_sine.csv")
names = ["sin1", "sin5", "sin10", "sin20", "sin40", "sin60", "sin90", "sin120", "random"]
print("\nsine, 300000 steps, means over seeds. For each gamma/lam1: ||E_g||, floor(high-pass), F_1, F_5  (||E|| first)")
for nm in names:
    line = f"{nm:7}"
    for g in (0.3, 1e-2, 1e-4, 1e-6):
        q = [r for r in S if r["target"] == nm and r["step"] == "300000" and abs(f(r, "gamma_rel") / g - 1) < 1e-6]
        m = lambda k: np.mean([f(r, k) for r in q])
        if g == 0.3:
            line += f" ||E||={m('err'):.3f} |"
        line += f" g={g:g}: {m('E_gamma'):.3f} {m('floor'):6.3f} {m('F_M1'):.3f} {m('F_M5'):.3f} |"
    print(line)
