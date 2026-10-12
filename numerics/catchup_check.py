"""Head start and catch-up on the saved Lorenz runs (results/soft_lorenz.csv; no computation beyond
reading the CSV).

head start      = log(||V_D|| / ||V_E||)              (errors at step 0, output layer zero)
log ratio(k)    = log(||E_D(k)|| / ||E_E(k)||)         (training errors, medians over 5 inits)
catch-up(k)     = head start - log ratio(k)

Coarse rate check. Under preconditioned gradient flow d/dt log||E|| = -lr * rho with
rho = <E_hat, K E_hat>. The training code uses Adam on the mean over all 3n entries, so one step at
learning rate lr_k changes log||E|| by about -(2/3) lr_k rho_k (momentum and clipping ignored), with
K built from Adam's preconditioner at that step. Printed: this predicted per-step rate at each saved
checkpoint, and the actual mean per-step rate over the interval to the next checkpoint."""
import csv
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = list(csv.DictReader(open(os.path.join(HERE, "results", "soft_lorenz.csv"))))
hs = sorted({r["h"] for r in L}, key=float)
steps = sorted({int(r["step"]) for r in L})
lr = lambda k: 3e-3 * 0.5 * (1 + np.cos(np.pi * k / 6000))


def med(key, **sel):
    return float(np.median([float(r[key]) for r in L if all(r[a] == str(b) for a, b in sel.items())]))


for W in ("32", "128"):
    print(f"\n=== width {W}: head start, catch-up(k) = head start - log ratio(k), final ratio")
    print("h      head   " + " ".join(f"A({k:>4})" for k in steps[1:]) + "   ratio_end")
    for h in hs:
        e = {s: [med("err_train", h=h, width=W, stepper=s, step=k) for k in steps] for s in ("direct", "euler")}
        lrat = np.log(np.array(e["direct"]) / np.array(e["euler"]))
        print(f"{h:6} {lrat[0]:5.2f}   " + " ".join(f"{lrat[0] - x:7.2f}" for x in lrat[1:]) + f"   {np.exp(lrat[-1]):6.2f}")
W = "32"
print(f"\n=== width {W}: per-step rates of log||E|| (x 1e-3). predicted = (2/3) lr_k rho_k at checkpoint k (Adam kernel);"
      f" actual = mean over [k, next checkpoint]")
for h in ("0.01", "0.03", "0.1"):
    print(f"h = {h}")
    print("  step   lr       direct: pred   actual  | euler: pred   actual  | pred D-E  actual D-E")
    for i, k in enumerate(steps[:-1]):
        out = {}
        for s in ("direct", "euler"):
            rho = med("E_rho_rel", h=h, width=W, stepper=s, step=k) * med("lam1", h=h, width=W, stepper=s, step=k)
            pred = (2 / 3) * lr(k) * rho
            e0, e1 = med("err_train", h=h, width=W, stepper=s, step=k), med("err_train", h=h, width=W, stepper=s, step=steps[i + 1])
            act = -(np.log(e1) - np.log(e0)) / (steps[i + 1] - k)
            out[s] = (pred, act)
        tag = "  (Pi = I at step 0: not Adam's kernel)" if k == 0 else ""
        print(f"  {k:5d} {lr(k):.2e}   {1e3 * out['direct'][0]:10.3f} {1e3 * out['direct'][1]:8.3f}  | "
              f"{1e3 * out['euler'][0]:10.3f} {1e3 * out['euler'][1]:8.3f}  | "
              f"{1e3 * (out['direct'][0] - out['euler'][0]):8.3f} {1e3 * (out['direct'][1] - out['euler'][1]):10.3f}{tag}")
