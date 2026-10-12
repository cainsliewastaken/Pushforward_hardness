"""Two-panel stepper figure from results/soft_lorenz.csv (no computation). Width 32 solid, 128 dashed;
medians over 5 initializations, bars = range; final step (6000).
(a) validation error of the direct and Euler steppers;
(b) best lower bound from roughness, in state units, on fitting each centred target with the trained
    kernel: the residual ||V_gamma|| at gamma = 1e-4 lambda_1, which the Krylov series reaches
    (the 50-term Krylov value is printed as a check), for the direct and Euler targets, and the same
    for the centred identity map under the kernel of the trained direct network.
Output: results/fig_stepper2.{png,pdf}."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = list(csv.DictReader(open(os.path.join(HERE, "results", "soft_lorenz.csv"))))
hs = sorted({r["h"] for r in L}, key=float)
hv = np.array([float(h) for h in hs])


def vals(key, **sel):
    return np.array([float(r[key]) for r in L if all(r[a] == b for a, b in sel.items())])


col = dict(direct="tab:blue", euler="tab:red", identity="tab:green")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
print("h     W    | Krylov50/residual: direct target, euler target, identity | residual D, E, id")
for W, ls, fill in (("32", "-", "full"), ("128", "--", "none")):
    for a, items in ((ax[0], (("direct", "err_val", "direct", "o"), ("euler", "err_val", "euler", "s"))),
                     (ax[1], (("direct", "V_res_0.0001", "direct target", "o"), ("euler", "V_res_0.0001", "Euler target", "s"),
                              ("direct", "id_res_0.0001", "identity map", "^")))):
        for s, key, lab, mk in items:
            c = col["identity"] if lab == "identity map" else col[s]
            st = [vals(key, h=h, width=W, stepper=s, step="6000") for h in hs]
            m = np.array([np.median(x) for x in st])
            lo, hi = np.array([x.min() for x in st]), np.array([x.max() for x in st])
            a.errorbar(hv, m, yerr=[m - lo, hi - m], color=c, ls=ls, marker=mk, ms=5, fillstyle=fill, capsize=2,
                       lw=1.2, label=f"{lab}, width {W}")
    for h in hs:
        chk = [np.median(vals(k50, h=h, width=W, stepper=s, step="6000")) / np.median(vals(kr, h=h, width=W, stepper=s, step="6000"))
               for s, k50, kr in (("direct", "V_F_M50", "V_res_0.0001"), ("euler", "V_F_M50", "V_res_0.0001"), ("direct", "id_F_M50", "id_res_0.0001"))]
        res = [np.median(vals(kr, h=h, width=W, stepper=s, step="6000")) for s, kr in (("direct", "V_res_0.0001"), ("euler", "V_res_0.0001"), ("direct", "id_res_0.0001"))]
        print(f"{h:5} {W:4} | " + " ".join(f"{c:6.3f}" for c in chk) + " | " + " ".join(f"{x:7.4f}" for x in res))
for a in ax:
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("step size $h$")
    a.legend(fontsize=7)
ax[0].set_ylabel("validation error, state units")
ax[0].set_title("(a) Error of the trained steppers", fontsize=10)
ax[1].set_ylabel("state units")
ax[1].set_title(r"(b) Krylov bound $\|V_\gamma\|$, trained kernel, $\gamma = 10^{-4}\lambda_1$", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "results", f"fig_stepper2.{ext}"), dpi=140)
print("ok")
