"""Redraws results/fig_soft_lorenz.{png,pdf} from results/soft_lorenz.csv and soft_lorenz_targets.csv
(no computation): the Krylov curve uses 50 terms, which have converged, and panel (b) shows two h."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
rd = lambda p: list(csv.DictReader(open(os.path.join(OUT, p))))
L, T = rd("soft_lorenz.csv"), rd("soft_lorenz_targets.csv")
H, last = "32", "6000"


def med(key, **sel):
    return float(np.median([float(r[key]) for r in L if all(r[a] == b for a, b in sel.items())]))


hs = [t["h"] for t in T]
hv = [float(h) for h in hs]
rat = lambda key: [med(key, h=h, width=H, stepper="direct", step=last) / med(key, h=h, width=H, stepper="euler", step=last) for h in hs]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
ax[0].loglog(hv, [float(t["R_direct"]) / float(t["R_euler"]) for t in T], color="#fd8d3c", marker="s", ms=4,
             label="roughness of the target (manifold high-pass, data only)")
ax[0].loglog(hv, rat("V_res_0.0001"), color="#08306b", marker="o", ms=5,
             label=r"residual of the target $\|V_\gamma\|$ (trained kernel)")
ax[0].loglog(hv, rat("V_F_M50"), color="#31a354", ls="none", marker="^", ms=7, mfc="none",
             label=r"its Krylov value with 50 terms")
ax[0].loglog(hv, rat("err_val"), "k-", marker="x", ms=6, lw=2, label="validation error")
ax[0].axhline(1, color="gray", lw=0.6)
ax[0].set_xlabel("step size $h$")
ax[0].set_ylabel("direct / Euler")
ax[0].set_title(f"(a) Ratio direct / Euler after training, width {H}", fontsize=10)
ax[0].legend(fontsize=7)
steps = sorted({r["step"] for r in L}, key=int)
xs = [max(int(k), 1) for k in steps]
for a, h, tag in zip(ax[1:], (hs[0], "0.1"), ("b", "c")):
    for s, c in (("direct", "tab:blue"), ("euler", "tab:red")):
        a.loglog(xs, [med("err_train", h=h, width=H, stepper=s, step=k) for k in steps], color=c, marker="o",
                 ms=4, label=rf"{s}: error $\|E\|$")
        a.loglog(xs, [med("E_res_0.0001", h=h, width=H, stepper=s, step=k) for k in steps], color=c, ls="--",
                 marker="^", ms=4, label=rf"{s}: residual $\|E_\gamma\|$")
    a.set_xlabel("training step (step 0 shown at 1)")
    a.set_ylabel("state units")
    a.set_title(f"({tag}) Along training, h = {h}, width {H}", fontsize=10)
    a.legend(fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_soft_lorenz.{ext}"), dpi=140)
print("ok", hs)
