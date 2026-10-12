"""Validation error against the Krylov bound of the target (Lorenz-63), all h, both methods, both
widths. Reads results/soft_lorenz.csv; no computation. The Krylov bound is the residual ||V_gamma||
of the centred target at gamma = 1e-4 lambda_1 under the kernel of the trained network, which the
50-term Krylov series reaches to three digits. Output: results/fig_error_vs_roughness.{png,pdf}."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = list(csv.DictReader(open(os.path.join(HERE, "results", "soft_lorenz.csv"))))
hs = sorted({r["h"] for r in L}, key=float)


def med(key, **sel):
    return float(np.median([float(r[key]) for r in L if all(r[a] == b for a, b in sel.items())]))


col = dict(direct="tab:blue", euler="tab:red")
fig, ax = plt.subplots(1, 1, figsize=(6, 4.6))
for W, fill, ls in (("32", "full", "-"), ("128", "none", "--")):
    for s in ("direct", "euler"):
        err = [med("err_val", h=h, width=W, stepper=s, step="6000") for h in hs]
        kb = [med("V_res_0.0001", h=h, width=W, stepper=s, step="6000") for h in hs]
        ax.loglog(kb, err, color=col[s], marker="o" if s == "direct" else "s", ms=6, fillstyle=fill, ls=ls, lw=1,
                  label=f"{s}, width {W}")
        if W == "32":
            for x, y, h in zip(kb, err, hs):
                if h in ("0.01", "0.05", "0.1", "0.3"):
                    ax.annotate(f"h={h}", (x, y), textcoords="offset points",
                                xytext=(6, -10) if s == "euler" else (-6, 7),
                                ha="left" if s == "euler" else "right", fontsize=7, color=col[s])
lo, hi = ax.get_xlim()
xx = np.array([lo, hi])
ax.loglog(xx, 0.2 * xx, color="gray", lw=0.8, ls=":", label="error = 0.2 x Krylov bound")
ax.set_xlim(lo, hi)
ax.set_xlabel(r"Krylov bound $\|V_\gamma\|$ of the target, trained kernel, state units")
ax.set_ylabel("validation error, state units")
ax.legend(fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "results", f"fig_error_vs_roughness.{ext}"), dpi=140)
print("ok")
