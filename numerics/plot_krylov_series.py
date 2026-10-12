"""Plot results/krylov_series.csv: (F - F_M)/(||E|| - F) against M, cut at the float64 noise level."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = list(csv.DictReader(open(os.path.join(RES, "krylov_series.csv"))))
for r in rows:
    for k in ("s", "norm_E", "F", "F_M", "gain", "rel_gap", "rel_bound"):
        r[k] = float(r[k])
CKPTS, KS = [3000, 30000, 300000], [100, 1000, 10000]
targets = ["sin5", "sin20", "sin60", "random"]
cols = {100: "tab:blue", 1000: "tab:orange", 10000: "tab:green"}
fig, axs = plt.subplots(3, 4, figsize=(11, 7), sharex=True, sharey=True)
for i, k in enumerate(CKPTS):
    for j, tg in enumerate(targets):
        ax = axs[i, j]
        for K in KS:
            for seed in ("0", "1", "2"):
                rs = [r for r in rows if int(r["step"]) == k and r["target"] == tg and int(r["K"]) == K and r["seed"] == seed]
                noise = 1e-14 * rs[0]["norm_E"] / rs[0]["gain"]      # rounding level of F - F_M, relative to the gain
                M, g = [], []
                for r in rs:
                    if r["rel_gap"] <= noise:
                        break
                    M.append(int(r["M"])); g.append(r["rel_gap"])
                ax.semilogy(M, g, "-o", ms=2, color=cols[K], alpha=1 if seed == "0" else 0.3,
                            label=f"s = path of next {K} steps" if seed == "0" else None)
        ax.set_ylim(1e-10, 1.5); ax.set_xlim(0.5, 20)
        ax.grid(alpha=0.3, which="major")
        if i == 0:
            ax.set_title({"sin5": r"$\sin 5x$", "sin20": r"$\sin 20x$", "sin60": r"$\sin 60x$", "random": "random labels"}[tg], fontsize=10)
        if j == 0:
            ax.set_ylabel(f"after {k} steps\n" + r"$(F - F_M)/(\|E\| - F)$")
        if i == 2:
            ax.set_xlabel("number of Krylov terms $M$")
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"fig_krylov_series.{ext}"), dpi=140)
print("ok")
