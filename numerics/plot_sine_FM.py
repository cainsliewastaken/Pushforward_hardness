"""Re-plot the Krylov series of part 2 from results/sine_krylov.csv as F_M(s)/F(s) (rises to 1 at M = q_E)."""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = list(csv.DictReader(open(os.path.join(RES, "sine_krylov.csv"))))
names = ["sin1", "sin5", "sin10", "sin20", "sin40", "sin60", "sin90", "sin120", "random"]
cols = {"3000": "tab:blue", "30000": "tab:orange", "300000": "tab:green"}
for mult in sorted({r["s_mult"] for r in rows}, key=float):
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    for i, nm in enumerate(names):
        ax = axs[i // 3, i % 3]
        grp = {}
        for r in rows:
            if r["target"] == nm and r["s_mult"] == mult:
                grp.setdefault((r["seed"], r["step"]), []).append(r)
        for (seed, step), rs in sorted(grp.items()):
            M = [int(r["M"]) for r in rs]
            y = [float(r["F_M"]) / float(r["F"]) for r in rs]
            ax.plot(M, y, color=cols[step], alpha=1 if seed == "0" else 0.35,
                    label=f"step {step}" if seed == "0" else None)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("M")
        ax.set_ylabel("F_M(s) / F(s)")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8)
    lab = "s = length of the next GD step" if float(mult) == 1 else f"SUPPLEMENTARY: s = {float(mult):g} x next GD step"
    fig.suptitle(f"Krylov series F_M(s)/F(s), {lab} (seed 0 solid, others faint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"fig_sine_FM{'' if float(mult) == 1 else f'_x{float(mult):g}'}.png"), dpi=130)
    plt.close(fig)
print("ok")
