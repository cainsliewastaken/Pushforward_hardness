"""Re-draw the numerics figures as PDF from the saved CSVs (no recomputation). Output: figures_pdf/."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "figures_pdf")
os.makedirs(OUT, exist_ok=True)
_save = mf.Figure.savefig
def savepdf(self, path, *a, **k):
    k.pop("dpi", None)
    return _save(self, os.path.join(OUT, os.path.basename(path).replace(".png", ".pdf")), *a, **k)
mf.Figure.savefig = savepdf

def read(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for key, v in r.items():
            if v in ("True", "False"):
                r[key] = v == "True"
                continue
            try:
                r[key] = float(v)
            except (TypeError, ValueError):
                pass
    return rows

import lorenz as L, replicate_rsk as R, sine as S
R1, R2 = os.path.join(HERE, "results"), os.path.join(HERE, "results_rsk")
L.figures(L.CFG, OUT, read(f"{R1}/lorenz_init_krylov.csv"), read(f"{R1}/lorenz_G.csv"),
          read(f"{R1}/lorenz_curves.csv"), read(f"{R1}/lorenz_G0.csv"))
cfg = S.CFG
A = read(f"{R2}/A_floor.csv"); M = read(f"{R2}/A_rough_subspaces.csv"); Cc = read(f"{R2}/A_ceiling.csv")
for r in A + M + Cc:
    r["seed"] = int(r["seed"])
R.figs_A(cfg, A, M, Cc, OUT)
R.figs_C(read(f"{R2}/C_table_steppers.csv"), read(f"{R2}/C_table_split.csv"), read(f"{R2}/C_targets.csv"), OUT)
# Krylov series F_M/F (same as plot_sine_FM.py)
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(f"{R1}/sine_krylov.csv")))
names = ["sin1", "sin5", "sin10", "sin20", "sin40", "sin60", "sin90", "sin120", "random"]
cols = {"3000": "tab:blue", "30000": "tab:orange", "300000": "tab:green"}
for mult in ("1", "10000"):
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.5))
    for i, nm in enumerate(names):
        ax = axs[i // 3, i % 3]
        grp = {}
        for r in rows:
            if r["target"] == nm and float(r["s_mult"]) == float(mult):
                grp.setdefault((r["seed"], r["step"]), []).append(r)
        for (seed, step), rs in sorted(grp.items()):
            ax.plot([int(r["M"]) for r in rs], [float(r["F_M"]) / float(r["F"]) for r in rs], color=cols[step],
                    alpha=1 if seed == "0" else 0.35, label=f"step {step}" if seed == "0" else None)
        ax.set_title(nm, fontsize=9); ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 20)
        if i // 3 == 2: ax.set_xlabel("M")
        if i % 3 == 0: ax.set_ylabel(r"$F_M(s)/F(s)$")
        if i == 0: ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"fig_sine_FM{'' if mult == '1' else '_x10000'}.png")
    plt.close(fig)
print(sorted(os.listdir(OUT)))
