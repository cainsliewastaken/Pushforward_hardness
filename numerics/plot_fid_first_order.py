"""Direct vs Euler at a single time derivative (Lorenz-63, initialization, raw targets c_D = 1):
first-order bound on the unbuilt identity ||id|| - s r_id, r_id = ||id||_{K_D}/||id||, against 2||Phi_h - id||.
Reads results/fid_lorenz.csv (F_id_M1(s) = max(0, ||id|| - s r_id)); no recomputation."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
R = list(csv.DictReader(open(os.path.join(RES, "fid_lorenz.csv"))))
for r in R:
    for k in r:
        r[k] = float(r[k])
hs = sorted({r["h"] for r in R})
rows = []
for h in hs:
    for seed in (0, 1, 2):
        rs = sorted([r for r in R if r["h"] == h and r["seed"] == seed], key=lambda r: r["s"])
        r1 = min(rs, key=lambda r: abs(r["s"] - 1.0))
        rid = (r1["norm_id"] - r1["F_id_M1"]) / r1["s"]
        sstar = (r1["norm_id"] - 2 * r1["dphi"]) / rid
        ok = all(q["F_id"] >= q["norm_id"] - q["s"] * rid - 1e-9 for q in rs)
        rows.append(dict(h=h, seed=seed, norm_id=r1["norm_id"], two_dphi=2 * r1["dphi"], r_id=rid, s_star=sstar, bound_below_exact=ok))
        print(f"h={h} seed={int(seed)}: ||id||={r1['norm_id']:.2f} 2||Phi-id||={2*r1['dphi']:.2f} r_id={rid:.3f} s*={sstar:.2f} first-order bound <= exact F_id: {ok}")
cols = dict(zip(hs, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
fig, ax = plt.subplots(figsize=(6.2, 4))
s = np.linspace(0, 20, 201)
for h in hs:
    q = [r for r in rows if r["h"] == h and r["seed"] == 0][0]
    ax.plot(s, q["norm_id"] - s * q["r_id"], color=cols[h], label=f"h={h}")
    ax.axhline(q["two_dphi"], color=cols[h], ls=":", lw=1)
ax.set_ylim(0, 36)
ax.set_xlabel(r"parameter step length $s$ ($c_{\mathrm{D}} = 1$)")
ax.set_ylabel("state units")
ax.set_title(r"$\|\mathrm{id}\| - s\,r_{\mathrm{id}}$ (solid) and $2\|\Phi_h - \mathrm{id}\|$ (dotted), seed 0", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"fig_fid_lorenz.{ext}"), dpi=140)
print("ok")
