"""Figure 1 (sine targets) from a sine_deep.py results folder and seminorm_compare.py, no computation.
(a) error along training against frequency: median of the logged error over a window around each
    checkpoint (Adam's loss has transient spikes; a single step can land on one);
(b) final checkpoint: the error, its slow part ||E_gamma|| and the slow part ||U_gamma|| of the target
    under the trained kernel, at gamma = 1e-4 lambda_1;
(c) final checkpoint, gamma = 1e-4 lambda_1: Krylov values F_M(s_gamma) against M, and the step bounds
    b_R(s_gamma) of the roughness functions of Section 'Examples' (best member of each family per run),
    all as fractions of F(s_gamma).
Output: <dir>/fig1_sine.{png,pdf}."""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results_deep_lr3e-4")
ap.add_argument("--gamma", type=float, default=1e-4)
a = ap.parse_args()
D = os.path.join(HERE, a.dir)
rd = lambda f: [{k: (v if k == "target" else float(v)) for k, v in r.items() if v != ""}
                for r in csv.DictReader(open(os.path.join(D, f)))]
S, CU, SN = rd("sine_rows.csv"), rd("sine_curves.csv"), rd("seminorm_rows.csv")
names = list(dict.fromkeys(r["target"] for r in S))
seeds = sorted({r["seed"] for r in S})
steps = sorted({int(r["step"]) for r in S})
last, g = steps[-1], a.gamma
XR = 500.0
x = np.array([XR if nm == "random" else float(nm[3:]) for nm in names])
o = np.argsort(x)
sines = [i for i in o if names[i] != "random"]
rnd = [i for i in o if names[i] == "random"]


def errbar(axx, vals, **kw):
    """vals: (targets, seeds) array; mean over seeds, bars = range; random labels unconnected."""
    m, lo, hi = vals.mean(1), vals.min(1), vals.max(1)
    lab = kw.pop("label", None)
    for idx, line in ((sines, True), (rnd, False)):
        axx.errorbar(x[idx], m[idx], yerr=[(m - lo)[idx], (hi - m)[idx]], capsize=2, lw=1.2,
                     label=lab if line else None, **(kw if line else {**kw, "ls": "none"}))


fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
# (a)
cols = ["#9ecae1", "#3182bd", "#08306b"]
curve = {}
for r in CU:
    curve.setdefault((r["target"], r["seed"]), []).append((r["step"], r["err"]))
for step, c in zip(steps, cols):
    lo_s, hi_s = 0.8 * step, min(1.2 * step, last)
    v = np.array([[np.median([e for k, e in curve[(nm, s)] if lo_s <= k <= hi_s]) for s in seeds] for nm in names])
    errbar(ax[0], v, color=c, marker="o", ms=3.5, label=f"{step} steps")
ax[0].set_yscale("log")
ax[0].set_ylabel("state units")
ax[0].set_title(r"(a) Error $\|E\|$ along training", fontsize=10)
# (b)
fin = {(r["target"], r["seed"]): r for r in S if r["step"] == last and r["gamma_rel"] == g}
get = lambda k: np.array([[fin[(nm, s)][k] for s in seeds] for nm in names])
errbar(ax[1], get("err"), color="k", marker="o", ms=4, label=r"error $\|E\|$")
errbar(ax[1], get("E_gamma"), color="#08519c", marker="s", ms=4, ls="--", mfc="none",
       label=r"slow part of the error $\|E_\gamma\|$")
errbar(ax[1], get("U_gamma"), color="#a50f15", marker="^", ms=4, label=r"slow part of the target $\|U_\gamma\|$")
errbar(ax[1], get("Yhat_gamma"), color="#fd8d3c", marker="v", ms=4, ls=":", mfc="none",
       label=r"slow part of the output $\|\hat Y_\gamma\|$")
ax[1].set_yscale("log")
ax[1].set_ylabel("state units")
ax[1].set_title(rf"(b) After {last} steps, $\gamma = 10^{{{int(np.log10(g))}}}\lambda_1$", fontsize=10)
# (c)
gs = sorted({r["gamma_rel"] for r in SN})
FLOOR = 1e-6


def frac(key, gg):
    q = [r for r in SN if r["step"] == last and r["gamma_rel"] == gg and r["F"] > 0]
    return np.array([max(r[key] / r["F"], FLOOR) for r in q])


SF = 1e-7          # shortfalls below this (rounding) are drawn at SF


def short(key, gg):
    q = [r for r in SN if r["step"] == last and r["gamma_rel"] == gg and r["F"] > 0]
    return np.array([min(max(1 - r[key] / r["F"], SF), 1.0) for r in q])


# (c) the lower bounds themselves, in state units, at one long step (gamma = g); nonpositive bounds
# (which certify nothing) are left out of the mean and range
finS = {(r["target"], r["seed"]): r for r in SN if r["step"] == last and r["gamma_rel"] == g}
getS = lambda k: np.array([[finS[(nm, s)][k] for s in seeds] for nm in names])
errbar(ax[2], getS("err"), color="k", marker="o", ms=4, label=r"error $\|E\|$")


def errbar_pos(axx, vals, **kw):
    v = np.where(vals > 0, vals, np.nan)
    m, lo, hi = np.nanmean(v, 1), np.nanmin(v, 1), np.nanmax(v, 1)
    lab = kw.pop("label", None)
    for idx, line in ((sines, True), (rnd, False)):
        idx = [i for i in idx if np.isfinite(m[i])]
        if idx:
            axx.errorbar(x[idx], m[idx], yerr=[(m - lo)[idx], (hi - m)[idx]], capsize=2, lw=1.2,
                         label=lab if line else None, **(kw if line else {**kw, "ls": "none"}))


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
for key, c, ls, mk, lab in (("F_M1", "#2ca25f", "none", "x", r"Krylov series, first term $F_1$"),
                            ("b_cluster", "#6a51a3", "--", "s", r"cluster seminorm (best $k$, $p$)"),
                            ("b_highpass", "#e6550d", "-.", "^", r"manifold high-pass (best $m$)"),
                            ("b_lipschitz", "#636363", ":", "d", "Lipschitz seminorm")):
    errbar_pos(ax[2], getS(key), color=c, ls=ls, marker=mk, ms=6 if mk == "x" else 4, mfc="none", label=lab)
ax[2].set_yscale("log")
ax[2].set_ylabel("state units")
ax[2].set_title(rf"(c) Lower bounds after one step of length $s_\gamma$, $\gamma = 10^{{{int(np.log10(g))}}}\lambda_1$",
                fontsize=10)
ticks = [1, 2, 5, 10, 20, 50, 100, 250, XR]
for axx in ax:
    axx.set_xscale("log")
    axx.set_xticks(ticks)
    axx.set_xticklabels([str(t) for t in ticks[:-1]] + ["random"])
    axx.minorticks_off()
    axx.set_xlabel(r"frequency $\omega$ of the target $\sin\omega x$")
for axx in ax:
    axx.legend(fontsize=7.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(D, f"fig1_sine.{ext}"), dpi=150)
# numbers for the text
print("final step, gamma", g)
for nm in names:
    e, sE, sU = get("err"), get("E_gamma"), get("U_gamma")
    i = names.index(nm)
    print(f"  {nm:8} err {e[i].mean():.2e}  slow E {sE[i].mean():.2e}  slow U {sU[i].mean():.3f}  ||U|| {get('U_norm')[i].mean():.3f}")
for key in ("b_cluster", "b_highpass", "b_lipschitz"):
    v = getS(key)
    print(f"  {key}: nonpositive in {int((v <= 0).sum())} of {v.size} runs; fraction of F: "
          f"median {np.median(np.clip(v, 0, None) / getS('F')):.3g}")
for gg in gs:
    print(f"gamma {gg:g}: max Krylov F_1 shortfall {max(1 - r['F_M1'] / r['F'] for r in SN if r['step'] == last and r['gamma_rel'] == gg and r['F'] > 0):.2e}")
    print(f"gamma {gg:g}: median fraction of F(s): " + ", ".join(
        f"{k} {np.median(frac(k, gg)):.3g} [IQR {np.percentile(frac(k, gg), 25):.2g}-{np.percentile(frac(k, gg), 75):.2g}]"
        for k in ("F_M1", "b_cluster", "b_highpass", "b_lipschitz")))
