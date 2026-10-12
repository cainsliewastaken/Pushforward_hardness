"""Figure 1 (sine targets) from results_deep/ (sine_deep.py, depth 2, Adam), no computation.
(a) error against frequency at every checkpoint (log scale); (b) error against the roughness not yet
supplied, R_m(U) - R_m(N), for every run and checkpoint (Theorem 'floor'); (c) at the last checkpoint,
the fraction of the error that a step of length s_gamma can remove, 1 - F_M(s_gamma)/||E||, from the
Krylov values. Also prints the numbers used in the text.
Output: results_deep/fig_deep_sine.{png,pdf}, results_deep/summary_deep.md."""
import argparse
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results_deep")
ap.add_argument("--step", type=int, default=5000)
a = ap.parse_args()
D = os.path.join(HERE, a.dir)
S = [{k: (v if k == "target" else float(v)) for k, v in r.items() if v != ""}
     for r in csv.DictReader(open(os.path.join(D, "sine_rows.csv")))]
meta = json.load(open(os.path.join(D, "sine_meta.json")))
names = list(dict.fromkeys(r["target"] for r in S))
steps = sorted({int(r["step"]) for r in S})
sel = lambda **kw: [r for r in S if all(r[k] == v for k, v in kw.items())]


def stat(key, step, g=0.3):
    x = np.array([[r[key] for r in sel(target=nm, step=step, gamma_rel=g)] for nm in names])
    return x.mean(1), x.min(1), x.max(1)


# x-axis: target frequency (log); random labels drawn separately at the right end. The manifold
# high-pass R_m(U) saturates at ~0.69 for every omega >= 40, so it cannot order the rough targets.
XR = 500.0
x = np.array([XR if nm == "random" else float(nm[3:]) for nm in names])
o = np.argsort(x)
sines = [i for i in o if names[i] != "random"]
rnd = [i for i in o if names[i] == "random"]


def draw(axx, key, step, **kw):
    m, lo, hi = stat(key, step)
    lab = kw.pop("label", None)
    for idx, ls_on in ((sines, True), (rnd, False)):
        if not idx:
            continue
        axx.errorbar(x[idx], m[idx], yerr=[(m - lo)[idx], (hi - m)[idx]], capsize=2, lw=1.1,
                     label=lab if ls_on else None, **({**kw, "ls": "none"} if not ls_on else kw))


def stat_fn(fn, step, g):
    v = np.array([[fn(r) for r in sel(target=nm, step=step, gamma_rel=g)] for nm in names])
    return v.mean(1), v.min(1), v.max(1)


def draw_fn(axx, fn, step, g, **kw):
    m, lo, hi = stat_fn(fn, step, g)
    lab = kw.pop("label", None)
    for idx, line in ((sines, True), (rnd, False)):
        axx.errorbar(x[idx], m[idx], yerr=[(m - lo)[idx], (hi - m)[idx]], capsize=2, lw=1.1,
                     label=lab if line else None, **(kw if line else {**kw, "ls": "none"}))


last = steps[-1]
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
cols = ["#9ecae1", "#3182bd", "#08306b"]
# (a) error over training, log scale: the late-training errors span four orders of magnitude
for step, c in zip(steps, cols):
    draw_fn(ax[0], lambda r: r["err"], step, 0.3, color=c, marker="o", ms=3.5, label=f"{step} steps")
ax[0].set_yscale("log")
ax[0].set_ylabel("state units")
ax[0].set_title(r"(a) Error $\|E\|$ along training", fontsize=10)
# (b) Theorem 'floor' at every run and checkpoint: ||E|| >= R_m(U) - R_m(N) - leak
for step, c in zip(steps, cols):
    q = sel(step=step, gamma_rel=0.3)
    for rr, mk in (([r for r in q if r["target"] != "random"], "o"), ([r for r in q if r["target"] == "random"], "s")):
        ax[1].plot([r["RU"] - r["RN"] for r in rr], [r["err"] for r in rr], mk, color=c, ms=4, mfc="none",
                   label=f"{step} steps" if mk == "o" else None)
ax[1].plot(np.linspace(1e-5, 1, 400), np.linspace(1e-5, 1, 400), "k:", lw=1, label=r"$\|E\| = \mathcal{R}_m(U) - \mathcal{R}_m(\mathcal{N}_\theta)$")
ax[1].set_xlim(-0.05, 1.0)
ax[1].set_yscale("log")
ax[1].set_ylim(1e-5, 1.5)
ax[1].set_xlabel(r"roughness not yet supplied, $\mathcal{R}_m(U) - \mathcal{R}_m(\mathcal{N}_\theta)$")
ax[1].set_ylabel(r"error $\|E\|$")
ax[1].set_title("(b) Error against unsupplied roughness (squares: random labels)", fontsize=10)
# (c) final step: fraction of the error that a step of length s_gamma can remove, from the Krylov series
eps = 1e-7
for g, c, mk in ((0.3, "#c7e9c0", "v"), (1e-2, "#41ab5d", "s"), (1e-4, "#00441b", "o")):
    draw_fn(ax[2], lambda r: max(1 - r["F_M100"] / r["err"], eps), last, g, color=c, marker=mk, ms=4,
            label=rf"$1 - F_{{100}}(s_\gamma)/\|E\|$, $\gamma = {g:g}\,\lambda_1$")
draw_fn(ax[2], lambda r: max(1 - r["F_M1"] / r["err"], eps), last, 1e-4, color="#00441b", marker="x", ms=6, ls="none",
        label=r"same with $F_1$, $\gamma = 10^{-4}\lambda_1$")
ax[2].set_yscale("log")
ax[2].set_ylim(eps / 2, 1)
ax[2].set_title(rf"(c) Fraction of the error one step of length $s_\gamma$ can remove, {last} steps", fontsize=10)
ticks = [1, 2, 5, 10, 20, 50, 100, 250, XR]
for axx in (ax[0], ax[2]):
    axx.set_xscale("log")
    axx.set_xticks(ticks)
    axx.set_xticklabels([str(t) for t in ticks[:-1]] + ["random"])
    axx.minorticks_off()
    axx.set_xlabel(r"frequency $\omega$ of the target $\sin\omega x$")
for axx in ax:
    axx.legend(fontsize=7.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(D, f"fig_deep_sine.{ext}"), dpi=150)

out = []
P = lambda *s: out.append(" ".join(str(x) for x in s))
P(f"config {meta['cfg']}\nchecks {meta['checks']}\nseconds {meta['seconds']:.0f}\n")
for step in steps:
    P(f"## step {step}, gamma = 0.3 lambda_1 (means over seeds; [min, max] for error and floor)")
    P("| target | R_m(U) | R_m(N) | error | slow | F_1 | F_5 | F_100 | floor |")
    P("|---|---|---|---|---|---|---|---|---|")
    for nm in names:
        q = sel(target=nm, step=step, gamma_rel=0.3)
        f = lambda k: np.mean([r[k] for r in q])
        rg = lambda k: f"{f(k):.3f} [{min(r[k] for r in q):.3f}, {max(r[k] for r in q):.3f}]"
        P(f"| {nm} | {f('RU'):.3f} | {f('RN'):.3f} | {rg('err')} | {f('E_gamma'):.3f} | {f('F_M1'):.3f} | "
          f"{f('F_M5'):.3f} | {f('F_M100'):.3f} | {rg('floor')} |")
    q = sel(step=step, gamma_rel=0.3)
    P(f"slow/err min {min(r['E_gamma'] / r['err'] for r in q if r['err'] > 1e-12):.4f}; "
      f"F1/slow min {min(r['F_M1'] / r['E_gamma'] for r in q if r['E_gamma'] > 1e-12):.4f}; "
      f"leak sigma*cs max {max(r['sigma'] * r['cs'] for r in q):.3g}; RN max {max(r['RN'] for r in q):.3f}")
    for g in sorted({r["gamma_rel"] for r in S}):
        q = [r for r in sel(step=step, gamma_rel=g) if r["E_gamma"] > 1e-12]
        P(f"  gamma {g:g}: F_M/slow min over runs " +
          " ".join(f"M{M}:{min(r[f'F_M{M}'] / r['E_gamma'] for r in q):.4f}" for M in meta["MS"]))
    P("")
open(os.path.join(D, "summary_deep.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
