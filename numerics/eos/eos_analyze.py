"""Summaries and figure for an eos_sine.py run: python eos/eos_analyze.py <out dir>.

Q1 sharpening against omega; Q2 plateau of eta*lambda_1 (kernel and Hessian) at 2; Q3 isotropy
(growth of lambda_k by k, band eigenvalue, overlap of v_1 with the target); Q4 predictors of the error
Delta steps later (soft residual, fixed-kernel forecast); Q5 monotonicity of the slow part through loss
spikes; chord-bound margins. Writes <out>/summary.md and <out>/fig_eos.png."""
import csv
import json
import os
import sys
import numpy as np

out = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(out, "eos_log.csv"))))
for r in rows:
    for k, v in r.items():
        if k not in ("target",):
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                r[k] = np.nan
meta = json.load(open(os.path.join(out, "eos_meta.json")))
loss = np.load(os.path.join(out, "eos_loss.npy"))
spec = np.load(os.path.join(out, "eos_spectra.npz"))
runs = [tuple(r) for r in meta["runs"]]
deltas = meta["deltas"]
lines = []


def P(s=""):
    print(s)
    lines.append(s)


def series(seed, target, lr, key):
    rr = sorted([r for r in rows if r["seed"] == seed and r["target"] == target and r["lr_rel"] == lr],
                key=lambda r: r["step"])
    return np.array([r["step"] for r in rr]), np.array([r.get(key, np.nan) for r in rr])


P(f"# EoS on sine targets: {out}\n")
P(f"eta = lr_rel / lambda_1(0); lambda_1(0) = {np.median(meta['lam1_0']):.4g}; steps = {meta['steps']}\n")

P("## Q1/Q2: sharpening and plateau (eta*lambda_1 of K; Hessian in brackets)\n")
P("| lr_rel | target | err(0) | err(end) | max eta*lam1 | first step eta*lam1>=1.9 | median eta*lam1, 2nd half | [median eta*lamH, 2nd half] |")
P("|---|---|---|---|---|---|---|---|")
for b, (seed, target, lr) in enumerate(runs):
    st, el = series(seed, target, lr, "eta_lam1")
    _, er = series(seed, target, lr, "err")
    sh, eh = series(seed, target, lr, "eta_hess_lam1")
    half = st >= st[-1] / 2
    hit = st[np.argmax(el >= 1.9)] if np.any(el >= 1.9) else None
    P(f"| {lr} | {target} | {er[0]:.3g} | {er[-1]:.3g} | {np.nanmax(el):.2f} | {hit if hit is not None else '-'} | "
      f"{np.nanmedian(el[half]):.2f} | {np.nanmedian(eh[half & np.isfinite(eh)]) if np.any(half & np.isfinite(eh)) else np.nan:.2f} |")

P("\n## Q3: is the sharpening isotropic? (end / start ratios)\n")
P("| lr_rel | target | lam1 | lam2 | lam5 | lam10 | lam50 | median lam | band/lam1 start->end | overlap v1.Y start->end | overlap v1.E end |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
lam = spec["lam"]  # (logs, B, n)
for b, (seed, target, lr) in enumerate(runs):
    l0, l1 = lam[0, b], lam[-1, b]
    rat = lambda i: l1[min(i, len(l1) - 1)] / l0[min(i, len(l0) - 1)]
    _, bl = series(seed, target, lr, "band_lam_lam1")
    _, oy = series(seed, target, lr, "ov_v1_Y")
    _, oe = series(seed, target, lr, "ov_v1_E")
    P(f"| {lr} | {target} | {rat(0):.2f} | {rat(1):.2f} | {rat(4):.2f} | {rat(9):.2f} | {rat(49):.2f} | "
      f"{np.median(l1) / np.median(l0):.2f} | {bl[0]:.3f}->{bl[-1]:.3f} | {oy[0]:.2f}->{oy[-1]:.2f} | {oe[-1]:.2f} |")

P("\n## Q4: predicting err(t + Delta) from the kernel at t\n")
P("Median and [10%, 90%] of log10(prediction / actual) over all logs t with t + Delta <= end.\n")
P("| Delta | soft residual gamma=1/(eta Delta) | fixed-K forecast (stable dirs) | fixed-K forecast (all dirs) |")
P("|---|---|---|---|")
for dl in deltas:
    acc = {"slow": [], "fcast_stable": [], "fcast": []}
    for b, (seed, target, lr) in enumerate(runs):
        st, er = series(seed, target, lr, "err")
        idx = {int(s): i for i, s in enumerate(st)}
        for key in acc:
            _, pr = series(seed, target, lr, f"{key}_{dl}")
            for i, s in enumerate(st):
                j = idx.get(int(s) + dl)
                if j is not None and er[j] > 0 and np.isfinite(pr[i]) and pr[i] > 0:
                    acc[key].append(np.log10(pr[i] / er[j]))
    cell = lambda v: (f"{np.median(v):+.2f} [{np.percentile(v, 10):+.2f}, {np.percentile(v, 90):+.2f}] (n={len(v)})"
                      if v else "-")
    P(f"| {dl} | {cell(acc['slow'])} | {cell(acc['fcast_stable'])} | {cell(acc['fcast'])} |")

P("\n## Q5: through loss spikes, does the slow part keep falling?\n")
P("Fraction of log-to-log intervals in which the quantity increased (at logs where eta*lam1 >= 1.8).\n")
P("| lr_rel | target | intervals | err increased | slow_{d} increased |".format(d=deltas[1]))
P("|---|---|---|---|---|")
for b, (seed, target, lr) in enumerate(runs):
    st, el = series(seed, target, lr, "eta_lam1")
    _, er = series(seed, target, lr, "err")
    _, sl = series(seed, target, lr, f"slow_{deltas[1]}")
    m = (el[:-1] >= 1.8)
    if m.sum() == 0:
        continue
    P(f"| {lr} | {target} | {m.sum()} | {np.mean(np.diff(er)[m] > 0):.2f} | {np.mean(np.diff(sl)[m] > 0):.2f} |")

cp = os.path.join(out, "eos_chord.csv")
if os.path.exists(cp):
    ch = list(csv.DictReader(open(cp)))
    P("\n## Chord bound: mean_tau sqrt(lam_max(P K P)) >= (||PY|| - ||PE||)/D\n")
    marg = np.array([float(c["margin"]) for c in ch])
    P(f"{len(ch)} checks, {np.sum(marg >= 0)} hold; smallest margin {marg.min():.3g}.\n")
    P("| lr_rel | target | step | D | lhs | rhs | demanded band lam (rhs^2) | max eta*lam1 on chord |")
    P("|---|---|---|---|---|---|---|---|")
    last = {}
    for c in ch:
        last[(c["lr_rel"], c["target"])] = c
    for (lr, t), c in sorted(last.items(), key=lambda z: (float(z[0][0]), float(z[1]["omega"]))):
        P(f"| {lr} | {t} | {c['step']} | {float(c['D']):.3g} | {float(c['avg_sqrt_band']):.3g} | {float(c['rhs']):.3g} | "
          f"{float(c['demand_band_lam']):.3g} | {float(c['eta_chord_lam1_max']):.2f} |")

open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lrs = sorted(set(r[2] for r in runs))
    fig, ax = plt.subplots(len(lrs), 4, figsize=(18, 4 * len(lrs)), squeeze=False)
    cmap = plt.get_cmap("viridis")
    targets = list(dict.fromkeys(r[1] for r in runs))
    for i, lr in enumerate(lrs):
        for j, t in enumerate(targets):
            col = cmap(j / max(len(targets) - 1, 1))
            b = runs.index(next(r for r in runs if r[1] == t and r[2] == lr))
            ax[i, 0].semilogy(np.sqrt(2 * loss[:, b]), color=col, lw=0.6, label=t)
            st, el = series(runs[b][0], t, lr, "eta_lam1")
            ax[i, 1].plot(st, el, color=col, label=t)
            _, bl = series(runs[b][0], t, lr, "band_lam_lam1")
            ax[i, 2].plot(st, bl, color=col)
            _, sl = series(runs[b][0], t, lr, f"slow_{deltas[1]}")
            ax[i, 3].semilogy(st, sl, color=col)
        ax[i, 1].axhline(2, color="k", ls="--", lw=0.8)
        ax[i, 0].set_title(f"lr_rel={lr}: error ||E|| (every step)")
        ax[i, 1].set_title("eta * lambda_1(K)")
        ax[i, 2].set_title("lambda_max(P K P) / lambda_1")
        ax[i, 3].set_title(f"slow part ||E_gamma||, gamma = 1/(eta*{deltas[1]})")
        ax[i, 0].legend(fontsize=8)
        for a_ in ax[i]:
            a_.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_eos.png"), dpi=110)
    print("wrote fig_eos.png")
except ImportError:
    print("matplotlib not available; no figure")
