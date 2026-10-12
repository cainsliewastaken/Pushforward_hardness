"""Figures and summary of the larger runs (no computation), from results_large/.

fig_large_sine.{png,pdf}     replaces Figure 1 (sine targets): (a) error, (b) roughness of the output,
                             (c) at gamma = 0.3 lambda_1, last checkpoint: error, slow part ||E_gamma||
                             (CG), Krylov values F_1, F_5, F_100 at s_gamma, manifold high-pass floor.
fig_large_stepper.{png,pdf}  replaces Figure 2 (Lorenz-63): (a) validation error, (b) Krylov value
                             F_200(s_gamma) at gamma = 1e-4 lambda_1 of the centred direct and Euler
                             targets under the trained kernel of each network, and of the centred
                             identity under the kernel of the trained direct network.
summary_large.md             tables, Mann-Whitney tests and the checks of every bound."""
import csv
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("PF_LARGE_DIR", os.path.join(HERE, "results_large"))
md = []
P = lambda *a: md.append(" ".join(str(x) for x in a))


def rd(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r[a] == b for a, b in kw.items())]


# ------------------------------------------------------------------------------ sine
if os.path.exists(os.path.join(D, "sine_rows.csv")):
    S = rd(os.path.join(D, "sine_rows.csv"))
    meta = json.load(open(os.path.join(D, "sine_meta.json")))
    names = list(dict.fromkeys(r["target"] for r in S))
    steps = sorted({r["step"] for r in S})
    last = steps[-1]
    seeds = sorted({r["seed"] for r in S})

    def stat(key, step, g=0.3):
        a = np.array([[r[key] for r in sel(S, target=nm, step=step, gamma_rel=g)] for nm in names])
        return a.mean(1), a.min(1), a.max(1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    cols = ["#9ecae1", "#3182bd", "#08306b"]
    for step, c in zip(steps, cols[-len(steps):]):
        x, _, _ = stat("RU", step)
        o = np.argsort(x)
        for a, key in zip(ax[:2], ("err", "RN")):
            m, lo, hi = stat(key, step)
            a.errorbar(x[o], m[o], yerr=[(m - lo)[o], (hi - m)[o]], color=c, marker="o", ms=3.5, capsize=2,
                       lw=1.1, label=f"{int(step)} steps")
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    x, _, _ = stat("RU", last)
    o = np.argsort(x)
    for key, c, ls, mk, lab in (("err", "k", "-", "o", r"error $\|E\|$"),
                                ("E_gamma", "#08306b", "-", "s", r"slow part $\|E_\gamma\| = F(s_\gamma)$"),
                                ("F_M100", "#006d2c", "none", "^", r"Krylov value $F_{100}(s_\gamma)$"),
                                ("F_M5", "#31a354", "--", "v", r"Krylov value $F_5(s_\gamma)$"),
                                ("F_M1", "#a1d99b", "--", "d", r"Krylov value $F_1(s_\gamma)$"),
                                ("floor", "#e6550d", ":", "o", "floor, manifold high-pass")):
        m, lo, hi = stat(key, last)
        ax[2].errorbar(x[o], m[o], yerr=[(m - lo)[o], (hi - m)[o]], color=c, ls=ls, marker=mk, ms=4,
                       mfc="none" if ls == "none" else c, capsize=2, lw=1.1, label=lab)
    ax[0].set_title(r"(a) Error $\|E\|$", fontsize=10)
    ax[1].set_title(r"(b) Roughness of the network output $\mathcal{R}_m(\mathcal{N}_\theta)$", fontsize=10)
    ax[2].set_title(rf"(c) Bounds at $\gamma = 0.3\lambda_1$, {int(last)} steps", fontsize=10)
    for a in ax:
        a.set_xlabel(r"roughness of the target $\mathcal{R}_m(U)$")
        a.legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(D, f"fig_large_sine.{ext}"), dpi=150)

    cfg = meta["cfg"]
    P("# Larger runs: sine targets (replaces Figure 1)\n")
    P(f"width {cfg['width']}, n = {cfg['n']}, {len(names)} targets, {len(seeds)} initializations, "
      f"{cfg['steps']} GD steps, lr {cfg['lr']}, manifold high-pass k = {cfg['knn']}, m = {cfg['m_hp']}. "
      f"Krylov values up to M = {meta['MMAX']}; slow part and s_gamma by matrix-free CG. "
      f"Wall time {meta['seconds'] / 60:.0f} min.\n")
    P(f"Last checkpoint ({int(last)} steps), gamma = 0.3 lambda_1, means over initializations [min, max]:\n")
    P("| target | R_m(U) | R_m(N) | error | slow part | F_1 | F_5 | F_20 | F_100 | exact F(s) | floor |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in names:
        q = sel(S, target=nm, step=last, gamma_rel=0.3)
        f = lambda k: f"{np.mean([r[k] for r in q]):.3f}"
        fr = lambda k: (f"{np.mean([r[k] for r in q]):.3f} [{min(r[k] for r in q):.3f}, "
                        f"{max(r[k] for r in q):.3f}]")
        P(f"| {nm} | {f('RU')} | {f('RN')} | {fr('err')} | {f('E_gamma')} | {f('F_M1')} | {f('F_M5')} | "
          f"{f('F_M20')} | {f('F_M100')} | {f('F_exact') if 'F_exact' in q[0] else '-'} | {fr('floor')} |")
    P("\nError along training (means over initializations):\n")
    P("| target | " + " | ".join(f"{int(s)} steps" for s in steps) + " |")
    P("|---|" + "---|" * len(steps))
    for nm in names:
        P(f"| {nm} | " + " | ".join(f"{np.mean([r['err'] for r in sel(S, target=nm, step=s, gamma_rel=0.3)]):.3f}"
                                    for s in steps) + " |")
    P("\nChecks over all rows (every checkpoint, run and gamma):\n")
    for k, v in meta["checks"].items():
        P(f"- {k}: {v}")
    # Krylov convergence: terms needed to reach all but 1e-3 of the slow part
    P("\nKrylov convergence at the last checkpoint: median over runs of F_M(s_gamma) / ||E_gamma||:\n")
    gs = sorted({r["gamma_rel"] for r in S})
    Ms = meta["MS"]
    P("| gamma / lambda_1 | " + " | ".join(f"M = {M}" for M in Ms) + " |")
    P("|---|" + "---|" * len(Ms))
    for g in gs:
        q = [r for r in sel(S, step=last, gamma_rel=g) if r["E_gamma"] > 0]
        P(f"| {g:g} | " + " | ".join(f"{np.median([r[f'F_M{M}'] / r['E_gamma'] for r in q]):.4f}" for M in Ms) + " |")

# ------------------------------------------------------------------------------ Lorenz
files = sorted(glob.glob(os.path.join(D, "lorenz_rows_w*.csv")))
if files:
    L = [r for f in files for r in rd(f)]
    T = rd(sorted(glob.glob(os.path.join(D, "lorenz_targets_w*.csv")))[0])
    widths = sorted({r["width"] for r in L})
    hs = sorted({r["h"] for r in L})
    g = "0.0001"
    col = dict(direct="tab:blue", euler="tab:red", identity="tab:green")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    lss = ["-", "--", ":"]
    for W, ls in zip(widths, lss):
        fill = "full" if ls == "-" else "none"
        for a, items in ((ax[0], (("direct", "err_val", "direct", "o"), ("euler", "err_val", "Euler", "s"))),
                         (ax[1], (("direct", f"V_g{g}_F_M200", "direct target", "o"),
                                  ("euler", f"V_g{g}_F_M200", "Euler target", "s"),
                                  ("direct", f"id_g{g}_F_M200", "identity map", "^")))):
            for s, key, lab, mk in items:
                c = col["identity"] if lab == "identity map" else col[s]
                hv = [h for h in hs if sel(L, h=h, width=W, stepper=s)]
                st = [np.array([r[key] for r in sel(L, h=h, width=W, stepper=s)]) for h in hv]
                m = np.array([np.median(x) for x in st])
                lo, hi = np.array([x.min() for x in st]), np.array([x.max() for x in st])
                a.errorbar(hv, m, yerr=[m - lo, hi - m], color=c, ls=ls, marker=mk, ms=4.5, fillstyle=fill,
                           capsize=2, lw=1.1, label=f"{lab}, width {int(W)}")
    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("step size $h$")
        a.legend(fontsize=7)
    ax[0].set_ylabel("validation error, state units")
    ax[0].set_title("(a) Error of the trained steppers", fontsize=10)
    ax[1].set_ylabel("state units")
    ax[1].set_title(r"(b) Krylov value $F_{200}(s_\gamma)$, trained kernel, $\gamma = 10^{-4}\lambda_1$", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(D, f"fig_large_stepper.{ext}"), dpi=150)

    meta = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "lorenz_meta_w*.json")))]
    c0 = (meta[0] if meta else {"cfg": {}})["cfg"]
    P("\n# Larger runs: Lorenz-63 direct vs Euler (replaces Figure 2)\n")
    P(f"widths {[int(w) for w in widths]}, {c0.get('n_tr')} training states, {c0.get('n_va')} validation states "
      f"from independent trajectories, {len(c0.get('inits', []))} initializations, {c0.get('steps')} Adam steps. "
      f"Krylov values up to M = 200; slow part and s_gamma by matrix-free CG.\n")
    P("Sizes of the centred targets (state units):\n")
    P("| h | direct | Euler | identity | ratio |")
    P("|---|---|---|---|---|")
    for t in T:
        P(f"| {t['h']:g} | {t['size_direct']:.3f} | {t['size_euler']:.4f} | {t['size_identity']:.3f} | "
          f"{t['size_direct'] / t['size_euler']:.1f} |")
    P("\nValidation error (medians [min, max] over initializations), two-sided Mann-Whitney p, and whether the "
      "ranges are disjoint:\n")
    P("| h | width | direct | Euler | D/E | p | Euler better (disjoint) | direct better (disjoint) |")
    P("|---|---|---|---|---|---|---|---|")
    for W in widths:
        for h in hs:
            vd = np.array([r["err_val"] for r in sel(L, h=h, width=W, stepper="direct")])
            ve = np.array([r["err_val"] for r in sel(L, h=h, width=W, stepper="euler")])
            if len(vd) == 0 or len(ve) == 0:
                continue
            p = mannwhitneyu(vd, ve, alternative="two-sided").pvalue
            P(f"| {h:g} | {int(W)} | {np.median(vd):.4f} [{vd.min():.4f}, {vd.max():.4f}] | {np.median(ve):.4f} "
              f"[{ve.min():.4f}, {ve.max():.4f}] | {np.median(vd) / np.median(ve):.2f} | {p:.2g} | "
              f"{ve.max() < vd.min()} | {vd.max() < ve.min()} |")
    P(f"\nKrylov value F_200(s_gamma) at gamma = 1e-4 lambda_1 (medians), its ratio to the CG slow part, "
      "and relative errors (median over runs):\n")
    P("| h | width | direct target | Euler target | identity | F_200 / slow (D, E, id) | train err D | train err E |")
    P("|---|---|---|---|---|---|---|---|")
    for W in widths:
        for h in hs:
            q = {s: sel(L, h=h, width=W, stepper=s) for s in ("direct", "euler")}
            if not q["direct"]:
                continue
            md_ = lambda s, k: np.median([r[k] for r in q[s]])
            rr = lambda s, u: np.median([r[f"{u}_g{g}_F_M200"] / r[f"{u}_g{g}_slow"] for r in q[s]])
            P(f"| {h:g} | {int(W)} | {md_('direct', f'V_g{g}_F_M200'):.4f} | {md_('euler', f'V_g{g}_F_M200'):.4f} | "
              f"{md_('direct', f'id_g{g}_F_M200'):.4f} | {rr('direct', 'V'):.4f}, {rr('euler', 'V'):.4f}, "
              f"{rr('direct', 'id'):.4f} | {md_('direct', 'err_train'):.4f} | {md_('euler', 'err_train'):.4f} |")
    # checks
    bad, tot, worst = 0, 0, {}
    for r in L:
        for u in ("E", "V", "id"):
            for gg in ("1e-05", "0.0001", "0.001", "0.01"):
                p = f"{u}_g{gg}"
                if p + "_F_M200" not in r or r[p + "_F_M200"] == "":
                    continue
                tot += 1
                Fs = [r[f"{p}_F_M{M}"] for M in (1, 2, 5, 10, 20, 50, 100, 200)]
                tol = 1e-9 * r[f"{u}_norm"]
                ok = all(a <= b + tol for a, b in zip(Fs, Fs[1:])) and Fs[-1] <= r[f"{p}_slow"] * (1 + 1e-6) + tol
                if f"{p}_F_exact" in r and r[f"{p}_F_exact"] != "":
                    ok = ok and Fs[-1] <= r[f"{p}_F_exact"] + tol
                    worst["max |slow_CG - exact| / |u|"] = max(worst.get("max |slow_CG - exact| / |u|", 0),
                                                                abs(r[f"{p}_slow"] - r[f"{p}_res_exact"]) / r[f"{u}_norm"])
                    worst["max |F(s) exact - slow_CG| / |u|"] = max(worst.get("max |F(s) exact - slow_CG| / |u|", 0),
                                                                    abs(r[f"{p}_F_exact"] - r[f"{p}_slow"]) / r[f"{u}_norm"])
                worst["max CG residual"] = max(worst.get("max CG residual", 0), r[f"{p}_cg_res"])
                worst["max Lanczos orthogonality loss"] = max(worst.get("max Lanczos orthogonality loss", 0), r[f"{u}_orth"])
                bad += not ok
    if "lam1_exact" in L[0]:
        worst["max |lam1 Ritz / exact - 1|"] = max(abs(r["lam1"] / r["lam1_exact"] - 1) for r in L)
    P(f"\nChecks: F_M nondecreasing in M and F_200 <= slow part (and <= exact F(s) where computed): "
      f"{tot - bad}/{tot} ok.")
    for k, v in worst.items():
        P(f"- {k}: {v:.2e}")
    P(f"- clip fraction: median {np.median([r['clip_frac'] for r in L]):.3f}, max {max(r['clip_frac'] for r in L):.3f}")
    P(f"- loss variation over the last 10% of steps: max {max(r['tail_loss_var'] for r in L):.4f}")

open(os.path.join(D, "summary_large.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
