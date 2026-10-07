"""Tables for follow-up 12 (checks of multistep_best_window.tex and nudged training). Prints markdown.

  python followup12_report.py [--out results/tables_followup12.md]
Reads rate_checks.pt, theory_checks_v.pt, robust_eval_*_part*.pt, and the new runs' meta.json / log.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from msr.config import load_cfg


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}g}"


def rng(vals, d=3):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return "--" if not v else (f"{min(v):.{d}g}" if len(v) == 1 else f"{min(v):.{d}g}-{max(v):.{d}g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.cfg, a.set)
    R = cfg["out_root"]
    lines = []
    P = lines.append

    rc = torch.load(os.path.join(R, "rate_checks.pt"), weights_only=False) if os.path.exists(os.path.join(R, "rate_checks.pt")) else {}
    runs = sorted({k[0] for k in rc})
    base = [r for r in runs if "nudge" not in r and "free_lr" not in r]
    new = [r for r in runs if r not in base]

    # ---- curvature vs window
    P("## Curvature of the window loss, free and nudged (lambda_max of the Gauss-Newton matrix, Euler, seed range)\n")
    for run in runs:
        ks = [k for k in rc if k[0] == run and "curv" in rc[k]]
        if not ks:
            continue
        cols = sorted({c for k in ks for c in rc[k]["curv"]}, key=lambda c: (c[0], -1 if c[1] is None else c[1]))
        P(f"**{run}**\n")
        P("| window | " + " | ".join(f"r0={'free' if c[1] is None else c[1]}" for c in cols if c[0] == cols[0][0]) + " |")
        P("|---|" + "---|" * len([c for c in cols if c[0] == cols[0][0]]))
        for T in sorted({c[0] for c in cols}):
            row = [rng([rc[k]["curv"].get((T, r0)) for k in ks]) for (_, r0) in [c for c in cols if c[0] == T]]
            P(f"| {T:g} | " + " | ".join(row) + " |")
        P("")

    # ---- rates
    P("## Rates per unit motion on the free error of the starts inside the radius (T = 1.28, Euler)\n")
    P("R_X = <a, g_X>/|g_X|; lower = certified lower bound on the nudged rate; upper = bound on the free rate; c = inside fraction.\n")
    P("| run | seed | r0 | c | |a| | R nudged | lower | R one-step | R free | upper free | delta/(c|a|) | |g_out|/(c|a|) | cos(a,g_out) | in-dep | cert > 1step | cert > free |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(rc):
        if "rates" not in rc[k]:
            continue
        for r0, r in sorted(rc[k]["rates"]["by_r0"].items()):
            if "note" in r:
                P(f"| {k[0]} | {k[2]} | {r0:g} | {fmt(r['c'])} | no partition |")
                continue
            P(f"| {k[0]} | {k[2]} | {r0:g} | {fmt(r['c'])} | {fmt(r['a_norm'])} | {fmt(r['R_nudged'])} | {fmt(r['lower_nudged'])} | "
              f"{fmt(r['R_onestep'])} | {fmt(r['R_free'])} | {fmt(r['upper_free'])} | {fmt(r['delta_over_ca'])} | "
              f"{fmt(r['gout_norm'] / (r['c'] * r['a_norm']))} | {r['cos_a_gout']:+.3f} | {fmt(r['in_departure'], 2)} | "
              f"{'yes' if r['cert_vs_onestep'] else 'no'} | {'yes' if r['cert_vs_free'] else 'no'} |")
    P("")

    # ---- surrogate
    P("## Nudged loss against the free T-step error (T = 1.28): gap and bound\n")
    P("| run | seed | r0 | V_T free | V_T nudged | gap | bound | max pre-nudge deviation |")
    P("|---|---|---|---|---|---|---|---|")
    for k in sorted(rc):
        if "surrogate" not in rc[k]:
            continue
        s = rc[k]["surrogate"]
        for r0, r in sorted(s["by_r0"].items()):
            P(f"| {k[0]} | {k[2]} | {r0:g} | {fmt(s['V_T'])} | {fmt(r['V_T_nudged'])} | {fmt(r['gap'])} | {fmt(r['bound'])} | {fmt(r['max_dev_nudged'])} |")
    P("")

    # ---- screen
    P("## Online rule: first-term screen against the exact decrease of the N-step loss (N = 2.56, fixed step size)\n")
    P("alpha such that the window-N step has length lr sqrt(P). exact = argmax_p of the measured decrease; fo = argmax of the first-order "
      "decrease alpha <g_V, g_p>; greedy = first p at which a_(p+1) - abar_p < 0.\n")
    P("| run | seed | r0 | lr | exact p* (decrease) | decrease at N | fo p* | greedy stop (decrease) | signs of a_(p+1)-abar_p |")
    P("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(rc):
        if "screen" not in rc[k]:
            continue
        for r0, s in sorted(rc[k]["screen"].items(), key=lambda kv: (-1 if kv[0] is None else kv[0])):
            signs = "".join("+" if x > 0 else "-" for x in s["first_term"])
            for row in s["steps"]:
                P(f"| {k[0]} | {k[2]} | {'free' if r0 is None else r0} | {row['lr']:.0e} | {row['argmax_exact']} ({row['best_dec']:.2e}) | "
                  f"{row['dec_at_N']:.2e} | {row['argmax_first_order']} | {row['greedy_stop']} ({row['dec_at_greedy']:.2e}) | `{signs}` |")
    P("")

    # ---- follow-up 11: unremovable fraction vs window at several step lengths
    tv = os.path.join(R, "theory_checks_v.pt")
    if os.path.exists(tv):
        res = torch.load(tv, weights_only=False)
        ks = [k for k in res if k[0] == "dt0.08_onestep" and k[1] == "euler" and k[2] == 0]
        if ks:
            P("## Follow-up 11: unremovable fraction F(s)/|e| of the window error against the window, one-step Euler model, seed 0\n")
            lrs = sorted({x["lr"] for k in ks for x in res[k]["steps"]})
            P("| window | lambda_max | tr/lambda | " + " | ".join(f"lr {lr:.0e}" for lr in lrs) + " |")
            P("|---|---|---|" + "---|" * len(lrs))
            for k in sorted(ks, key=lambda k: k[3]):
                r = res[k]
                F = {x["lr"]: x["F"] for x in r["steps"]}
                P(f"| {k[3]:g} | {r['lam_max']:.3g} | {r['tr_over_lam']:.3g} | " + " | ".join(fmt(F.get(lr)) for lr in lrs) + " |")
            P("")

    # ---- training: own loss, curvature at start, robust evaluation
    rob = {}
    for f in glob.glob(os.path.join(R, "robust_eval_*part*.pt")) + glob.glob(os.path.join(R, "robust_eval_part*.pt")):
        rob.update(torch.load(f, weights_only=False))
    runs_tr = sorted({os.path.basename(d) for pat in ("dt0.08_unroll*_s4k_*lr1e-*", "dt0.08_unroll*_s4k_*line*", "dt0.08_unroll*_s4k_*nudge*")
                      for d in glob.glob(os.path.join(R, "runs", pat))})
    ref = ["dt0.08_unroll032_ft_s4k", "dt0.08_unroll064_ft_s4k_fixed", "dt0.08_unroll128_ft_s4k_fixed", "dt0.08_unroll128_ft_s4k_inv"]
    P("## Nudged training at a fixed learning rate (4000 steps from the 0.64-window s4k fixed models, Euler, 3 seeds)\n")
    P("own: the run's own held-out loss, end/start and largest increase; lam0: curvature of the run's own loss at its start; "
      "errors |e_k(T)| on 4096 held-out starts, rms and median, seed ranges.\n")
    P("| run | lam0 | own end/start | own spike | median alpha | rms 0.32 | med 0.32 | rms 1.28 | med 1.28 | p99 1.28 | rms 2.56 | med 2.56 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in ref + runs_tr:
        d = os.path.join(R, "runs", run)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        names = [tuple(x) for x in meta["names"]]
        ej = [j for j, (c, s) in enumerate(names) if c == "euler"]
        lam0 = [meta["lam_start"][j] for j in ej] if meta.get("lam_start") else []
        own_es, own_sp, alphas = [], [], []
        if os.path.exists(os.path.join(d, "log.csv")):
            log = np.genfromtxt(os.path.join(d, "log.csv"), delimiter=",", names=True)
            for c, s in names:
                col = f"own_{c}{s}"
                if c == "euler" and col in log.dtype.names:
                    v = log[col]
                    own_es.append(float(v[-1] / v[0]))
                    own_sp.append(float(v.max() / v[0]))
                acol = f"alpha_{c}{s}"
                if c == "euler" and acol in log.dtype.names:
                    alphas.append(float(np.median(log[acol][1:])))
        stats = {}
        for T in (0.32, 1.28, 2.56):
            for q in ("rms", "med", "p99"):
                vals = []
                for s in (0, 1, 2):
                    k = (run, "euler", s)
                    if k in rob:
                        Ts = list(rob[k]["T"])
                        if T in Ts:
                            vals.append(float(rob[k][q][Ts.index(T)]))
                stats[(T, q)] = rng(vals)
        P(f"| {run} | {rng(lam0)} | {rng(own_es, 2)} | {rng(own_sp, 2)} | {rng(alphas, 2)} | {stats[(0.32, 'rms')]} | {stats[(0.32, 'med')]} | "
          f"{stats[(1.28, 'rms')]} | {stats[(1.28, 'med')]} | {stats[(1.28, 'p99')]} | {stats[(2.56, 'rms')]} | {stats[(2.56, 'med')]} |")
    P("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
