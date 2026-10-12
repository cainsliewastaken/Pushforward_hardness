"""Collect results/*.csv into results/summary.md: the pass/fail table for every checked inequality
(smallest relative margin, or largest relative difference for equalities, with the worst case) and
compact summary tables for parts 2 and 3. report.md is written from this file."""
import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                if v in ("True", "False"):
                    r[k] = v == "True"
    return rows


def g(x, p=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{p}g}"


out = []
out.append("## Pass/fail table\n")
out.append("Margin = (side that should be larger) - (other side), divided by the scale named in each CSV row "
           "(usually ||E|| or the larger side). Equalities report the largest relative difference. "
           "Rows starting with 'info:' are not consequences of the theory (first-order readings) and are "
           "reported for comparison only.\n")
out.append("| part | check | pass/total | smallest rel. margin (or largest rel. diff) | worst case |")
out.append("|---|---|---|---|---|")
for fn in ("unit_checks.csv", "sine_checks.csv", "lorenz_checks.csv"):
    rows = read(fn)
    groups = defaultdict(list)
    for r in rows:
        groups[r["check"]].append(r)
    for chk, rs in groups.items():
        eq = rs[0].get("kind") == "eq"
        npass = sum(1 for r in rs if r["passed"] is True)
        if eq:
            w = max(rs, key=lambda r: r["rel_diff"])
            val = f"diff {g(w['rel_diff'])}"
        else:
            w = min(rs, key=lambda r: r["rel_margin"])
            val = f"{g(w['rel_margin'])}"
        status = "PASS" if npass == len(rs) else ("**FAIL**" if not chk.startswith("info") else "info: fails")
        out.append(f"| {rs[0]['part']} | {chk} | {status} {npass}/{len(rs)} | {val} | {w['case']} |")

# ---------------- part 2 summary
sc = read("sine_checkpoints.csv")
if sc:
    out.append("\n## Sine targets: checkpoint summary (seed-averaged; s = next GD step)\n")
    out.append("| target | step | ‖E‖ | s | (‖E‖-F)/‖E‖ | F₁ share of gain | M for 1% | rank K | null part of E | "
               "q_E | ‖E‖_{K+,50}/‖E‖_{K+,1} | floor (best λc) | slow error there |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in sc:
        if r["s_mult"] == 1:
            agg[(r["target"], r["step"])].append(r)
    order = sorted(agg, key=lambda k: (k[0] != "random", float(k[0][3:]) if k[0] != "random" else 1e9, k[1]))
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    for key in order:
        rs = agg[key]
        out.append("| " + " | ".join([key[0], str(int(key[1]))] + [g(mean(rs, k)) for k in
                   ("norm_E", "s", "gain_frac", "F1_share", "M_1pct", "rank", "null_part", "q_E",
                    "kinv_Mmax_over_M1", "floor_best", "slow_err_at_best")]) + " |")
    out.append("\nSupplementary step lengths (s = mult x next GD step): median M for F_M within 1% of the gain, "
               "and median F₁ share of the gain, over seeds, targets and checkpoints.\n")
    out.append("| s mult | median M for 1% | max M for 1% | median F₁ share | median kappa |")
    out.append("|---|---|---|---|---|")
    for m in sorted({r["s_mult"] for r in sc}):
        rs = [r for r in sc if r["s_mult"] == m]
        med = lambda v: sorted(v)[len(v) // 2]
        out.append(f"| {g(m)} | {g(med([r['M_1pct'] for r in rs]))} | {g(max(r['M_1pct'] for r in rs))} | "
                   f"{g(med([r['F1_share'] for r in rs]))} | {g(med([r['kappa'] for r in rs if not math.isnan(r['kappa'])] or [float('nan')]))} |")

# ---------------- part 3 summary
ini = read("lorenz_init.csv")
if ini:
    out.append("\n## Lorenz-63 at initialization (Theorem 'The direct step needs the longer step'), ϱ = 0.5, "
               "seed-averaged where numeric\n")
    out.append("| h | convention | ‖Φ_h - id‖ | d₀ | (1+ϱ)‖Φ_h-id‖ | (b) holds (seeds) | best Krylov bound on s_D | "
               "exact s_D | exact s_E | upper bound s_E | sufficient condition (seeds) | s_D > s_E exact (seeds) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in ini:
        if r["rho"] == 0.5:
            agg[(r["h"], r["conv"])].append(r)
    for (h, conv), rs in sorted(agg.items()):
        m = lambda k: g(sum(r[k] for r in rs) / len(rs))
        cnt = lambda k: f"{sum(1 for r in rs if r[k] is True)}/{len(rs)}"
        out.append(f"| {g(h)} | {conv} | {m('norm_dPhi')} | {m('d0')} | {g(1.5 * sum(r['norm_dPhi'] for r in rs) / len(rs))} | "
                   f"{cnt('cond_b')} | {m('sD_bound_a_best')} | {m('sD_exact')} | {m('sE_exact')} | {m('sE_upper')} | "
                   f"{cnt('sufficient_condition_holds')} | {cnt('sD_gt_sE_exact')} |")
    rr = [r for r in ini if r["rho"] == 0.5 and r["conv"].startswith("raw")]
    out.append("\nd₀ sensitivity to the numerical-rank tolerance (relative eigenvalue cut 1e-8 / 1e-12 (used) / 1e-16), "
               "and rank of K₀:\n")
    out.append("| h | seed | rank K₀ | d₀ (1e-8) | d₀ (1e-12) | d₀ (1e-16) | ‖id‖_{K+,1} | ‖id‖_{K+,50} | ‖id‖_{K+} |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rr:
        out.append(f"| {g(r['h'])} | {int(r['seed'])} | {int(r['rank_K0'])} | {g(r['d0_rtol1e8'])} | {g(r['d0'])} | "
                   f"{g(r['d0_rtol1e16'])} | {g(r['kinv_id_M1'])} | {g(r['kinv_id_Mmax'])} | {g(r['kinv_id'])} |")
runs = read("lorenz_runs.csv")
if runs:
    out.append("\n## Lorenz-63 training runs (seed-averaged)\n")
    out.append("| h | scaling | method | c | initial error | final error | min error | loss increases (steps, of 6000) | "
               "max rel. increase | ever above initial (seeds) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in runs:
        agg[(r["h"], r["scaling"], r["method"])].append(r)
    for key, rs in sorted(agg.items()):
        m = lambda k: g(sum(r[k] for r in rs) / len(rs))
        out.append(f"| {g(key[0])} | {key[1]} | {key[2]} | {m('c')} | {m('err0')} | {m('err_final')} | {m('err_min')} | "
                   f"{m('n_increases')} | {g(max(r['max_rel_increase'] for r in rs))} | "
                   f"{sum(1 for r in rs if r['ever_above_initial'] is True)}/{len(rs)} |")
G = read("lorenz_G.csv")
if G:
    out.append("\n## Lorenz-63: integration theorem (G(t) approximated by a Riemann sum over Jacobians)\n")
    out.append("| h | scaling | seed | 2‖Φ_h - id‖ | last eval step with 2‖Φ_h-id‖ < G (lower cert.) | G at end [lo, up] (left rule) | "
               "G at end (right rule, lo) | ‖E_D‖ end | ‖E_E‖ end | Euler smaller at end |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    for r in G:
        agg[(r["h"], r["scaling"], r["seed"])].append(r)
    for key, rs in sorted(agg.items()):
        rs = sorted(rs, key=lambda r: r["step"])
        held = [r["step"] for r in rs if r.get("cond_2dphi_lt_G") is True]
        e = rs[-1]
        out.append(f"| {g(key[0])} | {key[1]} | {int(key[2])} | {g(e['two_dphi'])} | "
                   f"{g(max(held)) if held else 'never'} | "
                   f"[{g(e['G_left_lo'])}, {g(e['G_left_up'])}] | {g(e['G_right_lo'])} | {g(e['err_D'])} | {g(e['err_E'])} | "
                   f"{g(e['err_E'] < e['err_D'])} |")
ck = read("lorenz_ckpt.csv")
if ck:
    out.append("\n## Lorenz-63: Corollary 'Any training time' at checkpoints (seed 0; step lengths in parameter units)\n")
    out.append("| h | scaling | step | ‖E_D‖_{K+}/c (remove direct error) | ‖E_E‖_{K+}/c (remove Euler error) | "
               "‖u‖_{K+,1}/c | ‖u‖_{K+,50}/c | best Krylov bound on s_D (ϱ=.5) | exact s_D (ϱ=.5) | exact s_E (ϱ=.5) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    idx = {(r["h"], r["seed"], r["scaling"], r["method"], r["step"]): r for r in ck}
    for (h, seed, scaling, method, step), r in sorted(idx.items()):
        if method != "direct" or seed != 0:
            continue
        e = idx.get((h, seed, scaling, "euler", step), {})
        out.append(f"| {g(h)} | {scaling} | {int(step)} | {g(r['remove_len'])} | {g(e.get('remove_len', float('nan')))} | "
                   f"{g(r['kinv_u_M1'])} | {g(r['kinv_u_Mmax'])} | {g(r['sD_bound_c_best_rho0.5'])} | "
                   f"{g(r['sD_exact_rho0.5'])} | {g(e.get('sE_exact_rho0.5', float('nan')))} |")

with open(os.path.join(RES, "summary.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
