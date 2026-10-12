"""Direct vs Euler steppers on Lorenz-63 at larger widths.

Same protocol as part C of replicate_rsk.py (5000 training and 5000 validation states, inputs
standardized, targets centred and divided by their root-mean-square size, one-hidden-layer tanh
network with the output layer at zero, full-batch Adam for 6000 steps, cosine schedule from 3e-3,
gradient clipping at 0.05, 5 initializations shared by the two steppers). Only the width changes.
No kernel diagnostics. Results are written after every (width, h), so a time limit loses little.

Output: results/wide_steppers.csv (one row per run), results/wide_steppers_summary.csv."""
import os
import sys
import time
import numpy as np
import torch
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import replicate_rsk as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
WIDTHS = [int(w) for w in os.environ.get("WIDE_WIDTHS", "512,2048").split(",")]
TAG = "_w" + "_".join(str(w) for w in WIDTHS)   # separate files per process when widths run in parallel
HS = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
INITS, N_TR, N_VA, STEPS = list(range(5)), 5000, 5000, 6000

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={dev} widths={WIDTHS} hs={HS} inits={INITS} steps={STEPS}", flush=True)
data = {}
for h in HS:
    t0 = time.time()
    data[h] = R.stepper_data(h, N_TR, N_VA, seed=0)
    print(f"data h={h}: {time.time() - t0:.0f}s", flush=True)

rows, summ = [], []
for H in WIDTHS:
    for h in HS:
        t0 = time.time()
        X0, X1, V0, V1 = data[h]
        mu, sd = X0.mean(0), X0.std(0)
        xt = torch.tensor((X0 - mu) / sd, dtype=torch.float64, device=dev)
        xv = torch.tensor((V0 - mu) / sd, dtype=torch.float64, device=dev)
        tg = {"direct": X1, "euler": X1 - X0}
        cen = {s: tg[s].mean(0) for s in tg}
        sc = {s: np.sqrt(np.mean(np.sum((tg[s] - cen[s]) ** 2, 1))) for s in tg}
        Ys = [torch.tensor((tg[s] - cen[s]) / sc[s], dtype=torch.float64, device=dev)
              for s in ("direct", "euler") for _ in INITS]
        params, _, fclip, tail = R.train_batch(xt, Ys, H, INITS * 2, dev, steps=STEPS)
        for b, (s, i) in enumerate([(s, i) for s in ("direct", "euler") for i in INITS]):
            p = params[b]
            with torch.no_grad():
                def pred(xs, X):
                    o = R.net_out(p, xs).cpu().numpy() * sc[s] + cen[s]
                    return o if s == "direct" else X + o
                err_tr = float(np.sqrt(np.mean(np.sum((pred(xt, X0) - X1) ** 2, 1))))
                err_va = float(np.sqrt(np.mean(np.sum((pred(xv, V0) - V1) ** 2, 1))))
            rows.append(dict(h=h, width=H, stepper=s, init=i, scale=sc[s], err_train=err_tr, err_val=err_va,
                             rel_err_val=err_va / sc[s], clip_frac=float(fclip[b]), tail_loss_var=float(tail[b])))
        g = lambda s, k: np.array([r[k] for r in rows if r["h"] == h and r["width"] == H and r["stepper"] == s])
        vd, ve = g("direct", "err_val"), g("euler", "err_val")
        pval = float(mannwhitneyu(vd, ve, alternative="two-sided").pvalue)
        summ.append(dict(width=H, h=h, val_direct=float(np.median(vd)), val_euler=float(np.median(ve)),
                         train_direct=float(np.median(g("direct", "err_train"))),
                         train_euler=float(np.median(g("euler", "err_train"))),
                         ratio_direct_over_euler=float(np.median(vd) / np.median(ve)), p_mannwhitney=pval,
                         euler_better_all=bool(ve.max() < vd.min()), direct_better_all=bool(vd.max() < ve.min()),
                         rate_direct=float(np.median(vd) / h), rate_euler=float(np.median(ve) / h),
                         rel_direct=float(np.median(g("direct", "rel_err_val"))),
                         rel_euler=float(np.median(g("euler", "rel_err_val"))),
                         size_direct=float(sc["direct"]), size_euler=float(sc["euler"]),
                         clip_direct=float(np.median(g("direct", "clip_frac"))),
                         clip_euler=float(np.median(g("euler", "clip_frac"))),
                         tail_var_max=float(max(g("direct", "tail_loss_var").max(), g("euler", "tail_loss_var").max()))))
        q = summ[-1]
        print(f"width={H:5d} h={h:<5} val direct={q['val_direct']:.4f} euler={q['val_euler']:.4f} "
              f"D/E={q['ratio_direct_over_euler']:.2f} p={pval:.3f} | rate D={q['rate_direct']:.3f} E={q['rate_euler']:.3f} "
              f"| rel D={q['rel_direct']:.4f} E={q['rel_euler']:.4f} | clip D={q['clip_direct']:.2f} E={q['clip_euler']:.2f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        C.write_rows(os.path.join(OUT, f"wide_steppers{TAG}.csv"), rows)
        C.write_rows(os.path.join(OUT, f"wide_steppers_summary{TAG}.csv"), summ)
print("ok", flush=True)
