"""Summary of an eos_edge.py replay: python eos/eos_edge_summary.py <edge csv> <eta>.

Two-step sums (steps k and k+1) remove the part that flips sign with the oscillation. Compares:
  * the exact two-step change of lambda_1 with its parts: the drive (first-order term from the error off v_1),
    the oscillation term (first-order term from the error on v_1), and the remainder R;
  * the remainder with the prediction eta^2 lambda_1 x^2 d2, d2 = second derivative of lambda_1 along w_1
    (taken from the nearest step where it was measured);
  * the observed x^2 with the balance x_eq^2 = -(two-step drive) / (eta^2 lambda_1 d2);
  * gradient descent's error after each step with the best error F(s) of a step of the same length."""
import csv
import sys
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
eta = float(sys.argv[2])
f = lambda k: np.array([float(r[k]) if r.get(k, "") not in ("", None) else np.nan for r in rows])
lam, x2, drive, osc, rem, exact = f("lam1"), f("x2"), f("dlam1_drive"), f("dlam1_osc"), f("remainder"), f("dlam1_exact")
d2 = f("d2lam1_w1_a")
idx = np.where(np.isfinite(d2))[0]
d2n = d2[idx[np.abs(np.arange(len(rows))[:, None] - idx[None, :]).argmin(1)]]
two = lambda v: v[:-1] + v[1:]
pred = eta ** 2 * lam * x2 * d2n
D, O, R, X, P, x2m = two(drive), two(osc), two(rem), two(exact), two(pred) / 2, (x2[:-1] + x2[1:]) / 2
print(f"steps {len(rows)}, eta*lam1 {eta*lam.min():.3f}..{eta*lam.max():.3f}, x^2 mean {x2.mean():.3e}, "
      f"d2 (lambda_1 along w_1) median {np.median(d2[idx]):+.1f} [{d2[idx].min():+.1f}, {d2[idx].max():+.1f}], "
      f"3x spacing agrees to {np.nanmedian(np.abs(f('d2lam1_w1_b')[idx] / d2[idx] - 1)):.1%}")
print(f"one-step: |oscillation term| median {np.median(np.abs(osc)):.2e} vs |drive| {np.median(np.abs(drive)):.2e} "
      f"vs |remainder| {np.median(np.abs(rem)):.2e}")
print(f"two-step means: exact {X.mean():+.3e} = drive {D.mean():+.3e} + oscillation {O.mean():+.3e} + remainder {R.mean():+.3e}")
print(f"remainder vs prediction eta^2 lambda_1 x^2 d2: mean {R.mean():+.3e} vs {P.mean():+.3e}; "
      f"correlation {np.corrcoef(R, P)[0, 1]:+.2f}; median ratio {np.median(R / P):.2f}")
A = np.vstack([np.ones_like(x2m), x2m]).T
a, bfit = np.linalg.lstsq(A, X, rcond=None)[0]
print(f"fit two-step change = a + b x^2: a {a:+.3e}, b {bfit:+.3e}; predicted b = eta^2 lambda_1 d2 = "
      f"{np.median(eta ** 2 * lam * d2n):+.3e}")
xeq = -D / (eta ** 2 * lam[:-1] * d2n[:-1])
print(f"balance x_eq^2 = -drive/(eta^2 lambda_1 d2): median {np.median(xeq):.3e} vs observed x^2 median {np.median(x2):.3e}")
Fr, gs = f("F_ratio"), f("gap_share_top")
print(f"GD vs best step of the same length: (err_after - F)/(err - F) median {np.nanmedian(Fr):.2f} "
      f"(0 = best step, 1 = no progress); share of the linearized gap on v_1 median {np.nanmedian(gs):.2f}; "
      f"GD keeps {np.median(f('keep_gd')):+.2f} of x per step, the best step keeps {np.nanmedian(f('keep_best')):.2f}")
