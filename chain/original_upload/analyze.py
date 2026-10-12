"""
import numpy as _np
if not hasattr(_np, "trapezoid"): _np.trapezoid = _np.trapz   # numpy < 2
Post-process res_depth*.npy from l63test.py into the checks of the section
'The held rough error and the closure beyond the lazy regime'.

For each Krylov functional phi_gamma (rank-one seminorm, c = 1 exactly):
  (1) band check:  <phi,E(t)>^2 against the lower band Eq.(rough-floor-path) with the
      MEASURED sigma(t) and leak, and the upper band Eq.(rough-upper) with c = 1.
      (Both are exact consequences of the energy identity; this checks the logging and
      the size of the leak.)
  (2) a priori floor (Theorem 2):  Sigma(t) = sigma0 + a t with a = lam1bar^{1/2} H2bar ||E0||,
      against the measured sigma(t) and the measured <phi,E(t)>^2.  Also the realized
      drive d/dt sigma^2 (Prop. 15(a)) against its bound 2 sigma lam1^{1/2} H2 ||E||.
  (3) ignition times:  measured t_1/2 (component halves) against the a priori minimum
      time of Cor. 3 and against the reduced model of Prop. 8 (b = lam1^{1/2} H2, c = 1).
  (4) depth comparison: sigma0, H2, a, t_1/2 per depth.

Usage: python3 analyze.py res_depth1.npy res_depth3.npy
"""
import numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as ctz, quad

EPS = 0.5   # epsilon of the Gronwall step; exponent (2+eps), leak coefficient 1/eps

def lower_band(t, sig2, lam1, Es2, r0sq, eps=EPS):
    I = ctz((2+eps)*sig2, t, initial=0.)
    # integral_0^t lam1(s) Es2(s) exp(-(2+eps) int_s^t sig2) ds
    leak = np.array([np.trapezoid(lam1[:k+1]*Es2[:k+1]*np.exp(-(I[k]-I[:k+1])), t[:k+1]) for k in range(len(t))])
    return r0sq*np.exp(-I) - leak/eps

def upper_band(t, sig2, lam1, Es2, r0sq, c=1., eps=EPS):
    eps = min(eps, 2*c*0.9)
    I = ctz((2*c-eps)*sig2, t, initial=0.)
    leak = np.array([np.trapezoid(lam1[:k+1]*Es2[:k+1]*np.exp(-(I[k]-I[:k+1])), t[:k+1]) for k in range(len(t))])
    return r0sq*np.exp(-I) + leak/eps

def reduced_model_half_time(sigma0, r0, b, c=1.):
    # dr/dt = -c s^2 r, ds/dt = b r ;  r = r0 - c (s^3 - sigma0^3)/(3b) ;  t = int ds/(b r)
    s_half = (sigma0**3 + 3*b*r0/(2*c))**(1/3)
    f = lambda s: 1./(b*r0 - c*(s**3-sigma0**3)/3.)
    val, _ = quad(f, sigma0, s_half)
    return val

def main(files):
    summary = []
    for fn in files:
        res = np.load(fn, allow_pickle=True).item()
        logs = res['logs']; info = res['info']
        t = np.array([l['t'] for l in logs]); E = np.array([l['E'] for l in logs])
        lam1 = np.array([l['lam1'] for l in logs])
        ng = len(info)
        tag = fn.replace('.npy', '')
        # H2 along the gradient, measured along the run (lower estimate of H2bar)
        H2s = np.array([l.get('H2_g', np.nan) for l in logs]); H2bar = np.nanmax(H2s)
        lam1bar = lam1.max(); a = np.sqrt(lam1bar)*H2bar*E[0]
        fig, axs = plt.subplots(2, ng, figsize=(4*ng, 7)); axs = np.atleast_2d(axs)
        for j in range(ng):
            pE = np.array([l[f'phiE{j}'] for l in logs]); sig2 = np.array([l[f'sig2_{j}'] for l in logs])
            Es2 = np.array([l[f'Es{j}'] for l in logs])**2
            r0sq = pE[0]**2; sigma0 = np.sqrt(sig2[0])
            lo = lower_band(t, sig2, lam1, Es2, r0sq); up = upper_band(t, sig2, lam1, Es2, r0sq)
            Sig = sigma0 + a*t; I_ap = sigma0**2*t + sigma0*a*t**2 + a**2*t**3/3
            apriori = r0sq*np.exp(-(2+EPS)*I_ap)      # without the leak term (reported separately)
            # (1)+(2): energy of the component
            ax = axs[0, j]
            ax.plot(t, pE**2, 'k', lw=2, label=r'$\langle\phi,E\rangle^2$')
            ax.plot(t, np.maximum(lo, 1e-12), 'b--', label='lower band (meas. $\\sigma$)')
            ax.plot(t, up, 'r--', label='upper band (c=1)')
            ax.plot(t, np.maximum(apriori, 1e-12), 'g:', label='a priori floor (no leak)')
            ax.set_yscale('log'); ax.set_ylim(1e-4, 10); ax.set_xlabel('t')
            ax.set_title(f"$\\gamma/\\lambda_1$={info[j]['gamma']/res['lam1_0']:.0e}")
            if j == 0: ax.legend(fontsize=7)
            # (2): sigma^2 measured vs Sigma^2
            ax = axs[1, j]
            ax.plot(t, sig2, 'k', lw=2, label=r'$\sigma(t)^2$ measured')
            ax.plot(t, Sig**2, 'g:', label=r'$\Sigma(t)^2$ a priori')
            ax.set_yscale('log'); ax.set_xlabel('t'); ax.set_ylabel(r'$\|\phi\|_{K(t)}^2$')
            if j == 0: ax.legend(fontsize=7)
            # (3): ignition times
            half = np.where(pE**2 <= 0.5*r0sq)[0]
            t_half = t[half[0]] if len(half) else np.inf
            Esq_floor = 0.5*r0sq   # Cor. 3 with epsilon^2 = r0sq/2, leak dropped (optimistic)
            # solve (2+eps) I(t) = log 2 for t
            tt = np.linspace(0, t[-1], 20001); It = sigma0**2*tt + sigma0*a*tt**2 + a**2*tt**3/3
            k = np.where((2+EPS)*It >= np.log(2))[0]; t_min = tt[k[0]] if len(k) else np.inf
            # lazy prediction: sigma frozen at sigma0 -> t = log2/(2 sigma0^2)
            t_lazy = np.log(2)/(2*sigma0**2)
            # reduced model with b = lam1bar^{1/2} H2bar, c = 1, r0 = |<phi,E0>|
            t_red = reduced_model_half_time(sigma0, abs(pE[0]), np.sqrt(lam1bar)*H2bar)
            # realized drive vs bound, where logged
            dr = np.array([l.get(f'drive{j}', np.nan) for l in logs])
            bound = 2*np.sqrt(sig2)*np.sqrt(lam1)*H2s*E
            ratio = np.nanmedian(np.abs(dr)/bound)
            summary.append((tag, info[j]['gamma']/res['lam1_0'], sigma0**2, abs(pE[0]), t_half, t_min, t_lazy, t_red, ratio))
        fig.suptitle(f"{tag}: d={res['d_theta']}, lam1(0)={res['lam1_0']:.2f}, H2bar(along g)={H2bar:.2f}, a={a:.2f}")
        fig.tight_layout(); fig.savefig(f'{tag}_bands.png', dpi=120)
        print(f"{tag}: lam1bar={lam1bar:.3g}, H2bar={H2bar:.3g}, a={a:.3g}, final ||E||={E[-1]:.4f}")
    print('\nignition summary (per Krylov functional)')
    print(f"{'run':>12} {'gamma/l1':>9} {'sigma0^2':>10} {'<phi,E0>':>9} {'t_half meas':>12} {'t_min apriori':>14} {'t_lazy':>10} {'t_red model':>12} {'drive/bound':>12}")
    for s in summary:
        print(f"{s[0]:>12} {s[1]:>9.0e} {s[2]:>10.3e} {s[3]:>9.3f} {s[4]:>12.1f} {s[5]:>14.3f} {s[6]:>10.1f} {s[7]:>12.2f} {s[8]:>12.2e}")

if __name__ == '__main__':
    main(sys.argv[1:])
