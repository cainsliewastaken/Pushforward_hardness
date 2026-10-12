"""Spike theory, few-step test on lay_*.npy. For each run: hidden-layer spike amplitude a(tau) (max over hidden
layers of top sv of dW_l), kernel lift L(tau)=lam100(tau)/lam100(0), E_tr, E_val at the end.
(ii) Does a(tau_f) (and its log-growth rate) predict a(T), L(T), E_tr(T), E_val(T)? Spearman across runs.
(iii) Amplitude law: a(tau) per run, and a(tau)/a(T) collapse."""
import numpy as np, sys
from scipy.stats import spearmanr
R = []
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item(); L = len(r['snap_svW'][0]); taus = r['snap_t']*r['lam1_0']
    amp = np.array([max(r['snap_spike_amp'][j][1:L-1]) if L > 2 else r['snap_spike_amp'][j][1] for j in range(len(taus))])
    w0 = np.sort(np.linalg.eigvalsh(r['snap_K'][0].astype(np.float64)))[::-1]
    lift = np.array([np.sort(np.linalg.eigvalsh(r['snap_K'][j].astype(np.float64)))[::-1][99]/w0[99] for j in range(len(taus))])
    R.append(dict(tag=r['tag'], D=r['Delta'], depth=r['depth'], H=r['H'], s=r['outscale'], taus=taus, amp=amp, lift=lift, Etr=r['E'][-1], Ev=r['Ev'][-1], ampT=amp[-1], liftT=lift[-1], chi=r['chi'], H2=r['H2_0']))
print(f"{len(R)} runs")
print(f"{'tag':28s} {'D':>4s} {'d':>2s} {'s':>5s} | a(tau) at tau~1,10,100,1e3,1e4 | lift(T) E_tr(T) E_val(T) | chi")
for q in R:
    print(f"{q['tag']:28s} {q['D']:4.1f} {q['depth']:2d} {q['s']:5.2f} | " + ' '.join(f'{a:6.2f}' for a in q['amp'][:5]) + f" | {q['liftT']:8.1f} {q['Etr']:7.4f} {q['Ev']:7.4f} | {q['chi']:.2e}")
NOM = [1, 10, 100, 1000, 10000]
for D in sorted({q['D'] for q in R}):
    grp = [q for q in R if q['D'] == D]
    if len(grp) < 4: continue
    print(f"\n== Delta={D} ({len(grp)} runs): Spearman of few-step quantities with final quantities")
    print(f"   {'tau_f':>6s} | a(tau_f) vs: a(T)  lift(T)  E_tr(T)  E_val(T) | growth a(tau_f)/a(tau_f/10) vs: a(T) lift(T) E_tr(T) | chi vs: a(T) lift(T) E_tr(T) E_val(T)")
    for j in range(1, 5):
        a_f = np.array([q['amp'][j] if j < len(q['amp']) else np.nan for q in grp]); g_f = np.array([q['amp'][j]/q['amp'][j-1] if j < len(q['amp']) else np.nan for q in grp])
        aT = np.array([q['ampT'] for q in grp]); lT = np.array([q['liftT'] for q in grp]); eT = np.array([q['Etr'] for q in grp]); vT = np.array([q['Ev'] for q in grp]); chi = np.array([q['chi'] for q in grp])
        sp = lambda x, y: spearmanr(x, y, nan_policy='omit')[0]
        print(f"   {NOM[j]:6d} | {sp(a_f, aT):6.2f} {sp(a_f, lT):7.2f} {sp(a_f, -eT):7.2f} {sp(a_f, -vT):8.2f} | {sp(g_f, aT):6.2f} {sp(g_f, lT):6.2f} {sp(g_f, -eT):6.2f} | {sp(chi, aT):6.2f} {sp(chi, lT):6.2f} {sp(chi, -eT):6.2f} {sp(chi, -vT):6.2f}")
