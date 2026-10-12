"""Second-order-feature test (runs with L63_SOF=1). Per current mode k (ascending lam order as logged):
 predicted dlam_k/dt (Hellmann-Feynman) vs finite difference by rank; q_k = ||v_k^T Psi_g||^2/n (energy of the
 second-order features on mode k) vs the residual energy eE2_k and lam_k; alignment cosine cos_k."""
import numpy as np, sys
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    if 'sof_rate' not in r or len(r['sof_rate']) == 0: continue
    t = r['t']; lam = r['lam']; rate = r['sof_rate']; q = r['sof_q']; cs = r['sof_cos']; eE2 = r['eE2']; E = r['E']; H2 = r['H2_g']
    # finite-difference rate by rank (central), compare at interior logs
    fd = np.gradient(lam, t, axis=0)
    print(f"== {r['tag']}  (ranks given from the top: 1 = largest)")
    print("   t        corr(pred,fd) over ranks 1-300 | pred/fd at ranks 1,10,50,100,200 | q_k/H2^2 at 1,10,50,100,200,400 | cos_k at 1,10,50,100,200,400 | eE2 frac at 10,50,100,200,400")
    for i in range(10, len(t)-1, max(1, len(t)//8)):
        p = rate[i][::-1]; f = fd[i][::-1]; qq = q[i][::-1]; cc = cs[i][::-1]; ee = eE2[i][::-1]/eE2[i].sum()
        m = np.isfinite(p[:300]) & np.isfinite(f[:300]) & (np.abs(f[:300]) > 0)
        corr = np.corrcoef(np.log(np.abs(p[:300][m])+1e-300), np.log(np.abs(f[:300][m])+1e-300))[0, 1]
        print(f"  {t[i]:8.2f} {corr:6.3f} | " + ' '.join(f'{p[k]/f[k]:6.2f}' if f[k] != 0 else '   nan' for k in (0, 9, 49, 99, 199)) +
              ' | ' + ' '.join(f'{qq[k]/H2[i]**2:8.1e}' for k in (0, 9, 49, 99, 199, 399)) + ' | ' + ' '.join(f'{cc[k]:6.3f}' for k in (0, 9, 49, 99, 199, 399)) +
              ' | ' + ' '.join(f'{ee[k]:7.1e}' for k in (9, 49, 99, 199, 399)))
