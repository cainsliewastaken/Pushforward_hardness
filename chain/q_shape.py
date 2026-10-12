"""Self-similarity of Q(t) = dPsi dPsi^T/n and roughness of its eigenvectors (sweep C runs).
(1) shape distance: || Q(t)/trQ(t) - Q(T)/trQ(T) ||_F / ||Q(T)/trQ(T)||_F; overlap of top-10 eigenspaces;
(2) trQ(t) vs ell(t)^2 and vs t; (3) roughness index of eigenvectors vs rank bin for K0, K(T), Q(T)."""
import numpy as np, sys
bins = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 400)]
for fn in sys.argv[1:]:
    r = np.load(fn, allow_pickle=True).item()
    if 'snap_Q' not in r or len(r['snap_Q']) == 0: continue
    st = r['snap_t']; Qs = r['snap_Q']; n = r['n']; mu = r['lap_mu']; U = r['lap_U']; t = r['t']; g = r['gnorm']
    ell = np.concatenate([[0], np.cumsum(0.5*(g[1:]+g[:-1])*np.diff(t))])
    QT = Qs[-1].astype(np.float64); QTn = QT/np.trace(QT); wT, VT = np.linalg.eigh(QT); VT = VT[:, ::-1]
    print(f"== {r['tag']}")
    print("   t      trQ     trQ/ell^2   shape-dist  top10-overlap")
    for i in range(0, len(st), 3):
        Q = Qs[i].astype(np.float64); tr = np.trace(Q)
        if tr <= 0: continue
        Qn = Q/tr; dist = np.linalg.norm(Qn-QTn)/np.linalg.norm(QTn)
        V = np.linalg.eigh(Q)[1][:, ::-1][:, :10]; ov = np.linalg.norm(VT[:, :10].T @ V)**2/10
        j = np.searchsorted(t, st[i]); l = ell[min(j, len(ell)-1)]
        print(f"  {st[i]:8.2f} {tr:9.3e} {tr/max(l, 1e-12)**2:10.3e} {dist:10.3f} {ov:8.3f}")
    def rough_by_rank(M):
        w, V = np.linalg.eigh(M); V = V[:, ::-1]
        out = []
        for a, b in bins:
            idx = [((U.T @ V[:, k].reshape(n, 3))**2).sum(1) @ mu for k in range(a, b)]
            out.append(np.mean(idx))
        return out
    K0 = r['snap_K'][0].astype(np.float64); KT = r['snap_K'][-1].astype(np.float64)
    rY = r['rough_Y'] @ mu / r['rough_Y'].sum(); rE = r['rough_E'][-1] @ mu / r['rough_E'][-1].sum()
    print("   roughness of eigvecs by rank bin " + str(bins) + f"   [Y: {rY:.2f}, E(T): {rE:.2f}]")
    for name, M in [('K0  ', K0), ('K(T)', KT), ('Q(T)', QT)]:
        print(f"   {name}: " + ' '.join(f'{x:6.3f}' for x in rough_by_rank(M)))
