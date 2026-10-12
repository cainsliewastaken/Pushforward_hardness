"""
Analysis of late_*.npy from l63_late.py.
Tests, per run:
 (A) adiabatic model in the MOVING kernel eigenbasis: per-mode energy eE2_k(t) should follow
     d/dt eE2_k = -2 lam_k(t) eE2_k if the eigenbasis rotates slowly. Reports the fraction of energy
     change explained, and the 'transfer' (energy that moved between modes) per interval.
 (B) late-stage power law ||E|| ~ t^-alpha and lam_eff * t.
 (C) spectral picture: at several times, (lam_k, eE2_k) scatter = residual roughness spectrum in the current kernel.
 (D) kernel change dK = K(t)-K(0): spectrum (effective rank), alignment of top eigvecs with E(0), E(t), Y.
 (E) roughness: index of E(t), of Y, of top kernel eigvecs vs time; RMT: outliers above the K(0) bulk edge.
usage: python analyze_late.py OUTFIG late_a.npy [late_b.npy ...]   -> figures in OUTFIG, summary table on stdout + summary.csv
"""
import numpy as np, sys, os, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def eff_rank(ev):
    ev = np.abs(ev); p = ev/ev.sum(); p = p[p > 0]
    return float(np.exp(-(p*np.log(p)).sum()))

def analyze(fn, outfig):
    r = np.load(fn, allow_pickle=True).item()
    t = r['t']; E = r['E']; lam = r['lam'][:, ::-1]; eE2 = r['eE2'][:, ::-1]; yV2 = r['yV2'][:, ::-1]   # descending
    lam_eff = r['lam_eff']; n = r['n']; tag = r['tag']
    lam1 = lam[:, 0]
    # ---- (A) adiabatic test between consecutive logs
    E2 = E**2
    pred_tot = np.zeros(len(t)); transfer = np.zeros(len(t)); adiab_err = np.zeros(len(t))
    for i in range(len(t)-1):
        dt = t[i+1]-t[i]
        pred = eE2[i]*np.exp(-2*lam[i]*dt)             # per-mode prediction (basis of time i)
        pred_tot[i+1] = pred.sum()
        # compare to actual energies at i+1 (in basis of time i+1): sorted-mode comparison is basis-dependent,
        # so compare totals and the top-k partial sums
        act = eE2[i+1]
        adiab_err[i+1] = (pred.sum()-act.sum())/max(act.sum(), 1e-300)
        transfer[i+1] = 0.5*np.abs(np.sort(pred)[::-1]-np.sort(act)[::-1]).sum()/max(act.sum(), 1e-300)
    # ---- (B) late power law: fit log E vs log t on the last decade of t
    m = t > t[-1]/10
    alpha = -np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]
    lam_eff_t = lam_eff*t
    # energy-weighted mode index of E in current basis, and top-k fractions
    ranks = np.arange(1, lam.shape[1]+1)
    mean_rank = (eE2*ranks).sum(1)/eE2.sum(1)
    top1 = eE2[:, 0]/eE2.sum(1); top10 = eE2[:, :10].sum(1)/eE2.sum(1)
    # ---- (D) kernel change
    K0 = r['snap_K'][0].astype(np.float64); snapK = r['snap_K']; st = r['snap_t']
    Ev = r['Evec']; Y = r['Y'].reshape(-1); E0 = Ev[0]
    dK_rank = []; dK_top = []; al_E0 = []; al_Et = []; al_Y = []; al_curE = []
    for s, Ks in zip(st, snapK):
        dK = Ks.astype(np.float64)-K0
        w, V = np.linalg.eigh(dK); order = np.argsort(-np.abs(w)); w = w[order]; V = V[:, order]
        dK_rank.append(eff_rank(w)); dK_top.append(w[0])
        i = np.searchsorted(t, s); Et = Ev[min(i, len(t)-1)]
        u = V[:, 0]
        al_E0.append((u@E0)**2/(E0@E0)); al_Et.append((u@Et)**2/(Et@Et)); al_Y.append((u@Y)**2/(Y@Y))
        # fraction of top-5 dK subspace captured by span of {E0, Y}
        al_curE.append(((V[:, :5].T@Et)**2).sum()/(Et@Et))
    # ---- (E) roughness
    mu = r['lap_mu']; rE = r['rough_E']; rY = r['rough_Y']
    rough_idx_E = (rE@mu)/rE.sum(1); rough_idx_Y = (rY@mu)/rY.sum()
    rough_top = r['rough_top'][:, ::-1]    # top-5 eigvecs, index 0 = top
    # RMT: outliers above the initial bulk edge: count eigenvalues of K(t) above lam_1(0)
    n_out = (lam > lam1[0]).sum(1)
    bulk_edge0 = lam1[0]
    late = slice(len(t)*3//4, None)
    # ---- (F) multi-step prediction from a late time t0: frozen-spectrum vs moving-spectrum (adiabatic) kernel regression
    def predict_from(i0, moving):
        e0 = eE2[i0].copy(); pred = np.zeros(len(t)); pred[i0] = e0.sum(); I = np.zeros_like(e0)
        for i in range(i0, len(t)-1):
            lam_i = lam[i] if moving else lam[i0]
            I += 2*lam_i*(t[i+1]-t[i]); pred[i+1] = (e0*np.exp(-I)).sum()
        return np.sqrt(np.maximum(pred, 0))
    i0 = int(np.searchsorted(E[::-1], 0.5*E[0]))      # first index where ||E|| <= E0/2 (E is nonincreasing)
    i0 = len(t)-1-i0 if i0 < len(t) else len(t)//2
    i0 = max(1, min(i0, len(t)-20))
    p_frozen = predict_from(i0, False); p_moving = predict_from(i0, True)
    i1 = int(np.searchsorted(t, t[-1]/6.)); p_lfro = predict_from(i1, False); p_lmov = predict_from(i1, True)
    err_lfro = float(p_lfro[-1]/E[-1]); err_lmov = float(p_lmov[-1]/E[-1])
    err_frozen = float(np.abs(np.log(p_frozen[-1]/E[-1]))); err_moving = float(np.abs(np.log(p_moving[-1]/E[-1])))
    # ---- (G) spectral exponents at late time: lam_k ~ k^-a over the modes carrying the residual; eE2_k ~ k^-b
    iL = len(t)-1; cum = np.cumsum(eE2[iL])/eE2[iL].sum()
    k_lo = max(2, int(np.searchsorted(cum, 0.05))); k_hi = max(k_lo+5, int(np.searchsorted(cum, 0.95)))
    kk = np.arange(k_lo, k_hi+1)
    a_exp = -np.polyfit(np.log(kk), np.log(lam[iL, k_lo-1:k_hi]), 1)[0]
    b_exp = -np.polyfit(np.log(kk), np.log(np.maximum(eE2[iL, k_lo-1:k_hi], 1e-300)), 1)[0]
    # ---- (H) per-mode scale factor of the spectrum
    ok0 = lam[0] > 1e-9*lam[0][0]
    scale = np.where(ok0, lam[-1]/np.where(ok0, lam[0], 1.0), np.nan)
    # ---- (I) current-kernel regression estimate: E0 (or Y) energy in the CURRENT basis, decayed with the current
    #      spectrum over the elapsed time (instantaneous) or with the integrated spectrum by rank (adiabatic)
    has_e0 = 'e0V2' in r
    src = r['e0V2'][:, ::-1] if has_e0 else yV2
    I_rank = np.zeros(lam.shape[1]); S_inst = np.zeros(len(t)); S_adia = np.zeros(len(t))
    for i in range(len(t)):
        if i > 0: I_rank += (lam[i]+lam[i-1])*(t[i]-t[i-1])
        S_inst[i] = (src[i]*np.exp(-2*lam[i]*t[i])).sum(); S_adia[i] = (src[i]*np.exp(-I_rank)).sum()
    pos = lambda x: np.maximum(x, 1e-300)
    # hard-cutoff version: energy of the source on current modes with lam_k(t) t < 1; cutoff rank k*(t)
    kstar = np.array([int((lam[i]*t[i] >= 1).sum()) for i in range(len(t))])
    S_cut = np.array([src[i][lam[i]*t[i] < 1].sum() for i in range(len(t))])
    S_instY = np.array([(yV2[i]*np.exp(-2*lam[i]*t[i])).sum() for i in range(len(t))])
    # ---- figures
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    a = ax[0, 0]; a.loglog(t[1:], E[1:], 'k', label='||E||'); a.loglog(t[1:], np.sqrt(np.maximum(pred_tot[1:], 1e-300)), 'r:', label='adiabatic 1-step pred'); a.set_title(f'{tag}: ||E||, late slope -{alpha:.2f}'); a.legend(); a.set_xlabel('t')
    a = ax[0, 1]; a.loglog(t[1:], lam1[1:], 'k', label='lam1'); a.loglog(t[1:], lam_eff[1:], 'b', label='lam_eff=<e,Ke>'); a.loglog(t[1:], lam[1:, 9], 'g--', label='lam10'); a.loglog(t[1:], lam[1:, 99], 'c--', label='lam100'); a.legend(); a.set_title('kernel eigenvalues'); a.set_xlabel('t')
    a = ax[0, 2]; a.loglog(t[1:], E[1:], 'k', lw=2, label='||E||'); a.loglog(t[1:], np.sqrt(pos(S_inst[1:])), 'b--', label='current-kernel regression (inst.)'); a.loglog(t[1:], np.sqrt(pos(S_adia[1:])), 'r:', lw=2, label='by-rank integrated spectrum'); a.legend(fontsize=7); a.set_title(('source=E0' if has_e0 else 'source=Y') + f' in current basis; lam_eff*t late={np.median(lam_eff_t[late]):.2f}'); a.set_xlabel('t')
    a = ax[1, 0]
    for i in [len(t)//2, len(t)-1]:
        pm = src[i]*np.exp(-2*lam[i]*t[i]); mk = (eE2[i] > 1e-12*eE2[i].sum()) & (pm > 0)
        a.loglog(pos(pm[mk]), pos(eE2[i][mk]), '.', ms=3, label=f't={t[i]:.1f}')
    lim = [1e-10, 1]; a.plot(lim, lim, 'k:'); a.set_xlabel('src_k(t) exp(-2 lam_k(t) t)'); a.set_ylabel('eE2_k(t)'); a.legend(fontsize=7); a.set_title('per-mode current-kernel regression test')
    a = ax[1, 1]
    for i in [0, len(t)//2, len(t)-1]:
        a.loglog(ranks, lam[i], label=f't={t[i]:.1f}')
    a.axhline(bulk_edge0, color='gray', ls=':'); a.legend(fontsize=7); a.set_xlabel('k'); a.set_title('kernel spectrum lam_k; dotted = lam1(0)')
    a = ax[1, 2]; a.semilogx(t[1:], n_out[1:], 'k'); a.set_title('# eigenvalues above lam1(0) (outliers)'); a.set_xlabel('t')
    a = ax[2, 0]; a.semilogx(st, dK_rank, 'ko-', label='eff. rank of K(t)-K(0)'); a.set_xlabel('t'); a.legend(); a2 = a.twinx(); a2.semilogx(st, al_E0, 'r.-', label='align top dK-vec with E0'); a2.semilogx(st, al_Y, 'g.-', label='with Y'); a2.semilogx(st, al_curE, 'b.-', label='E(t) in top-5 dK'); a2.legend(fontsize=7, loc='center right'); a.set_title('kernel change: rank and alignment')
    a = ax[2, 1]; a.semilogx(t[1:], rough_idx_E[1:], 'k', label='roughness idx of E(t)'); a.axhline(rough_idx_Y, color='g', ls='--', label='of Y'); a.semilogx(t[1:], rough_top[1:, 0], 'r', label='of top kernel eigvec'); a.semilogx(t[1:], rough_top[1:, 4], 'r:', label='of 5th'); a.legend(fontsize=7); a.set_xlabel('t'); a.set_title('graph-Laplacian roughness')
    a = ax[2, 2]; a.loglog(t[i0:], E[i0:], 'k', lw=2, label='||E|| actual'); a.loglog(t[i0:], pos(p_frozen[i0:]), 'b--', label=f'frozen spectrum from t0={t[i0]:.1f}'); a.loglog(t[i0:], pos(p_moving[i0:]), 'r:', lw=2, label='moving spectrum (adiabatic)'); a.loglog(t[i1:], pos(p_lfro[i1:]), 'g--', label=f'frozen from t1={t[i1]:.0f}'); a.legend(fontsize=7); a.set_xlabel('t'); a.set_title(f'prediction from t0; a={a_exp:.2f} b={b_exp:.2f} -> alpha_pred={(b_exp-1)/(2*a_exp):.2f}')
    a = ax[1, 2]; a.cla(); a.semilogx(ranks[ok0], scale[ok0], 'k.', ms=2); a.set_xlabel('k'); a.set_ylabel('lam_k(T)/lam_k(0)'); a.set_yscale('log'); a.set_title(f'spectrum scale factor; # outliers above lam1(0): {n_out[-1]}')
    try: fig.tight_layout(); fig.savefig(f'{outfig}/{tag}.png', dpi=100)
    except Exception as ex: print('figure failed for', tag, ex)
    plt.close(fig)
    return dict(tag=tag, H=r['H'], depth=r['depth'], Delta=r['Delta'], d_theta=r['d_theta'], E0=E[0], ET=E[-1], lam1_0=lam1[0], lam1_T=lam1[-1],
                alpha=alpha, lam_eff_t_late=float(np.median(lam_eff_t[late])), top1_late=float(np.median(top1[late])), top10_late=float(np.median(top10[late])),
                mean_rank_late=float(np.median(mean_rank[late])), adiab_err_late=float(np.median(np.abs(adiab_err[late]))), transfer_late=float(np.median(transfer[late])),
                dK_rank_T=dK_rank[-1], al_E0_T=al_E0[-1], al_Y_T=al_Y[-1], n_out_T=int(n_out[-1]), rough_E_T=rough_idx_E[-1], rough_Y=rough_idx_Y, rough_top_T=rough_top[-1, 0],
                H2_late=float(np.nanmedian(r['H2_g'][late])), wall=r['wall_s'], steps=int(r['steps']),
                t0=float(t[i0]), err_frozen=err_frozen, err_moving=err_moving, a_exp=float(a_exp), b_exp=float(b_exp), alpha_pred=float((b_exp-1)/(2*a_exp)),
                scale_top=float(scale[0]), scale_med=float(np.nanmedian(scale[:300])), scale_k100=float(scale[99]), S_inst_ratio=float(np.median(np.sqrt(S_inst[late])/E[late])), S_adia_ratio=float(np.median(np.sqrt(S_adia[late])/E[late])), S_cut_ratio=float(np.median(np.sqrt(S_cut[late])/E[late])), S_instY_ratio=float(np.median(np.sqrt(S_instY[late])/E[late])), kstar_late=float(np.median(kstar[late])), p_moving_ratio=float(p_moving[-1]/E[-1]), p_frozen_ratio=float(p_frozen[-1]/E[-1]), t1=float(t[i1]), p_latefrozen_ratio=err_lfro, p_latemoving_ratio=err_lmov,
                lam_tail0=[float(lam[0][k]) for k in (9, 49, 99, 199, 399)], lam_tailT=[float(lam[-1][k]) for k in (9, 49, 99, 199, 399)])

if __name__ == '__main__':
    outfig = sys.argv[1]; os.makedirs(outfig, exist_ok=True)
    rows = [analyze(fn, outfig) for fn in sys.argv[2:]]
    keys = list(rows[0].keys())
    with open(f'{outfig}/summary.csv', 'w') as f:
        f.write(','.join(keys)+'\n')
        for r in rows: f.write(','.join(str(r[k]) for k in keys)+'\n')
    print(f"{'tag':24s} {'ET':>8s} {'l1T/l10':>8s} {'alpha':>6s} {'leff*t':>7s} {'top1':>5s} {'top10':>5s} {'mrank':>6s} {'adiab':>8s} {'transf':>7s} {'dKrank':>6s} {'alE0':>5s} {'alY':>5s} {'nout':>4s} {'rE_T':>5s} {'rY':>5s} {'rtop':>5s}")
    for r in rows:
        print(f"{r['tag']:24s} {r['ET']:8.4f} t0={r['t0']:6.1f} errF={r['err_frozen']:5.2f} errM={r['err_moving']:5.2f} a={r['a_exp']:4.2f} b={r['b_exp']:4.2f} sc1={r['scale_top']:5.1f} sc100={r['scale_k100']:5.1f} Sinst/E={r['S_inst_ratio']:5.2f} Sadia/E={r['S_adia_ratio']:5.2f} Scut/E={r['S_cut_ratio']:5.2f} SY/E={r['S_instY_ratio']:5.2f} k*={r['kstar_late']:5.0f} pM/E={r['p_moving_ratio']:5.2f} pF/E={r['p_frozen_ratio']:5.2f} late-t1={r['t1']:5.0f}: pF/E={r['p_latefrozen_ratio']:5.2f} pM/E={r['p_latemoving_ratio']:5.2f} | {r['lam1_T']/r['lam1_0']:8.1f} {r['alpha']:6.2f} {r['lam_eff_t_late']:7.2f} {r['top1_late']:5.2f} {r['top10_late']:5.2f} {r['mean_rank_late']:6.1f} {r['adiab_err_late']:8.1e} {r['transfer_late']:7.3f} {r['dK_rank_T']:6.1f} {r['al_E0_T']:5.2f} {r['al_Y_T']:5.2f} {r['n_out_T']:4d} {r['rough_E_T']:5.2f} {r['rough_Y']:5.2f} {r['rough_top_T']:5.2f}")
