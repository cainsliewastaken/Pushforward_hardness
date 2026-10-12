"""Item 5: is hyperparameter transfer a size-invariance of the feature-learning/lazy ratio? For each run: R(tau) = (E'KE - E'K0E)/E'K0E
along lazy time tau = s^2 t, final error, and the optimal output scale per width under 'standard' (s fixed) vs 'muP-like' (s ~ 1/sqrt(H)) labelling."""
import numpy as np, glob, sys, collections
runs=collections.defaultdict(dict)
for fn in sorted(glob.glob(sys.argv[1]+'/des_*.npy')):
    r=np.load(fn,allow_pickle=True).item(); s=float(r.get('outscale') or 1.0); H=int(r['H']); d=int(r['depth'])
    t=np.asarray(r['t']); EKE=np.asarray(r['EKE']); EK0E=np.asarray(r['EK0E']); E=np.asarray(r['E']); tau=s*s*t
    R=(EKE-EK0E)/np.clip(EK0E,1e-300,None)
    runs[(d,H)][s]=dict(tau=tau,R=R,ET=float(E[-1]),E=E)
for (d,H),byS in sorted(runs.items()):
    print(f"depth {d} width {H}:")
    for s,x in sorted(byS.items()):
        pts=" ".join(f"tau={q:5.0f}:R={np.interp(q,x['tau'],x['R']):6.2f}" for q in (10,100,1000,3000))
        print(f"   s={s:<5}: E(T)={x['ET']:.4f}   feature/lazy ratio {pts}")
    best=min(byS.items(),key=lambda kv: kv[1]['ET'])[0]; print(f"   -> optimal s at this width: {best}   (muP-like label s*sqrt(H/96) = {best*np.sqrt(H/96):.2f})")
print("\nTransfer test: optimal s vs width.  Standard transfer <=> s* constant;  muP-like transfer <=> s*·sqrt(H) constant.")
print("Criterion test: does R(tau) at matched tau collapse across widths when s is relabelled muP-style? (compare rows with equal s*sqrt(H/96))")
