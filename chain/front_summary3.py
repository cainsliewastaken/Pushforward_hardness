"""Wave-keyed variants (runs/front/p3_w*.jsonl) vs the equal-push baseline at the same lr (p1/p2, rho inf, beta fixed)."""
import json, glob, collections
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
R = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(L63_ROOT + '/runs/front/p[123]_w*.jsonl'):
    for line in open(f):
        d = json.loads(line); a = d['args'].split(); g = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt
        key = (g('--Delta'), int(g('--signmode')), float(g('--lr')), g('--front', 'rayleigh'), g('--rho'), g('--boost', '1'), g('--beta_mode'))
        R[key][d['metric']].append((d['step'], d['value']))
last = lambda m, k: m[k][-1][1] if m[k] else float('nan')
for D in ('0.5', '1.0'):
    for sm, lr in ((1, 1e-3), (1, 3e-3), (0, 3e-2), (0, 9e-2)):
        base = R.get((D, sm, lr, 'rayleigh', 'inf', '1', 'fixed'))
        bs = f"{last(base,'E_train_final'):.4f} / {last(base,'E_val_final'):.4f}" if base else 'n/a'
        print(f"\nDelta {D}, {'sign' if sm else 'RMS'} lr {lr:g}: equal push baseline train/val {bs}")
        for (rho, boost, bm) in (('3','1','fixed'), ('10','1','fixed'), ('30','1','fixed'), ('10','3','fixed'), ('inf','1','front')):
            m = R.get((D, sm, lr, 'wave', rho, boost, bm))
            if not m or not m['E_train_final']: print(f"   rho {rho:>3} boost {boost} beta {bm:5s}: missing"); continue
            wl = [x[1] for x in m['lamf']]; wq = [x[1] for x in m['wave_q25']]
            print(f"   rho {rho:>3} boost {boost} beta {bm:5s}: train {last(m,'E_train_final'):.4f} val {last(m,'E_val_final'):.4f}"
                  f"   wave median start/mid/end {wl[2]:.2g}/{wl[len(wl)//2]:.2g}/{wl[-1]:.2g}  q25 end {wq[-1]:.2g}  lam_r100 end {last(m,'lam_r100'):.3g}")
