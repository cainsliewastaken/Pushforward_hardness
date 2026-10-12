"""Item 1 (runs/front/p4_w*.jsonl): exponent q < 0, Gauss-Newton cap, trust-region lr (Taylor remainder kappa), with
gamma(t), the remainder nu and the effective lr along the run."""
import json, glob, collections
import os as _os; L63_ROOT = _os.environ.get('L63_ROOT', _os.path.expandvars('$SCRATCH/l63_held_rough'))  # run tree on scratch
R = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(L63_ROOT + '/runs/front/p4_w*.jsonl'):
    for line in open(f):
        d = json.loads(line); a = d['args'].split(); g = lambda k, dflt='0': a[a.index(k) + 1] if k in a else dflt
        R[(g('--Delta'), g('--qmode'), g('--cap_gn'), g('--tr_kappa'), g('--decay', '1'))][d['metric']].append((d['step'], d['value']))
last = lambda m, k: m[k][-1][1] if m[k] else float('nan')
def at(m, k, s):
    v = [x for x in m[k] if x[0] <= s]; return v[-1][1] if v else float('nan')
for D in ('1.0', '0.5'):
    print(f'\n### Delta {D}')
    print('| q | GN cap | TR kappa | decay | E_train | E_val | E_train at 300 / 1000 steps | gamma(t) 100/1000/3000 | remainder nu 1000/3000 | lr_eff 1000/3000 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for k in sorted([k for k in R if k[0] == D], key=lambda k: (k[3], k[2], -float(k[1]))):
        m = R[k]
        if not m['E_train_final']: print(f'| {k[1]} | {k[2]} | {k[3]} | {k[4]} | diverged at {m["diverged"]} |'); continue
        print(f"| {k[1]} | {k[2]} | {k[3]} | {k[4]} | {last(m,'E_train_final'):.4f} | {last(m,'E_val_final'):.4f} | {at(m,'E_train',300):.3f} / {at(m,'E_train',1000):.3f} | "
              f"{at(m,'gamma_t',100):+.2f} / {at(m,'gamma_t',1000):+.2f} / {last(m,'gamma_t'):+.2f} | {at(m,'nu',1000):.3f} / {last(m,'nu'):.3f} | {at(m,'lr_eff',1000):.2g} / {last(m,'lr_eff'):.2g} |")
