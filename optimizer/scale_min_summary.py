"""Summary of the minimal-method test (scale_min.sh): per target and preset the mean final validation MSE over seeds, the
seed values, max/min, and mean wall."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_min')
def key(args):
    t = 'rough' if '--p 2' in args else 'smooth'
    cap = '--cap 1' in args; rem = args.split('--rem ')[1].split()[0]
    return t, ('A reference (cap on, budget 0.2)' if cap else f'{"B" if rem == "0.2" else "C"} cap off, budget {rem}')
res = collections.defaultdict(list)
for fn in glob.glob(f'{F}/res_j*.jsonl'):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    vf = [r['value'] for r in rows if r['metric'] == 'val_final']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
    if vf: res[key(rows[0]['args'])].append((vf[0], w[0] if w else float('nan')))
for t in ('rough', 'smooth'):
    print(f'==== {t}')
    for k in sorted(res):
        if k[0] != t: continue
        v = [x[0] for x in res[k]]; w = [x[1] for x in res[k]]
        print(f"   {k[1]:36s} mean {sum(v) / len(v):.4g}  seeds ({' '.join(f'{x:.3g}' for x in v)})  max/min {max(v) / min(v):.2f}  wall {sum(w) / len(w):.0f} s")
