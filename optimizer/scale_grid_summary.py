"""Grid summary (scale_grid.sh): per method / lr / (budget | steps) / target: mean over seeds, max/min, n."""
import json, os, re, glob, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_grid')
rows = collections.defaultdict(list)
for f in glob.glob(f'{F}/res_j*.jsonl'):
    recs = [json.loads(l) for l in open(f)]
    if not recs: continue
    a = recs[0]['args']; opt = re.search(r'--opt (\w+)', a).group(1); p = float(re.search(r'--p (\d)', a).group(1)); lr = float(re.search(r'--lr ([0-9.e-]+)', a).group(1))
    st = int(re.search(r'--steps (\d+)', a).group(1)); rem = re.search(r'--rem ([0-9.]+)', a); rem = float(rem.group(1)) if rem else None
    v = [r['value'] for r in recs if r['metric'] == 'val_final']; w = [r['value'] for r in recs if r['metric'] == 'wall_total']
    if v: rows[(opt, p, lr, rem, st)].append((v[0], w[0] if w else float('nan')))
for p in (2.0, 0.0):
    print(f'==== {"rough" if p == 2 else "smooth"}: method, lr, budget/steps -> mean (seeds), max/min, mean wall s')
    for key in sorted(rows, key=lambda t: (t[0], t[4], t[2], t[3] or 0)):
        if key[1] != p: continue
        v = [x[0] for x in rows[key]]; m = sum(v) / len(v)
        print(f'   {key[0]:7s} lr {key[2]:<8g} {"rem " + str(key[3]) if key[3] is not None else "steps " + str(key[4]):10s}  mean {m:.4g} ({" ".join(f"{x:.3g}" for x in v)})  max/min {max(v) / min(v):.2f}  wall {sum(x[1] for x in rows[key]) / len(v):.0f}')
