"""Multi-seed table (scale_seeds.sh): per method and target, the validation MSE at 5000 steps for each seed, mean and
spread (max/min), state per parameter, and isolated wall per step. Job index i = 8*(seed-1) + 4*(p==0) + m, m in
{0: KronQ eager, 1: KronQ fused, 2: AdamW, 3: SOAP b256}."""
import json, os, re, glob, collections, math
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_seeds')
M = {0: 'KronQ fast (eager)', 1: 'KronQ fast (fused)', 2: 'AdamW', 3: 'SOAP b256'}
rows = collections.defaultdict(list)
for f in glob.glob(f'{F}/res_j*.jsonl'):
    i = int(re.search(r'res_j(\d+)', f).group(1)); seed = i // 8 + 1; p = 2 if (i % 8) < 4 else 0; m = i % 4
    recs = [json.loads(l) for l in open(f)]
    val = [r['value'] for r in recs if r['metric'] == 'val_final']; wall = [r['value'] for r in recs if r['metric'] == 'wall_total']
    sb = [r['value'] for r in recs if r['metric'] == 'state_per_param']
    if val: rows[(p, m)].append((seed, val[0], wall[0] if wall else float('nan'), sb[0] if sb else float('nan')))
for p in (2, 0):
    print(f'== {"rough" if p == 2 else "smooth"} target: val at 5000 steps (seeds), mean, max/min, state/P, ms per step (isolated)')
    for m in range(4):
        r = sorted(rows.get((p, m), []))
        if not r: print(f'   {M[m]:20s} missing'); continue
        v = [x[1] for x in r]; mean = sum(v) / len(v); spread = max(v) / min(v)
        print(f'   {M[m]:20s} ' + ' '.join(f'{x:.4g}' for x in v) + f'   mean {mean:.4g}  max/min {spread:.2f}  state/P {r[0][3]:.2f}  ms/step {sum(x[2] for x in r) / len(r) / 5000 * 1000:.1f}')
