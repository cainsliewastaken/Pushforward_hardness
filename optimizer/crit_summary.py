"""Critical batch size on the field: best final validation over settings per (optimizer, target, batch, steps); steps to reach targets by
log-log interpolation along the step budgets; scaling efficiency S(B) / S(2B) (2 = perfect scaling, 1 = no gain); wall per run."""
import json, glob, os, math, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_crit')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
best = collections.defaultdict(dict); wall = collections.defaultdict(dict); n = 0
for fn in glob.glob(f'{F}/res_j*.jsonl'):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    a = rows[0]['args']; vf = [r['value'] for r in rows if r['metric'] == 'val_final']; wt = [r['value'] for r in rows if r['metric'] == 'wall_total']
    if not vf or not math.isfinite(vf[0]): continue
    n += 1; key = ('KronQ' if 'kronqf' in a else 'AdamW', 'rough' if arg(a, 'p') == '2' else 'smooth', int(arg(a, 'batch'))); T = int(arg(a, 'steps'))
    if T not in best[key] or vf[0] < best[key][T][0]: best[key][T] = (vf[0], f"lr {arg(a, 'lr')}" + (f" budget {arg(a, 'rem')}" if 'kronqf' in a else '')); wall[key][T] = wt[0] if wt else float('nan')
print(f'{n} finished runs')
TGT = {('KronQ', 'rough'): [0.1, 0.05, 0.03, 0.02], ('AdamW', 'rough'): [0.5, 0.4, 0.3, 0.25], ('KronQ', 'smooth'): [1e-3, 3e-4, 1.5e-4, 1e-4], ('AdamW', 'smooth'): [3e-3, 1e-3, 5e-4, 3e-4]}
def steps_to(curve, tgt):
    pts = sorted(curve.items())
    if not pts or pts[0][1][0] <= tgt: return (pts[0][0] if pts else float('nan')) if pts and pts[0][1][0] <= tgt else float('nan')
    for (t0, (v0, _)), (t1, (v1, _)) in zip(pts, pts[1:]):
        if v1 <= tgt < v0: return math.exp(math.log(t0) + (math.log(tgt) - math.log(v0)) / (math.log(v1) - math.log(v0)) * (math.log(t1) - math.log(t0)))
    return float('nan')
for opt in ('KronQ', 'AdamW'):
    for tg in ('rough', 'smooth'):
        Bs = sorted(B for (o, t, B) in best if o == opt and t == tg)
        if not Bs: continue
        print(f'==== {opt} {tg}: best final validation per (batch, steps)  [setting]')
        for B in Bs:
            c = best[(opt, tg, B)]; print(f'  B {B:5d}: ' + '  '.join(f'{T}:{c[T][0]:.3g}' for T in sorted(c)) + f'   [{c[max(c)][1]} at {max(c)}; wall {wall[(opt, tg, B)][max(c)]:.0f} s]')
        for tv in TGT[(opt, tg)]:
            S = {B: steps_to(best[(opt, tg, B)], tv) for B in Bs}
            eff = '  '.join(f'{B}->{2 * B}: {S[B] / S[2 * B]:.2f}' for B in Bs if 2 * B in S and math.isfinite(S[B]) and math.isfinite(S[2 * B]))
            print(f'  target {tv:g}: steps ' + '  '.join(f'{B}:{S[B]:.0f}' if math.isfinite(S[B]) else f'{B}:-' for B in Bs) + f'   | S(B)/S(2B) {eff}')
