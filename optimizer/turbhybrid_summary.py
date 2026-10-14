"""The hybrid on turbulence (scale_turbhybrid): final / best validation, end train-batch loss, spikes (> 3x running best after 10 %),
validation at a few steps, wall; next to the KronQFast and AdamW references."""
import json, glob, os, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_turbhybrid')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
REF = {('delta', 'cap'): 'KronQFast capped wd 1.0: 0.1109', ('delta', 'nob'): 'KronQFast no budget lr 2e-4: 0.1148', ('direct', 'cap'): 'KronQFast capped wd 0.3: 0.0628 / 0.0631', ('direct', 'nob'): 'none'}
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl'), key=lambda f: int(f.split('_j')[1].split('.')[0])):
    m = collections.defaultdict(dict); a = None
    for l in open(fn):
        r = json.loads(l); a = a or r.get('args'); m[r['metric']][r['step']] = r['value']
    v = m['val']; st = sorted(v); best = float('inf'); sp = 0; T = int(arg(a, 'steps'))
    for s in st:
        if s > 0.1 * T and v[s] > 3 * best: sp += 1
        best = min(best, v[s])
    tg = arg(a, 'turb_target'); mode = 'cap' if '--sa_cap 1' in a else 'nob'; tb = m.get('train_batch_loss', {})
    lab = f"{tg:6s} {'capped budget wd ' + arg(a, 'wd') if mode == 'cap' else 'no budget lr 2e-4 decay 1e-3/step':34s} seed {arg(a, 'seed')}{' +fp32 rot' if arg(a, 'rot_hp') == '1' else '          '}"
    traj = ' '.join(f"{s}:{v[s]:.3g}" for s in (2000, 10000, 20000) if s in v)
    print(f"{lab}: final {v[st[-1]]:.4f} best {min(v.values()):.4f} train {tb[max(tb)] if tb else float('nan'):.4f} spikes {sp} [{traj}] wall {list(m['wall_total'].values())[0]:.0f} s   [{REF[(tg, mode)]}]")
print('AdamW best: increment 0.1563, direct 0.0600')
