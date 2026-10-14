"""The refactor (scale_orig): per run final / best validation, spikes (> 3x running best after 10 %), wall, and with --spike_diag the
largest function change at a refresh and the largest basis error; next to the KronQFast references from earlier jobs."""
import json, glob, os, sys, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_orig')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
REF = {('rough', '256', '5000', '1'): '0.0622', ('rough', '256', '5000', '2'): '0.0626', ('smooth', '256', '5000', '1'): '1.42e-4', ('smooth', '256', '5000', '2'): '1.38e-4',
       ('smooth', '1024', '10000', '1'): 'TF32 spiked 11x, final 5.5e-3; guard: 2 small, final 4.4e-5', ('smooth', '1024', '10000', '2'): 'TF32 11 spikes, 4.8e-4; guard: 1, 3.8e-5',
       ('smooth', '1024', '10000', '3'): 'TF32 6 spikes, 2.7e-2; guard: 0, 4.1e-5', ('smooth', '1024', '10000', '4'): 'TF32 9 spikes, 1.3e-3; guard: 3, 6.0e-5',
       ('rough', '1024', '10000', '1'): 'TF32 3 spikes, best 0.0073; fp32 0.0033', ('rough', '1024', '10000', '2'): 'TF32 1 spike, best 0.0078; fp32 0.0034',
       ('rough', '1024', '5000', '1'): 'wall: KronQFast 59 s, AdamW 15 s', ('rough', '4096', '5000', '1'): 'wall: KronQFast 127 s, AdamW 30 s'}
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl'), key=lambda f: int(f.split('_j')[1].split('.')[0])):
    m = collections.defaultdict(dict); a = None
    for l in open(fn):
        r = json.loads(l); a = a or r.get('args'); m[r['metric']][r['step']] = r['value']
    T = int(arg(a, 'steps')); v = m['val']; st = sorted(v); best = float('inf'); sp = 0
    for s in st:
        if s > 0.1 * T and v[s] > 3 * best: sp += 1
        best = min(best, v[s])
    key = ('rough' if arg(a, 'p') == '2' else 'smooth', arg(a, 'batch'), arg(a, 'steps'), arg(a, 'seed'))
    rd = m.get('sd_refresh_df', {}); ua = m.get('sd_Uacc', {}); fb = m.get('sd_qr_fallbacks', {})
    extra = f" | refresh change max {max(rd.values()):.1e}, basis error max {max(ua.values()):.1e}, fallbacks {fb[max(fb)]:.0f}" if rd else ''
    print(f"{key[0]:6s} B{key[1]:>4s} T{key[2]:>5s} seed {key[3]}: final {v[st[-1]]:.3g} best {min(v.values()):.3g} spikes {sp} wall {list(m['wall_total'].values())[0]:.0f} s{extra}   [ref {REF.get(key, '-')}]")
