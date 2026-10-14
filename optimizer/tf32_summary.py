"""Spike numerics test (scale_tf32): per run spikes (> 3x running best after 10 % of training), best / final validation, and over the
instrumented window the function change per refresh (median, 99th percentile, max; relative to the probe residual) and the accumulated-basis
orthogonality (max, end). Grouped by configuration and TF32 on / off."""
import json, glob, os, sys, collections, statistics as stt
F = os.path.expandvars(sys.argv[1] if len(sys.argv) > 1 else '$SCRATCH/l63_held_rough/runs/scale_tf32')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
out = []
for fn in glob.glob(f'{F}/res_j*.jsonl'):
    m = collections.defaultdict(dict); a = None
    for l in open(fn):
        r = json.loads(l); a = a or r.get('args'); m[r['metric']][r['step']] = r['value']
    T = int(arg(a, 'steps')); v = m['val']; st = sorted(v); best = float('inf'); sp = []
    for s in st:
        if s > 0.1 * T and v[s] > 3 * best: sp.append(s)
        best = min(best, v[s])
    rd = sorted(m.get('sd_refresh_df', {}).values()); ua = m.get('sd_Uacc', {})
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')
    cfg = f"{'smooth' if arg(a, 'p') == '0' else 'rough'} B{arg(a, 'batch')} T{T}"
    out.append((cfg, arg(a, 'tf32'), int(arg(a, 'seed')), len(sp), sp[0] if sp else '-', min(v.values()), v[st[-1]], q(rd, .5), q(rd, .99), rd[-1] if rd else float('nan'), max(ua.values()) if ua else float('nan'), ua[max(ua)] if ua else float('nan')))
print(f"{'config':22s} tf32 seed spikes first   best      final    | refresh_df median  p99      max     | Uacc max  end")
for r in sorted(out, key=lambda r: (r[0], -int(r[1]), r[2])):
    print(f"{r[0]:22s} {r[1]:>4s} {r[2]:4d} {r[3]:6d} {str(r[4]):>6s} {r[5]:.3g} {r[6]:9.3g}   | {r[7]:9.2e} {r[8]:9.2e} {r[9]:9.2e} | {r[10]:8.2e} {r[11]:8.2e}")
for cfg in sorted({r[0] for r in out}):
    for t in ('1', '0'):
        R = [r for r in out if r[0] == cfg and r[1] == t]
        if R: print(f"  {cfg} tf32 {t}: {sum(1 for r in R if r[3])} of {len(R)} seeds spiked; median refresh_df {stt.median(r[7] for r in R):.2e}")
