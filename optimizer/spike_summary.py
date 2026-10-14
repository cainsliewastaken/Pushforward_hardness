"""Spike defect (scale_spikefield): per run the final / best validation, the probe-loss jumps (> 3x the running minimum, from the
instrument's start), and at each jump and in the 5 steps before it: refresh_df (function change across the basis refresh), Vorth / Uorth,
re-orthonormalisation, rho_raw / rho_real (proposed / realised output change over the budget), umax. Plus the run-wide maxima."""
import json, glob, os, collections, math
import sys; F = os.path.expandvars(sys.argv[1] if len(sys.argv) > 1 else '$SCRATCH/l63_held_rough/runs/scale_spikefield')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl'), key=lambda f: int(f.split('_j')[1].split('.')[0])):
    rows = [json.loads(l) for l in open(fn)]; a = rows[0]['args']
    m = collections.defaultdict(dict)
    for r in rows: m[r['metric']][r['step']] = r['value']
    lab = f"{'smooth' if arg(a, 'p') == '0' else 'rough'} B{arg(a, 'batch')} T{arg(a, 'steps')} lr {arg(a, 'lr')}" + ''.join(f" {k} {arg(a, k)}" for k in ('rem_iter', 'cholqr', 'u_clamp') if arg(a, k) is not None)
    v = m['val']; pr = m.get('sd_probe', {})
    print(f"=== {os.path.basename(fn)} {lab}: final {list(m['val_final'].values())[0]:.3g} best {min(v.values()):.3g}")
    if not pr: continue
    st = sorted(pr); run_min = float('inf'); jumps = []
    for s in st:
        if pr[s] > 3 * run_min: jumps.append(s)
        run_min = min(run_min, pr[s])
    rem = float(arg(a, 'rem'))
    def g(k, s): return m.get(f'sd_{k}', {}).get(s, float('nan'))
    print(f"   run-wide: max refresh_df {max(m['sd_refresh_df'].values()):.3g}  max Vorth {max(m['sd_Vorth'].values()):.3g}  max Uorth {max(m['sd_Uorth'].values()):.3g}  "
          f"max rho_raw/budget {max(x for x in m['sd_rho_raw'].values() if math.isfinite(x)) / rem:.3g}  max rho_real/budget {max([x for x in m['sd_rho_real'].values() if math.isfinite(x)] or [float('nan')]) / rem:.3g}  max umax {max(m['sd_umax'].values()):.3g}")
    first = []; prev = -10
    for s in jumps:
        if s - prev > 20: first.append(s)
        prev = s
    print(f"   {len(jumps)} steps with probe > 3x running min; onsets: {first[:8]}")
    for s in first[:4]:
        for t in range(s - 5, s + 2):
            if t in pr: print(f"     step {t}: probe {pr[t]:.3g} refresh_df {g('refresh_df', t):.2g} nref {g('nref', t):.0f} reorth {g('reorth', t):.0f} Vorth {g('Vorth', t):.2g} Uorth {g('Uorth', t):.2g} rho_raw/b {g('rho_raw', t) / rem:.3g} rho_real/b {g('rho_real', t) / rem:.3g} umax {g('umax', t):.3g} wnorm {g('wnorm', t):.4g}")
