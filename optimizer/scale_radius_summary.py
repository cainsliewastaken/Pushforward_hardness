"""Taylor-radius scan summary (kronq_fast --radius_scan): per scan step, layer and direction, the step's own length s0, the
leverage ||J u|| (output change per unit parameter length), the radius s* where the relative remainder reaches 0.1, in
parameter units and in output units, and the remainder curve on the length grid s0 * 10^(-3 .. 2)."""
import json, os, sys, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_radius')
for tag in sys.argv[1:] or ('rough', 'smooth'):
    p = f'{F}/res_{tag}.jsonl'
    if not os.path.exists(p): print(tag, 'missing'); continue
    rows = [json.loads(l) for l in open(p)]
    print(f'==== {tag}: val', [(r['step'], round(r['value'], 5)) for r in rows if r['metric'] == 'val'])
    d = collections.defaultdict(dict)
    for r in rows:
        m = r['metric']
        if m.startswith('rscan_'):
            parts = m.split('_'); d[(r['step'], parts[2], parts[1])]['_'.join(parts[3:])] = r['value']
    order = ['step', 'sign', 'grad', 'ridge2', 'ridge4', 'rand', 'mode']
    print('  grid: k=0..12 is s = s0 * 10^(-3 + 5k/12); s0 at k ~ 7.2.   columns: lev, s*, s*/s0, s*.lev, rf at k=4..11')
    for (step, l, name), v in sorted(d.items(), key=lambda t: (t[0][0], t[0][1], order.index(t[0][2]))):
        s0 = v['s0']; rfs = [v.get(f'rf{k}', float('nan')) for k in range(13)]
        print(f"  step {step:5d} {l} {name:7s} s0 {s0:8.3g} lev {v['lev']:8.3g} cosE {v.get('cosE', float('nan')):6.3f}  s* {v['sstar']:8.3g}  s*/s0 {v['sstar'] / s0:8.3g}  s*.lev {v['sstar_out']:8.3g}  rf: " + ' '.join(f'{x:.2g}' for x in rfs[4:12]))
