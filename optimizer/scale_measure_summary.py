"""Summary of the horizon measurement (scale_measure.sh; SIGN_PROCESS.md section 7). Per run: the window table (c, r, concentration,
kappa_W for W = 1..40, batch / held-out rho ratio) and, at the diag steps, per layer the capped output-energy share, the cap-active
fraction and the lag-1 sign agreement of consecutive steps, with the per-decile profile at a few steps."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_measure')
WS = (1, 2, 5, 10, 20, 40)
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl')):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    args = rows[0]['args']; name = 'coord raw' if '--problem coord' in args else ('field rough' if '--p 2' in args else 'field smooth')
    vf = [r['value'] for r in rows if r['metric'] == 'val_final']
    print(f"==== {name}  ({os.path.basename(fn)})  val_final {vf[0] if vf else float('nan'):.4g}")
    by = collections.defaultdict(dict)
    for r in rows: by[r['step']][r['metric']] = r['value']
    print('   step    c      r     conc   ' + ' '.join(f'kW{W:<3d}' for W in WS) + '  rho_b/rho_v')
    for s in sorted(by):
        d = by[s]
        if 'sa_kappaW1' not in d: continue
        print(f"   {s:5d}  {d.get('sa_c', float('nan')):.3f}  {d.get('sa_r', float('nan')):+.3f}  {d.get('sa_conc', float('nan')):5.1f}  "
              + ' '.join(f"{d.get(f'sa_kappaW{W}', float('nan')):5.2f}" for W in WS) + f"   {d.get('sa_rho_batch_over_heldout', float('nan')):.2f}")
    L = 1 + max(int(m.split('_l')[1]) for m in rows[0]['metric'].split() if False) if False else 3
    print('   diag step | per layer: cap_eshare / capfrac / dsign_acf1 | rem_rho')
    for s in sorted(by):
        d = by[s]
        if 'cap_eshare_l0' not in d: continue
        cells = []
        for l in range(3):
            if f'cap_eshare_l{l}' in d: cells.append(f"l{l}: {d[f'cap_eshare_l{l}']:.2f} / {d[f'capfrac_l{l}']:.3f} / {d.get(f'dsign_acf1_l{l}', float('nan')):+.2f}")
        print(f"   {s:5d} | " + ' | '.join(cells) + f" | rho {d.get('rem_rho', float('nan')):.3f}")
    for s in (250, 1000, 2500, 4000, 5000):
        d = by.get(s, {})
        if 'prof_eshare_d0_l1' not in d: continue
        print(f'   per-decile profile at step {s} (decile 0 = fastest):')
        for l in range(3):
            if f'prof_eshare_d0_l{l}' not in d: continue
            print(f"      l{l} eshare   " + ' '.join(f"{d[f'prof_eshare_d{k}_l{l}']:.3f}" for k in range(10)))
            print(f"      l{l} capfrac  " + ' '.join(f"{d[f'prof_capfrac_d{k}_l{l}']:.3f}" for k in range(10)))
            if f'prof_dsign_acf1_d0_l{l}' in d: print(f"      l{l} dsign1   " + ' '.join(f"{d[f'prof_dsign_acf1_d{k}_l{l}']:+.2f}" for k in range(10)))
