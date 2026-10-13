"""Mechanism summary (scale_mech.sh): per rule and target, the per-step loss decomposition, the residual split by the
current kernel, the leverage of the residual direction and its change (Cross, Q), the alignment, and the minibatch SNR,
along the run."""
import json, os, sys, glob, re, collections
F = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_mech')
LAB = {0: 'rough sign+cap+budget', 1: 'rough adam rule', 2: 'rough frozen ridge', 3: 'rough AdamW', 4: 'smooth sign+cap+budget', 5: 'smooth adam rule', 6: 'smooth frozen ridge', 7: 'smooth AdamW'}
for j in range(8):
    p = f'{F}/res_j{j}.jsonl'
    if not os.path.exists(p): print(f'== j{j} {LAB[j]}: missing'); continue
    rows = [json.loads(l) for l in open(p)]
    by = collections.defaultdict(dict)
    for r in rows: by[r['metric']][r['step']] = r['value']
    steps = sorted(by['mech_dL']); vals = by['val']
    print(f'\n==== j{j} {LAB[j]}: val_final {by.get("val_final", {}).get(max(by.get("val_final", {0: 0})), float("nan")):.4g}')
    print('  step    val      dL        first     gn        rem      |first_fast first_slow| cosE  Jd/E  fastfrac Jd_fast | levE   dlev2   cross    Q     | snr   lam1   rayleigh')
    for s in steps:
        g = lambda m: by[m].get(s, float('nan'))
        print(f"  {s:5d} {vals.get(s, float('nan')):8.3g} {g('mech_dL'):+9.2e} {g('mech_first'):+9.2e} {g('mech_gn'):+8.1e} {g('mech_rem'):+8.1e} | {g('mech_first_fast'):+8.1e} {g('mech_first_slow'):+8.1e} | {g('mech_cosE'):5.2f} {g('mech_Jd_over_E'):5.2f} {g('mech_fastfrac'):6.3f} {g('mech_Jd_fastfrac'):6.3f} | {g('mech_levE'):6.3g} {g('mech_dlev2'):+8.2g} {g('mech_cross_E'):+8.2g} {g('mech_Q_E'):+7.2g} | {g('mech_snr'):5.2f} {g('mech_lam1'):7.3g} {g('mech_rayleigh'):7.3g}")
    sn = {m: by[m] for m in by if m.startswith('mech_snr_d')}
    if sn:
        for l in ('l0', 'l1'):
            for s in (250, 2500, 5000):
                row = [sn.get(f'mech_snr_d{k}_{l}', {}).get(s, float('nan')) for k in range(10)]
                if not all(x != x for x in row): print(f'  snr by lambda decile {l} @{s}: ' + ' '.join(f'{x:.2f}' for x in row))
