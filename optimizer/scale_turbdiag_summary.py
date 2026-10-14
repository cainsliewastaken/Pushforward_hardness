"""Summary of the turbulence diagnostics (scale_turbdiag.sh): per run the final and best validation MSE, the training batch loss
against the validation loss along the run (generalisation gap), and for the KronQ runs the measurement windows (residual
concentration on the true kernel's top directions, per-step held-out alignment)."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_turbdiag')
def arg(args, k): return args.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in args else None
for fn in sorted(glob.glob(f'{F}/res_j*.jsonl'), key=lambda s: int(s.split('_j')[1].split('.')[0])):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    a = rows[0]['args']; t = arg(a, 'turb_target')
    m = 'adamw' if '--opt adamw' in a else ('kronq adam-rule' if '--rule adam' in a else ('kronq soft sign' if '--sign_floor' in a else 'kronq hard sign'))
    by = collections.defaultdict(dict)
    for r in rows: by[r['step']][r['metric']] = r['value']
    vf = [r['value'] for r in rows if r['metric'] == 'val_final']; vb = [r['value'] for r in rows if r['metric'] == 'val_best']
    print(f"==== {t} | {m}  final {vf[0] if vf else float('nan'):.4g}  best {vb[0] if vb else float('nan'):.4g}  ({os.path.basename(fn)})")
    print('   step   train_batch   val     gap(val/train)' + ('   conc   c' if m != 'adamw' else ''))
    for s in sorted(by):
        d = by[s]
        if 'val' in d and 'train_batch_loss' in d and (s % 1000 == 0 or s in (250, 500)):
            line = f"   {s:5d}  {d['train_batch_loss']:.4g}     {d['val']:.4g}   {d['val'] / max(d['train_batch_loss'], 1e-12):.2f}"
            if m != 'adamw':
                w = by.get(s - 210, {}) if (s - 210) in by else {}
                sa = [(k, v) for k, v in by.items() if 'sa_conc' in v and abs(k - s) <= 60]
                if sa: line += f"   {sa[0][1]['sa_conc']:.1f}x  {sa[0][1].get('sa_c', float('nan')):.3f}"
            print(line)
    if m != 'adamw':
        sa = sorted((k, v) for k, v in by.items() if 'sa_conc' in v)
        if sa: print('   windows (step: conc, c): ' + ' '.join(f"{k}: {v['sa_conc']:.1f}x, {v.get('sa_c', float('nan')):.3f}" for k, v in sa[:6]) + (' ...' if len(sa) > 6 else ''))
