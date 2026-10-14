"""Item 53: cross-fitting. Per run: final / best validation and the end train-batch loss; then the agreement diagnostic per layer, fast deciles
(0-2) and slow deciles (7-9), normalised (same and cross by mean |g|, val by mean |g_val|), early (steps <= 1500, epoch 1 on turbulence) and late
(steps > 3000)."""
import json, glob, os, collections
S = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_xfold')
def arg(a, k): return a.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in a else None
for fn in sorted(glob.glob(f'{S}/res_j*.jsonl'), key=lambda f: int(f.split('_j')[1].split('.')[0])):
    rows = [json.loads(l) for l in open(fn)]
    if not rows: continue
    a = rows[0]['args']
    lab = (f"turb {arg(a, 'turb_target')}" if '--problem turb' in a else ('field rough' if arg(a, 'p') == '2' else 'field smooth')) + (' RULE' if arg(a, 'xfold_rule') == '1' else ' diag') + (f" shrink {arg(a, 'u_decay')}" if arg(a, 'u_decay') else '')
    m = collections.defaultdict(dict)
    for r in rows: m[r['metric']][r['step']] = r['value']
    vf = list(m.get('val_final', {}).values()); vb = list(m.get('val_best', {}).values()); tb = m.get('train_batch_loss', {})
    print(f"{os.path.basename(fn):14s} {lab:34s} final {vf[0] if vf else float('nan'):.4g} best {vb[0] if vb else float('nan'):.4g} train_end {tb[max(tb)] if tb else float('nan'):.4g}")
    for l in range(3):
        if f'xf_same_l{l}_d0' not in m: continue
        out = []
        for win, sel in (('early', lambda s: s <= 1500), ('late', lambda s: s > 3000)):
            for band, ds in (('fast', (0, 1, 2)), ('slow', (7, 8, 9))):
                def nz(num, den):
                    st = [s for s in m[f'xf_{num}_l{l}_d0'] if sel(s)]
                    if not st: return float('nan')
                    return sum(m[f'xf_{num}_l{l}_d{d}'][s] for s in st for d in ds) / (sum(m[f'xf_{den}_l{l}_d{d}'][s] for s in st for d in ds) + 1e-30)
                out.append(f"{win} {band}: same {nz('same', 'absg'):+.3f} cross {nz('cross', 'absg'):+.3f} val {nz('val', 'absgv'):+.3f}")
        print(f"      l{l}  " + ' | '.join(out))
