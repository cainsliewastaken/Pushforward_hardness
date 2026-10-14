"""Summary of both batch-size passes (scale_bs.sh + scale_bs2.sh): KronQ cells merged over runs/scale_bs, runs/scale_bs2 and the
batch-256 grid (runs/scale_grid); per target the lr x budget table at each batch, the optimum per batch and the scaling of the
optimal constants; AdamW at batch 1024 (scale_bs2) against KronQ at batch 1024 at equal wall."""
import json, glob, os, sys, collections
S = os.path.expandvars('$SCRATCH/l63_held_rough/runs')
def arg(args, k): return args.split(f'--{k} ')[1].split()[0] if f'--{k} ' in args else None
kq = collections.defaultdict(list); ad = collections.defaultdict(list)
for folder in ('scale_grid', 'scale_bs', 'scale_bs2'):
    for fn in glob.glob(f'{S}/{folder}/res_j*.jsonl'):
        rows = [json.loads(l) for l in open(fn)]
        if not rows: continue
        a = rows[0]['args']; vf = [r['value'] for r in rows if r['metric'] == 'val_final']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
        if not vf: continue
        t = 'rough' if arg(a, 'p') == '2' else 'smooth'; bs = int(arg(a, 'batch') or 256)
        if '--opt kronqf' in a and (folder != 'scale_grid' or arg(a, 'cap') != '0'):
            kq[(t, bs, float(arg(a, 'lr')), float(arg(a, 'rem')))].append((vf[0], w[0] if w else float('nan')))
        elif '--opt adamw' in a:
            ad[(t, bs, float(arg(a, 'lr')), int(arg(a, 'steps') or 5000))].append((vf[0], w[0] if w else float('nan')))
mean = lambda v: sum(x[0] for x in v) / len(v)
for t in ('rough', 'smooth'):
    print(f'==== {t}')
    best = {}
    for bs in (256, 512, 1024, 2048):
        cells = {k: v for k, v in kq.items() if k[0] == t and k[1] == bs}
        if not cells: continue
        lrs = sorted(set(k[2] for k in cells)); rems = sorted(set(k[3] for k in cells))
        print(f'   KronQ batch {bs}: rows lr, cols budget ' + ' '.join(f'{r:>9.2f}' for r in rems))
        for lr in lrs:
            print(f'      lr {lr:<8g} ' + ' '.join(f"{mean(cells[(t, bs, lr, r)]):9.3g}" if (t, bs, lr, r) in cells else f"{'-':>9s}" for r in rems))
        kb = min(cells, key=lambda k: mean(cells[k])); best[bs] = (kb[2], kb[3], mean(cells[kb]), sum(x[1] for x in cells[kb]) / len(cells[kb]))
        print(f"      optimum lr {kb[2]:g} budget {kb[3]:g}: {best[bs][2]:.4g} (wall {best[bs][3]:.0f} s)")
    for bs in sorted(best):
        if bs != 256 and 256 in best:
            print(f"   {bs}/256: optimal lr x{best[bs][0] / best[256][0]:.2f}, optimal budget x{best[bs][1] / best[256][1]:.2f}, val x{best[bs][2] / best[256][2]:.2f}")
    for bs in (256, 1024):
        cells = {k: v for k, v in ad.items() if k[0] == t and k[1] == bs}
        if not cells: continue
        for k in sorted(cells): print(f"   AdamW batch {bs} lr {k[2]:g} steps {k[3]}: {mean(cells[k]):.4g} (wall {sum(x[1] for x in cells[k]) / len(cells[k]):.0f} s)")
        long = {k: v for k, v in cells.items() if k[3] > 5000}
        if long and bs in best:
            kb = min(long, key=lambda k: mean(long[k]))
            print(f"   equal wall at batch {bs}: best AdamW {mean(long[kb]):.4g} vs KronQ optimum {best[bs][2]:.4g}: ratio {mean(long[kb]) / best[bs][2]:.2f}")
