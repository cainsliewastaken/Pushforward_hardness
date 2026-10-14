"""Summary of the batch-size scaling grid (scale_bs.sh): per target and batch, the mean final validation MSE over seeds for each
(lr, budget), the optimum, and the batch-256 optimum from runs/scale_grid (item 40) for the scaling ratio."""
import json, glob, os, sys, collections
F = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_bs')
G = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale_grid')
def arg(args, k): return args.split(f'--{k} ')[1].split()[0] if f'--{k} ' in args else None
def load(folder, want_kronq=True):
    res = collections.defaultdict(list)
    for fn in glob.glob(f'{folder}/res_j*.jsonl'):
        rows = [json.loads(l) for l in open(fn)]
        if not rows: continue
        a = rows[0]['args']
        if want_kronq and '--opt kronqf' not in a: continue
        vf = [r['value'] for r in rows if r['metric'] == 'val_final']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
        if not vf: continue
        t = 'rough' if arg(a, 'p') == '2' else 'smooth'; bs = int(arg(a, 'batch') or 256)
        res[(t, bs, float(arg(a, 'lr')), float(arg(a, 'rem')))].append((vf[0], w[0] if w else float('nan')))
    return res
res = load(F); ref = load(G)
for t in ('rough', 'smooth'):
    print(f'==== {t}')
    best = {}
    for bs in (256, 512, 1024):
        src = ref if bs == 256 else res
        cells = {k: v for k, v in src.items() if k[0] == t and k[1] == bs}
        if not cells: continue
        lrs = sorted(set(k[2] for k in cells)); rems = sorted(set(k[3] for k in cells))
        print(f'   batch {bs}: rows lr, cols budget ' + ' '.join(f'{r:>9.2f}' for r in rems))
        for lr in lrs:
            row = []
            for r in rems:
                v = cells.get((t, bs, lr, r))
                row.append(f"{sum(x[0] for x in v) / len(v):9.3g}" if v else f"{'-':>9s}")
            print(f'      lr {lr:<8g} ' + ' '.join(row))
        k_best = min(cells, key=lambda k: sum(x[0] for x in cells[k]) / len(cells[k])); v = cells[k_best]
        best[bs] = (k_best[2], k_best[3], sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v))
        print(f"      optimum: lr {k_best[2]:g}, budget {k_best[3]:g}, val {best[bs][2]:.4g}, wall {best[bs][3]:.0f} s")
    if 256 in best:
        for bs in (512, 1024):
            if bs in best: print(f"   scaling {bs}/256: lr x{best[bs][0] / best[256][0]:.2f}, budget x{best[bs][1] / best[256][1]:.2f}, val ratio {best[bs][2] / best[256][2]:.2f} (sqrt(n_B) prediction: x{(bs / 256) ** 0.5:.2f})")
