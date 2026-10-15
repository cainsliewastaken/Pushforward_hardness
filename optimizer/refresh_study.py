"""Refresh study (2026-10-15): which problems need KronQ's basis refreshed often, and which can refresh sparsely, as a function of target
roughness and network width; and does the need change from early to late training (THEORY.md sec. 18-19: the frame rotates as
A_kj = v_j^T K' v_k / (lambda_k - lambda_j)).
  roughness: field p in {0 (smooth), 1, 2, 3 (rough)}; turbulence increment (rough) / direct (smooth control)
  width: field 512 / 1024 / 2048 / 4096, turbulence 1024 / 4096 (depth 2)
  refresh: fixed every 5 / 20 / 80 / 320 steps, or KronQ's adaptive rule; KronQ exactly otherwise (kronqx mode probe, which equals
           kronq_main.KronQ), with the basis drift logged at every refresh (kronqx log_refresh)
  CIFAR-100 ResNet-18 (fixed width; roughness not measured, an architecture check): the same five refresh settings, 2 seeds
Stage lr: per (problem, width), lr in a 3-point grid at the adaptive rule, seed 1. Stage grid: the best lr, five settings, seeds 1-2.
  python refresh_study.py jobs ROOT lr|grid     python refresh_study.py work ROOT STAGE     python refresh_study.py summary ROOT
"""
import json, math, os, sys, statistics as S
import kronm_compare as K

FIELD = '--problem field --depth 2 --batch 256 --steps 5000 --eval_every 250 --nval 4096 --wd_abs 1e-3'
TURB = ('--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --depth 2 '
        '--eval_every 250 --nval 4096 --batch 256 --steps 31240 --rollout 100 --rollout_starts 64 --wd_abs 3e-3')
OPT = '--preset current --opt kronqx --kx_mode probe --kx_log 1'
REFRESH = {'f5': '--refresh 5 --adapt_refresh 0', 'f20': '--refresh 20 --adapt_refresh 0', 'f80': '--refresh 80 --adapt_refresh 0',
           'f320': '--refresh 320 --adapt_refresh 0', 'adapt': ''}
CREFRESH = {'f5': '--refresh_fixed 5', 'f20': '--refresh_fixed 20', 'f80': '--refresh_fixed 80', 'f320': '--refresh_fixed 320', 'adapt': ''}
PROBLEMS = [(f'field_p{p}_w{w}', f'{FIELD} --p {p} --width {w}', [3e-4, 1e-3, 3e-3]) for p in (3, 2, 1, 0) for w in (4096, 2048, 1024, 512)] + \
           [(f'turb_{tg}_w{w}', f'{TURB} --turb_target {tg} --width {w}', [5e-5, 1e-4, 2e-4]) for tg in ('delta', 'direct') for w in (4096, 1024)]


def tag(task, setting, lr, seed): return f'{task}__{setting}__lr{lr:g}__s{seed}'


def read(root, stage, t):
    """dict(val, roll (h -> nMSE), rlog, ms) of an MLP run, or dict(acc, ms) of a CIFAR run; None if missing; {'fail': rc} if crashed."""
    if t.startswith('cifar'):
        f = f'{root}/{stage}/res/{t}/summary.json'
        if os.path.exists(f): s = json.load(open(f)); return dict(acc=s['final_test_acc'] if not s['diverged'] else math.nan, ms=s['ms_per_step'])
    else:
        f = f'{root}/{stage}/res/{t}.jsonl'
        if os.path.exists(f):
            r = [json.loads(l) for l in open(f)]; m = {x['metric']: x['value'] for x in r if x['metric'] in ('val_final', 'wall_total', 'refresh_log')}
            steps = 5000 if t.startswith('field') else 31240
            return dict(val=m.get('val_final', math.nan), ms=1e3 * m.get('wall_total', math.nan) / steps, rlog=m.get('refresh_log', []),
                        roll={x['step']: x['value'] for x in r if x['metric'] == 'roll_nmse'})
    d = f'{root}/{stage}/done/{t}'
    return {'fail': json.load(open(d))['rc']} if os.path.exists(d) else None


def best_lr(root, task, lrs):
    sc = [((read(root, 'lr', tag(task, 'adapt', lr, 1)) or {}).get('val', math.nan), lr) for lr in lrs]
    sc = [x for x in sc if math.isfinite(x[0])]
    return min(sc)[1] if sc else None


def grid_lrs(root, task, lrs):
    """lrs run in stage grid: the stage-lr best, plus one step past the grid edge when the best sits on an edge (added 2026-10-15:
    7 of 16 field settings had their best at an edge, the higher lrs failing to learn at all)."""
    b = best_lr(root, task, lrs)
    if b is None: return []
    if b == lrs[0]: return [b, b / 3 if task.startswith('field') else b / 2]
    if b == lrs[-1]: return [b, b * 3 if task.startswith('field') else b * 2]
    return [b]


def chosen_lr(root, task, lrs):
    """the grid-stage lr with the lower mean val at the adaptive rule (seeds 1-2)."""
    sc = []
    for lr in grid_lrs(root, task, lrs):
        v = [(read(root, 'grid', tag(task, 'adapt', lr, s)) or {}).get('val', math.nan) for s in (1, 2)]; v = [x for x in v if math.isfinite(x)]
        if v: sc.append((S.mean(v), lr))
    return min(sc)[1] if sc else best_lr(root, task, lrs)


def jobs(root, stage):
    out = []
    if stage == 'lr':
        for task, base, lrs in PROBLEMS:
            for lr in lrs: out.append((tag(task, 'adapt', lr, 1), f'scale_study.py {OPT} {base} --lr {lr} --seed 1 --out $ROOT/lr/res/{tag(task, "adapt", lr, 1)}.jsonl'))
    else:
        for st in CREFRESH:                                               # CIFAR first: the longest runs
            for seed in (1, 2):
                t = tag('cifar', st, 3e-3, seed)
                out.append((t, f'bench/cifar_kx.py {CREFRESH[st]} --opt kronq --epochs 100 --lr 3e-3 --wd 1e-3 --seed {seed} --name {t} --out $ROOT/grid/res/{t}'))
        for task, base, lrs in sorted(PROBLEMS, key=lambda x: not x[0].startswith('turb')):
            gl = grid_lrs(root, task, lrs)
            if not gl: print(f'{task}: no finished lr run, skipped'); continue
            for lr in gl:
                for st, flags in REFRESH.items():
                    for seed in (1, 2):
                        t = tag(task, st, lr, seed); out.append((t, f'scale_study.py {OPT} {base} {flags} --lr {lr} --seed {seed} --out $ROOT/grid/res/{t}.jsonl'))
    os.makedirs(f'{root}/{stage}/res', exist_ok=True)
    with open(f'{root}/{stage}/jobs.txt', 'w') as fh:
        for t, c in out: fh.write(f'{t}\t{c}\n')
    print(f'{len(out)} jobs -> {root}/{stage}/jobs.txt')


def drift(rlog, steps):
    """mean off-diagonal share per refresh step, per (side) over the early (first 20 %) and late (last 50 %) parts of training."""
    out = {}
    for side in ('A', 'S'):
        for name, lo, hi in (('early', 0, 0.2), ('late', 0.5, 1.0)):
            xs = [r[3] / max(r[4], 1) for r in rlog if r[2] == side and lo * steps < r[0] <= hi * steps and r[4] > 0]
            out[f'{side}_{name}'] = S.mean(xs) if xs else math.nan
    return out


def summary(root):
    fmt = lambda xs: (f'{S.mean(xs):.4g}' + (f' +- {S.stdev(xs):.1g}' if len(xs) > 1 else '')) if xs else '-'
    print('error by refresh setting (mean of seeds; field / turb: val MSE, turb also rollout nMSE mean over h1-20; cifar: test acc)')
    print('drift: off-diagonal share of the factor in the old basis per step since the last refresh, at f20, early (first 20 %) / late (last 50 %)')
    for task, _, lrs in PROBLEMS + [('cifar', '', [3e-3])]:
        lr = 3e-3 if task == 'cifar' else chosen_lr(root, task, lrs)
        if lr is None: continue
        line = f'{task:18s} lr {lr:<7g}'
        for st in REFRESH:
            rs = [read(root, 'grid', tag(task, st, lr, s)) for s in (1, 2)]; rs = [r for r in rs if r and 'fail' not in r]
            if task == 'cifar': line += f' | {st} {fmt([r["acc"] for r in rs if math.isfinite(r["acc"])])}'
            else:
                line += f' | {st} {fmt([r["val"] for r in rs if math.isfinite(r["val"])])}'
                if task.startswith('turb'): line += f' ({fmt([S.mean(r["roll"].get(h, math.nan) for h in range(1, 21)) for r in rs if r["roll"]])})'
        print(line)
        if task != 'cifar':
            steps = 5000 if task.startswith('field') else 31240
            ds = [drift(r['rlog'], steps) for r in (read(root, 'grid', tag(task, 'f20', lr, s)) for s in (1, 2)) if r and 'fail' not in r]
            if ds: print(' ' * 18 + '  drift ' + '  '.join(f'{k} {S.mean(d[k] for d in ds):.3g}' for k in ds[0]))


if __name__ == '__main__':
    cmd, root = sys.argv[1], sys.argv[2]
    if cmd == 'jobs': jobs(root, sys.argv[3])
    elif cmd == 'work': K.work(root, sys.argv[3])
    else: summary(root)
