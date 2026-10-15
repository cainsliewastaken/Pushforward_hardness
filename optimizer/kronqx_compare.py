"""Cheaper curvature statistics for KronQ (kronqx.py, and the main optimizer's own stats_every) against KronQ with the probe every step,
at KronQ's best settings from kronm_compare (field lr 1e-3 decay 1e-3; turbulence lr 1e-4 decay 3e-3, both targets with a 100-step
rollout; CIFAR-100 lr 3e-3 decay 1e-3). The reference runs are kronm_compare's KronQ seeds 1-3 at those settings (same main code).

  python kronqx_compare.py jobs ROOT       write ROOT/run/jobs.txt
  python kronqx_compare.py work ROOT       a worker (one GPU), as kronm_compare.work
  python kronqx_compare.py summary ROOT REF  table: mean +- sd over seeds of each variant and of the reference (REF = kronm_compare root)
"""
import json, math, os, sys, statistics as S
import kronm_compare as K

VARIANTS = {'kx_fused': '--preset current --opt kronqx --kx_mode fused', 'kx_se4': '--preset current --stats_every 4',
            'kx_se16': '--preset current --stats_every 16', 'kx_gradfac': '--preset current --opt kronqx --kx_mode gradfac',
            'kx_aonly': '--preset current --opt kronqx --kx_mode aonly'}
K.METHODS.update(VARIANTS)
K.CMETHODS['kx_se4'] = '--opt kronq --stats_every 4'
BEST = {'field': (1e-3, 1e-3), 'turb_delta': (1e-4, 3e-3), 'turb_direct': (1e-4, 3e-3), 'cifar': (3e-3, 1e-3)}
REF_STAGE = {1: ('grid', 'seeds'), 2: ('seeds', 'seeds2'), 3: ('seeds', 'seeds2')}   # where kronm_compare's KronQ seeds live (MLP, CIFAR)


def jobs(root):
    out = [K.job('cifar', 'kx_se4', *BEST['cifar'], s, 'run') for s in (1, 2)]
    for task in ('turb_delta', 'turb_direct', 'field'):
        for m in VARIANTS:
            for s in ((1,) if m == 'kx_fused' else (1, 2, 3)): out.append(K.job(task, m, *BEST[task], s, 'run'))
    os.makedirs(f'{root}/run/res', exist_ok=True)
    with open(f'{root}/run/jobs.txt', 'w') as fh:
        for tag, cmd in out: fh.write(f'{tag}\t{cmd}\n')
    print(f'{len(out)} jobs -> {root}/run/jobs.txt')


def stats(root, stage, tag):
    """(score, val_final or test acc, ms per step, rollout nMSE at h10) of one run, or None."""
    sc = K.result(root, stage, tag)
    if sc is None: return None
    if tag.startswith('cifar'):
        f = f'{root}/{stage}/res/{tag}/summary.json'
        if not os.path.exists(f): return (sc, math.nan, math.nan, math.nan)
        s = json.load(open(f)); return (sc, s['final_test_acc'], s['ms_per_step'], math.nan)
    f = f'{root}/{stage}/res/{tag}.jsonl'
    if not os.path.exists(f): return (sc, math.nan, math.nan, math.nan)
    r = [json.loads(l) for l in open(f)]; m = {x['metric']: x for x in r if x['metric'] in ('val_final', 'wall_total')}
    ro = {x['step']: x['value'] for x in r if x['metric'] == 'roll_nmse'}
    steps = 5000 if tag.startswith('field') else 31240
    return (sc, m.get('val_final', {}).get('value', math.nan), 1e3 * m.get('wall_total', {}).get('value', math.nan) / steps, ro.get(10, math.nan))


def summary(root, ref):
    fmt = lambda xs: (f'{S.mean(xs):.4g} +- {S.stdev(xs):.2g}' if len(xs) > 1 else f'{xs[0]:.4g}') if xs else '-'
    for task in ('field', 'turb_delta', 'turb_direct', 'cifar'):
        print(f'== {task} at lr {BEST[task][0]:g}, decay {BEST[task][1]:g}   (columns: score | val / test acc | ms per step | rollout h10)')
        rows = [('KronQ (probe every step)', [stats(ref, REF_STAGE[s][task == 'cifar'], K.job(task, 'kronq', *BEST[task], s, '')[0]) for s in (1, 2, 3)])]
        for m in (['kx_se4'] if task == 'cifar' else VARIANTS):
            rows.append((m, [stats(root, 'run', K.job(task, m, *BEST[task], s, 'run')[0]) for s in (1, 2, 3)]))
        for name, rs in rows:
            rs = [x for x in rs if x is not None]
            cols = [fmt([x[i] for x in rs if math.isfinite(x[i])]) for i in range(4)]
            print(f'  {name:26s} n={len(rs)}  ' + ' | '.join(cols))


if __name__ == '__main__':
    cmd, root = sys.argv[1], sys.argv[2]
    if cmd == 'jobs': jobs(root)
    elif cmd == 'work': K.work(root, 'run')
    else: summary(root, sys.argv[3])
