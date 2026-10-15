"""KronQ (main: learned multiplier x sign) against KronM (TEST: per-mode heavy-ball momentum, kronm_main.py) and KronM-const (one momentum
for every mode), on the rough field, 2D turbulence (increment and direct targets, with a forward rollout) and CIFAR-100 ResNet-18. Decoupled
weight decay on every run (the method's own, per step at peak lr, following the schedule); the field also at zero decay.

  python kronm_compare.py jobs  ROOT STAGE        write ROOT/<STAGE>/jobs.txt (stage grid: lr x decay, seed 1; stage seeds: the best
                                                  setting per method and task from the grid, seeds 2 and 3)
  python kronm_compare.py work  ROOT STAGE        a worker (one GPU): claims jobs by mkdir (atomic on Lustre) and runs them in order
  python kronm_compare.py summary ROOT            tables from the saved results (no retraining)
Outputs: ROOT/<STAGE>/{jobs.txt, claim/, log/, done/, res/}. Code is run from ROOT/code (a git archive of the commit in ROOT/COMMIT).
"""
import json, math, os, subprocess, sys, time, glob

FIELD = '--problem field --p 2 --width 4096 --depth 2 --batch 256 --steps 5000 --eval_every 250 --nval 4096'
TURB = ('--problem turb --turb_cache $SCRATCH/l63_held_rough/data/turb64_lead10_400k.npz --turb_cpu 1 --turb_onepass 1 --width 4096 --depth 2 '
        '--eval_every 250 --nval 4096 --batch 256 --steps 31240 --rollout 100 --rollout_starts 64')
CIFAR = '--epochs 100'
METHODS = {'kronq': '--preset current', 'kronm': '--preset current --opt kronm_main --km_rule crit',
           'kronm_const': '--preset current --opt kronm_main --km_rule const'}
CMETHODS = {'kronq': '--opt kronq', 'kronm': '--opt kronm --km_rule crit', 'kronm_const': '--opt kronm --km_rule const',
            'kronm_b90': '--opt kronm --km_rule crit --km_beta_max 0.9', 'kronm_b99': '--opt kronm --km_rule crit --km_beta_max 0.99'}
# added 2026-10-15 after the grid's CIFAR KronM (beta_max 0.999) runs stalled at chance by epoch 2 at every lr (stopped at epoch 2-3):
# the per-mode rule with its momentum capped, run in stage seeds (seed 1, two lrs each) and the best one again in stage seeds2 (seeds 2, 3)
# added 2026-10-15: at width 4096 KronM diverged within 10 steps at lr >= 0.01 on turbulence (16 x 16 blocks: the block-diagonal Kronecker
# curvature underestimates the true top curvature, so "lr x Newton on the top estimated mode" is far more than lr x Newton); a probe at
# lr 1e-3 / 3e-3 was stable. Lower-lr KronM grid in stage seeds; its best (over grid and seeds) gets seeds 2, 3 in stage seeds2.
GRID2 = {'turb_delta': ([3e-4, 1e-3, 3e-3], [3e-4, 1e-3, 3e-3]), 'turb_direct': ([3e-4, 1e-3, 3e-3], [3e-4, 1e-3, 3e-3]),
         'field': ([3e-4, 1e-3, 3e-3, 1e-2], [0.0, 1e-3])}
CIFAR_BM = {'kronm_b90': [0.003, 0.01], 'kronm_b99': [0.003, 0.01]}
# added 2026-10-15 (user): CIFAR at the decay strengths of the older KronQ runs, wd 1 and 3 at lr 1e-3 = 1e-3 and 3e-3 per step, in stage
# seeds for KronQ and KronM-const (at their grid-best lr) and the capped variants (both lrs); every CIFAR seed repeat is in stage seeds2,
# chosen over grid + seeds. Groups compared: KronQ, KronM-const, KronM capped (b90 or b99, whichever is better).
CIFAR_HIWD = [1e-3, 3e-3]
CIFAR_GROUPS = {'kronq': ('kronq',), 'kronm_const': ('kronm_const',), 'kronm_capped': ('kronm_b90', 'kronm_b99')}


def seed1_runs(root, stages=('grid', 'seeds')):
    """(stage, task, method, lr, wd) of every seed-1 job listed in the given stages."""
    out = []
    for st in stages:
        f = f'{root}/{st}/jobs.txt'
        if not os.path.exists(f): continue
        for line in open(f):
            task, method, lr, wd, seed = line.split('\t')[0].split('__')
            if seed == 's1': out.append((st, task, method, float(lr[2:]), float(wd[2:])))
    return out


def best(root, task, methods, stages=('grid', 'seeds')):
    """(score, method, lr, wd) of the best finished seed-1 run of task among methods, or None."""
    sc = [(result(root, st, job(t, m, lr, wd, 1, st)[0]), m, lr, wd) for st, t, m, lr, wd in seed1_runs(root, stages) if t == task and m in methods]
    sc = [x for x in sc if x[0] is not None and math.isfinite(x[0])]
    return min(sc) if sc else None
GRID = {   # task -> method -> (lrs, decays)
    'field': {'kronq': ([3e-4, 1e-3, 3e-3], [0.0, 1e-3]), 'kronm': ([0.01, 0.03, 0.1, 0.3, 1.0], [0.0, 1e-3]),
              'kronm_const': ([0.003, 0.01, 0.03, 0.1], [1e-3])},
    'turb_delta': {'kronq': ([1e-4, 2e-4, 5e-4], [3e-4, 1e-3, 3e-3]), 'kronm': ([0.01, 0.03, 0.1, 0.3], [3e-4, 1e-3, 3e-3]),
                   'kronm_const': ([0.003, 0.01, 0.03], [1e-3])},
    'cifar': {'kronq': ([1e-3, 3e-3], [5e-5, 5e-4]), 'kronm': ([0.003, 0.01, 0.03, 0.1], [5e-5, 5e-4]), 'kronm_const': ([0.003, 0.01], [5e-5])},
}
GRID['turb_direct'] = GRID['turb_delta']


def job(task, method, lr, wd, seed, stage):
    tag = f'{task}__{method}__lr{lr:g}__wd{wd:g}__s{seed}'
    if task == 'cifar':
        cmd = f'bench/cifar.py {CMETHODS[method]} {CIFAR} --lr {lr} --wd {wd} --seed {seed} --name {tag} --out $ROOT/{stage}/res/{tag}'
    else:
        base = FIELD if task == 'field' else TURB + f' --turb_target {task.split("_")[1]}'
        cmd = f'scale_study.py {METHODS[method]} {base} --lr {lr} --wd_abs {wd} --seed {seed} --out $ROOT/{stage}/res/{tag}.jsonl'
    return tag, cmd


def result(root, stage, tag):
    """the score of one run (lower is better): field val_final; turb mean rollout nMSE over horizons 1-100 (and val_final); cifar
    1 - final test accuracy. None if missing; inf if diverged."""
    if tag.startswith('cifar'):
        f = f'{root}/{stage}/res/{tag}/summary.json'
        if not os.path.exists(f):
            return math.inf if os.path.exists(f'{root}/{stage}/done/{tag}') else None
        s = json.load(open(f)); return math.inf if s['diverged'] else 1 - s['final_test_acc']
    f = f'{root}/{stage}/res/{tag}.jsonl'
    if not os.path.exists(f): return math.inf if os.path.exists(f'{root}/{stage}/done/{tag}') else None
    rows = [json.loads(l) for l in open(f)]; m = {}
    for r in rows: m.setdefault(r['metric'], []).append(r)
    if 'val_final' not in m or not math.isfinite(m['val_final'][-1]['value']): return math.inf
    if tag.startswith('turb'):
        ro = [r['value'] for r in m.get('roll_nmse', [])]
        return sum(min(v, 10.0) if math.isfinite(v) else 10.0 for v in ro) / len(ro) if ro else math.inf
    return m['val_final'][-1]['value']


def jobs(root, stage):
    out = []
    if stage == 'grid':
        for task in ('cifar', 'turb_delta', 'turb_direct', 'field'):            # longest first
            for method, (lrs, wds) in GRID[task].items():
                for lr in lrs:
                    for wd in wds: out.append(job(task, method, lr, wd, 1, stage))
    elif stage == 'seeds2':
        for methods in CIFAR_GROUPS.values():
            b = best(root, 'cifar', methods)
            if b:
                for seed in (2, 3): out.append(job('cifar', b[1], b[2], b[3], seed, stage))
        for task in GRID2:
            b = best(root, task, ('kronm',))
            if b:
                for seed in (2, 3): out.append(job(task, 'kronm', b[2], b[3], seed, stage))
    else:
        for m, lrs in CIFAR_BM.items():
            for lr in lrs:
                for wd in [5e-5] + CIFAR_HIWD: out.append(job('cifar', m, lr, wd, 1, stage))
        for m in ('kronq', 'kronm_const'):
            b = best(root, 'cifar', (m,), ('grid',))
            lr = b[2] if b else GRID['cifar'][m][0][-1]
            for wd in CIFAR_HIWD: out.append(job('cifar', m, lr, wd, 1, stage))
        for task in ('turb_delta', 'turb_direct', 'field'):
            for lr in GRID2[task][0]:
                for wd in GRID2[task][1]: out.append(job(task, 'kronm', lr, wd, 1, stage))
        for task in ('turb_delta', 'turb_direct', 'field'):           # CIFAR seed repeats: stage seeds2
            for method, (lrs, wds) in GRID[task].items():
                if method == 'kronm': continue                              # its seeds come in stage seeds2, over grid and GRID2           # its seeds come in stage seeds2, over grid and GRID2
                sc = [(result(root, 'grid', job(task, method, lr, wd, 1, 'grid')[0]), lr, wd) for lr in lrs for wd in wds]
                sc = [x for x in sc if x[0] is not None and math.isfinite(x[0])]
                if not sc: continue
                _, lr, wd = min(sc)
                for seed in (2, 3): out.append(job(task, method, lr, wd, seed, stage))
    d = f'{root}/{stage}'; os.makedirs(d + '/res', exist_ok=True)
    with open(d + '/jobs.txt', 'w') as fh:
        for tag, cmd in out: fh.write(f'{tag}\t{cmd}\n')
    print(f'{len(out)} jobs -> {d}/jobs.txt')


def work(root, stage):
    d = f'{root}/{stage}'
    for sub in ('claim', 'log', 'done'): os.makedirs(f'{d}/{sub}', exist_ok=True)
    env = dict(os.environ, ROOT=root, OMP_NUM_THREADS='8')
    me = f"{os.uname().nodename}:{os.environ.get('SLURM_PROCID', '?')}:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
    for line in open(f'{d}/jobs.txt'):
        tag, cmd = line.rstrip('\n').split('\t')
        try: os.mkdir(f'{d}/claim/{tag}')
        except FileExistsError: continue
        cmd = os.path.expandvars(cmd.replace('$ROOT', root)); t0 = time.time()
        with open(f'{d}/log/{tag}.txt', 'w') as lg:
            lg.write(f'# {me} {time.strftime("%F %T")}\n# python -u {cmd}\n'); lg.flush()
            rc = subprocess.call(['python', '-u'] + cmd.split(), cwd=f'{root}/code/optimizer', env=env, stdout=lg, stderr=subprocess.STDOUT)
        with open(f'{d}/done/{tag}', 'w') as fh: fh.write(json.dumps(dict(rc=rc, wall=round(time.time() - t0, 1), worker=me)) + '\n')
    print(f'{me} no more jobs', flush=True)


def summary(root):
    for stage in ('grid', 'seeds', 'seeds2'):
        if not os.path.exists(f'{root}/{stage}/jobs.txt'): continue
        print(f'== {stage}  (score: field val MSE; turb mean rollout nMSE h1-100; cifar test error)')
        for line in open(f'{root}/{stage}/jobs.txt'):
            tag = line.split('\t')[0]; r = result(root, stage, tag)
            extra = ''
            if tag.startswith('turb') and r is not None and os.path.exists(f'{root}/{stage}/res/{tag}.jsonl'):
                rows = [json.loads(l) for l in open(f'{root}/{stage}/res/{tag}.jsonl')]
                ro = {x['step']: x['value'] for x in rows if x['metric'] == 'roll_nmse'}; vf = [x['value'] for x in rows if x['metric'] == 'val_final']
                extra = f"  val {vf[-1]:.4g}" + ''.join(f'  h{h} {ro[h]:.3g}' for h in (1, 10, 50, 100) if h in ro) if vf else ''
            print(f"{tag:55s} {('-' if r is None else f'{r:.4g}'):>8s}{extra}")


if __name__ == '__main__':
    cmd, root = sys.argv[1], sys.argv[2]
    if cmd == 'jobs': jobs(root, sys.argv[3])
    elif cmd == 'work': work(root, sys.argv[3])
    else: summary(root)
