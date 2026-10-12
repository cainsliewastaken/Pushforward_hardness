"""Blocked / identity Kronecker-Adam vs Adam on the field problem (n = 20k, widths 512,512, GELU, batch 100), seed 1, one GPU.
Stage 1 (per step):   Adam (tuned) and each Kronecker-Adam variant, 6000 steps -> accuracy, FLOPs, steady ms/step.
Stage 2 (per FLOP):   each variant with steps = Adam FLOPs / its FLOPs per step, at the tuned lr and 1.5x.
Stage 3 (per second): each variant with steps = Adam wall / its steady ms per step, same two lrs.
Resumable.  Usage: python block_sweep.py OUT_DIR [p ...]
"""
import os, sys, json, subprocess
OUT = sys.argv[1]; PS = [int(x) for x in sys.argv[2:]] or [1, 0]
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(os.environ['SCRATCH'], 'l63_held_rough', 'runs', 'kadam')
os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
VARIANTS = [('b0', ''), ('b256', '--block 256'), ('b128', '--block 128'), ('b64', '--block 64'),
            ('b0_id', '--last_S identity'), ('b64_id', '--block 64 --last_S identity')]
STEPS = 6000
TUNED = {  # from the field20k tuning (2026-10-12): (lr, b2, warmup)
    'adam': {0: (2.5e-3, 0.99, 0), 1: (3.8e-3, 0.99, 0), 2: (1.7e-3, 0.99, 100)},
    'ks':   {0: (3.8e-3, 0.99, 0), 1: (1.9e-2, 0.999, 0), 2: (1.1e-3, 0.999, 0)},
}
def prob(p): return f'--data {D}/field20k_p{p}.npz --widths 512,512 --act gelu --batch 100 --standardize none --seed 1'

def run(tag, args):
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        r = {}
        if os.path.exists(path):
            for l in open(path):
                d = json.loads(l); r[d['metric']] = d['value']
        return r
    if 'E_val_final' not in read():
        if os.path.exists(path): os.remove(path)
        subprocess.run(f'python -u {HERE}/train_ks.py {args} --out {path}', shell=True, check=False)
    r = read()
    rec = dict(tag=tag, E_val=r.get('E_val_final', float('nan')), E_train=r.get('E_train_final', float('nan')),
               flops=r.get('flops_total', float('nan')), wall=r.get('wall_s', float('nan')),
               ms_step=r.get('ms_per_step_steady', float('nan')), steps=int(args.split('--steps ')[1].split()[0]))
    with open(os.path.join(OUT, 'summary.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print('RESULT', json.dumps(rec), flush=True)
    return rec

for p in PS:
    lr_a, b2_a, wu_a = TUNED['adam'][p]; lr_k, b2_k, wu_k = TUNED['ks'][p]
    A = run(f'p{p}_adam_s{STEPS}', f'--opt adam --lr {lr_a} --b2 {b2_a} --warmup {wu_a} --steps {STEPS} {prob(p)}')
    K = {}
    for name, extra in VARIANTS:
        K[name] = run(f'p{p}_ks_{name}_s{STEPS}', f'--opt kronsign --rule adam --lr {lr_k} --b2 {b2_k} --warmup {wu_k} {extra} --steps {STEPS} {prob(p)}')
    for name, extra in VARIANTS:
        fps = K[name]['flops'] / STEPS
        n_flop = max(50, int(round(A['flops'] / fps)))
        n_time = max(50, int(round(1000 * A['wall'] / K[name]['ms_step'])))
        for mult in (1.0, 1.5):
            lr = lr_k * mult
            run(f'p{p}_ks_{name}_eqflop_s{n_flop}_lr{mult:g}x', f'--opt kronsign --rule adam --lr {lr:.6g} --b2 {b2_k} --warmup {wu_k} {extra} --steps {n_flop} {prob(p)}')
            run(f'p{p}_ks_{name}_eqtime_s{n_time}_lr{mult:g}x', f'--opt kronsign --rule adam --lr {lr:.6g} --b2 {b2_k} --warmup {wu_k} {extra} --steps {n_time} {prob(p)}')
print('SWEEP DONE', flush=True)
