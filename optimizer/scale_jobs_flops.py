"""Equal-FLOP job list for the scaling study (REPORT 61 item 6): every run gets the same FLOP budget, B = AdamW for
BUDGET_STEPS steps at that width, and its step count is B / (its FLOPs per step). Per-step FLOPs are the analytic
counts below (what an efficient implementation costs; the harness's blocked rotations are a Python loop, so wall time
does not follow them). Writes $L63_ROOT/runs/scale_flops/jobs.txt, longest (in wall) first.
usage: python scale_jobs_flops.py [BUDGET_STEPS=20000]"""
import os, sys
ROOT = os.environ.get('L63_ROOT', os.path.expandvars('$SCRATCH/l63_held_rough'))
OUT = f'{ROOT}/runs/scale_flops'; os.makedirs(OUT, exist_ok=True)
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
M, NIN, NOUT = 256, 256, 256
def P_of(w): return NIN * w + w + w * w + w + w * NOUT + NOUT
def cost(opt, w, block=0, diag=False, rem=False, gnls=False):
    """FLOPs per step in units of F0 = 6 P m (forward + backward)."""
    P = P_of(w); F0 = 6 * P * M; sq = 2 * w ** 3
    if opt == 'adamw': c = 1.0
    elif opt == 'muon': c = 1.0 + 60 * w ** 3 / F0                       # NS5 on the two w x w matrices (fp32 count)
    elif diag: c = 1.2
    elif block == 0: c = 1.0 + (12 * sq + 1.0 * sq) / F0                 # rotations of C, Gr, D (12 w^3 per square layer x 2) + eigh amortised
    else: c = 1.0 + 2 * (12 * w * w * block) / F0 + 0.05
    if rem: c += 1.0 / 3
    if gnls: c += 1.3
    return c
jobs = []
for w in (4096, 2048, 1024):
    for p in ('0', '2'):
        def add(opt, lr, extra='', **kw):
            c = cost(opt, w, **kw); steps = max(100, int(BUDGET / c)); ev = max(10, steps // 100)
            jobs.append((w, c * steps, f'--opt {opt} --lr {lr} {extra} --width {w} --depth 2 --p {p} --steps {steps} --batch {M} --refresh 20 --eval_every {ev}'))
        for lr in ('3e-4', '1e-3', '3e-3'): add('adamw', lr)
        for lr in ('2e-2', '5e-2'): add('muon', lr, '--lr_adam 1e-3')
        for lr in ('1e-3', '3e-3'): add('soap', lr, '--block 256', block=256)
        add('soap', '3e-3', block=0)
        add('kronq', '1e-3', block=0)
        add('kronq', '1e-3', '--block 1024 --gnls 1', block=1024, gnls=True)
        for lr in ('1e-3', '3e-3'):
            add('kronq', lr, '--block 256 --compress 1 --mom_bf16 1 --gnls 1', block=256, gnls=True)
            add('kronq', lr, '--block 256 --compress 1 --mom_bf16 1 --gnls 1 --gmask 1e-3', block=256, gnls=True)
            add('kronq', lr, '--block 256 --compress 1 --mom_bf16 1 --rem 0.1', block=256, rem=True)
jobs.sort(key=lambda j: (-j[0], -j[1]))
with open(f'{OUT}/jobs.txt', 'w') as fh:
    for _, _, j in jobs: fh.write(' '.join(j.split()) + '\n')
print(len(jobs), 'jobs ->', f'{OUT}/jobs.txt')
for w in (4096, 2048, 1024):
    print(f'w={w}: steps per method at budget {BUDGET} x F0:', {k: int(BUDGET / cost(*v)) for k, v in
          {'adamw': ('adamw', w), 'muon': ('muon', w), 'soap full': ('soap', w), 'soap b256': ('soap', w, 256), 'kronq b1024 gnls': ('kronq', w, 1024, False, False, True),
           'kronq b256 gnls': ('kronq', w, 256, False, False, True), 'kronq b256 rem': ('kronq', w, 256, False, True)}.items()})
