"""Write the scaling-study job list (runs/scale/jobs.txt), longest runs first (width, then optimizer cost)."""
import os
OUT = os.path.expandvars('$SCRATCH/l63_held_rough/runs/scale')
jobs = []
for w in (4096, 2048, 1024, 512, 256):
    for p in ('0', '2'):
        base = f'--width {w} --depth 2 --p {p} --steps 5000 --batch 256 --refresh 20 --eval_every 250'
        kq = [('', 'full'), ('--block 1024', 'b1024'), ('--block 256', 'b256'), ('--block 64', 'b64'),
              ('--block 256 --compress 1 --mom_bf16 1', 'b256c'), ('--max_dim 1024', 'diag1024')]
        for lr in ('1e-3', '3e-3'):
            for flag, tag in kq:
                jobs.append((w, 3, f'--opt kronq --lr {lr} {flag} {base}'))
        for lr in ('1e-3', '3e-3'):
            for flag in ('', '--block 256'):
                jobs.append((w, 2, f'--opt soap --lr {lr} {flag} {base}'))
        for lr in ('5e-3', '2e-2', '5e-2'):
            jobs.append((w, 1, f'--opt muon --lr {lr} --lr_adam 1e-3 {base}'))
        for lr in ('3e-4', '1e-3', '3e-3'):
            jobs.append((w, 0, f'--opt adamw --lr {lr} {base}'))
jobs.sort(key=lambda j: (-j[0], -j[1]))
with open(f'{OUT}/jobs.txt', 'w') as fh:
    for _, _, j in jobs: fh.write(' '.join(j.split()) + '\n')
print(len(jobs), 'jobs ->', f'{OUT}/jobs.txt')
