"""Critical batch size on the field (fresh data). Writes the job list with an estimated cost per job and assigns jobs to NSLOT GPU slots by
longest-processing-time first; each slot then runs its jobs one at a time (one run per GPU: the wall times stay isolated).
usage: python crit_plan.py OUTDIR NSLOT [--dry]"""
import sys, os
F, NSLOT = sys.argv[1], int(sys.argv[2])
base = "--width 4096 --depth 2 --tf32 1 --seed 1 --nval 4096"
K = "--opt kronqf --block 256 --compress 1 --mom_bf16 1 --stats_every 1 --eig_iters 2 --refresh 20 --adapt_refresh 0.1 --refresh_min 5 --reltol 1e-12 --hyper 1 --fused 1 --cap 0"
# KronQ settings per batch from the batch law (items 44, 46): rough lr 1e-3 with budget rising, smooth budget 0.3 with lr rising; a second
# value where the optimum is unmeasured (2048 edge, 4096 extrapolated)
KR = {256: [(1e-3, 0.15), (1e-3, 0.2)], 512: [(1e-3, 0.2)], 1024: [(1e-3, 0.3)], 2048: [(1e-3, 0.4), (1e-3, 0.8)], 4096: [(1e-3, 0.8), (1e-3, 1.6)]}
KS = {256: [(1e-3, 0.3)], 512: [(2e-3, 0.3)], 1024: [(2e-3, 0.3)], 2048: [(5.6e-3, 0.3), (1.1e-2, 0.3)], 4096: [(1.1e-2, 0.3), (2.2e-2, 0.3)]}
AL = {256: [1e-4, 3e-4, 1e-3], 512: [1e-4, 3e-4, 1e-3], 1024: [3e-4, 1e-3, 3e-3], 2048: [3e-4, 1e-3, 3e-3], 4096: [3e-4, 1e-3, 3e-3]}
TK = {256: [1250, 2500, 5000, 10000, 20000], 512: [1250, 2500, 5000, 10000, 20000], 1024: [625, 1250, 2500, 5000, 10000, 20000], 2048: [625, 1250, 2500, 5000, 10000], 4096: [625, 1250, 2500, 5000]}
TA = {256: [2500, 5000, 10000, 20000, 40000], 512: [2500, 5000, 10000, 20000, 40000], 1024: [1250, 2500, 5000, 10000, 20000], 2048: [1250, 2500, 5000, 10000, 20000], 4096: [1250, 2500, 5000, 10000, 20000]}
ms_k = lambda B: 7.5 + 2.5 * B / 1024; ms_a = lambda B: 1.5 + 2.0 * B / 1024        # per-step cost model (ms), from items 40-46 walls
jobs = []
for p in (2, 0):
    for B, sets in (KR if p == 2 else KS).items():
        for lr, rem in sets:
            for T in TK[B]:
                ev = 125 if T <= 2500 else 250
                jobs.append((30 + T * ms_k(B) / 1000, f"{base} {K} --p {p} --batch {B} --steps {T} --eval_every {ev} --lr {lr} --rem {rem}"))
    for B, lrs in AL.items():
        for lr in lrs:
            for T in TA[B]:
                ev = 125 if T <= 2500 else 250
                jobs.append((15 + T * ms_a(B) / 1000, f"{base} --opt adamw --p {p} --batch {B} --steps {T} --eval_every {ev} --lr {lr}"))
order = sorted(range(len(jobs)), key=lambda i: -jobs[i][0]); load = [0.0] * NSLOT; slots = [[] for _ in range(NSLOT)]
for i in order:
    s = min(range(NSLOT), key=lambda k: load[k]); slots[s].append(i); load[s] += jobs[i][0]
print(f"{len(jobs)} jobs, {sum(j[0] for j in jobs) / 60:.0f} GPU-min estimated; {NSLOT} slots, makespan {max(load) / 60:.1f} min (min slot {min(load) / 60:.1f})")
if '--dry' not in sys.argv:
    os.makedirs(F, exist_ok=True)
    for s in range(NSLOT):
        with open(f"{F}/slot_{s}.txt", 'w') as fh:
            for i in slots[s]: fh.write(f"{i} {jobs[i][1]}\n")
    open(f"{F}/njobs", 'w').write(f"{len(jobs)} jobs\n")
