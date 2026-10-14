"""Summary of the divergence mechanism test (scale_turbfull res_j100+) and the capped-budget fix (scale_capfix): per run a label built
from the distinguishing flags (cap, rem_anchor, slow_adapt / sa_cap / kappa_max, wd, lr, target or field p), the final and best
validation, the validation at a few steps (to see degradation), and the wall."""
import json, glob, os, sys, collections
def arg(args, k): return args.split(f'--{k} ')[-1].split()[0] if f'--{k} ' in args else None
def label(a):
    parts = []
    if '--problem turb' in a: parts.append(f"turb {arg(a, 'turb_target')}" + (' N=100k' if '100k' in (arg(a, 'turb_cache') or '') else '') + (f" seed {arg(a, 'seed')}" if arg(a, 'seed') not in (None, '1') else ''))
    else: parts.append('field ' + ('rough' if arg(a, 'p') == '2' else 'smooth') + f" seed {arg(a, 'seed')}")
    if '--whiten' in a and arg(a, 'whiten') != '0': parts.append(f"whiten {arg(a, 'whiten')}")
    if '--opt adamw' in a: parts.append(f"adamw lr {arg(a, 'lr')}"); return ' | '.join(parts)
    if '--freeze_after' in a and arg(a, 'freeze_after') != '0': parts.append(f"freeze after {arg(a, 'freeze_after')}")
    if '--stats_every' in a and arg(a, 'stats_every') != '1': parts.append(f"stats every {arg(a, 'stats_every')}")
    parts.append(f"lr {arg(a, 'lr')} budget {arg(a, 'rem') or '0'}")
    parts.append('cap on' if arg(a, 'cap') == '1' else 'cap off')
    if '--rem_anchor' in a: parts.append(f"anchor {arg(a, 'rem_anchor')}")
    if '--slow_adapt 1' in a: parts.append(('held-out CAP' if '--sa_cap 1' in a else 'held-out budget') + f" kappa {arg(a, 'sa_kappa_max') or '1.0'}")
    if '--wd' in a and arg(a, 'wd') not in (None, '0.0', '0'): parts.append(f"wd {arg(a, 'wd')}")
    for k in ('width', 'wd_abs', 'wd_gate', 'wd_gate_q0', 'lr_track', 'batch', 'rem_iter', 'clip_grad', 'u_clamp', 'rem_check', 'qr_guard', 'tf32'):
        if f'--{k} ' in a and arg(a, k) not in ('0', '0.0', '4.6', '0.5', '256', '4096', '1'): parts.append(f"{k} {arg(a, k)}")
        if k == 'tf32' and arg(a, k) == '0': parts.append('fp32')
    if '--steps' in a and arg(a, 'steps') not in ('5000', '31240'): parts.append(f"steps {arg(a, 'steps')}")
    if '--u_decay' in a and arg(a, 'u_decay') not in (None, '0.0', '0'): parts.append(f"u_decay {arg(a, 'u_decay')} floor {arg(a, 'u_floor') or '-2.3'}")
    return ' | '.join(parts)
def show(files, title):
    print(f'==== {title}')
    rows_all = []
    for fn in files:
        rows = [json.loads(l) for l in open(fn)]
        if not rows: continue
        a = rows[0]['args']; v = {r['step']: r['value'] for r in rows if r['metric'] == 'val'}
        vf = [r['value'] for r in rows if r['metric'] == 'val_final']; vb = [r['value'] for r in rows if r['metric'] == 'val_best']; w = [r['value'] for r in rows if r['metric'] == 'wall_total']
        steps = sorted(v); marks = [s for s in steps if s in (1000, 2000, 5000, 10000, 20000, 30000)] or steps[-3:]
        traj = ' '.join(f"{s}:{v[s]:.3g}" for s in marks)
        rows_all.append((label(a), vf[0] if vf else float('nan'), vb[0] if vb else float('nan'), w[0] if w else float('nan'), traj))
    for lab, f_, b_, w_, traj in sorted(rows_all):
        print(f"   {lab:70s} final {f_:9.4g}  best {b_:9.4g}  wall {w_:5.0f} s   [{traj}]")
S = os.path.expandvars('$SCRATCH/l63_held_rough/runs')
add = [f for f in glob.glob(f'{S}/scale_turbfull/res_j*.jsonl') if int(os.path.basename(f).split('_j')[1].split('.')[0]) >= 100]
if add: show(sorted(add), 'mechanism test (scale_turbfull, indices 100+)')
fix = glob.glob(f'{S}/scale_capfix/res_j*.jsonl')
if fix: show(sorted(fix), 'capped-budget fix (scale_capfix)')
sn = glob.glob(f'{S}/scale_smallnet/res_j*.jsonl')
if sn: show(sorted(sn), 'small network, small and large batch, no cap or budget, guarded fp32 (scale_smallnet)')
nb = glob.glob(f'{S}/scale_nobudget/res_j*.jsonl')
if nb: show(sorted(nb), 'no budget, no cap: lr + decay only (scale_nobudget)')
gt = glob.glob(f'{S}/scale_gate/res_j*.jsonl')
if gt: show(sorted(gt), 'gated decay and budget-tracking lr on turbulence (scale_gate)')
fs = glob.glob(f'{S}/scale_fieldspike/res_j*.jsonl')
if fs: show(sorted(fs), 'field spikes: decay, gated decay, cap (scale_fieldspike)')
sp = glob.glob(f'{S}/scale_spike/res_j*.jsonl')
if sp: show(sorted(sp), 'spike mechanism: realised output change, iterated rescale, u clamp, gradient clip (scale_spike)')
bg = glob.glob(f'{S}/scale_budget/res_j*.jsonl')
if bg: show(sorted(bg), 'budget with strong decay, no slow adaptation (scale_budget)')
fg = glob.glob(f'{S}/scale_forget/res_j*.jsonl')
if fg: show(sorted(fg), 'forgetting-rate test: decay vs dataset size (scale_forget)')
sm = glob.glob(f'{S}/scale_simple/res_j*.jsonl')
if sm: show(sorted(sm), 'simplifications: whitened AdamW, frozen bases, probe every k (scale_simple)')
ud = glob.glob(f'{S}/scale_udecay/res_j*.jsonl')
if ud: show(sorted(ud), 'weight shrink toward the floor and weight-decay sweep (scale_udecay)')
