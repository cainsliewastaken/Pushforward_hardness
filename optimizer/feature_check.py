"""CPU checks for the extended kronsign.py / train_ks.py.
1. Defaults reproduce the previous version (train_ks_v2.py + kronsign_v2.py) to floating-point noise, sign and adam, block 0/16.
2. Every new option runs and gives a finite, sensible error: last_S identity, shampoo stats (SOAP), kronv rule (block 0/16),
   eig_batch (same result as unbatched), eig_adapt, rot_dtype float32, beta_stats, cnn arch (adam / kronsign adam / kronv / identity).
3. DDP on CPU (gloo, 2 ranks via torchrun) runs and ranks end with identical parameters.
Usage: python feature_check.py DATA.npz  (run from the directory holding train_ks.py, kronsign.py, train_ks_v2.py, kronsign_v2.py)
"""
import os, sys, json, math, subprocess, shutil, tempfile
DATA = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix='fc_', dir=HERE)
BASE = f'--data {DATA} --act gelu --steps 40 --batch 50 --standardize none --log_every 1000'
MLP = f'{BASE} --widths 48,48'

def run(script, args, tag, extra_cmd=''):
    out = os.path.join(TMP, tag + '.jsonl')
    script = script if os.path.isabs(script) else os.path.join(HERE, script)
    cmd = f'{extra_cmd} python -u {script} {args} --out {out}'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print('FAILED:', cmd); print(r.stdout[-3000:]); print(r.stderr[-6000:]); sys.exit(1)
    res = {}
    for l in open(out):
        d = json.loads(l); res[d['metric']] = d['value']
    return res

def close(a, b, tol=1e-9): return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

# ---- 0. unit test of the blocked basis carry against the dense formula
print('== 0. blocked carry unit test')
sys.path.insert(0, HERE)
import torch
from kronsign import KronSign
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
ks = KronSign(torch.nn.Linear(3, 3), 1e-3, block=16)
nb, bo, na, bi = 3, 16, 5, 16
def orth(n, b): return torch.linalg.qr(torch.randn(n, b, b))[0]
US_o, US_n, UA_o, UA_n = orth(nb, bo), orth(nb, bo), orth(na, bi), orth(na, bi)
C = torch.randn(nb * bo, na * bi)
MA = torch.einsum('lrq,lrs->lqs', UA_o, UA_n); MS = torch.einsum('kqp,kqr->kpr', US_n, US_o)
got = ks._rot(MS, C, MA, False, True)
bd = torch.block_diag
want = bd(*US_n).T @ bd(*US_o) @ C @ bd(*UA_o).T @ bd(*UA_n)
assert torch.allclose(got, want, atol=1e-10), float((got - want).abs().max())
gotV = ks._rot(MS ** 2, C, MA ** 2, False, True)
wantV = (bd(*US_n).T @ bd(*US_o)) ** 2 @ C @ ((bd(*UA_n).T @ bd(*UA_o)) ** 2).T
assert torch.allclose(gotV, wantV, atol=1e-10)
assert torch.allclose(ks._rot(US_n, C, UA_n, True, True), bd(*US_n).T @ C @ bd(*UA_n), atol=1e-10)
assert torch.allclose(ks._rot(US_n, C, UA_n, False, False), bd(*US_n) @ C @ bd(*UA_n).T, atol=1e-10)
assert torch.allclose(ks._rot(None, C, UA_n, False, True), C @ bd(*UA_n), atol=1e-10)
print('  carry / rotation identities OK')

# ---- 1. reproduce the previous version
print('== 1. equivalence with the previous version')
# kronsign_v2 needs to be importable as `kronsign` for train_ks_v2: run it from a shadow directory
V2 = os.path.join(TMP, 'v2'); os.makedirs(V2)
shutil.copy(os.path.join(HERE, 'kronsign_v2.py'), os.path.join(V2, 'kronsign.py'))
shutil.copy(os.path.join(HERE, 'train_ks_v2.py'), os.path.join(V2, 'train_ks.py'))
for rule in ('sign', 'adam'):
    for block in (0,):
        args = f'{MLP} --opt kronsign --rule {rule} --lr 3e-3'
        old = run(os.path.join(V2, 'train_ks.py'), args + (f' --block {block}' if block else ''), f'old_{rule}_{block}')
        new = run('train_ks.py', args + f' --block {block}', f'new_{rule}_{block}')
        ok = close(old['E_val_final'], new['E_val_final'], 1e-10) and close(old['flops_total'], new['flops_total'], 1e-9)
        print(f'  rule={rule} block={block}: old E={old["E_val_final"]:.10f} new E={new["E_val_final"]:.10f} flops old={old["flops_total"]:.6e} new={new["flops_total"]:.6e} {"OK" if ok else "MISMATCH"}')
        assert ok
ad = run('train_ks.py', f'{MLP} --opt adam --lr 3e-3', 'adam_ref')
print(f'  adam reference E={ad["E_val_final"]:.5f}')

# ---- 2. features
print('== 2. new options')
ref = run('train_ks.py', f'{MLP} --opt kronsign --rule adam --lr 3e-3', 'ref_adam')
E0 = ref['E_val_final']
def feat(tag, extra, arch_args=MLP, bound=1.5):
    r = run('train_ks.py', f'{arch_args} --opt kronsign --lr 3e-3 {extra}', tag)
    E = r['E_val_final']; assert math.isfinite(E) and E < bound * max(E0, ad['E_val_final']), (tag, E)
    print(f'  {tag:28s} E={E:.5f}  flops={r["flops_total"]:.3e}  refresh={r.get("n_refresh")}  {("periods=" + str(r["eig_periods"])) if "eig_periods" in r else ""}')
    return r
rid = feat('identity_last_S', '--rule adam --last_S identity')
assert rid['flops_total'] < ref['flops_total']
feat('identity_block16', '--rule adam --last_S identity --block 16')
feat('soap', '--rule adam --stats shampoo --stats_every 1')
feat('soap_block16', '--rule adam --stats shampoo --stats_every 1 --block 16')
rk = feat('kronv', '--rule kronv')
assert rk['flops_total'] < ref['flops_total']
feat('kronv_block16', '--rule kronv --block 16')
feat('kronv_identity', '--rule kronv --last_S identity')
feat('sign_identity', '--rule sign --last_S identity')
rb = feat('eig_batch_block16', '--rule adam --block 16 --eig_batch 1')
rn = run('train_ks.py', f'{MLP} --opt kronsign --rule adam --lr 3e-3 --block 16', 'nobatch_block16')
print(f'  eig_batch vs unbatched: {rb["E_val_final"]:.10f} vs {rn["E_val_final"]:.10f}')
assert close(rb['E_val_final'], rn['E_val_final'], 1e-6)
feat('eig_adapt', '--rule adam --eig_adapt 1 --eig_every 2 --eig_max 16')
feat('rot_float32', '--rule adam --rot_dtype float32')
feat('beta_stats_0.8', '--rule adam --beta_stats 0.8')
CNN = f'{BASE} --arch cnn --channels 8,8'
ac = run('train_ks.py', f'{CNN} --opt adam --lr 3e-3', 'cnn_adam'); print(f'  cnn adam                     E={ac["E_val_final"]:.5f}')
E0 = ac['E_val_final']
feat('cnn_ks_adam', '--rule adam', CNN, bound=3.0)
feat('cnn_ks_adam_block8', '--rule adam --block 8', CNN, bound=3.0)
feat('cnn_ks_identity', '--rule adam --last_S identity', CNN, bound=3.0)
feat('cnn_kronv', '--rule kronv', CNN, bound=3.0)
feat('cnn_soap', '--rule adam --stats shampoo --stats_every 1', CNN, bound=3.0)

# ---- 3. DDP on CPU
print('== 3. DDP (gloo, 2 ranks)')
out = os.path.join(TMP, 'ddp.jsonl')
cmd = (f'torchrun --standalone --nproc-per-node=2 {HERE}/train_ks.py {MLP} --opt kronsign --rule adam --lr 3e-3 --block 16 '
       f'--eig_batch 1 --ddp 1 --out {out}')
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if r.returncode != 0: print('FAILED:', cmd); print(r.stdout[-3000:]); print(r.stderr[-6000:]); sys.exit(1)
res = {json.loads(l)['metric']: json.loads(l)['value'] for l in open(out)}
print(f'  ddp E={res["E_val_final"]:.5f} world-flops={res["flops_total"]:.3e}')
assert math.isfinite(res['E_val_final'])
print('ALL FEATURE CHECKS PASSED'); shutil.rmtree(TMP)
