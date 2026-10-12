"""Cross-check: same small config through l63test.py (numpy) and l63test_torch.py; compare logs."""
import sys, numpy as np
a = np.load(sys.argv[1], allow_pickle=True).item(); b = np.load(sys.argv[2], allow_pickle=True).item()
print('lam1_0', a['lam1_0'], b['lam1_0'], ' H2_g0', a['H2_g0'], b['H2_g0'], ' H2_rand', a['H2_rand'], b['H2_rand'])
for ia, ib in zip(a['info'], b['info']):
    print(' info', {k: (round(ia[k], 8), round(ib[k], 8)) for k in ('phiE0', 'sigma0', 'Egamma_norm', 'Upsilon', 'cg_iters')})
worst = {}
for la, lb in zip(a['logs'], b['logs']):
    for k in la:
        if k in lb:
            d = abs(la[k]-lb[k])/max(abs(la[k]), 1e-12)
            worst[k] = max(worst.get(k, 0), d)
print('n logs', len(a['logs']), len(b['logs']))
print('max relative diff per logged key:')
for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:12]: print(f'   {k:10s} {v:.2e}')
print('phis max abs diff', np.abs(a['phis']-b['phis']).max())
