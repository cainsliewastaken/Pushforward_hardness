"""Markdown table of SegFlow runs vs saved GD runs (reads runs/segflow/*_target_flow_m30_d0.25.npy)."""
import glob, numpy as np, os
rows = []
for f in sorted(glob.glob('../runs/segflow/*_target_flow_m30_d0.25.npy')):
    r = np.load(f, allow_pickle=True).item(); tag = r['tag']; h = r['hist']
    d = np.load(f'../runs/heldout/des_{tag}.npy', allow_pickle=True).item()
    rows.append((tag, d['steps'], d['wall_s'], float(d['E'][-1]), h[-1]['E'], h[-1]['jvp'], h[-1]['segments'], r['wall'], d['steps']/h[-1]['jvp'], r.get('Ev_gd', np.nan), r.get('Ev_sf', np.nan)))
print('| design | GD steps | GD wall (GPU) | GD E(T) | SegFlow E | Lanczos iters (jvp+vjp pairs) | segments | SegFlow wall (CPU, 8 thr) | passes ratio GD/SegFlow | val E GD | val E SegFlow |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| %s | %d | %.0f s | %.4f | %.4f | %d | %d | %.0f s | %.0fx | %.3f | %.3f |' % r)
