"""Run a list of train_ks.py configurations and summarise.  Usage:
  python runlist.py OUT_DIR "COMMON ARGS" tag1 "args1" tag2 "args2" ...
Each run writes OUT_DIR/runs/<tag>.jsonl (skipped if finished) and one summary line to OUT_DIR/summary.jsonl.
"""
import os, sys, json, subprocess
OUT, COMMON = sys.argv[1], sys.argv[2]; items = sys.argv[3:]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, 'runs'), exist_ok=True)
for tag, args in zip(items[::2], items[1::2]):
    path = os.path.join(OUT, 'runs', tag + '.jsonl')
    def read():
        r = {}
        if os.path.exists(path):
            for l in open(path):
                d = json.loads(l); r[d['metric']] = d['value']
        return r
    if 'E_val_final' not in read():
        if os.path.exists(path): os.remove(path)
        subprocess.run(f'python -u {HERE}/train_ks.py {COMMON} {args} --out {path}', shell=True, check=False)
    r = read()
    rec = dict(tag=tag, args=args, E_val=r.get('E_val_final'), E_train=r.get('E_train_final'), flops=r.get('flops_total'),
               wall=r.get('wall_s'), ms_step=r.get('ms_per_step_steady'), periods=r.get('eig_periods'),
               prof={k[17:]: round(v, 3) for k, v in r.items() if k.startswith('prof_ms_per_step_')})
    with open(os.path.join(OUT, 'summary.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print('RESULT', json.dumps(rec), flush=True)
print('LIST DONE', flush=True)
