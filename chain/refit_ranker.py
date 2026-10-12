"""Refit/evaluate the init-only ranker on the combined feature tables. usage: python refit_ranker.py base2.csv ext2.csv"""
import numpy as np, csv, itertools, re, sys
from scipy.stats import spearmanr
def load(fn):
    rows=[x for x in csv.DictReader(open(fn)) if not re.search(r'_b\d+$|_momentum',x['tag'])]
    for x in rows:
        L=int(x['depth'])+1; f=lambda k: float(x[k]); x['H2slow_hidden']=sum(f(f'H2_slow_l{l}') for l in range(1,L-1)) if L>2 else f('H2_slow_l0')
        x['sc_']=f('sc'); x['y']=np.log(f('ET')); x['lam1T']=f('lam1')*f('T'); x['H2n']=f('H2_desc')/x['sc_']; x['H2hn']=x['H2slow_hidden']/x['sc_']; x['gn']=f('gnorm_desc')/x['sc_']
    return rows
base=load(sys.argv[1]); ext=load(sys.argv[2]); seen={x['tag'] for x in base}; ext=[x for x in ext if x['tag'] not in seen]
val=lambda x,k: x[k] if isinstance(x.get(k),float) else float(x[k])
def X_(data,cols): return np.column_stack([np.ones(len(data))]+[[np.log(val(x,c)) for x in data] for c in cols])
def report(pred,y,label):
    pairs=list(itertools.combinations(range(len(y)),2)); gaps=np.array([abs(y[i]-y[j]) for i,j in pairs]); ok=np.array([np.sign(y[i]-y[j])==np.sign(pred[i]-pred[j]) for i,j in pairs])
    bins=" ".join(f"{100*ok[(gaps>np.log(lo))&(gaps<=np.log(hi))].mean():.0f}%({((gaps>np.log(lo))&(gaps<=np.log(hi))).sum()})" if ((gaps>np.log(lo))&(gaps<=np.log(hi))).sum() else "-" for lo,hi in [(1,1.1),(1.1,1.3),(1.3,2),(2,1e9)])
    print(f"{label:58s} n={len(y)} Spearman {spearmanr(pred,y).correlation:.3f} med {np.exp(np.median(np.abs(pred-y))):.2f} p90 {np.exp(np.percentile(np.abs(pred-y),90)):.2f} max {np.exp(np.max(np.abs(pred-y))):.2f} pairs {bins}")
def loo(cols,data,label):
    y=np.array([x['y'] for x in data]); X=X_(data,cols); pred=np.zeros(len(y))
    for i in range(len(y)):
        m=np.ones(len(y),bool); m[i]=False; pred[i]=X[i]@np.linalg.lstsq(X[m],y[m],rcond=None)[0]
    report(pred,y,label+" [LOO]")
def transfer(cols,train,test,label):
    b=np.linalg.lstsq(X_(train,cols),np.array([x['y'] for x in train]),rcond=None)[0]; yt=np.array([x['y'] for x in test]); pt=X_(test,cols)@b; report(pt,yt,label+" [fit->test]")
    for x,p in zip(test,pt): print(f"      {x['tag']:28s} meas {np.exp(x['y']):.4f} pred {np.exp(p):.4f} ratio {np.exp(p-x['y']):.2f}")
b1=[x for x in base if x['sc_']==1.0]; e1=[x for x in ext if x['sc_']==1.0]; bo=[x for x in base if x['sc_']!=1.0]; eo=[x for x in ext if x['sc_']!=1.0]
print(f"scale-1: {len(b1)} old + {len(e1)} new (deep/wide);  other scales: {len(bo)} old + {len(eo)} new (rich, s=0.5)")
for cols in [['H2slow_hidden','Elazy','lam1'],['H2_desc','Elazy','lam1']]:
    print("==", '+'.join(cols)); transfer(cols,b1,e1,"  old 53 -> new deep/wide (out-of-sample)"); loo(cols,b1+e1,"  all scale-1 combined")
print("== rich regime: fit on all scale-1, predict all other-scale runs with normalised features")
for cols in [['H2_desc','Elazy','lam1'],['H2n','Elazy','lam1T'],['H2hn','Elazy','lam1T'],['H2n','Elazy','lam1T','gn']]: transfer(cols,b1+e1,bo+eo,"  "+'+'.join(cols))
print("== joint fit with an explicit scale term, LOO over everything")
for x in base+ext: x['scf']=x['sc_']
for cols in [['H2_desc','Elazy','lam1','scf'],['H2n','Elazy','lam1T','scf'],['H2hn','Elazy','lam1T','scf']]: loo(cols,base+ext,"  "+'+'.join(cols))
