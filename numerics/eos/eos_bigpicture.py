"""Two tests of what the roughness theory adds: python eos/eos_bigpicture.py <run dir> [...] --device cuda.

Test 1 (difficulty predicts sharpening). For each network, from the kernel K0 at initialization (snapshot 0), the
paper's measures of the target Y: the slow part ||Y_gamma||, the squared step length Upsilon_Y(gamma), and
gamma*Upsilon_Y(gamma) (content of Y in the band lambda ~ gamma), at gamma = 1/(eta*T) (T = training steps) and
gamma = 1e-3 lambda_1(0). Against the sharpening: max lambda_1 / lambda_1(0) over training, and whether
eta*lambda_1 reached 1.9. Spearman rank correlations, per step size and pooled.

Test 2 (the edge redirects kernel growth). The two step sizes differ by a factor 2 in eta for the same
initialization, so the larger step at step s and the smaller at step 2s have the same eta*t. At such pairs it
compares lambda_1, lambda_10, lambda_50 (relative to their initial values), lambda_10/lambda_1, lambda_50/lambda_1, and
the error. Output: printed tables; <run dir>/bigpicture_t1.csv, bigpicture_t2.csv."""
import argparse
import csv
import json
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import core as C
import eos_sine as ES
from eos_decompose import jac_scaled

torch.set_default_dtype(torch.float64)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_default_device(a.device)
    t1, t2 = [], []
    for d in a.runs:
        meta = json.load(open(os.path.join(d, "eos_meta.json")))
        ES.set_arch(meta.get("arch", "tanh"))
        runs, eta = [tuple(r) for r in meta["runs"]], np.array(meta["eta"])
        T = meta["steps"]
        spec = np.load(os.path.join(d, "eos_spectra.npz"))
        steps, lamall, Eall = spec["steps"], spec["lam"], spec["E"]
        X = torch.tensor(spec["x"])[:, :, None]
        Yall = spec["y"]
        B, n = Yall.shape
        z = np.load(os.path.join(d, "snaps", "params_0000000.npz"))
        P = [torch.tensor(z[f"p{i}"]) for i in range(len(z.files))]
        for b, (sd, tg, lr) in enumerate(runs):
            J = jac_scaled([p[b:b + 1] for p in P], X[b:b + 1]).cpu().numpy()
            lam, U = np.linalg.eigh(J @ J.T / n)
            lam = np.maximum(lam, 0.0)
            yk = U.T @ Yall[b] / np.sqrt(n)
            l1 = lam[-1]
            r = dict(seed=sd, target=tg, omega=float(tg[3:]), lr_rel=lr, lam1_0=l1)
            for tag, g in (("T", 1.0 / (eta[b] * T)), ("1e-3", 1e-3 * l1)):
                r[f"slow_{tag}"] = float(np.sqrt(np.sum((g / (lam + g)) ** 2 * yk ** 2)))
                r[f"ups_{tag}"] = float(np.sum(lam * yk ** 2 / (lam + g) ** 2))
                r[f"gups_{tag}"] = g * r[f"ups_{tag}"]
            l1t = lamall[:, b, 0]
            r["max_growth"] = float(l1t.max() / l1t[0])
            r["reached_edge"] = bool(eta[b] * l1t.max() >= 1.9)
            r["final_err"] = float(np.linalg.norm(Eall[-1, b]) / np.sqrt(n))
            t1.append(r)
        # test 2: pair lr 1.0 at step s with lr 0.5 at step 2s, same seed and target
        idx = {(runs[b][0], runs[b][1], float(runs[b][2])): b for b in range(B)}
        sidx = {int(s): i for i, s in enumerate(steps)}
        for (sd, tg, lr), b in idx.items():
            if lr != 1.0 or (sd, tg, 0.5) not in idx:
                continue
            b2 = idx[(sd, tg, 0.5)]
            for s in range(1000, T // 2 + 1, 1000):
                if s not in sidx or 2 * s not in sidx:
                    continue
                L1, L2 = lamall[sidx[s], b], lamall[sidx[2 * s], b2]
                L10, L20 = lamall[0, b], lamall[0, b2]
                t2.append(dict(seed=sd, target=tg, eta_t=eta[b] * s,
                               big_lam1=L1[0] / L10[0], small_lam1=L2[0] / L20[0],
                               big_lam10=L1[9] / L10[9], small_lam10=L2[9] / L20[9],
                               big_lam50=L1[49] / L10[49], small_lam50=L2[49] / L20[49],
                               big_r10=L1[9] / L1[0], small_r10=L2[9] / L2[0],
                               big_r50=L1[49] / L1[0], small_r50=L2[49] / L2[0],
                               big_err=np.linalg.norm(Eall[sidx[s], b]) / np.sqrt(n),
                               small_err=np.linalg.norm(Eall[sidx[2 * s], b2]) / np.sqrt(n),
                               big_edge=bool(eta[b] * L1[0] >= 1.9)))
    out = a.runs[-1]
    C.write_rows(os.path.join(out, "bigpicture_t1.csv"), t1)
    C.write_rows(os.path.join(out, "bigpicture_t2.csv"), t2)

    print("TEST 1: difficulty under the initial kernel vs sharpening (Spearman rank correlation with max lambda_1/lambda_1(0))")
    for lr in (0.5, 1.0):
        rr = [r for r in t1 if float(r["lr_rel"]) == lr]
        g = np.array([r["max_growth"] for r in rr])
        print(f"  lr_rel {lr} (n={len(rr)}): " + ", ".join(
            f"{k} {spearman(np.array([r[k] for r in rr]), g):+.2f}" for k in
            ("slow_T", "ups_T", "gups_T", "slow_1e-3", "ups_1e-3", "gups_1e-3", "omega")))
    print("  per target (median over inits): omega, ||Y_gamma||(gamma=1/eta T, lr 1.0), max growth lr 0.5 / lr 1.0, reached edge lr 1.0, final err lr 1.0")
    for tg in dict.fromkeys(r["target"] for r in t1):
        r5 = [r for r in t1 if r["target"] == tg and float(r["lr_rel"]) == 0.5]
        r1 = [r for r in t1 if r["target"] == tg and float(r["lr_rel"]) == 1.0]
        print(f"    {tg:6s} slow {np.median([r['slow_T'] for r in r1]):.3f}  gUps {np.median([r['gups_T'] for r in r1]):.3e}  "
              f"growth {np.median([r['max_growth'] for r in r5]):.2f} / {np.median([r['max_growth'] for r in r1]):.2f}  "
              f"edge {sum(r['reached_edge'] for r in r1)}/{len(r1)}  err {np.median([r['final_err'] for r in r1]):.3f}")
    print("\nTEST 2: at equal eta*t, larger step (big) vs smaller step (small); medians over pairs")
    for tg in dict.fromkeys(r["target"] for r in t2):
        rr = [r for r in t2 if r["target"] == tg]
        m = lambda k: np.median([r[k] for r in rr])
        print(f"  {tg:6s} (n={len(rr)}, big at edge {np.mean([r['big_edge'] for r in rr]):.0%}): lam1 x{m('big_lam1'):.2f} vs x{m('small_lam1'):.2f} | "
              f"lam10 x{m('big_lam10'):.1e} vs x{m('small_lam10'):.1e} | lam50 x{m('big_lam50'):.1e} vs x{m('small_lam50'):.1e} | "
              f"lam50/lam1 {m('big_r50'):.1e} vs {m('small_r50'):.1e} | err {m('big_err'):.3f} vs {m('small_err'):.3f}")
    os._exit(0)


if __name__ == "__main__":
    main()
