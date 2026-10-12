"""Check of the multistep note (multistep_v3.tex) on the 1D KS FNO models of modular_training_stack:
multistep_fno1d_ks1d_w256_m256_direct (G~ = N) and _euler (G~ = u + dt H), both dt = 1e-3, trained on
the first 1.2M steps of KS_1024_full.h5; analysis on held-out states after them. No training.

For each model and each of S start states (normalized units, as in training), along the TRUE
trajectory u_T(t_i), i < P:
  eps_i  = u_T(t_{i+1}) - G~[u_T(t_i)]                 injected (generalization) error
  a_i    = ||grad G~(t_i) - I||_2                       power iteration on JVP/VJP
  drift_i = ||eps_i - eps_0||                           decorrelation along the flow
and along the ROLLOUT from the same start: e(t_q), and the exact remainders
  r_i = e(t_{i+1}) - eps_i - grad G~(t_i) e(t_i).
Theorem 1 (first-order parts with the measured remainders in place of the L-bound), at p in PS:
  upper  = e^{A_p} * sum_{i<p}(||eps_i|| + ||r_i||),                A_p = sum_{1<=j<p} a_j
  lower  = p||eps_0|| - sum_i drift_i - (e^{A_p}-1) sum_i ||eps_i|| - e^{A_p} sum_i ||r_i||
  measured ||e(t_p)||.
Rates: sigma_hat = mean(a_i)/dt; full Jacobians at JAC_STATES states give |lambda_max| and
sigma = ln|lambda_max|/dt (eigenanalysis paper) and ||grad G~ - I||_2 exactly (checks power iteration).
Output: <out>/ks_check_rows.csv, ks_check_summary.json."""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.func import jvp, vjp

ROOT = os.path.expanduser("~/modular_training_stack")
sys.path.insert(0, ROOT)
from stack.plugins.loader import instantiate_plugin  # noqa: E402

CK = os.path.join(os.environ.get("SCRATCH", ""), "model_chkpts_scratch")
H5 = os.path.join(os.environ.get("SCRATCH", ""), "training_data/1d_KS/KS_1024_full.h5")
STATS = os.path.join(os.environ.get("SCRATCH", ""), "training_data/1d_KS/KS_1024_full_stats.npz")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def build(name, dev):
    ck = torch.load(os.path.join(CK, name, f"chkpt_{name}_best_chkpt.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mp, mpar = instantiate_plugin(cfg["model"], section="model")
    from types import SimpleNamespace
    ctx = SimpleNamespace(device=torch.device(dev), use_distributed=False, local_rank=0, current_rank=0,
                          world_size=1, root_dir=ROOT)
    model = mp.build(params=mpar, config=cfg, context=ctx)
    sp, spar = instantiate_plugin(cfg["step_method"], section="step_method")
    step = sp.build(model=model, data=None, params=spar, config=cfg, context=ctx)
    step.load_state_dict(ck["model_state_dict"], strict=True)      # saved from the step module ("network." prefix)
    step = step.to(dev).eval()
    for p in step.parameters():
        p.requires_grad_(False)
    return step, cfg, ck["epoch"]


def make_G(step):
    """G~ on a batch of states u (B, L) -> (B, L), via the stack's step module (input (B, 1, L))."""
    def G(u):
        return step(u.unsqueeze(1)).squeeze(1)
    return G


def opnorm_minus_I(G, u, iters, gen):
    """||grad G~(u) - I||_2 for each state in the batch, by power iteration on (J - I)^T (J - I)."""
    B, L = u.shape
    v = torch.randn(B, L, generator=gen, device=u.device, dtype=u.dtype)
    v = v / v.norm(dim=1, keepdim=True)
    _, vjp_fn = vjp(G, u)
    s = torch.zeros(B, device=u.device, dtype=u.dtype)
    for _ in range(iters):
        Jv = jvp(G, (u,), (v,))[1] - v
        w = vjp_fn(Jv)[0] - Jv
        s = w.norm(dim=1).sqrt()                      # sqrt of Rayleigh-type growth -> sigma_max
        v = w / w.norm(dim=1, keepdim=True).clamp(min=1e-30)
    Jv = jvp(G, (u,), (v,))[1] - v
    return Jv.norm(dim=1)                             # ||(J - I) v|| at the converged v


def full_jacobian(G, u1, chunk=128):
    """Full (L, L) Jacobian of G~ at one state u1 (L,)."""
    L = u1.shape[0]
    _, vjp_fn = vjp(lambda x: G(x[None])[0], u1)
    eye = torch.eye(L, device=u1.device, dtype=u1.dtype)
    rows = [torch.func.vmap(lambda c: vjp_fn(c)[0])(eye[i:i + chunk]) for i in range(0, L, chunk)]
    return torch.cat(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=16)
    ap.add_argument("--P", type=int, default=1000)
    ap.add_argument("--jac-every", type=int, default=1, help="compute a_i every k steps (a_i held constant in between)")
    ap.add_argument("--power-iters", type=int, default=30)
    ap.add_argument("--jac-states", type=int, default=4)
    ap.add_argument("--ps", default="1,2,5,10,20,50,100,200,500,1000")
    ap.add_argument("--models", default="multistep_fno1d_ks1d_w256_m256_direct,multistep_fno1d_ks1d_w256_m256_euler")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    log("device", dev)
    st = np.load(STATS)
    mean, std = float(st["mean"]), float(st["std"])
    import h5py
    f = h5py.File(H5, "r")
    ucol = f["u"]
    n_train, n_test = 1_200_000, 400_000
    P = a.P
    starts = np.linspace(n_train + 10_000, n_train + n_test - P - 10, a.starts).astype(int)
    log(f"reading {len(starts)} windows of {P + 1} states from {H5}")
    traj = np.stack([np.asarray(ucol[:, s:s + P + 1]).T for s in starts])      # (S, P+1, L)
    traj = torch.tensor((traj - mean) / std, dtype=dtype, device=dev)
    S, _, L = traj.shape
    dt = 1e-3
    ps = [int(x) for x in a.ps.split(",") if int(x) <= P]
    summary, rows = {}, []
    gen = torch.Generator(device=dev).manual_seed(0)
    for name in a.models.split(","):
        step, cfg, epoch = build(name, dev)
        G = make_G(step)
        kind = cfg["step_method"]["params"]["name"]
        log(f"model {name} (step {kind}, epoch {epoch})")
        # along the true trajectory
        with torch.no_grad():
            eps = torch.stack([traj[:, i + 1] - G(traj[:, i]) for i in range(P)], 1)        # (S, P, L)
        eps_n = eps.norm(dim=2)
        drift = (eps - eps[:, :1]).norm(dim=2)
        a_i = torch.zeros(S, P, device=dev)
        for i in range(0, P, a.jac_every):
            a_i[:, i:i + a.jac_every] = opnorm_minus_I(G, traj[:, i], a.power_iters, gen)[:, None]
        log(f"  true-trajectory quantities done; mean ||eps|| {eps_n.mean():.3e}, mean a_i {a_i.mean():.3e}")
        # rollout and exact remainders
        e = torch.zeros(S, P + 1, L, device=dev, dtype=dtype)
        r_n = torch.zeros(S, P, device=dev)
        up = traj[:, 0].clone()
        for i in range(P):
            with torch.no_grad():
                up_next = G(up)
            e[:, i + 1] = traj[:, i + 1] - up_next
            Je = jvp(G, (traj[:, i],), (e[:, i],))[1]
            r_n[:, i] = (e[:, i + 1] - eps[:, i] - Je).norm(dim=1)
            up = up_next
        e_n = e.norm(dim=2)
        log(f"  rollout done; ||e(t_P)|| median {e_n[:, -1].median():.3e}")
        # Theorem 1 at each p
        for p in ps:
            A = a_i[:, 1:p].sum(1)
            eA = torch.exp(A)
            se, sr, sd = eps_n[:, :p].sum(1), r_n[:, :p].sum(1), drift[:, :p].sum(1)
            upper = eA * (se + sr)
            lower = p * eps_n[:, 0] - sd - (eA - 1) * se - eA * sr
            meas = e_n[:, p]
            for s in range(S):
                rows.append(dict(model=name, kind=kind, start=int(starts[s]), p=p, measured=float(meas[s]),
                                 upper=float(upper[s]), lower=float(lower[s]), linear=float(p * eps_n[s, 0]),
                                 A_p=float(A[s]), mean_eps=float(eps_n[s, :p].mean()), drift_sum=float(sd[s]),
                                 rem_sum=float(sr[s]), mean_a=float(a_i[s, :p].mean())))
        # full Jacobians at a few states
        lam, opex = [], []
        for s in range(min(a.jac_states, S)):
            Jf = full_jacobian(G, traj[s, 0]).double()
            ev = torch.linalg.eigvals(Jf)
            lam.append(float(ev.abs().max()))
            opex.append(float(torch.linalg.matrix_norm(Jf - torch.eye(L, device=dev, dtype=Jf.dtype), ord=2)))
        pi_at = a_i[: len(opex), 0].tolist()
        summary[name] = dict(kind=kind, epoch=epoch, dt=dt, mean_eps=float(eps_n.mean()), mean_a=float(a_i.mean()),
                             sigma_hat=float(a_i.mean() / dt), lambda_max=lam,
                             sigma_eig=[float(np.log(x) / dt) for x in lam],
                             opnorm_exact=opex, opnorm_power_iter=pi_at,
                             e_P_median=float(e_n[:, -1].median()))
        log(f"  {kind}: mean||eps|| {summary[name]['mean_eps']:.3e}  mean a {summary[name]['mean_a']:.3e}  "
            f"sigma_hat {summary[name]['sigma_hat']:.3g}  |lambda_max| {np.round(lam, 5)}  "
            f"sigma_eig {np.round(summary[name]['sigma_eig'], 3)}  ||J-I|| exact {np.round(opex, 5)} vs power {np.round(pi_at, 5)}")
        import csv
        with open(os.path.join(a.out, "ks_check_rows.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        json.dump(summary, open(os.path.join(a.out, "ks_check_summary.json"), "w"), indent=1)
        del step, G
    # print the theorem table
    for name in summary:
        log(f"== {name}: p | median measured | median lower | median upper | linear p||eps0|| | frac lower<=meas<=upper")
        for p in ps:
            q = [r for r in rows if r["model"] == name and r["p"] == p]
            med = lambda k: np.median([r[k] for r in q])
            ok = np.mean([r["lower"] <= r["measured"] * (1 + 1e-6) and r["measured"] <= r["upper"] * (1 + 1e-6) for r in q])
            log(f"   {p:5d} | {med('measured'):.3e} | {med('lower'):.3e} | {med('upper'):.3e} | {med('linear'):.3e} | {ok:.2f}")
    log("ok")


if __name__ == "__main__":
    main()
