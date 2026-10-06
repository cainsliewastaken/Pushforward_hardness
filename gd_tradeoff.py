"""Horizon trade-off along an actual gradient-descent run (note, Lemma tradeoff).

One Euler model (width W, seed s), fine-tuned from the line-rule 1.28 model (dt0.08_unroll128_ft_s4k_linec025) on the
window loss L_w (w_q = 1/p) by plain gradient descent on minibatches: theta <- theta - alpha_k g_k, with g_k the
minibatch gradient and alpha_k = c g.g / g^T G g (quadratic model along g, G the Gauss-Newton matrix of the batch loss).
The direction is the gradient, so the lemma applies with the time change dt = alpha_k (preconditioner Pi = I).

Every `meas_every` steps, on held-out starts: V_T' = (1/2) mean_k |e_k(T')|^2 for each evaluation horizon T', the median,
90th and 99th percentile of |e_k(T')|, the exact gradient of V_T' and its first-order part (-Phi^T y_T'). Between
measurements the predicted change of V_T' is accumulated step by step:
   exact:  -sum_k alpha_k <grad V_T', g_k>        first order:  -sum_k alpha_k <grad^(1) V_T', g_k>
(held-out gradients held at the last measurement), and compared with the measured change at the next measurement."""
import argparse, json, os, time, numpy as np, torch
from torch.func import vjp, jvp
from msr.config import load_cfg
from msr.core import Arch, Model, make_data, loss_unrolled, nsteps
from train import _rollseq

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--window", type=float, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--meas-every", type=int, default=40)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--n-held", type=int, default=1024)
ap.add_argument("--out", required=True)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
R = os.path.expandvars("$SCRATCH/multistep_roughness")
root = R if a.width == 256 else f"{R}/pilot/w{a.width}"
cfg = load_cfg("config.yaml", [f"hidden=[{a.width},{a.width},{a.width}]", f"n_test={a.n_held}"] + a.set)
dt = 0.08; evs = [0.64, 1.28, 1.92, 2.56]; p = nsteps(a.window, dt); pmax = nsteps(max(evs), dt)
ck = torch.load(f"{root}/runs/dt0.08_unroll128_ft_s4k_linec025/ckpts.pt", map_location=dev, weights_only=False)
j = [tuple(x) for x in ck["names"]].index(("euler", a.seed))
data = make_data(cfg, dt, dev, train=True, horizon=max(evs))
for k_, v_ in ck["stats"].items():
    assert torch.allclose(v_.to(dev), data["stats"][k_], rtol=1e-12, atol=0), k_
arch = Arch(cfg["hidden"], cfg["act"]); m = Model(arch, "euler", dt, data["stats"])
th = ck["theta"][max(ck["theta"])][j].detach().clone().to(dev)
uTr, uTe = data["uT_train"], data["uT_test"]; ntr, B = uTr.shape[1], cfg["batch"]; n = uTe.shape[1]
gen = torch.Generator(device=dev).manual_seed(1000 * a.seed + p)


def heldout(th):
    """V, quantiles, exact and first-order gradients of V_T' for every T', and the held-out window gradient."""
    with torch.no_grad():
        J = torch.stack([m.Jstate(th, uTe[i]) for i in range(pmax)])
        eps = uTe[1:pmax + 1] - m.Gt(th, uTe[:pmax])
        e1 = [torch.zeros_like(eps[0])]
        for i in range(pmax): e1.append(torch.einsum("nab,nb->na", J[i], e1[-1]) + eps[i])
        e1 = torch.stack(e1)
        en = (uTe[1:pmax + 1] - _rollseq(arch, th, m.c, uTe[0], pmax)).norm(dim=-1)
    _, vf = vjp(lambda t: m.Gt(t, uTe[:pmax]), th)
    out = {}
    for T in evs + ["window"]:
        r = p if T == "window" else nsteps(T, dt)
        wq = None if T == "window" else [1.0 if q == r else 0.0 for q in range(1, r + 1)]
        ws = torch.ones(r, device=dev) if wq is None else torch.tensor(wq, device=dev)
        with torch.no_grad():
            c = ws[:, None, None] * e1[1:r + 1] / (ws.sum() * n)
            y = torch.zeros(pmax, n, 3, device=dev); y[r - 1] = c[r - 1]
            for i in range(r - 2, -1, -1): y[i] = c[i] + torch.einsum("nba,nb->na", J[i + 1], y[i + 1])
        g1 = -vf(y)[0]
        t_ = th.clone().requires_grad_(True)
        V = loss_unrolled(arch, t_, m.c, uTe[:r + 1], r, wq)
        g = torch.autograd.grad(V, t_)[0]
        x = en[r - 1]
        out[T] = dict(V=float(V.detach()), g=g, g1=g1, med=float(x.median()), p90=float(torch.quantile(x, 0.9)),
                      p99=float(torch.quantile(x, 0.99)))
    return out


log, t0 = [], time.time()
acc = {T: [0.0, 0.0] for T in evs}; cos_bw = []
H = heldout(th)
for k in range(a.steps + 1):
    if k % a.meas_every == 0 and k > 0 or k == a.steps:
        H2 = heldout(th)
        log.append(dict(step=k, **{str(T): dict(V0=H[T]["V"], V1=H2[T]["V"], dV=H2[T]["V"] - H[T]["V"], pred=acc[T][0], pred1=acc[T][1],
                                               med=H2[T]["med"], p90=H2[T]["p90"], p99=H2[T]["p99"]) for T in evs},
                        window=dict(V=H2["window"]["V"]), cos_batch_heldout=float(np.mean(cos_bw))))
        H = H2; acc = {T: [0.0, 0.0] for T in evs}; cos_bw = []
    if k == a.steps: break
    idx = torch.randint(ntr, (B,), generator=gen, device=dev)
    U = uTr[:p + 1, idx]
    t_ = th.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_unrolled(arch, t_, m.c, U, p), t_)[0]
    _, tan = jvp(lambda t: _rollseq(arch, t, m.c, U[0], p), (th,), (g,))
    alpha = a.c * float(g @ g) / float((tan ** 2).sum() / (p * B))
    for T in evs:
        acc[T][0] -= alpha * float(H[T]["g"] @ g)
        acc[T][1] -= alpha * float(H[T]["g1"] @ g)
    cos_bw.append(float(g @ H["window"]["g"] / (g.norm() * H["window"]["g"].norm())))
    th = th - alpha * g
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(args=vars(a), log=log, wall_s=time.time() - t0, theta_path=None), open(a.out, "w"), indent=1)
torch.save(th.cpu(), a.out.replace(".json", "_theta.pt"))
print(f"done W{a.width} window {a.window} seed {a.seed}: {time.time() - t0:.0f}s, {len(log)} measurements", flush=True)
