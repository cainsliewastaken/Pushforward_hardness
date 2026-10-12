"""Seconds-long check of train_kappa.kappa_step: its hypergradient h against central finite differences of
D(kappa) = V(theta) - V(theta+(kappa)), with theta+ the Adam step from a non-trivial optimizer state
(clip factor and second moment frozen at kappa0, as in the derivative), on a tiny network (CPU, single process)."""
import torch
from msr.config import load_cfg
from msr.core import Arch, Model, make_data
from train_kappa import adam_lookahead, kappa_step, loss_kappa

torch.manual_seed(0)
cfg = load_cfg("config.yaml", ["n_train=64", "n_test=8", "transient=2.0"])
dt, p, lr, clip = 0.02, 8, 1e-3, 1.0
data = make_data(cfg, dt, "cpu", horizon=0.32)
arch = Arch([16, 16], "silu")
c = Model(arch, "euler", dt, data["stats"]).c
th = (arch.init(0, "cpu") + 0.3 * torch.randn(arch.P)).requires_grad_(True)
U = data["uT_train"][:p + 1, :6]
opt = torch.optim.Adam([th], lr=lr)
for _ in range(3):                                   # non-trivial Adam state
    opt.zero_grad(); loss_kappa(arch, th, c, U, p, None).backward(); opt.step()
kap0 = 0.5 * torch.rand(p, U.shape[1])


def grad_at(kap):
    th_ = th.detach().clone().requires_grad_(True)
    return torch.autograd.grad(loss_kappa(arch, th_, c, U, p, kap), th_)[0]


# freeze the clip factor and Adam's second moment at kappa0, as the derivative does
g0 = grad_at(kap0)
b1, b2 = opt.defaults["betas"]; eps = opt.defaults["eps"]
st = opt.state[th]; n = int(st["step"]) + 1
cf = min(1.0, clip / float(torch.linalg.norm(g0)))
den = ((b2 * st["exp_avg_sq"] + (1 - b2) * (cf * g0) ** 2) / (1 - b2 ** n)).sqrt() + eps
look_frozen = lambda gb: (th.detach() - lr * ((b1 * st["exp_avg"] + (1 - b1) * cf * gb) / (1 - b1 ** n)) / den,
                          lr * (1 - b1) * cf / ((1 - b1 ** n) * den))
# the frozen lookahead equals the real one at kappa0
tp_real, _ = adam_lookahead(opt, th, lr, g0, clip)
tp_frz, _ = look_frozen(g0)
print("lookahead equals frozen lookahead at kappa0:", float((tp_real - tp_frz).abs().max()))


def D(kap):
    tp = look_frozen(grad_at(kap))[0]
    with torch.no_grad():
        return loss_kappa(arch, th.detach(), c, U, p, None) - loss_kappa(arch, tp, c, U, p, None)


_, h = kappa_step(arch, th, c, U, p, look_frozen, 1, 1, 0.0, kap0=kap0)
worst = 0.0
for (i, b) in [(1, 0), (3, 2), (5, 4), (7, 5), (2, 1)]:
    e = torch.zeros_like(kap0); e[i, b] = 1e-5
    fd = (D(kap0 + e) - D(kap0 - e)) / 2e-5
    rel = abs(float(h[i, b]) - float(fd)) / (abs(float(fd)) + 1e-30)
    worst = max(worst, rel)
    print(f"kappa[{i},{b}]  hypergrad {float(h[i, b]): .6e}  finite diff {float(fd): .6e}  rel err {rel:.1e}")
print("PASS" if worst < 1e-5 else "FAIL", f"worst rel err {worst:.1e}")
