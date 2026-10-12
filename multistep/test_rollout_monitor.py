"""Checks of rollout_monitor: series coefficients against the exact window loss, l_1 against a finite difference,
first-order gradient at short windows, and a generic nn.Module with matrix-valued states."""
import math, torch
from torch.utils._pytree import tree_map
from rollout_monitor import RolloutMonitor, window_loss, from_module, _axpy, _scale, _norm
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# generic model: states [B, 4, 5], Euler step with an MLP acting on the flattened state
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(20, 64), torch.nn.Linear(64, 20)
    def forward(self, u):
        x = u.flatten(1)
        return self.l2(torch.nn.functional.silu(self.l1(x))).view_as(u)

net = Net()
dt = 0.05
step_fn, params = from_module(net, step=lambda f, u: u + dt * f(u))
# "true" trajectories from a perturbed copy of the network, so the errors are small but nonzero
true_net = Net(); true_net.load_state_dict(net.state_dict())
with torch.no_grad():
    for p in true_net.parameters():
        p.add_(0.05 * torch.randn_like(p))
true_step, true_params = from_module(true_net, step=lambda f, u: u + dt * f(u))
B, N = 64, 12
u0 = torch.randn(B, 4, 5)
with torch.no_grad():
    U = [u0]
    for _ in range(N):
        U.append(true_step(true_params, U[-1]))
    U = torch.stack(U)
precond = tree_map(lambda x: 0.5 + torch.rand_like(x), params)
mon = RolloutMonitor(step_fn, params, precond=precond)
out = mon.profile(U, N, order=4, grid=[1, 2, 6, 12], krylov=True, higher_order=True)

g = mon.grad_window(U, N)
d = _scale(g, -1.0 / float(_norm(g)))
dtheta = mon._rescale(d)
for p in (3, N):
    for z in (1e-3, 1e-2):
        exact = float(window_loss(step_fn, _axpy(params, dtheta, z), U, p))
        series = float(window_loss(step_fn, params, U, p)) + sum(out["ell"][m][p - 1] * z ** (m + 1) for m in range(4))
        print(f"p={p} z={z:g}: exact {exact:.12e}  series(order 4) {series:.12e}  rel diff {abs(exact - series) / abs(exact):.1e}")
    h = 1e-6
    fd = (float(window_loss(step_fn, _axpy(params, dtheta, h), U, p)) - float(window_loss(step_fn, _axpy(params, dtheta, -h), U, p))) / (2 * h)
    print(f"p={p}: l_1 {out['ell'][0][p - 1]:.10e}  central difference {fd:.10e}")
print("cos(g_p, g_target):", out["cos_target"], " higher-order ratio:", out["higher_order_ratio"])
print("Krylov bounds on F_p(s)/|e|:", out["krylov_lo"], out["krylov_up"])
assert all(lo <= up + 1e-9 for lo, up in zip(out["krylov_lo"], out["krylov_up"])), "Krylov lower bound exceeds upper bound"
print("ok")
