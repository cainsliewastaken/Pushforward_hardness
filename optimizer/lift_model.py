"""Model test of the lookahead theory (paper/notes/LIFT_GRADIENT.md, LOOKAHEAD_THEORY.md) in the EXACT parameter space of a
small network: no Kronecker basis, no blocks, every quantity by autograd (torch.func). Standalone: shares nothing with
scale_study.py / kronq_fast.py except the field problem's definition (copied, 8 lines).

Objects (per step, on the batch; E = f - y, N = m ny, L = ||E||^2 / N):
  g      = grad L                                   the loss gradient
  s_E^2  = ||J^T E_hat||^2 = E_hat^T K E_hat          leverage of the residual direction (one vjp)
  S      = grad_theta s_E^2 = 2 M_E_hat J^T E_hat     the sharpening gradient (double backprop: grad of the vjp's norm)
  Cross  = <S, d>, Q = ||M_E_hat d||^2                the two terms of the stage identity on E_hat along the step d
  c(d)   = <E, f_dd> = d^T M_E d                      residual curvature along the step (hvp of <E, f>)
Rules, all at the same per-step length rho_t (linear decay) so only the DIRECTION differs:
  gd         d = -rho g / |g|                                        greedy first-order optimal at fixed length (Krylov M = 1)
  lookahead  d = -rho g~ / |g~|,  g~ = g - kappa S,  kappa = ||E|| Lambda_rem / (N s_E)   (LIFT_GRADIENT.md section 4)
  sign       d = -rho sign(g) / sqrt(P)                              l_inf steepest descent at the same l_2 length (Adam-like)
  liftonly   d = +rho S / |S|                                        pure lift (control: should raise s_E and not the error)
  momentum   heavy ball on g (beta 0.9) at the same length                 larger steps on the persistent (slow) modes, by accident
Direction lifted (--dir): 'full' = the whole residual E_hat; 'slow' = the Krylov residual r_M = E - K x_M after M matrix-free
CG iterations on K x = E (the part an M-term Krylov step cannot remove, Thm stepkrylov), normalised. kappa is zero when
the direction is already completable within the remaining path (s Lambda_rem >= ||r||), the completed regime of Prop. 4.
--check 1 verifies at a few steps: E_hat^T (K_after - K_before) E_hat = <S, d> + ||M d||^2 + O(d^3), and c(d) = d^T (hvp) d.
Outputs: jsonl rows (metric, value, step) in --out; prints a summary line. Runs on the login node in ~1 min at width 128.
"""
import argparse, json, math, os, sys, time, subprocess
import torch
from torch.func import functional_call, vjp, jvp, grad

p = argparse.ArgumentParser()
p.add_argument('--width', type=int, default=128); p.add_argument('--depth', type=int, default=2)
p.add_argument('--steps', type=int, default=1500); p.add_argument('--batch', type=int, default=64); p.add_argument('--nval', type=int, default=2048)
p.add_argument('--p', type=float, default=2.0, help='target roughness exponent (2 rough, 0 smooth)')
p.add_argument('--rho', type=float, default=0.3, help='per-step Euclidean length in parameter space at t = 0 (linear decay to 0)')
p.add_argument('--rule', default='lookahead', choices=['gd', 'lookahead', 'sign', 'liftonly', 'momentum'])
p.add_argument('--dir', default='slow', choices=['full', 'slow'], help='direction whose leverage the lookahead lifts')
p.add_argument('--cg', type=int, default=5, help='Krylov terms defining the slow residual (dir slow)')
p.add_argument('--beta', type=float, default=0.9, help='momentum')
p.add_argument('--kappa_scale', type=float, default=1.0, help='multiplier on kappa (lookahead); 0 reduces to gd')
p.add_argument('--check', type=int, default=0, help='1: verify the stage-identity decomposition and the curvature at a few steps')
p.add_argument('--eval_every', type=int, default=100); p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default=os.path.expandvars('$SCRATCH/l63_held_rough/runs/lift_model/res.jsonl'))
a = p.parse_args()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
try: GIT = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
except Exception: GIT = 'unknown'

# ---- the field problem of scale_study.py (same definition; 16 x 16 fields, target = tanh(2 v) with v the p-th power of the Laplacian spectrum)
G = 16; NIN = NOUT = G * G
kx = torch.fft.fftfreq(G, device=DEV) * G; KX, KY = torch.meshgrid(kx, kx, indexing='ij'); K2 = KX ** 2 + KY ** 2
AMP = (1 + K2) ** -2.0; PK = (K2 ** a.p) if a.p > 0 else None
def draw(m, gen):
    z = torch.complex(torch.randn(m, G, G, device=DEV, generator=gen), torch.randn(m, G, G, device=DEV, generator=gen))
    u = torch.fft.ifft2(z * AMP).real; u = u / u.std(dim=(1, 2), keepdim=True)
    v = torch.fft.ifft2(torch.fft.fft2(u) * PK).real if PK is not None else u.clone(); v = v / v.std(dim=(1, 2), keepdim=True)
    return u.reshape(m, -1).float(), torch.tanh(2 * v).reshape(m, -1).float()
gtrain = torch.Generator(device=DEV); gtrain.manual_seed(1000 + a.seed)
gval = torch.Generator(device=DEV); gval.manual_seed(999); Xv, Yv = draw(a.nval, gval)

# ---- network as a functional
dims = [NIN] + [a.width] * a.depth + [NOUT]
net = torch.nn.Sequential(*[m for i in range(len(dims) - 1) for m in (torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.GELU())][:-1]).to(DEV)
params = {k: v.detach().clone() for k, v in net.named_parameters()}
names = list(params); P = sum(v.numel() for v in params.values())
def flat(d): return torch.cat([d[k].reshape(-1) for k in names])
def unflat(x):
    out, i = {}, 0
    for k in names: n = params[k].numel(); out[k] = x[i:i + n].view_as(params[k]); i += n
    return out
def f(theta, x): return functional_call(net, theta, (x,))
def loss_fn(theta, x, y): return ((f(theta, x) - y) ** 2).mean()

rows = []
def emit(metric, value, step): rows.append(dict(git=GIT, args=' '.join(sys.argv[1:]), rule=a.rule, width=a.width, p=a.p, rho=a.rho, step=step, metric=metric, value=float(value)))
def val_mse(theta):
    with torch.no_grad(): return float(((f(theta, Xv) - Yv) ** 2).mean())

# ---- per-step quantities (all exact on the batch)
def quantities(theta, xb, yb):
    """returns g (flat), E, E_hat (the direction lifted), s^2 along it, S (flat), with S = grad_theta ||J^T E_hat||^2 by double backprop."""
    out, vjp_fn = vjp(lambda th: f(th, xb), theta)
    E = out - yb; Edir = slow_residual(theta, xb, E.detach(), a.cg) if a.dir == 'slow' else E.detach()
    Ehat = Edir / (Edir.norm() + 1e-30)
    def lev2(th):
        _, vf = vjp(lambda t: f(t, xb), th)
        JtE = vf(Ehat)[0]
        return sum((t * t).sum() for t in JtE.values())
    s2 = lev2(theta)
    S = flat(grad(lev2)(theta))
    g = flat(grad(loss_fn)(theta, xb, yb))
    return g, E, Ehat, float(s2), S, out, Edir

def K_apply(theta, xb, v):
    _, vf = vjp(lambda t: f(t, xb), theta); Jtv = vf(v)[0]
    _, Jv = jvp(lambda t: f(t, xb), (theta,), (Jtv,)); return Jv
def slow_residual(theta, xb, E, M):
    """least-squares Krylov residual: r = E - K x with x in span{E, K E, ..., K^(M-1) E} minimising ||r|| (the part no M-term
    Krylov step removes, Thm stepkrylov); ||r|| <= ||E|| by construction. (CG's residual is not monotone in this norm.)"""
    V = []; v = E / (E.norm() + 1e-30); V.append(v)
    for _ in range(M - 1):
        w = K_apply(theta, xb, V[-1])
        for u in V: w = w - (w * u).sum() * u
        nw = w.norm()
        if nw < 1e-12: break
        V.append(w / nw)
    W = torch.stack([K_apply(theta, xb, v).reshape(-1) for v in V], 1)   # N x M (outputs flattened)
    y = torch.linalg.lstsq(W, E.reshape(-1, 1)).solution[:, 0]
    return (E.reshape(-1) - W @ y).view_as(E)
def leverage2(theta, xb, v):
    _, vf = vjp(lambda t: f(t, xb), theta); JtV = vf(v)[0]
    return float(sum((t * t).sum() for t in JtV.values()))

def check_identity(theta, xb, yb, d_flat, step):
    """E_hat^T (K(theta + d) - K(theta)) E_hat against <S, d> + ||M d||^2; c(d) against the hvp; at the actual step d."""
    g, E, Ehat, s2, S, out, Edir = quantities(theta, xb, yb)
    d = unflat(d_flat); theta2 = {k: theta[k] + d[k] for k in names}
    s2_after = leverage2(theta2, xb, Ehat)
    cross = float((S * d_flat).sum())
    # exact stage identity on E_hat: s2_after - s2 = 2 <J^T v, dJ^T v> + ||dJ^T v||^2 with the FULL change dJ (two vjps)
    _, vf1 = vjp(lambda t: f(t, xb), theta); _, vf2 = vjp(lambda t: f(t, xb), theta2)
    J1 = flat(vf1(Ehat)[0]); J2 = flat(vf2(Ehat)[0]); dJ = J2 - J1
    cross_exact = float(2 * (J1 * dJ).sum()); Q_exact = float((dJ * dJ).sum())
    # M d = d/ds [J^T E_hat](theta + s d): jvp of the vjp
    def JtE(th):
        _, vf = vjp(lambda t: f(t, xb), th); return vf(Ehat)[0]
    _, Md = jvp(JtE, (theta,), (d,))
    q = float(sum((t * t).sum() for t in Md.values()))
    # residual curvature along d: d^T M_E d with M_E the hessian of <E, f> (E fixed) = hvp of the scalar <E_fixed, f(theta)>
    Efix = E.detach()
    _, Hd = jvp(grad(lambda th: (f(th, xb) * Efix).sum()), (theta,), (d,))   # hvp = jvp of the gradient
    c_d = float(sum((Hd[k] * d[k]).sum() for k in names))
    # finite-difference second derivative of <E_fixed, f> along d (central, two step sizes) for the hvp check
    def h_of(s):
        th = {k: theta[k] + s * d[k] for k in names}; return float((f(th, xb) * Efix).sum())
    h0 = h_of(0.0); fd = [(h_of(s) - 2 * h0 + h_of(-s)) / (s * s) for s in (0.5, 0.25)]
    rec = dict(dlam_true=s2_after - s2, cross_exact=cross_exact, Q_exact=Q_exact, identity_err=s2_after - s2 - cross_exact - Q_exact,
               cross_1st=cross, Q_1st=q, c_d=c_d, c_fd_half=fd[0], c_fd_quarter=fd[1], s2=s2, dnorm=float(d_flat.norm()))
    for k, v in rec.items(): emit(f'check_{k}', v, step)
    return rec

# ---- training loop
theta = params
T = a.steps; rho_t = lambda t: a.rho * (1 - (t - 1) / T)
Lam_rem = lambda t: sum(rho_t(s) for s in range(t, T + 1))   # remaining path under the schedule
t_start = time.time(); emit('val', val_mse(theta), 0)
check_steps = {1, T // 4, T // 2, 3 * T // 4} if a.check else set()
mom = None
for step in range(1, T + 1):
    xb, yb = draw(a.batch, gtrain); rho = rho_t(step)
    g, E, Ehat, s2, S, out, Edir = quantities(theta, xb, yb)
    N = E.numel(); sE = math.sqrt(max(s2, 1e-30)); e_dir = float(Edir.norm())
    completable = sE * Lam_rem(step) >= e_dir                          # Prop. 4 completed regime: extra leverage is worth nothing
    kappa = 0.0 if completable else a.kappa_scale * e_dir * Lam_rem(step) / (N * sE)
    if a.rule == 'gd': dirv = -g
    elif a.rule == 'lookahead': dirv = -(g - kappa * S)
    elif a.rule == 'sign': dirv = -torch.sign(g)
    elif a.rule == 'liftonly': dirv = S
    elif a.rule == 'momentum': mom = g if mom is None else a.beta * mom + g; dirv = -mom
    d_flat = dirv * (rho / (dirv.norm() + 1e-30))
    if step in check_steps: check_identity(theta, xb, yb, d_flat, step)
    if step % a.eval_every == 0 or step == T or step == 1:
        emit('s2_E', s2, step); emit('kappa', kappa, step); emit('loss_batch', float((E * E).mean()), step)
        emit('slow_frac', e_dir / (float(E.norm()) + 1e-30), step); emit('completable', float(completable), step)
        emit('cross_step', float((S * d_flat).sum()), step)                       # first-order lift of the residual's leverage by this step
        emit('cos_g_S', float((g * S).sum() / (g.norm() * S.norm() + 1e-30)), step)   # alignment of descent and sharpening
        emit('frac_S', float(kappa * S.norm() / (g.norm() + 1e-30)), step)         # weight of the lookahead term relative to the gradient
    d = unflat(d_flat); theta = {k: (theta[k] + d[k]).detach() for k in names}
    if step % a.eval_every == 0 or step == T:
        v = val_mse(theta); emit('val', v, step); emit('wall', time.time() - t_start, step)
        if not math.isfinite(v): emit('diverged', 1, step); break
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, 'a') as fh:
    for r in rows: fh.write(json.dumps(r) + '\n')
vals = [r for r in rows if r['metric'] == 'val']; s2s = [r['value'] for r in rows if r['metric'] == 's2_E']
print(f"lift_model rule={a.rule} p={a.p:g} w={a.width} rho={a.rho:g} kappa_scale={a.kappa_scale:g}: val_final={vals[-1]['value']:.4g} (val_0 {vals[0]['value']:.4g}) "
      f"s2_E first/last {s2s[0]:.3g}/{s2s[-1]:.3g}  P={P} wall={time.time() - t_start:.0f}s")
if a.check:
    for r in rows:
        if r['metric'].startswith('check_'): print(f"  step {r['step']:5d} {r['metric'][6:]:16s} {r['value']:+.4e}")
