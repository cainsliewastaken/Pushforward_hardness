"""Scaling study harness: memory, time and accuracy of KronQ (Kronecker sign push + Gauss-Newton cap + per-mode
hypergradient push weights) against AdamW, Muon and SOAP on the field problem, as width grows.

Problem (online, no reuse): u = Gaussian random field on a 16x16 periodic grid (spectrum (1+k^2)^-2, unit std);
target y = tanh(2 N[(-Lap)^p u]) (p = 0 smooth ... p = 2 rough), as in mkfield.py / muon_field.py. Every minibatch is a
fresh draw, so there is no training set to overfit; validation is a fixed held-out draw of --nval fields.
Model: MLP 256 -> w x depth -> 256, GELU, fp32. Loss: mean squared error. Linear lr decay to 0 after --warmup steps.

Optimizers (--opt):
  adamw   torch AdamW (betas 0.9/--b2, weight decay --wd on matrices)
  muon    Nesterov momentum 0.95 + Newton-Schulz(5) orthogonalisation, scale sqrt(max(1, out/in)), on hidden matrices;
          AdamW (lr --lr_adam) on input / output matrices and biases
  soap    per layer L = EMA(G G^T), R = EMA(G^T G) (Shampoo factors), eigenbases every --refresh steps, Adam (betas
          0.9/--b2) in the rotated basis; optional blocks (--block) and one-sided (sides wider than --max_dim stay unrotated)
  kronq   per layer Kronecker Gauss-Newton basis: A = EMA(a~ a~^T) of layer inputs (with bias column), S = EMA(d d^T) of
          backprop signals for one random +-1 output combination per sample (unbiased GN estimate; one extra backward per
          statistics update); D_ij = w_ij sign(C_ij) on live modes (lam_ij = eS_i eA_j > reltol lam_max), capped at the
          Gauss-Newton step |Gr_ij| / (lam_ij lr L); per-mode push weights w = exp(u) learned online by the one-step
          hypergradient (--hyper 1): u -= beta_u hg / sqrt(EMA hg^2), hg = -lr (Gr_t)_ij D_{t-1,ij}.
          --block, --max_dim as for SOAP; --compress 1 stores u and D_{t-1} in bf16 and factors the normaliser EMA by
          rows and columns (Adafactor style); --mom_bf16 1 keeps the momentum in bf16.
Logs (jsonl): validation MSE on a step grid with wall time; per-step time split (fwd/bwd, statistics, eigh, update);
peak GPU memory; exact optimizer-state bytes (/4 bytes per parameter = "state per parameter").
"""
import os, sys, json, math, time, argparse
import numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument('--opt', required=True, choices=['adamw', 'muon', 'soap', 'kronq'])
p.add_argument('--width', type=int, default=512); p.add_argument('--depth', type=int, default=2)
p.add_argument('--p', type=float, default=2.0, help='target roughness exponent (0 smooth, 2 rough)')
p.add_argument('--steps', type=int, default=5000); p.add_argument('--batch', type=int, default=256)
p.add_argument('--lr', type=float, required=True); p.add_argument('--lr_adam', type=float, default=3e-3)
p.add_argument('--b2', type=float, default=0.99); p.add_argument('--wd', type=float, default=0.0); p.add_argument('--warmup', type=int, default=0)
p.add_argument('--refresh', type=int, default=10, help='statistics + eigenbasis refresh interval (soap, kronq)')
p.add_argument('--block', type=int, default=0); p.add_argument('--max_dim', type=int, default=10 ** 9)
p.add_argument('--hyper', type=int, default=1); p.add_argument('--beta_u', type=float, default=0.01)
p.add_argument('--compress', type=int, default=0); p.add_argument('--mom_bf16', type=int, default=0)
p.add_argument('--reltol', type=float, default=1e-10); p.add_argument('--beta_stats', type=float, default=0.95)
p.add_argument('--nval', type=int, default=4096); p.add_argument('--eval_every', type=int, default=250)
p.add_argument('--seed', type=int, default=1); p.add_argument('--out', required=True)
p.add_argument('--u_clamp', type=float, default=4.6, help='clamp on the log push weight u (push w = exp(u) in [e^-c, e^c])')
p.add_argument('--diag', type=int, default=0, help='1: at every eval step log per layer the u distribution, clamp fraction, cap-active fraction, '
               'step energy on the lowest decile of modes by Kronecker eigenvalue, weight norm, and hidden-activation saturation')
a = p.parse_args()
import subprocess
try: GIT = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception: GIT = 'unknown'
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(a.seed)
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

dims = [NIN] + [a.width] * a.depth + [NOUT]
torch.manual_seed(a.seed)
lins = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]).to(DEV)
L1 = len(lins); P = sum(q.numel() for q in lins.parameters())
acts_cache = [None] * L1; zs_cache = [None] * L1
def forward(x, keep=False):
    h = x
    for l, lin in enumerate(lins):
        if keep: acts_cache[l] = h
        z = lin(h)
        if keep: z.retain_grad(); zs_cache[l] = z
        h = torch.nn.functional.gelu(z) if l < L1 - 1 else z
    return h
def mse(out, y): return ((out - y) ** 2).mean()
@torch.no_grad()
def val_mse():
    return float(sum(mse(forward(Xv[i:i + 1024]), Yv[i:i + 1024]) * len(Xv[i:i + 1024]) for i in range(0, len(Xv), 1024)) / len(Xv))
def nbytes(x): return x.numel() * x.element_size()

# ------------------------------------------------------------------ blocked / one-sided eigenbasis helpers
def blocks_of(dim):
    if dim > a.max_dim: return None
    b = a.block if 0 < a.block < dim else dim
    return [slice(i, min(i + b, dim)) for i in range(0, dim, b)]
class Side:
    """EMA'd factor on one side of a layer (full, blocked, or diagonal when wider than max_dim) and its eigenbasis."""
    def __init__(self, dim):
        self.dim, self.bl = dim, blocks_of(dim); self.F = None; self.U = None; self.e = None
    def update(self, Z, beta):           # Z: samples x dim, factor = E[Z^T Z] (or its diagonal)
        m = Z.shape[0]
        new = (Z * Z).mean(0) if self.bl is None else [Z[:, s].T @ Z[:, s] / m for s in self.bl]
        if self.F is None: self.F = new
        elif self.bl is None: self.F.mul_(beta).add_(new, alpha=1 - beta)
        else:
            for F_, N_ in zip(self.F, new): F_.mul_(beta).add_(N_, alpha=1 - beta)
    def eig(self):
        if self.bl is None: self.U = None; self.e = self.F.clamp(min=0).clone(); return
        self.U = []; self.e = torch.zeros(self.dim, device=DEV)
        for s, F_ in zip(self.bl, self.F):
            ev, V = torch.linalg.eigh(F_); self.U.append(V); self.e[s] = ev.clamp(min=0)
    def overlap(self, Uold):           # squared overlaps (new x old) per block, for carrying per-mode state
        if self.U is None or Uold is None: return None
        return [(Vn.T @ Vo) ** 2 for Vn, Vo in zip(self.U, Uold)]
    def carry(self, T, O, side):       # T indexed by modes on this side; side 'L' rows / 'R' columns
        if O is None: return T
        out = T.clone()
        for s, Ob in zip(self.bl, O):
            if side == 'L': out[s, ...] = Ob @ T[s, ...]
            else: out[..., s] = T[..., s] @ Ob.T
        return out
    def rot(self, M, side, inv):          # side 'L': rows; 'R': columns. inv=False: into the eigenbasis
        if self.U is None: return M
        out = torch.empty_like(M)
        for s, V in zip(self.bl, self.U):
            if side == 'L': out[s, :] = (V @ M[s, :]) if inv else (V.T @ M[s, :])
            else: out[:, s] = (M[:, s] @ V.T) if inv else (M[:, s] @ V)
        return out
    def state_bytes(self):
        if self.F is None: return 0
        if self.bl is None: return nbytes(self.F)
        return sum(nbytes(F_) for F_ in self.F) + (sum(nbytes(V) for V in self.U) if self.U else 0)
def ns5(Gm, steps=5, eps=1e-7):
    aa, bb, cc = (3.4445, -4.7750, 2.0315); Xm = Gm / (Gm.norm() + eps); tr = Gm.shape[0] > Gm.shape[1]
    if tr: Xm = Xm.T
    for _ in range(steps):
        A_ = Xm @ Xm.T; B_ = bb * A_ + cc * (A_ @ A_); Xm = aa * Xm + B_ @ Xm
    return Xm.T if tr else Xm

# ------------------------------------------------------------------ optimizer state
mats = [lin.weight for lin in lins]; biases = [lin.bias for lin in lins]
if a.opt == 'adamw':
    opt = torch.optim.AdamW([{'params': mats, 'weight_decay': a.wd}, {'params': biases, 'weight_decay': 0.0}], lr=a.lr, betas=(0.9, a.b2))
elif a.opt == 'muon':
    hidden = mats[1:-1]; others = [mats[0], mats[-1]] + biases
    opt = torch.optim.AdamW([{'params': others, 'weight_decay': a.wd}], lr=a.lr_adam, betas=(0.9, a.b2))
    mbuf = [torch.zeros_like(w) for w in hidden]
else:
    # soap / kronq act on Wt = [W | b] per layer
    st = []
    for lin in lins:
        o, i = lin.weight.shape
        s_ = dict(L=Side(o), R=Side(i + 1), m=torch.zeros(o, i + 1, device=DEV, dtype=torch.bfloat16 if a.mom_bf16 else torch.float32))
        if a.opt == 'soap': s_['v'] = torch.zeros(o, i + 1, device=DEV)
        if a.opt == 'kronq' and a.hyper:
            dt = torch.bfloat16 if a.compress else torch.float32
            s_['u'] = torch.zeros(o, i + 1, device=DEV, dtype=dt); s_['Dprev'] = None
            s_['hv'] = (torch.zeros(o, device=DEV), torch.zeros(i + 1, device=DEV)) if a.compress else torch.zeros(o, i + 1, device=DEV)
        st.append(s_)
def opt_state_bytes():
    if a.opt == 'adamw':
        return sum(nbytes(t) for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.numel() > 1)
    if a.opt == 'muon':
        return sum(nbytes(t) for t in mbuf) + sum(nbytes(t) for s in opt.state.values() for t in s.values() if torch.is_tensor(t) and t.numel() > 1)
    tot = 0
    for s_ in st:
        tot += s_['L'].state_bytes() + s_['R'].state_bytes() + nbytes(s_['m'])
        for k in ('v', 'u', 'Dprev'):
            if s_.get(k) is not None: tot += nbytes(s_[k])
        if 'hv' in s_: tot += sum(nbytes(t) for t in s_['hv']) if isinstance(s_['hv'], tuple) else nbytes(s_['hv'])
    return tot

rows = []
def emit(metric, value, step=None): rows.append(dict(git=GIT, args=' '.join(sys.argv[1:]), opt=a.opt, width=a.width, depth=a.depth, p=a.p, P=P, step=step, metric=metric, value=value))
tm = dict(fb=0.0, stats=0.0, eig=0.0, upd=0.0)
def sync(): torch.cuda.synchronize() if DEV == 'cuda' else None
t_start = time.time(); emit('val', val_mse(), 0); emit('wall', 0.0, 0)
for step in range(1, a.steps + 1):
    fac = min(1.0, step / a.warmup) if a.warmup else 1.0; fac *= 1 - (step - 1) / a.steps; lr = a.lr * fac
    xb, yb = draw(a.batch, gtrain)
    sync(); t0 = time.time()
    need_stats = a.opt in ('soap', 'kronq') and (step - 1) % a.refresh == 0
    for q in lins.parameters(): q.grad = None
    out = forward(xb, keep=need_stats and a.opt == 'kronq'); loss = mse(out, yb); loss.backward(retain_graph=need_stats and a.opt == 'kronq')
    sync(); tm['fb'] += time.time() - t0
    with torch.no_grad():
        if a.opt == 'adamw':
            for g_ in opt.param_groups: g_['lr'] = lr
            t0 = time.time(); opt.step(); sync(); tm['upd'] += time.time() - t0
        elif a.opt == 'muon':
            t0 = time.time()
            for g_ in opt.param_groups: g_['lr'] = a.lr_adam * fac
            opt.step()
            for w, mb in zip(mats[1:-1], mbuf):
                mb.mul_(0.95).add_(w.grad); gn = w.grad + 0.95 * mb
                w.sub_(lr * ns5(gn) * max(1.0, w.shape[0] / w.shape[1]) ** 0.5)
            sync(); tm['upd'] += time.time() - t0
        else:
            Gs = [torch.cat([lin.weight.grad, lin.bias.grad[:, None]], 1) for lin in lins]
            if need_stats:
                t0 = time.time()
                if a.opt == 'soap':
                    for s_, Gw in zip(st, Gs): s_['L'].update(Gw.T, a.beta_stats); s_['R'].update(Gw, a.beta_stats)   # E[G G^T], E[G^T G]
                else:
                    with torch.enable_grad():
                        eps_ = torch.randint(0, 2, out.shape, device=DEV, generator=gtrain).float() * 2 - 1
                        gz = torch.autograd.grad((out * eps_).sum(), [zs_cache[l] for l in range(L1)])
                    for l, s_ in enumerate(st):
                        at = torch.cat([acts_cache[l], torch.ones(acts_cache[l].shape[0], 1, device=DEV)], 1)
                        s_['R'].update(at, a.beta_stats); s_['L'].update(gz[l], a.beta_stats)
                sync(); tm['stats'] += time.time() - t0; t0 = time.time()
                for s_ in st:
                    oldL, oldR = s_['L'].U, s_['R'].U
                    s_['L'].eig(); s_['R'].eig()
                    if a.opt == 'kronq' and a.hyper:
                        OL, OR = s_['L'].overlap(oldL), s_['R'].overlap(oldR)
                        s_['u'] = s_['R'].carry(s_['L'].carry(s_['u'].float(), OL, 'L'), OR, 'R').to(s_['u'].dtype)
                        if a.compress: s_['hv'] = (s_['L'].carry(s_['hv'][0][:, None], OL, 'L')[:, 0], s_['R'].carry(s_['hv'][1][None, :], OR, 'R')[0])
                        else: s_['hv'] = s_['R'].carry(s_['L'].carry(s_['hv'], OL, 'L'), OR, 'R')
                        s_['Dprev'] = None    # the pairing of the last step with the next gradient restarts in the new basis
                sync(); tm['eig'] += time.time() - t0
            t0 = time.time()
            for l, (s_, lin, Gw) in enumerate(zip(st, lins, Gs)):
                m = s_['m']; m.mul_(0.9).add_(Gw.to(m.dtype), alpha=0.1)
                C = s_['R'].rot(s_['L'].rot(m.float(), 'L', False), 'R', False)
                Gr = s_['R'].rot(s_['L'].rot(Gw, 'L', False), 'R', False)
                if a.opt == 'soap':
                    s_['v'].mul_(a.b2).add_(Gr * Gr, alpha=1 - a.b2)
                    bc1 = 1 - 0.9 ** step; bc2 = 1 - a.b2 ** step
                    Dr = (C / bc1) / ((s_['v'] / bc2).sqrt() + 1e-8)
                else:
                    lam = s_['L'].e[:, None] * s_['R'].e[None, :] * (2.0 / NOUT); lmax = float(lam.max()); live = lam > a.reltol * lmax   # GN eigenvalue of the mean-squared loss
                    w = torch.ones_like(lam)
                    if a.hyper:
                        if s_['Dprev'] is not None:
                            hg = -lr * Gr * s_['Dprev'].float()
                            if a.compress:
                                r_, c_ = s_['hv']; r_.mul_(0.99).add_((hg * hg).mean(1), alpha=0.01); c_.mul_(0.99).add_((hg * hg).mean(0), alpha=0.01)
                                den = (r_[:, None] * c_[None, :] / (r_.mean() + 1e-30)).sqrt()
                            else:
                                s_['hv'].mul_(0.99).add_(hg * hg, alpha=0.01); den = s_['hv'].sqrt()
                            s_['u'].copy_((s_['u'].float() - a.beta_u * hg / (den + 1e-30)).clamp(-a.u_clamp, a.u_clamp))
                        w = torch.exp(s_['u'].float())
                    Dr = torch.where(live, w * torch.sign(C), torch.zeros_like(C))
                    capv = Gr.abs() / (lam.clamp(min=1e-30) * max(lr, 1e-30) * L1)
                    Dr = torch.sign(Dr) * torch.minimum(Dr.abs(), capv)
                    if a.diag and (step % a.eval_every == 0 or step == a.steps):
                        E2 = Dr * Dr; tot = float(E2.sum()) + 1e-30; lf = lam.flatten()
                        thr = float(lf.kthvalue(max(1, int(0.1 * lf.numel())))[0])
                        emit(f'e_lowdec_l{l}', float(E2[lam <= thr].sum()) / tot, step)          # step energy on the lowest decile of modes
                        emit(f'capfrac_l{l}', float((w > capv).float().mean()), step)            # fraction of modes where the GN cap binds
                        emit(f'livefrac_l{l}', float(live.float().mean()), step)
                        emit(f'wmean_l{l}', float(w.mean()), step); emit(f'wmax_l{l}', float(w.max()), step)
                        if a.hyper:
                            uf = s_['u'].float().flatten(); sub = uf[torch.randperm(uf.numel(), device=DEV)[:200000]]
                            for qn, qv in zip(('10', '50', '90'), torch.quantile(sub, torch.tensor([0.1, 0.5, 0.9], device=DEV)).tolist()): emit(f'u_q{qn}_l{l}', qv, step)
                            emit(f'u_clampfrac_l{l}', float((uf.abs() >= a.u_clamp - 1e-3).float().mean()), step)
                    if a.hyper: s_['Dprev'] = Dr.to(s_['u'].dtype)
                Dw = s_['L'].rot(s_['R'].rot(Dr, 'R', True), 'L', True)
                lin.weight.sub_(lr * Dw[:, :-1]); lin.bias.sub_(lr * Dw[:, -1])
                if a.wd: lin.weight.mul_(1 - lr * a.wd)
            sync(); tm['upd'] += time.time() - t0
    if step % a.eval_every == 0 or step == a.steps:
        v = val_mse(); emit('val', v, step); emit('wall', time.time() - t_start, step)
        if a.diag:
            with torch.no_grad():
                for l, lin in enumerate(lins): emit(f'wnorm_l{l}', float(lin.weight.norm()), step)
                h = Xv[:1024]
                for l, lin in enumerate(lins[:-1]):
                    z = lin(h); emit(f'sat_l{l}', float((z.abs() > 3).float().mean()), step); emit(f'zrms_l{l}', float(z.pow(2).mean().sqrt()), step)
                    h = torch.nn.functional.gelu(z)
        if not math.isfinite(v): emit('diverged', 1, step); break
peak = torch.cuda.max_memory_allocated() if DEV == 'cuda' else 0
sb = opt_state_bytes()
VAL_FINAL = [r_['value'] for r_ in rows if r_['metric'] == 'val'][-1]
for k_, v_ in (('val_final', VAL_FINAL), ('state_bytes', sb), ('state_per_param', sb / (4 * P)),
               ('peak_mem_bytes', peak), ('t_fb', tm['fb']), ('t_stats', tm['stats']), ('t_eig', tm['eig']), ('t_upd', tm['upd']), ('wall_total', time.time() - t_start)):
    emit(k_, v_, a.steps)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'a') as fh:
    for rw in rows: fh.write(json.dumps(rw) + '\n')
print(f"{a.opt} w={a.width} d={a.depth} p={a.p} lr={a.lr} block={a.block} max_dim={a.max_dim} hyper={a.hyper} compress={a.compress} mom_bf16={a.mom_bf16}: "
      f"val={VAL_FINAL:.5g} state/P={sb / (4 * P):.2f} peak={peak / 2**30:.2f}GiB time fb/stats/eig/upd = "
      f"{tm['fb']:.1f}/{tm['stats']:.1f}/{tm['eig']:.1f}/{tm['upd']:.1f}s wall={time.time() - t_start:.0f}s", flush=True)
