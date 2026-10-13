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
p.add_argument('--opt', required=True, choices=['adamw', 'muon', 'soap', 'kronq', 'kronqf', 'krylov'])
p.add_argument('--kr_M', type=int, default=3, help='krylov: CG iterations (Krylov terms) on the batch'); p.add_argument('--kr_mom', type=float, default=0.0, help='krylov: heavy-ball momentum on the scaled step'); p.add_argument('--kr_gamma', type=float, default=1e-3, help='krylov: damping relative to the Rayleigh quotient E^T K E / E^T E'); p.add_argument('--kr_decay', type=int, default=1, help='krylov: 1 = linear decay of the output-space radius rem ||E|| to 0')
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
p.add_argument('--gmask', type=float, default=0.0, help='>0: push only modes with |G_ij| > gmask * rms(G) (gradient-energy mask, basis independent)')
p.add_argument('--bcurv', type=int, default=0, help='1: live mask and GN cap from the CURRENT batch Kronecker curvature (extra backward per step)')
p.add_argument('--rem', type=float, default=0.0, help='>0: scale the whole step so that ||f(theta+d) - f(theta)|| <= rem * ||E|| on the batch (extra forward per step)')
p.add_argument('--rem_ctrl', type=float, default=0.0, help='>0: two-sided controller: scale the whole step by a running factor so that the measured output change '
               '||f(theta+d) - f(theta)|| / ||E|| on the batch is held at this value (replaces --rem and disables the lr decay)')
p.add_argument('--mom_adapt', type=int, default=0, help='kronqf: per-mode momentum window T_k = clip((v - m^2)/m^2, T_min, T_max) from the mode\'s online SNR; beta_k = 1 - 1/T_k'); p.add_argument('--mom_snr_beta', type=float, default=0.99, help='kronqf mom_adapt: EMA factor of the per-mode signal^2 estimate'); p.add_argument('--mom_T_min', type=float, default=2.0); p.add_argument('--mom_T_max', type=float, default=1000.0); p.add_argument('--alloc2_oneshot', type=int, default=0, help='kronqf alloc2: T_eff = 1 (the completion step, then the budget sets the length)'); p.add_argument('--acf', type=str, default='', help='kronqf: comma-separated steps at which to record the rotated gradient on 200 modes per lambda decile for --acf_len steps (basis held) and emit its autocorrelation'); p.add_argument('--acf_len', type=int, default=64); p.add_argument('--mech', type=int, default=0, help='1: per-step mechanism diagnostics (mechanism.py) every --mech_every steps'); p.add_argument('--mech_every', type=int, default=250); p.add_argument('--refresh_min', type=int, default=1, help='kronqf: floor on the adaptive refresh period'); p.add_argument('--radius_scan', type=str, default='', help='kronqf: comma-separated steps at which to scan the Taylor radius along several directions (diag)'); p.add_argument('--check_curv', type=int, default=0, help='kronqf: exact autograd check of the residual curvature on a few modes (diag)'); p.add_argument('--alloc2', type=int, default=0, help='kronqf: LOOKAHEAD_THEORY Thm 7 step: ridge with the full curvature lam + c on convex modes, full push (box) on concave ones; no cap/budget/hyper'); p.add_argument('--curv', type=int, default=0, help='kronqf: compute the residual curvature c per mode (for the diagnostics) without using it'); p.add_argument('--cfac', type=float, default=1.0, help='kronqf alloc2: multiplier on the residual curvature (0 = frozen-GN ridge control)'); p.add_argument('--box', type=float, default=1.0, help='kronqf alloc2: per-mode box in units of lr eta; >1 lets the sphere budget sum D^2 <= N bind')
p.add_argument('--alloc', type=int, default=0, help='1: derived per-mode allocation in place of the sign push: D = C / (lam + gamma) / T_rem with gamma solved so that '
               'the layer step length over the remaining T_rem steps equals the budget (T_rem R)^2; no cap, no hypergradient, no live mask')
p.add_argument('--sphere', type=float, default=0.0, help='>0: Euclidean step. Each layer\'s direction is scaled to Frobenius length R_l, and R_l is set by a '
               'controller holding the measured relative Taylor remainder r_f = ||f(th+d) - f(th) - J d|| / ||J d|| of that layer\'s step at this value '
               '(one jvp + one forward, round-robin over layers every --sphere_every steps). The lr decay is replaced by decaying the target.')
p.add_argument('--sphere_every', type=int, default=10)
p.add_argument('--stats_every', type=int, default=0, help='accumulate the factor statistics every k steps (0 = only on refresh steps, the old behaviour)')
p.add_argument('--ekfac', type=int, default=0, help='1: per-mode curvature = EMA of the exact GN diagonal in the Kronecker basis (same-sample pairing), used in the cap and live mask')
p.add_argument('--rot_once', type=int, default=0, help='1: keep the momentum in the rotated basis (exact block transport at refresh): one rotation in, one out per step')
p.add_argument('--tf32', type=int, default=0, help='1: allow TF32 matmuls (tensor cores) for every method')
p.add_argument('--eig_iters', type=int, default=0, help='>0: warm-started subspace iteration (k QR steps per block) instead of eigh after the first refresh')
p.add_argument('--adapt_refresh', type=float, default=0.0, help='>0: error-controlled refresh interval: e = |sum|Gr_new| - sum|Gr_old|| / sum|Gr_new| at a refresh, R <- R clip(kappa/e, 1/2, 2) in [2, 200]')
p.add_argument('--rule', default='sign', choices=['sign', 'adam'], help='kronqf: sign push (KronQ) or the Adam rule in the Kronecker basis (ks_adam)')
p.add_argument('--S_mode', default='mc', choices=['mc', 'emp'], help='kronqf: backprop factor from the MC probe or the empirical signal (rescaled to the GN trace once)')
p.add_argument('--last_S', default='stats', choices=['stats', 'identity'], help='kronqf: output layer S factor from statistics or identity')
p.add_argument('--qa', type=float, default=0.0, help='kronqf: input-side weight exponent on every hidden layer, w *= (alpha_j / alpha_max)^(qa/2) (TRUST_REGION: -1 = equal pre-activation change per mode)'); p.add_argument('--q_floor', type=float, default=1e-4, help='kronqf: floor on alpha_j / alpha_max (and lam / lam_max) in the weight exponents'); p.add_argument('--q_in', type=float, default=0.0, help='kronqf: per-mode weight exponent on the INPUT layer only (decile scan: its box radius goes as 1/leverage, so -1 = equal output change per mode)'); p.add_argument('--q', type=float, default=0.0, help='kronqf: fixed per-mode weight (lam/lam_max)^(q/2) when --hyper 0')
p.add_argument('--b1', type=float, default=0.9); p.add_argument('--eps', type=float, default=1e-8); p.add_argument('--cap', type=int, default=1)
p.add_argument('--centred', type=int, default=0, help='1: train f(x) - f_0(x) with f_0 the frozen initial network (one extra forward per step)')
p.add_argument('--horizon', type=float, default=1.0, help='kronqf hypergradient: eligibility-trace horizon (1 = one-step)')
p.add_argument('--hyper_shape', type=int, default=0, help='kronqf: 1 = learn the per-mode shape only (mean of u removed)')
p.add_argument('--sign_floor', type=float, default=0.0, help='kronqf: soft sign C / (|C| + floor max|C|) instead of sign(C)')
p.add_argument('--topk', type=int, default=0, help='kronqf: push only the k modes with the largest curvature per layer (0 = all live)')
p.add_argument('--lr_kappa', type=float, default=0.0, help='kronqf: per-layer step multiplier eta_l from the measured function-space remainder, held at kappa (KronQ2 --lr_kappa)')
p.add_argument('--check_every', type=int, default=10)
p.add_argument('--eig_dtype', default='float32', choices=['float32', 'float64'], help='kronqf: dtype of the block eigendecompositions (float64 makes the null-mode exclusion real)')
p.add_argument('--eig_adapt', type=int, default=0, help='kronqf: kronsign per-layer refresh period by basis overlap (x2 above 0.99 up to --eig_max, /2 below 0.9)')
p.add_argument('--eig_max', type=int, default=80)
p.add_argument('--dprev_bits', type=int, default=0, help='1: store the previous step as int8 signs and reconstruct its magnitude from the current push weight (hypergradient state 0.25 P instead of 0.5-1 P)')
p.add_argument('--gnls', type=int, default=0, help='1: per-layer line search of the sign step against the EXACT per-sample GN curvature on the batch, '
               'q = (2/(m NOUT)) sum_x (gz(x)^T dW a(x))^2 (needs --bcurv-style per-sample gz; one matmul per layer); scale = min(1, g.D / (lr q))')
p.add_argument('--diag', type=int, default=0, help='1: at every eval step log per layer the u distribution, clamp fraction, cap-active fraction, '
               'step energy on the lowest decile of modes by Kronecker eigenvalue, weight norm, and hidden-activation saturation')
a = p.parse_args()
import subprocess
try: GIT = subprocess.run(['git', '-C', os.path.dirname(os.path.abspath(__file__)), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
except Exception: GIT = 'unknown'
torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32); torch.backends.cudnn.allow_tf32 = bool(a.tf32)
if a.tf32: torch.set_float32_matmul_precision('high')
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
lins0 = None
if a.centred:
    import copy
    lins0 = copy.deepcopy(lins)
    for q in lins0.parameters(): q.requires_grad_(False)
KQ = None; KR = None
if a.opt == 'krylov':
    from krylov_step import KrylovStep
    KR = KrylovStep(lins, a, DEV, torch.nn.functional.gelu)
if a.opt == 'kronqf':
    from kronq_fast import KronQFast
    KQ = KronQFast(lins, a, DEV, NOUT, torch.nn.functional.gelu, lambda shape: torch.randint(0, 2, shape, device=DEV, generator=gtrain).float() * 2 - 1)
def forward0(x):
    h = x
    for l, lin in enumerate(lins0):
        z = lin(h); h = torch.nn.functional.gelu(z) if l < L1 - 1 else z
    return h
def forward(x, keep=False):
    if KQ is not None:
        y = KQ.forward(x, keep)
        return y - forward0(x) if lins0 is not None else y
    h = x
    for l, lin in enumerate(lins):
        if keep: acts_cache[l] = h
        z = lin(h)
        if keep: z.retain_grad(); zs_cache[l] = z
        h = torch.nn.functional.gelu(z) if l < L1 - 1 else z
    return h - forward0(x) if lins0 is not None else h
def mse(out, y): return ((out - y) ** 2).mean()
@torch.no_grad()
def val_mse():
    return float(sum(mse(forward(Xv[i:i + 1024]), Yv[i:i + 1024]) * len(Xv[i:i + 1024]) for i in range(0, len(Xv), 1024)) / len(Xv))
def nbytes(x): return x.numel() * x.element_size()

# ------------------------------------------------------------------ blocked / one-sided eigenbasis helpers
class Side:
    """EMA'd factor on one side of a layer: full / blocked (batched full blocks + one remainder block) / diagonal (dim > max_dim)."""
    def __init__(self, dim):
        self.dim = dim; self.diag = dim > a.max_dim
        b = a.block if 0 < a.block < dim else dim
        self.b, self.nf = b, dim // b; self.rem = dim - self.nf * b
        self.F = None; self.Fr = None; self.U = None; self.Ur = None; self.e = None
    def update(self, Z, beta):           # Z: samples x dim; factor = E[Z^T Z] (or its diagonal)
        m = Z.shape[0]
        if self.diag:
            new = (Z * Z).mean(0)
            if self.F is None: self.F = new
            else: self.F.mul_(beta).add_(new, alpha=1 - beta)
            return
        nb = self.nf * self.b; Zf = Z[:, :nb].reshape(m, self.nf, self.b)
        newF = torch.einsum('mkb,mkc->kbc', Zf, Zf) / m
        newR = (Z[:, nb:].T @ Z[:, nb:] / m) if self.rem else None
        if self.F is None: self.F, self.Fr = newF, newR
        else:
            self.F.mul_(beta).add_(newF, alpha=1 - beta)
            if self.rem: self.Fr.mul_(beta).add_(newR, alpha=1 - beta)
    def eig(self, iters=0):
        if self.diag: self.U = None; self.e = self.F.clamp(min=0).clone(); return
        if self.U is None or iters <= 0:
            ev, V = torch.linalg.eigh(self.F); self.U = V; e = ev.clamp(min=0).reshape(-1)
            if self.rem: evr, Vr = torch.linalg.eigh(self.Fr); self.Ur = Vr; e = torch.cat([e, evr.clamp(min=0)])
        else:                            # warm-started subspace iteration from the previous basis
            V = self.U
            for _ in range(iters): V, _ = torch.linalg.qr(torch.bmm(self.F, V))
            self.U = V; e = torch.einsum('kbc,kbd,kdc->kc', V, self.F, V).clamp(min=0).reshape(-1)
            if self.rem:
                Vr = self.Ur
                for _ in range(iters): Vr, _ = torch.linalg.qr(self.Fr @ Vr)
                self.Ur = Vr; e = torch.cat([e, torch.diagonal(Vr.T @ self.Fr @ Vr).clamp(min=0)])
        self.e = e
    def basis(self): return None if self.U is None else (self.U, self.Ur)
    def overlap(self, Bold, signed=False):   # per-block overlaps (new^T old): squared for second moments, signed for first moments
        if self.U is None or Bold is None: return None
        O = torch.bmm(self.U.transpose(1, 2), Bold[0]); Or = (self.Ur.T @ Bold[1]) if self.rem else None
        return (O, Or) if signed else (O * O, (Or * Or) if self.rem else None)
    def _apply(self, M, side, Bf, Br, transpose):
        """block-diagonal (Bf, Br) applied to M: side 'L' -> B @ M (rows), 'R' -> M @ B^T (columns); transpose=True uses B^T in place of B."""
        nb = self.nf * self.b
        if side == 'L':
            n = M.shape[1]; Mf = M[:nb].reshape(self.nf, self.b, n)
            out = torch.bmm(Bf.transpose(1, 2) if transpose else Bf, Mf).reshape(nb, n)
            if self.rem: out = torch.cat([out, (Br.T if transpose else Br) @ M[nb:]], 0)
        else:
            n = M.shape[0]; Mf = M[:, :nb].reshape(n, self.nf, self.b)
            out = torch.einsum('nkb,kcb->nkc' if not transpose else 'nkb,kbc->nkc', Mf, Bf).reshape(n, nb)
            if self.rem: out = torch.cat([out, M[:, nb:] @ (Br if transpose else Br.T)], 1)
        return out
    def carry(self, T, O, side):
        if O is None: return T
        return self._apply(T, side, O[0], O[1], False)
    def rot(self, M, side, inv):             # inv=False: into the eigenbasis (U^T M rows / M U columns); inv=True: back
        if self.U is None: return M
        return self._apply(M, side, self.U, self.Ur, transpose=not inv)
    def state_bytes(self):
        if self.F is None: return 0
        if self.diag: return nbytes(self.F)
        tot = nbytes(self.F) + (nbytes(self.Fr) if self.rem else 0)
        if self.U is not None: tot += nbytes(self.U) + (nbytes(self.Ur) if self.rem else 0)
        return tot
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
    if a.opt == 'kronqf': return KQ.state_bytes()
    if a.opt == 'krylov': return KR.state_bytes()
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
g_scale = 1.0          # two-sided controller state (--rem_ctrl): multiplier on the whole step
R_l = [None] * L1      # --sphere: per-layer Euclidean step length (initialised from the first step's own length)
def layer_jvp(l, dW, db, x):
    """(f(x), J_l [dW, db]): forward-mode derivative of the network output w.r.t. layer l's weight and bias along (dW, db)."""
    from torch.func import jvp
    def f_l(Wl, bl):
        h = x
        for k, lin in enumerate(lins):
            z = torch.nn.functional.linear(h, Wl, bl) if k == l else lin(h)
            h = torch.nn.functional.gelu(z) if k < L1 - 1 else z
        return h
    return jvp(f_l, (lins[l].weight.detach(), lins[l].bias.detach()), (dW, db))
def rel_remainder(l, dW, db, x, f0):
    """relative Taylor remainder r_f and leverage ||J d|| / ||d|| of the step (dW, db) in layer l, on inputs x (f0 = f(x) before the step)."""
    with torch.no_grad():
        f_lin, Jd = layer_jvp(l, dW, db, x)
        W0, b0 = lins[l].weight.clone(), lins[l].bias.clone()
        lins[l].weight.add_(dW); lins[l].bias.add_(db); f1 = forward(x); lins[l].weight.copy_(W0); lins[l].bias.copy_(b0)
        rf = float((f1 - f0 - Jd).norm() / (Jd.norm() + 1e-30)); lev = float(Jd.norm() / (torch.sqrt(dW.pow(2).sum() + db.pow(2).sum()) + 1e-30))
    return rf, lev
def solve_gamma(Cabs, lam, target2, iters=40):
    """gamma such that sum (Cabs / (lam + gamma))^2 = target2 (decreasing in gamma); bisection in log gamma."""
    lo, hi = -30.0, 30.0
    for _ in range(iters):
        g = 0.5 * (lo + hi); v = float((Cabs / (lam + math.exp(g))).pow(2).sum())
        if v > target2: lo = g
        else: hi = g
    return math.exp(0.5 * (lo + hi))
MECH = None
if a.mech:
    from mechanism import Mechanism
    MECH = Mechanism(torch.nn.functional.gelu, L1, DEV)
def params_orig_any():
    if KQ is not None: return KQ.params_orig()
    return [t.detach().clone() for lin in lins for t in (lin.weight, lin.bias)]
for step in range(1, a.steps + 1):
    mech_now = MECH is not None and (step % a.mech_every == 0 or step == 1)
    if mech_now: p_before = params_orig_any()
    fac = min(1.0, step / a.warmup) if a.warmup else 1.0
    decay = 1 - (step - 1) / a.steps
    if not a.rem_ctrl and not a.sphere: fac *= decay
    lr = a.lr * fac
    xb, yb = draw(a.batch, gtrain)
    sync(); t0 = time.time()
    need_stats = a.opt in ('soap', 'kronq') and (step - 1) % a.refresh == 0
    for q in lins.parameters(): q.grad = None
    if KQ is not None:
        for W in KQ.Wt: W.grad = None
    keep_now = (a.opt == 'kronq' and (need_stats or a.bcurv or a.gnls)) or a.opt == 'kronqf'
    if a.opt == 'krylov':
        t0 = time.time(); KR.step(step, (decay if a.kr_decay else 1.0), xb, yb, emit if a.diag else None, step % a.eval_every == 0 or step == a.steps); sync(); tm['upd'] += time.time() - t0
        out = None
    else:
        out = forward(xb, keep=keep_now); loss = mse(out, yb); loss.backward(retain_graph=keep_now)
    bc = None; gz_b = None
    if a.opt == 'kronq' and (a.bcurv or a.gnls):   # per-sample backprop signal of a random output combination (unbiased GN probe)
        with torch.enable_grad():
            eps_b = torch.randint(0, 2, out.shape, device=DEV, generator=gtrain).float() * 2 - 1
            gz_b = torch.autograd.grad((out * eps_b).sum(), [zs_cache[l] for l in range(L1)], retain_graph=need_stats)
    if a.opt == 'kronq' and a.bcurv:        # current-batch Kronecker curvature per mode: diag(U_S^T S_b U_S) x diag(U_A^T A_b U_A)
        bc = []
        for l, s_ in enumerate(st):
            if s_['L'].U is None: bc.append(None); continue
            at_b = torch.cat([acts_cache[l], torch.ones(acts_cache[l].shape[0], 1, device=DEV)], 1)
            dA = (s_['R'].rot(at_b, 'R', False) ** 2).mean(0); dS = (s_['L'].rot(gz_b[l].T, 'L', False) ** 2).mean(1)
            bc.append((dS[:, None] * dA[None, :]) * (2.0 / NOUT))
    sync(); tm['fb'] += time.time() - t0
    if a.opt == 'kronqf':
        KQ.step(step, lr, out, xb, yb, mse, tm, time.time, emit if a.diag else None)
    with torch.no_grad():
        if a.opt in ('kronqf', 'krylov'): pass
        elif a.opt == 'adamw':
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
                    oldL, oldR = s_['L'].basis(), s_['R'].basis()
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
                    if a.alloc:
                        # derived T-step allocation (frozen-kernel, isotropic budget): per-step length x_k / T_rem with x_k = C_k / (lam_k + gamma),
                        # gamma from sum x_k^2 = (T_rem R)^2, R = lr sqrt(#modes) (the sign push's per-step length); the controller scales R.
                        T_rem = max(10, a.steps - step + 1); R = lr * math.sqrt(lam.numel())
                        if a.sphere and R_l[l] is not None: R = R_l[l] * decay          # the sphere's radius is the budget
                        gam = solve_gamma(C.abs(), lam, (T_rem * R) ** 2)
                        Dr = C / (lam + gam) / T_rem / max(lr, 1e-30)          # in units of lr, like the sign push (the step is -lr * Dw)
                        if a.diag and (step % a.eval_every == 0 or step == a.steps):
                            emit(f'alloc_gamma_l{l}', gam, step); emit(f'alloc_gamma_over_lmax_l{l}', gam / lmax, step)
                            emit(f'alloc_fitfrac_l{l}', float((lam > gam).float().mean()), step)
                        if a.diag and (step % a.eval_every == 0 or step == a.steps):
                            ld = (Dr.abs() / (Dr.pow(2).mean().sqrt() + 1e-30) + 1e-30).log().flatten(); lf = lam.flatten()
                            order = lf.argsort(); nb = 10; n_ = lf.numel()
                            for b_ in range(nb):
                                idx = order[b_ * n_ // nb:(b_ + 1) * n_ // nb]
                                emit(f'ld_bin{b_}_l{l}', float(ld[idx].mean()), step); emit(f'lam_bin{b_}_l{l}', float(lf[idx].mean()), step)
                        if a.hyper: s_['Dprev'] = None
                        Dw = s_['L'].rot(s_['R'].rot(Dr, 'R', True), 'L', True)
                        if a.sphere:
                            if R_l[l] is None: R_l[l] = float(lr * Dw.norm())
                            Dw = Dw * (R_l[l] * decay / (lr * float(Dw.norm()) + 1e-30))
                        lin.weight.sub_(lr * Dw[:, :-1]); lin.bias.sub_(lr * Dw[:, -1])
                        if a.wd: lin.weight.mul_(1 - lr * a.wd)
                        if a.rem > 0 or a.rem_ctrl > 0 or a.sphere: s_['Dw_last'] = Dw
                        continue
                    if bc is not None and bc[l] is not None:
                        lam = bc[l]; lmax = float(lam.max()); live = lam > a.reltol * lmax
                    if a.gmask > 0:
                        gl = Gr.abs() > a.gmask * Gr.pow(2).mean().sqrt()
                        if a.diag and (step % a.eval_every == 0 or step == a.steps):
                            emit(f'glive_l{l}', float(gl.float().mean()), step); emit(f'glive_and_live_l{l}', float((gl & live).float().mean()), step)
                        live = live & gl
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
                        # per-mode weight profile against the eigenvalue: mean log(|D| / rms|D|) per lambda decile (tests the peaked-profile prediction)
                        ld = (Dr.abs() / (Dr.pow(2).mean().sqrt() + 1e-30) + 1e-30).log().flatten(); order = lf.argsort(); nb = 10; n_ = lf.numel()
                        for b_ in range(nb):
                            idx = order[b_ * n_ // nb:(b_ + 1) * n_ // nb]
                            emit(f'ld_bin{b_}_l{l}', float(ld[idx].mean()), step); emit(f'lam_bin{b_}_l{l}', float(lf[idx].mean()), step)
                            if a.hyper: emit(f'u_bin{b_}_l{l}', float(s_['u'].float().flatten()[idx].mean()), step)
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
                if a.opt == 'kronq' and a.gnls:   # exact-GN line search along the step: alpha = min(1, g.D / (lr q)), q from the per-sample probe
                    at_b = torch.cat([acts_cache[l], torch.ones(acts_cache[l].shape[0], 1, device=DEV)], 1)
                    sx = (gz_b[l] * (at_b @ Dw.T)).sum(1)                       # gz(x)^T dW a(x) per sample
                    q = float((sx * sx).mean()) * (2.0 / NOUT); gD = float((Gw * Dw).sum())
                    alpha = min(1.0, gD / (lr * q + 1e-30)) if gD > 0 else 1.0
                    Dw = alpha * Dw
                    if a.diag and (step % a.eval_every == 0 or step == a.steps):
                        emit(f'gnls_alpha_l{l}', alpha, step)
                        qk = float((lam * Dr * Dr).sum()) if a.opt == 'kronq' else float('nan')     # the Kronecker model's curvature along the same step
                        emit(f'gnls_qratio_l{l}', q / (qk + 1e-30), step)
                if a.rem_ctrl > 0: Dw = g_scale * Dw          # two-sided controller: running multiplier on the whole step
                if a.sphere and a.opt == 'kronq':             # Euclidean step: scale the direction to the layer's radius (decayed)
                    if R_l[l] is None: R_l[l] = float(lr * Dw.norm())
                    Dw = Dw * (R_l[l] * decay / (lr * float(Dw.norm()) + 1e-30))
                lin.weight.sub_(lr * Dw[:, :-1]); lin.bias.sub_(lr * Dw[:, -1])
                if a.wd: lin.weight.mul_(1 - lr * a.wd)
                if a.rem > 0 or a.rem_ctrl > 0 or a.sphere: s_['Dw_last'] = Dw
            if a.opt == 'kronq' and (a.rem > 0 or a.rem_ctrl > 0):   # measured output change of the step on the batch, relative to the residual
                if a.alloc and a.rem_ctrl > 0 and g_scale != 1.0:      # alloc applied its step unscaled above; apply the controller's extra factor now
                    for s_, lin in zip(st, lins):
                        lin.weight.sub_((g_scale - 1.0) * lr * s_['Dw_last'][:, :-1]); lin.bias.sub_((g_scale - 1.0) * lr * s_['Dw_last'][:, -1]); s_['Dw_last'] = g_scale * s_['Dw_last']
                df = (forward(xb) - out).norm(); En = (out - yb).norm(); rho = float(df / (En + 1e-30))
                # one-sided budget (--rem): undo the fraction of the step beyond the budget (exact semantics of the fix runs);
                # controller (--rem_ctrl): safety undo beyond twice the target only. NOTE: between 2026-10-12 20:27 and this
                # edit, --rem undid only down to 2 x rem when rho > 2 rem (REPORT 61 item 8); runs with git a36fd54/6e29d3a are affected.
                lim = a.rem if a.rem > 0 else 2.0 * a.rem_ctrl
                if rho > lim:
                    sc = 1.0 - lim / rho
                    for s_, lin in zip(st, lins): lin.weight.add_(sc * lr * s_['Dw_last'][:, :-1]); lin.bias.add_(sc * lr * s_['Dw_last'][:, -1])
                if a.rem_ctrl > 0:                           # damped multiplicative update of the running factor toward the target
                    g_scale *= min(2.0, max(0.5, (a.rem_ctrl / max(rho, 1e-12)) ** 0.5))
                if a.diag and (step % a.eval_every == 0 or step == a.steps): emit('rem_rho', rho, step); emit('g_scale', g_scale, step)
            if a.opt == 'kronq' and a.sphere and step % a.sphere_every == 0:
                # radius controller: measure the relative Taylor remainder of ONE layer's step (round-robin) and move its radius toward the target
                l = (step // a.sphere_every) % L1; s_ = st[l]
                if s_.get('Dw_last') is not None:
                    dW, db = -lr * s_['Dw_last'][:, :-1], -lr * s_['Dw_last'][:, -1]
                    with torch.no_grad():
                        lins[l].weight.sub_(dW); lins[l].bias.sub_(db)              # evaluate the remainder at the pre-step point
                        f0 = forward(xb); rf, lev = rel_remainder(l, dW, db, xb, f0)
                        lins[l].weight.add_(dW); lins[l].bias.add_(db)
                    target = a.sphere * max(decay, 0.05)
                    R_l[l] *= min(2.0, max(0.5, (target / max(rf, 1e-12)) ** 0.5))
                    if a.diag and (step % a.eval_every < a.sphere_every or step == a.steps):
                        emit(f'rf_step_l{l}', rf, step); emit(f'lev_step_l{l}', lev, step); emit(f'R_l{l}', R_l[l], step)
            if a.opt == 'kronq' and a.diag and (step % a.eval_every == 0 or step == a.steps):
                # shape of the Taylor radius in the square layer: relative remainder and leverage along three unit directions at the current step length
                l = 1 if L1 > 2 else 0; s_ = st[l]
                if s_.get('Dw_last') is not None:
                    with torch.no_grad():
                        Dl = s_['Dw_last']; nrm = float(lr * Dl.norm()); f0 = forward(xb)
                        Gd = torch.cat([lins[l].weight.grad, lins[l].bias.grad[:, None]], 1); Rd = torch.randn_like(Dl)
                        for name, Dir in (('step', Dl), ('grad', Gd), ('rand', Rd)):
                            Dn = Dir * (nrm / (float(Dir.norm()) + 1e-30))
                            rf, lev = rel_remainder(l, -Dn[:, :-1], -Dn[:, -1], xb, f0)
                            emit(f'shape_rf_{name}_l{l}', rf, step); emit(f'shape_lev_{name}_l{l}', lev, step)
            sync(); tm['upd'] += time.time() - t0
    if mech_now:
        t0 = time.time(); MECH.run(p_before, params_orig_any(), xb, yb, mse, emit, step, rotate=(KQ.rotate_grads if KQ is not None else None), xv=Xv[:1024], yv=Yv[:1024]); sync(); tm['stats'] += time.time() - t0
    if step % a.eval_every == 0 or step == a.steps:
        v = val_mse(); emit('val', v, step); emit('wall', time.time() - t_start, step)
        if a.diag:
            with torch.no_grad():
                if KQ is not None:                                   # fast path: the trained weights are KQ.Wt (rotated), lins is untouched
                    for l in range(len(lins)): emit(f'wnorm_l{l}', float(KQ.Wt[l].norm()), step)
                    zs = KQ.preacts(Xv[:1024])
                else:
                    for l, lin in enumerate(lins): emit(f'wnorm_l{l}', float(lin.weight.norm()), step)
                    h = Xv[:1024]; zs = []
                    for l, lin in enumerate(lins[:-1]): z = lin(h); zs.append(z); h = torch.nn.functional.gelu(z)
                for l, z in enumerate(zs): emit(f'sat_l{l}', float((z.abs() > 3).float().mean()), step); emit(f'zrms_l{l}', float(z.pow(2).mean().sqrt()), step)
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
