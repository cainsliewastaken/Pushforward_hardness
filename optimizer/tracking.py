"""Does the Kronecker basis track the true per-layer kernel? (2026-10-13; REPORT 61 item 25 plan.) Diagnostics on the
batch, per hidden layer, in the fast path's rotated coordinates (a Kronecker mode is a unit coordinate of Wt[l]):
  (a) coupling: for 64 random modes per lambda decile, the responses J u_k (one jvp each) and their Gram G = R R^T; per
      decile the median over its modes of sum_{k' != k} G_kk'^2 / G_kk^2 (how much of a mode's response is shared with the
      other sampled modes: 0 = the basis decouples them) and the median G_kk / lam_k (the Kronecker curvature's accuracy);
  (b) tracking: the top-k eigen-directions of the true per-layer kernel K_l = J_l J_l^T by Lanczos (vjp + jvp per matvec),
      each mapped to parameter space by one vjp and expanded in the Kronecker basis: the fraction of its energy on the
      top-K Kronecker modes (lambda-sorted) for K in (256, 4096, 65536, 1048576), averaged over the k directions;
  (c) mispricing of the actual step: ||J D||^2 / sum_k lam_k D_k^2 (one jvp)."""
import math, torch
from torch.func import jvp, vjp
from mechanism import lanczos_topk

class Tracker:
    def __init__(self, KQ, k=32, per_dec=64, Ks=(256, 4096, 65536, 1048576), seed=11):
        self.KQ, self.k, self.per_dec, self.Ks = KQ, k, per_dec, Ks
        self.gen = torch.Generator(device=KQ.dev); self.gen.manual_seed(seed)
    @torch.no_grad()
    def run(self, xb, emit, step, Dws=None):
        KQ = self.KQ; L = KQ.L
        for l in range(L - 1):
            if KQ.S[l].e is None or KQ.A[l].e is None: continue
            W0 = KQ.Wt[l].data; shp = W0.shape; lam = (KQ.S[l].e[:, None] * KQ.A[l].e[None, :] * (2.0 / KQ.nout)).flatten()
            order = torch.argsort(lam, descending=True); n = order.numel()
            f_l = lambda Wl: KQ.forward_with(xb, l, Wl)
            # (a) coupling and curvature accuracy per decile
            idx = []; decs = []
            for dec in range(10):
                seg = order[dec * n // 10:(dec + 1) * n // 10]; pick = seg[torch.randperm(seg.numel(), device=seg.device, generator=self.gen)[:self.per_dec]]
                idx.append(pick); decs += [dec] * pick.numel()
            idx = torch.cat(idx); R = []
            with torch.enable_grad():
                for kk in idx.tolist():
                    e = torch.zeros(shp, device=W0.device); e.view(-1)[kk] = 1.0
                    _, Ju = jvp(f_l, (W0,), (e,)); R.append(Ju.reshape(-1))
            R = torch.stack(R, 0); G = R @ R.T; d = torch.diagonal(G); off = (G * G).sum(1) - d * d
            decs_t = torch.tensor(decs, device=W0.device)
            for dec in range(10):
                m = decs_t == dec
                emit(f'trk_coupling_d{dec}_l{l}', float((off[m] / (d[m] ** 2 + 1e-30)).median()), step)
                emit(f'trk_curv_ratio_d{dec}_l{l}', float((d[m] * (2.0 / KQ.nout) / xb.shape[0] / (lam[idx[m]] + 1e-30)).median()), step)
            # (b) tracking of the true kernel's top-k directions
            with torch.enable_grad():
                _, vjp_fn = vjp(f_l, W0)
            N = R.shape[1]
            def matvec(v):
                g = vjp_fn(v.reshape(xb.shape[0], -1))[0]
                with torch.enable_grad():
                    _, Kv = jvp(f_l, (W0,), (g,))
                return Kv.reshape(-1)
            ev, ritz = lanczos_topk(matvec, N, W0.device, m=min(2 * self.k, 64), k=self.k, gen=self.gen)
            fr = {K: 0.0 for K in self.Ks}; cnt = 0
            for i in range(ritz.shape[1]):
                u = vjp_fn(ritz[:, i].reshape(xb.shape[0], -1))[0].flatten(); c2 = u * u; tot = float(c2.sum()) + 1e-30
                cum = torch.cumsum(c2[order], 0)
                for K in self.Ks:
                    if K <= n: fr[K] += float(cum[K - 1]) / tot
                cnt += 1
            for K in self.Ks:
                if K <= n: emit(f'trk_topk_on_K{K}_l{l}', fr[K] / max(cnt, 1), step)
            emit(f'trk_lam1_true_l{l}', float(ev[0]), step); emit(f'trk_lamk_true_l{l}', float(ev[-1]), step)
            # (c) mispricing of the actual step direction
            if Dws is not None:
                D = Dws[l]
                with torch.enable_grad():
                    _, JD = jvp(f_l, (W0,), (D,))
                emit(f'trk_misprice_l{l}', float((JD * JD).sum() * (2.0 / KQ.nout) / xb.shape[0]) / (float((lam.reshape(shp) * D * D).sum()) + 1e-30), step)
