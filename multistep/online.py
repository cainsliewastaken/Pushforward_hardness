"""Online choice of the nudge weight kappa (best_kappa_theorem.tex, rules R1-R3), shared by ks/train.py and
turb2d/train.py. kappa = 1 is the one-step loss, kappa = 0 the free loss (the objective V).

At the start of every window of m steps, at the current state theta and Adam state:
  rates:  for each candidate kappa, the exact Adam step the optimizer would take for L_kappa (current moments, bias
          correction, gradient-norm clipping) and the first-order decrease of V per unit of motion,
          r_kappa = -<g_V, dtheta_kappa> / |dtheta_kappa|          (Lemma onestep; one backward pass per candidate);
  R1 admissibility: the candidate run on the previous window is admissible iff the held-out V did not increase over
          it; any other candidate is admissible iff lr * (Adam-preconditioned top eigenvalue of Hess L_kappa) is
          below margin * (2 + 2 b1)/(1 - b1) at the state; kappa = 1 is always admissible;
  R2 choice: the admissible candidate with the largest rate;
  R3 back-off: after a window on which the held-out V increased, that kappa and every smaller one are inadmissible on
          the next window.
Logged per window: rates, sharpness (where computed), admissibility, the choice, the held-out V, the rms drift of the
nudged rollout for each candidate on the rate batch (so the dependence of the choice on the current drift is visible),
and the rms one-step error.
"""
import json
import time

import torch


class OnlineKappa:
    def __init__(self, loss_fn, rollout_fn, th, opt, p, grid, margin=0.8, sharp_iters=10, clip=1.0, log_path=None, dev="cpu"):
        self.loss_fn, self.rollout_fn, self.th, self.opt, self.p = loss_fn, rollout_fn, th, opt, p
        self.grid = sorted([float(k) for k in grid], reverse=True)          # 1 first
        assert self.grid[0] == 1.0
        b1 = opt.defaults["betas"][0]
        self.thr = margin * (2 + 2 * b1) / (1 - b1)
        self.sharp_iters, self.clip, self.dev = sharp_iters, clip, dev
        self.prev_kappa, self.prev_V, self.banned = None, None, set()
        self.log_path = log_path
        self.rows = []

    # ---- exact Adam step for a gradient g from the current optimizer state (clipping, momentum, bias correction)
    def adam_step(self, g, lr):
        b1, b2 = self.opt.defaults["betas"]
        eps = self.opt.defaults["eps"]
        cf = min(1.0, self.clip / (float(g.norm()) + 1e-300))
        gc = cf * g
        st = self.opt.state.get(self.th, {})
        n = int(st["step"]) + 1 if "step" in st else 1
        m = st["exp_avg"] if "exp_avg" in st else torch.zeros_like(g)
        v = st["exp_avg_sq"] if "exp_avg_sq" in st else torch.zeros_like(g)
        m1 = b1 * m + (1 - b1) * gc
        v1 = b2 * v + (1 - b2) * gc * gc
        return -lr * (m1 / (1 - b1 ** n)) / ((v1 / (1 - b2 ** n)).sqrt() + eps)

    def grad(self, U, kap):
        t = self.th.detach().clone().requires_grad_(True)
        L = self.loss_fn(t, U, self.p, kap)
        return torch.autograd.grad(L, t)[0], float(L.detach())

    # ---- Adam-preconditioned sharpness of L_kappa at the state (power iteration on P^{1/2} H P^{1/2})
    def sharpness(self, U, kap):
        st = self.opt.state.get(self.th, {})
        if "exp_avg_sq" not in st or int(st.get("step", 0)) == 0:
            s = torch.ones_like(self.th)
        else:
            b2 = self.opt.defaults["betas"][1]
            vhat = st["exp_avg_sq"] / (1 - b2 ** int(st["step"]))
            s = (1.0 / (vhat.sqrt() + self.opt.defaults["eps"])).sqrt()

        def mv(v):
            x = self.th.detach().clone().requires_grad_(True)
            gr = torch.autograd.grad(self.loss_fn(x, U, self.p, kap), x, create_graph=True)[0]
            return s * torch.autograd.grad(gr @ (s * v), x)[0]

        def power(f):
            g = torch.Generator(device=self.dev).manual_seed(0)
            v = torch.randn(self.th.numel(), device=self.dev, generator=g, dtype=self.th.dtype)
            v /= v.norm()
            lam = 0.0
            for _ in range(self.sharp_iters):
                w = f(v)
                lam = float(v @ w)
                v = w / (w.norm() + 1e-300)
            return lam
        lam1 = power(mv)
        return lam1 if lam1 >= 0 else power(lambda v: mv(v) - lam1 * v) + lam1

    @torch.no_grad()
    def drift(self, U, kap):
        seq = self.rollout_fn(self.th.detach(), U, self.p, kap)              # rows 1..p
        d = (U[1:self.p + 1] - seq)
        return float(torch.sqrt((d * d).flatten(2).mean(-1).mean()))

    # ---- one decision
    def choose(self, U, lr, V_held, step):
        t0 = time.time()
        gV, LV = self.grad(U, 0.0)
        row = dict(step=step, lr=lr, V_held=V_held, rate={}, sharp={}, adm={}, drift={}, L={})
        with torch.no_grad():
            d = U[1:self.p + 1] - self.rollout_fn(self.th.detach(), U, self.p, 1.0)
            row["rms_eps"] = float(torch.sqrt((d * d).flatten(2).mean(-1).mean()))
        # R1 / R3
        increased = self.prev_V is not None and V_held is not None and V_held > self.prev_V
        banned = set()
        if increased and self.prev_kappa is not None:
            banned = {k for k in self.grid if k <= self.prev_kappa}
        best, best_rate = 1.0, -float("inf")
        for k in self.grid:
            g, L = self.grad(U, k)
            dth = self.adam_step(g, lr)
            r = float(-(gV @ dth) / (dth.norm() + 1e-300))
            row["rate"][k], row["L"][k], row["drift"][k] = r, L, self.drift(U, k)
            if k == 1.0:
                adm = True
            elif k in banned:
                adm = False
            elif k == self.prev_kappa and self.prev_V is not None and V_held is not None:
                adm = not increased
            else:
                sh = self.sharpness(U, k)
                row["sharp"][k] = sh
                adm = lr * sh < self.thr
            row["adm"][k] = adm
            if adm and r > best_rate:
                best, best_rate = k, r
        row["choice"], row["increased"], row["secs"] = best, increased, time.time() - t0
        self.prev_kappa, self.prev_V = best, V_held
        self.rows.append(row)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        rates = " ".join(f"{k:g}:{row['rate'][k]:+.2e}{'' if row['adm'][k] else 'x'}" for k in self.grid)
        print(f"[online] step {step} V_held {V_held if V_held is None else round(V_held, 5)} inc {increased} -> kappa {best:g} | rates {rates} | "
              f"sharp*lr {' '.join(f'{k:g}:{lr * v:.1f}' for k, v in row['sharp'].items())} | drift {' '.join(f'{k:g}:{v:.3g}' for k, v in row['drift'].items())} | eps {row['rms_eps']:.3g} ({row['secs']:.0f}s)", flush=True)
        return best
