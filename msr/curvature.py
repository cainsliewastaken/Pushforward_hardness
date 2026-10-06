"""Largest Gauss-Newton eigenvalue of an unrolled training loss (same quantity as trainability.py's lam0).

L = sum_q w_q (1/2) ||e_q||_n^2 / sum_q w_q on the starts uT[:, :n]; its Gauss-Newton matrix is
G = sum_q (w_q / sum w) (1/n) Psi_q^T Psi_q, applied matrix-free (one JVP and one VJP of the weighted rollout).
"""
import math

import torch
from torch.func import jvp, vjp

from .measure import _top_eig


def gn_lam_max(m, th, uT, p, w=None, tol=1e-4):
    n = uT.shape[1]
    W = torch.ones(p, device=uT.device) if w is None else torch.tensor(w, device=uT.device)
    sw = torch.sqrt(W / W.sum())
    th = th.detach()
    f = lambda t: (sw[:, None, None] * m.rollout(t, uT[0], p)[1:]).reshape(-1) / math.sqrt(n)
    _, vjp_fn = vjp(f, th)
    return _top_eig(lambda x: vjp_fn(jvp(f, (th,), (x,))[1])[0], m.P, uT.device, tol=tol)
