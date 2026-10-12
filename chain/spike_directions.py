"""Are the final spike directions in the span of initial per-mode gradient features? For each layer l:
 top-r right/left singular vectors of dW_l = W_T - W_0; init gradient feature of K0-mode k: G_k^(l) = reshape(Psi_l^T v_k)
 (an H_out x H_in matrix). Report: fraction of ||dW_l||_F^2 (and of its rank-r part) captured by projection onto
 the span of {G_k, k<=m} for m = 10, 50, 100, 300, 600, and onto the single direction G(E0) (init gradient)."""
import numpy as np, sys, torch
from l63test import make_data
from l63test_torch import MLP, kernel, DEV
torch.set_default_dtype(torch.float64)
fn, Delta, depth, H = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = np.load(fn, allow_pickle=True).item(); W0 = [torch.tensor(w, device=DEV) for w in d['W0']]; WT = [torch.tensor(w, device=DEV) for w in d['WT']]
n = 200; Xn, Yn = make_data(n, Delta); X = torch.tensor(Xn, device=DEV); Y = torch.tensor(Yn, device=DEV); Yv = Y.reshape(-1)
net = MLP(3, H, depth, 3, seed=0)
for l in range(len(W0)): assert torch.allclose(net.W[l], W0[l])
J, out = net.jacobian(X); E0 = out.reshape(-1) - Yv; K0 = kernel(J, n); w, V = torch.linalg.eigh(K0); V = V.flip(1); w = w.flip(0)
sizesW = [W.numel() for W in net.W]; offs = np.cumsum([0] + sizesW)
G = J.T @ V            # d x 600 : column k = gradient feature of mode k (all params)
gE = J.T @ E0 / n
print(f"{fn}: H{H} d{depth} D{Delta}")
for l in range(len(W0)):
    dW = (WT[l]-W0[l]).reshape(-1); dWn = dW/dW.norm()
    Gl = G[offs[l]:offs[l+1], :]                      # layer-l block, (H_out*H_in) x 600
    U, S, Vh = torch.linalg.svd((WT[l]-W0[l]), full_matrices=False)
    top4 = sum((U[:, i:i+1]*S[i]) @ Vh[i:i+1] for i in range(min(4, S.numel()))).reshape(-1); top4n = top4/top4.norm()
    res = []
    for m in [1, 10, 50, 100, 300, 600]:
        Q, _ = torch.linalg.qr(Gl[:, :m])                # orthonormal basis of span of first m mode-gradients
        res.append((float(((Q.T @ dWn)**2).sum()), float(((Q.T @ top4n)**2).sum())))
    gl = gE[offs[l]:offs[l+1]]; gl = gl/gl.norm()
    # also: span of ALL mode gradients = row space of Psi_l (dimension <= 600); complement = directions invisible to K0
    print(f"  layer {l} (|dW|={float(dW.norm()):.2f}, top-4 frac {float(top4.norm()**2/dW.norm()**2):.2f}): overlap^2 with init full-gradient dir: dW {float((gl@dWn)**2):.3f}, top4 {float((gl@top4n)**2):.3f}")
    print("     fraction of dW (top4) in span of init gradient features of modes 1..m:  " + '  '.join(f"m={m}: {a:.2f} ({b:.2f})" for m, (a, b) in zip([1, 10, 50, 100, 300, 600], res)))
