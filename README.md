# multistep_roughness

Lorenz-63 experiments on multistep (unrolled) training of learned steppers: how unrolling changes
the learned map's contraction along the rollout error, the injected and retained error, and the
rollout training kernel, at a fixed network and several emulator steps $\Delta t$.

Code: `~/multistep_roughness` on Perlmutter (git). Outputs: `$SCRATCH/multistep_roughness`
(symlink `results`). Results and every number: `results.md`.

## System and data

- Lorenz-63, $\sigma=10$, $\rho=28$, $\beta=8/3$. Truth map $G$: RK4 with 10 substeps per emulator step $\Delta t$.
- Starts on the attractor: independent random initial conditions $\mathcal N((0,0,25), 8^2 I)$ run for 20 time units (RK4, step 0.01).
- Training: 131,072 starts (`data_seed_train`), each with its true trajectory over the training window (0.64).
  Held-out: 4096 starts (`data_seed_test`) with true trajectories to $T=1.28$.
- Float64 everywhere.

## Models

Same network for every $\Delta t$: MLP $3\to256\to256\to256\to3$, SiLU, inputs standardized
(mean/std of training states). Hidden layers PyTorch-default uniform init, output layer zero; for a
given seed both classes start from the same parameters.

- Integration-constrained (Euler): $\tilde G(u) = u + \Delta t\,H(u;\theta)$, $H = s_f\odot N$.
- Direct: $\tilde G(u) = \mu + s\odot N(u;\theta)$.

$N$ is the network; $s_f$ (std of $f$), $\mu$, $s$ (mean/std of states) are fixed output scales so
$N$ is $O(1)$. The tendency of either model is $H(u) = (\tilde G(u)-u)/\Delta t$ (for the direct model
this is the definition used in M5 and M6).

## Notation (names used in code)

- $u_T(t_i)$ true trajectory, $u_p(t_0)=u_T(t_0)$, $u_p(t_{i+1})=\tilde G(u_p(t_i))$ (`up`).
- $e(t_i) = u_T(t_i)-u_p(t_i)$ (`e`); $\epsilon_i = G(u_T(t_i)) - \tilde G(u_T(t_i))$ (`eps`).
- $|a| = (\frac1n\sum_k |a_k|^2)^{1/2}$ over starts (`norm`); per-start Euclidean norms where a max over starts is taken (M5, M6).
- $J_i = \nabla\tilde G(u_p(t_i))$; $\bar B_i = \int_0^1 \nabla\tilde G(u_p(t_i)+\tau e(t_i))\,d\tau$ by Gauss–Legendre (`gl_points`, 10; see results.md for 5).
- $\hat P(t_p,t_i) = J_{p-1}\cdots J_{i+1}$, $\hat P(t_p,t_{p-1}) = I$. $\mathbf q_i = (\bar B_i - J_i)\,e(t_i)$.
- Exact identity: $e(t_{i+1}) = J_i e(t_i) + \epsilon_i + \mathbf q_i$, so $e(t_p)=\sum_{i<p}\hat P(t_p,t_i)(\epsilon_i+\mathbf q_i)$.
- $\Psi^{(p)} = \partial_\theta u_p(t_p)$; $K^{(p)} = \frac1n\Psi^{(p)}\Psi^{(p)\top}$.
- Contraction along the error: $n_i = e(t_i)/|e(t_i)|$ per start, $\rho_i = n_i^\top J_i n_i$, $\bar\rho_i = n_i^\top \bar B_i n_i$, $\rho^f_i = n_i^\top\nabla G(u_T(t_i)) n_i$, $i\ge1$. For Euler $\rho_i - 1 = \Delta t\, n_i^\top\nabla H n_i$.

## Losses and velocities

- One-step: $\frac12\langle|\epsilon|^2\rangle$ on true pairs $(u_T(t_i),u_T(t_{i+1}))$, one pair per start in the batch, $t_i$ uniform in $[0, 0.64)$.
- Unrolled: $\mathcal L_p = \frac1p\sum_{q=1}^p \frac12|e(t_q)|^2$ from the batch starts, no per-step weighting.
- Velocities (negative gradients): $\hat g^{(q)} = -\nabla_\theta\frac12|e(t_q)|^2$; $\hat g^{(1)}$; $\hat g^{(\Sigma)}(p) = \frac1p\sum_{q\le p}\hat g^{(q)}$.

## Training protocols (per class, $\Delta t\in\{0.01,0.02,0.04,0.08\}$, seeds 0, 1, 2)

1. `onestep`: one-step loss.
2. `unroll032_scratch`, `unroll064_scratch`: $\mathcal L_p$ with $T_{\rm train} = 0.32, 0.64$ ($p = T_{\rm train}/\Delta t$), from the seed's initialization.
3. `unroll032_ft`, `unroll064_ft`: the same losses, fine-tuned from the converged `onestep` model.

Optimizer: Adam, minibatch 1024 starts drawn afresh each step (independently per model), gradient-norm clipping at 1 per model, no weight decay.
Schedule: linear warmup (1000 steps) then cosine to $10^{-6}$. From scratch: peak $10^{-3}$, 100,000 steps. Fine-tuning: peak $10^{-4}$, 20,000 steps.
Checkpoints: step 0, 10 log-spaced steps in $[10, S)$, and the final step $S$.

## Measurements (every checkpoint, held-out starts, $T\in\{0.08,0.16,0.32,0.64,1.28\}$, $p=T/\Delta t$)

- M1: $I(T) = |\sum_{i<p}\epsilon_i|$, $R(T) = |e(t_p)| - I(T)$.
- M2: means over starts and $1\le i\le p-1$ of $\rho_i-1$, $\bar\rho_i-1$, $\rho^f_i-1$; $\langle\sum_{1\le i<p}\ln|\bar\rho_i|\rangle_k$ (and the fraction of $\bar\rho_i\le0$).
- M3: $c_\Sigma(p) = \cos\angle(\hat g^{(1)},\hat g^{(\Sigma)}(p))$ and $\cos\angle(\hat g^{(1)},\hat g^{(q)})$ for every $p, q \le 1.28/\Delta t$ (one forward graph, batched backward passes); first 1024 held-out starts.
- M4: $|\sum_i\hat P\epsilon_i|$, $|\sum_i\hat P\mathbf q_i|$, $|e(t_p)|$, ratio $|\sum\hat P\mathbf q|/|e(t_p)|$ (recursions $S_{i+1} = J_iS_i+\epsilon_i$), and the reconstruction error.
- M5: $\mathcal R_e(H) = \max_{k,i}|n_i^\top(H(v_i)-H(\bar u_i))|/|e_k(t_i)|$, $v_i=u_p(t_i)$, $\bar u_i = u_T(t_i)$; compared with $\max_{k,i}(1-\rho_i)/\Delta t$; and $\mathcal R_e(H-f_{\Delta t})$, $f_{\Delta t}(u) = (G(u)-u)/\Delta t$.
- M6: $M$ with rows $n_i^\top(\partial_\theta H(v_i)-\partial_\theta H(\bar u_i))/|e_k(t_i)|$ over $(k, 1\le i\le p-1)$, first 1024 held-out starts: spectral norm (Lanczos on $M^\top M$, matrix-free) and largest row norm.
- M7: $\lambda_{\max}(K^{(p)})$ (Lanczos on $\Psi\Psi^\top$ via JVP/VJP), $\mathrm{tr}K^{(p)}/\lambda_{\max}$ (exact, batched VJPs), $|\mathcal P^\top\hat e|$ (adjoint pass with the 3×3 $J_i$); first 512 held-out starts.

Everything is matrix-free; $K^{(p)}$ is applied through JVP/VJP rather than formed.

## Departures from the original brief (and why)

| Brief | Here | Why |
|---|---|---|
| 1 hidden layer, tanh, $w=64$ | 3 × 256 SiLU (133k params), same at every $\Delta t$ | A realistic emulator; smooth activation keeps $\bar B$, $\mathbf q$ meaningful. Width check in results.md. |
| $n=128$ training starts, batch 1024 | 131,072 starts, fresh batch of 1024 each step | 128 starts is too little data to defend; a batch larger than the dataset is not defined. |
| 128 held-out starts | 4096 (M1, M2, M4, M5), 1024 (M3, M6), 512 (M7) | Smaller sampling error; M7 matrix-free so not limited to 128. |
| Adam 1e-3 → 1e-4, 8000 + 4000 steps | Warmup + cosine, 100k steps (scratch), 20k (fine-tune), clipping 1 | Runs trained to a plateau (training curves in `figs/fig0_training.pdf`). |
| Protocols (ii), (iii) | Each run both from scratch and fine-tuned from the one-step model | Fine-tuning is standard practice; from scratch is the cleaner comparison. |
| 5-point Gauss–Legendre | 10-point | With attractor-sized errors 5 points leaves ~1e-7 quadrature error; 10 gives ≤1e-11 (identity check). |
| CPU, SLURM array | One GPU node (4 A100s), `sweep.py` scheduler | Training cost. |

## Running

All Python runs in Shifter `nersc/pytorch:26.01.01`.

```
python checks.py                       # identity checks (writes results/checks.json)
python sweep.py --gpus 4 --per-gpu 4   # all training + measurement (skips finished tasks)
python aggregate.py                    # figs/*.pdf, tables/*.csv, tables.md
```

Single pieces: `train.py --dt 0.01 --proto unroll064 --init ft`, `measure_run.py` with the same
arguments. Any config entry can be overridden with `--set key=value` (e.g. `--set "hidden=[512,512,512]" out_root=...`).
