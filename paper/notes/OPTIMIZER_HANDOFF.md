# Optimizer line: big-picture summary and handoff (written 2026-10-12)

For a model or person continuing this work. Read with `THEORY.md` (chain: sections 8-20), `REPORT.md` (step log;
optimizer sections 31-60), `OPTIMIZER_SUMMARY.md` (what worked / did not, with numbers) and `FEASIBILITY.md` (benefit vs
online cost per component). Standing rules are in the repo `CLAUDE.md`; the ones that bite most often are repeated in
section 9.

## 1. The question, in the theory's terms

Training on the squared loss obeys the exact residual equation dE/dt = -K E with the training kernel K = Psi Psi^T / n,
and K is a sum over parameter-linear blocks of (input Gram) * (backprop Gram) (THEORY.md Thm 1-2). An optimizer changes
two things only:

1. **the metric** P: preconditioned flow gives dE/dt = -K_P E with K_P = Psi P Psi^T / n (THEORY.md Thm 17); in the
   eigenbasis of K a per-mode metric changes each mode's rate, so it decides how fast the residual moves down the
   spectrum (the "wave" / front, THEORY.md Prop 8);
2. **the time integration**: step size, momentum, schedule.

The chain theory adds what neither a fixed-kernel nor an infinite-width picture has: the kernel itself moves. Every
stage changes it by Cross + Q with Q >= 0 (stage identity); the weight change of every block is a low-rank coherent core
driven by the shared residual (spike theorem); along the directions the residual has driven, the kernel at T equals the
squared accumulated feature change (feature-sum identity, bracket). Expansions are valid inside a Taylor radius, set by
the second derivative along the step (H_2; THEORY.md Thm 16). The roughness theory says the rough part of the error
loses energy only through the peak feature roughness sigma_R, i.e. only as the kernel is lifted on rough directions.

So the optimizer question is: **choose the per-mode metric and the step so that the slow (deep, rough) modes are fitted
as fast as possible while every step stays inside the Taylor radius** — knowing that the deep modes' eigenvalues are
themselves being lifted by the steps.

## 2. What is established (theory + measurement)

1. **Momentum is gradient flow in rescaled time** (REPORT 53, n = 200). Heavy ball with beta at step eta follows gradient
   flow at t = eta k / (1 - beta) on the overdamped modes; stiff modes oscillate and decay at (1 - sqrt beta) per step;
   stability needs eta lambda_1 < 2(1 + beta). No sqrt(lambda) speedup at fixed beta. Near its stability edge momentum
   beats gradient flow (sharpness capped, top modes damped, tail lifted 2-12x more). This accounts for ~34x of the 31x
   momentum gain over GD. The chain theory applies to momentum with the time rescaling; the edge effect is outside it.
2. **Equal (sign) push per kernel eigenmode compresses the spectrum.** A per-mode step a_k = lr (s_k/s_1)^q sign(m_k)
   (s_k = sqrt(n lambda_k), the Jacobian singular value) gives per-step residual decrease ~ lambda_k^((1+q)/2): GD is
   q = 1, sign push q = 0 (rate ~ sqrt lambda), Gauss-Newton q = -1 (no wave). In the exact per-layer GN eigenbasis this is
   the strongest per-step rule found: at n = 20000, 4x Adam (rough target) and ~50x (smooth) per step (REPORT 58).
3. **The Gauss-Newton cap**: never move a mode further than its own GN step |g_k| / lambda_k (split across layer blocks).
   It stops fitted stiff modes from overshooting; free; worth 4-5x where isolated (REPORT 56).
4. **What limits the push is the network's Taylor radius, not the kernel spectrum.** A frozen-kernel (lazy) simulation
   of the same rule improves monotonically down to q = -1 (REPORT 58, lazy_q.py); the real network fails from q ~ -0.5
   on. The function-space Taylor remainder rf = ||f(th + d) - f(th) - J d|| / ||J d|| measures it cheaply (one jvp + one
   forward on a subset, round-robin over layers). Over a 6 x 7 grid of (q, lr) the optimal ridge lies at roughly constant
   remainder (~0.1 in the exact basis for both targets; REPORT 60): q and lr are two coordinates of one quantity, the
   per-step nonlinearity budget. "q = -0.25 everywhere" was an artefact of a fixed lr.
5. **Per-mode curvature** (REPORT 57). |f''[V_k, V_k]| is nearly flat in leverage (c ~ s^0.14-0.19), so a per-mode
   radius from the SIZE of the remainder predicts q ~ +0.8, opposite to what works. The residual-projected curvature
   r . f''[V_k, V_k] / n (the non-GN part of the Hessian) is negative on 100 % of modes at initialisation (the network
   unlearning its own initial output; THEORY 12.1, Lemma 15) and positive on 93-100 % from step 300 on. The theory gives
   the identity; the sign is measured, not derived.
6. **Local models of the step lose; measuring the step's effect per mode wins.** A trust-region Newton step built from
   the exact GN matrix plus measured residual curvature (Levenberg-Marquardt with measured damping; trgn.py) loses 3-4x
   per step to sign push (REPORT 58). The one-step hypergradient per mode, dL_t/du_k = -lr (G_t)_k D_{t-1,k} for push
   weights w_k = exp(u_k) (IDBD / Autostep style; kronq3h.py), needs no extra passes and matches the best of the whole
   tuned (q, lr) grid on the rough target and beats it by 27 % on the smooth one, insensitive to the base lr over 10x
   (REPORT 59). The per-mode shape it learns is worth 10-45 % over q = 0 at equal lr.
7. **The benefit of pushing deep modes harder is realised within a few steps**: the best q does not become more negative
   at longer horizons, and a one-step hypergradient is as good as a 100-step trace (REPORT 60). An earlier claim that it
   came from delayed, multi-step lift is withdrawn.
8. **Per-mode values cannot be computed ahead of training** (consistent with THEORY 6 / chain section "hidden state",
   visibility and amplification): they are state-dependent and must be measured online; the chain's stage criterion
   says current-state quantities are valid for one stage, which is also the theory for how long cached optimizer
   quantities (factors, bases, weights) can be reused (error-controlled refresh; see section 6).
9. **The fitting wave can be located for free**: the per-mode share of the loss removal, g_k u_k (= lambda_k e_k under
   GD), has a median eigenvalue tracking 1/t late in training, band about one decade wide (REPORT 54-55). Locating it
   did not, however, give a better push rule (windows keyed to it were negative for training error).
10. **Basis**: per-layer bases beat the global (inter-layer, exact GN) basis for equal push — equal push per layer also
    balances layers. The Kronecker (K-FAC) basis, the scalable form, loses ~3x per step to the exact basis on the rough
    L63 target (0.22 vs 0.065) but still beats Adam per FLOP at equal cost on L63 (rough 0.18-0.22 vs 0.41; smooth 0.003
    vs 0.029; Adam lr grid of only 3 values) and, in the other instance's tuned runs, tuned Adam by 1.4x (L63, 10k steps)
    and 2.2-3x on the field problem (equal steps).

Withdrawn / corrected claims are listed in OPTIMIZER_SUMMARY.md ("Corrected claims", "Correction").

Which of the items above are theorems and which are measurements organised by the identities: THEORY.md section 21 (added
2026-10-12). Short version: the basis, the leverage, the cap's form and the remainder detector are theory-derived; sign
push, q, the hypergradient weights, per-layer-beats-global, the remainder level and every speedup factor are measured.
Sign-plus-cap steps are not a kernel flow (no energy identity), the metric moves (dP term in the stage identity for K_P),
and the working regime is at or beyond the stability edge, so the chain's conditional results do not apply to KronQ.

## 3. The current algorithm (KronQ; optimizer/kronq2.py, kronq3h.py, scale_study.py --opt kronq)

Per linear layer with Wt = [W | b] (out x (in+1)):
* factors A = EMA(a~ a~^T) (layer inputs + bias column) and S = EMA(d d^T) (backprop signal; exact over outputs for
  few outputs, one random +-1 output combination per sample otherwise); eigenbases U_A, U_S; mode (i, j) has GN
  eigenvalue lam_ij = eS_i eA_j (times the loss normalisation); modes below reltol * lam_max are null and get no push;
* rotated momentum C = U_S^T M U_A and gradient Gr; step D_ij = w_ij sign(C_ij), capped at |Gr_ij| / (lam_ij lr L);
* w = exp(u) with u learned by the one-step hypergradient (normalised); or w = (lam/lam_max)^(q/2) with fixed q;
* optional per-layer lr from the function-space remainder (round-robin check), optional error-controlled refresh;
* memory reductions: blocks of size b, diagonal treatment of sides wider than max_dim, bf16 / factored hypergradient
  state, bf16 momentum.
Cost: ~3.1 forward-pass equivalents per step at width 96 (Adam: 3). Memory (state per parameter): full ~8 (square
layers) to ~12.5 (4x-expansion layers); blocked + compressed ~1.1-2.2 (smoke values: full 12.0, b + compress 2.2 at
width 64). Full Kronecker factors are quadratic in width: one factor on a 32768-wide side is ~4 GB fp32 per layer; this
is the scaling risk (section 5).

## 4. What did not work (pointers)

SegFlow / Krylov segments (wins on passes, loses per FLOP; REPORT 31-41); Krylov recycling; minibatch segments; Muon on
smooth field regression (loses 2-8x; 42-47); roughness-supply objective, truncated spectral update, per-layer supply
allocation, band-wise output-roughness preconditioning (48-52); push windows keyed to the Rayleigh front or the wave
(54-55); front-damped and per-mode critically damped momentum (54); loss-based trust-region lr controller (56);
trust-region Newton, exact and low-rank (58); global GN basis for equal push; RMS push (persistence-weighted).

## 5. Scaling: status and the study that is set up but not yet run

The method must keep its advantage once its memory is brought down to Adam's level; if it cannot, it does not scale
(user criterion, 2026-10-12). Optimizer-state memory per parameter: AdamW 2, Muon ~1, SOAP ~6 (square) / ~10.5 (4x),
KronQ full ~8 / ~12.5, KronQ blocked + compressed ~1.1-1.4 (analytic, at b = 256-1024). Untested: the accuracy cost of
blocking at large width, of the Monte-Carlo S factor, of compressed hypergradient state, of minibatch noise.

**Scaling study** (optimizer/scale_study.py, scale_jobs.py, scale_launch.sh): field problem (16x16 GRF input,
target tanh(2 N[(-Lap)^p u]), p = 0 smooth / 2 rough), fresh minibatch every step (online, no overfitting), MLP
256 -> w x 2 -> 256 GELU fp32, widths 256-4096, 5000 steps, batch 256; AdamW (3 lr), Muon (3 lr), SOAP (2 lr x full /
block 256), KronQ with hypergradient (2 lr x full / b1024 / b256 / b64 / b256+compressed+bf16 / diagonal sides > 1024);
220 jobs in $SCRATCH/l63_held_rough/runs/scale/jobs.txt (longest first). Logs: validation MSE vs step and wall time,
time split (fwd/bwd, statistics, eigh, update), peak GPU memory, exact optimizer-state bytes. Launch on an allocation
(one srun task per node, 3 runs per GPU):
  salloc --no-shell -A m5326 -C gpu -q interactive -N <nodes> -t <time> --image=nersc/pytorch:26.01.01 -J l63_scale
  srun --jobid=<id> --overlap -N <nodes> --ntasks-per-node=1 --gpus-per-node=4 shifter --image=nersc/pytorch:26.01.01 \
       bash optimizer/scale_launch.sh 3
Estimated ~9 GPU-hours total; RUN 2026-10-12 on job 59722234 (m5326, interactive, 4 nodes, 36 min, approved in chat).
RESULTS: REPORT.md section 61 (tables in runs/scale/summary/). Verdict against the user's criterion: KronQ as built does NOT
scale. At width 4096 the variant at Adam's memory (b256 + compress, 2.0 state/P) collapses to the constant predictor on the
rough target and is 32x worse than full KronQ on the smooth one; every memory reduction's cost grows with width; the usable
lr shrinks with width (3e-3 fails everywhere >= 1024, only 1e-3 works at 4096 and only for full / b1024); and SOAP with
blocks of 256 (2.5 state/P) is the best method at 4096 on the rough target (0.083 vs KronQ full 0.108) with SOAP full best
on the smooth one (4.5e-5 vs 1.2e-4). KronQ's lead over SOAP at widths 256-2048 reverses at 4096. DIAGNOSED (job 59723886, REPORT 61 item 4): not the
hypergradient; the blocked / diagonal factors have no null modes, so in the first refresh windows every mode is sign-pushed
while the full basis pushes ~6-50 %, the second hidden layer's preactivations blow up (rms 50-185, 100 % GELU saturation by
step 250), the GN cap then binds everywhere and the run is frozen at the mean predictor. Fixes to test (needs a yes):
reltol from the sampling noise floor / no push before dim samples, a per-layer step budget from the function-space
remainder, or the cap from full-factor eigenvalues; ~1 GPU-node x 45 min; if the reductions cannot be made to work at Adam's memory, the line stops here per the criterion.
CAVEAT on the baselines: three fixed lrs, no warmup / schedule / weight decay, beta2 = 0.99 for AdamW, the same lr grid at
every width. The study decides whether the memory reductions cost accuracy at large width (they do); it does NOT decide
"beats tuned AdamW / Muon / SOAP" by the standard of section 7 (THEORY.md 21(d)).

### Pending decisions (awaiting the user, as of 2026-10-12 evening)

1. **Scaling-study allocation: DONE (job 59722234, 2026-10-12).** Was: account m5326, QOS interactive, 4 GPU nodes (16 GPUs,
   3 runs per GPU), 1 hour limit, released automatically by chaining `scancel` after the launcher finishes; expected
   use ~30-40 min (~9 GPU-hours). Commands (fill in the job id):
   `salloc --no-shell -A m5326 -C gpu -q interactive -N 4 -t 01:00:00 --image=nersc/pytorch:26.01.01 -J l63_scale`, then
   `nohup bash -c 'srun --jobid=<id> --overlap -N 4 --ntasks-per-node=1 --gpus-per-node=4 shifter --image=nersc/pytorch:26.01.01 bash optimizer/scale_launch.sh 3 > $SCRATCH/l63_held_rough/runs/scale/all.log 2>&1; echo "exit=$?" > $SCRATCH/l63_held_rough/runs/scale/all.log.done; scancel <id>' &`.
   Do not launch without a fresh explicit yes (the global rules require it per run).
2. **Pushing** commits 1ce5411 and 569bcbe (and later ones) to origin: not yet approved.
3. **Removing** `$SCRATCH/l63_held_rough/code` and `report/`: only after the other pre-repository session has ended and
   `scripts/sync_l63_code.sh` reports nothing to copy; needs the user's OK. The run tree `runs/` stays.

## 6. Open questions, in priority order

1. Scaling study (above): does KronQ keep an advantage over tuned AdamW / Muon / SOAP at <= Adam's memory and equal
   wall-clock, at large width, with minibatch noise? This decides whether the line continues.
2. The Kronecker-vs-exact gap (~3x per step on rough L63): what the K-FAC approximation loses (inter-layer terms? the
   Kronecker factorisation of each layer's GN?), and whether it grows or shrinks with width.
3. What sets the optimal remainder level (exact basis ~0.1 on both targets; Kronecker basis ~0.1-0.2 rough, ~0.02
   smooth). If it is set by a measurable quantity, the per-layer remainder controller needs no tuning at all.
4. Theory: derive the per-mode value that the hypergradient measures from the chain (stage identity with the step's
   own second-order term); derive the sign of the residual curvature after the transient (Lemma 15); explain why sign
   push beats magnitude-based steps (both are steepest descent, in l_inf vs l_2 geometry of the mode coordinates).
5. Tuned baselines with weight decay and schedules; per-FLOP and per-second comparisons; the other instance's SOAP
   tuning (runs/kadam/blocktest/tunesmoke) and block sweep (block_sweep.py) results.
6. Field / climate-like problems at large output dimension: number of stiff modes (spikes above the damping) vs output
   dimension; Martin-Mahoney-type spectra (bulk + spikes vs heavy tail) of the per-layer GN factors along training.

## 7. State of the external landscape (searched 2026-10-12)

LLM pretraining: AdamW default; Muon, SOAP and Kron give ~1.1-1.4x over a well-tuned AdamW (Fantastic Pretraining
Optimizers, arXiv 2509.02046), up to 20-30 % for a Muon variant growing with training length (arXiv 2606.16899); at
multi-billion scale SOAP and Muon beat AdamW and stay efficient at batches where AdamW degrades (arXiv 2607.20548).
Weather / climate models train with AdamW (GraphCast: beta2 0.95, wd 0.1, clipping, 12-step rollouts). On fluid-dynamics
diffusion training Muon / SOAP reach 18 % lower loss than AdamW (arXiv 2510.19376); a later controlled benchmark finds no
optimizer wins everywhere (arXiv 2609.23055). The bar for this line: beat tuned AdamW, Muon and SOAP per FLOP at scale.

## 8. Files and runs

optimizer/: kadam_front.py (exact-basis sign push, q, GN cap, windows, wave, remainder detector, per-layer lr
controller, centred init), trgn.py (trust-region Newton, exact and low-rank), kronq.py / kronq2.py (Kronecker sign push;
kronq2: blocks, diagonal sides, per-layer lr from the remainder, error-controlled refresh), kronq3h.py (kronq2 +
per-mode hypergradient weights), lazy_q.py (frozen-kernel simulation), momentum_modes.py, mode_curvature*.py,
mode_stats.py, gamma_init.py, q_grid_summary.py, horizon_q.py, front_summary*.py, scale_study.py / scale_jobs.py /
scale_launch.sh. Run trees: $SCRATCH/l63_held_rough/runs/{momentum, front (p1-p7: windows, wave, q, n = 20000, q grid,
hypergradient), scale}. The other instance's KAdam / KronSign / SOAP tuning is in runs/kadam/.

## 9. Lessons and pitfalls (from this line's history)

* Ask before every allocation (account m5326 so far; state nodes, GPUs, time, QOS); release by chaining scancel after
  the work; pass --image to shifter inside srun (salloc's --image is not inherited by `shifter` in a step).
* Perlmutter GPU nodes in this allocation pool can have 40 GB A100s: two exact-basis jobs plus a trust-region job on one
  GPU ran out of memory; chunk Jacobians.
* `pkill -f <pattern>` matches the invoking shell when the pattern is in its command line; kill by PID.
* Use n in the thousands; at n = 2000 the rough target overfits 2-3x and "validation gains" were artefacts. Even at
  n = 20000 the rough L63 map overfits for strong optimizers; the scaling study uses fresh data every step.
* The residual Rayleigh quotient is not the fitting front under sign push (oscillating fitted modes dominate it); use
  the loss-removal share. Loss-ratio step controllers are fragile when the step is misaligned with the gradient; use
  the function-space remainder.
* When a design choice is not derived, say so; several "theory-based" predictions here failed because the second-order
  term was treated as pure error. Measure the quantities the theory leaves open (signs, levels) rather than modelling them.
