# optimizer/llm/: KronQ for transformers and language models

`kronq_lm.py` (class `KronQLM`) runs the main optimizer's method (`../kronq_main.py`, class `KronQ`) on any model built from
`nn.Linear` layers. It imports kronq_main's basis, QR and per-mode update (`_Basis`, `_update_c`, `_fp32`) and leaves that file
untouched, so the method is the same code. The learned per-mode weight, the hypergradient (rate 0.01) and the clamp (|u| <= 4.6) are
kronq_main's. There is no budget and no cap. Added here:

| Need | What KronQLM does |
|---|---|
| Any architecture | Finds every `nn.Linear` by module traversal and reads it with forward hooks. The bias is handled as in kronq_main (`[W \| b]`). Weight sharing is refused. |
| Parameter split (as SOAP / Muon) | KronQ on the hidden matrices (attention, MLP). AdamW on embeddings, the output head (tied, or listed in `exclude`), norms, and any Linear wider than `max_dim`. AdamW runs at `lr_adam * lr / lr_max`, with no decay on 1-D parameters. |
| Curvature probe for the loss | `loss='mse'`: random +-1 output signs, kronq_main's Gauss-Newton probe. `loss='ce'`: labels sampled from the model's softmax (Monte Carlo Fisher). `probe='emp'`: the loss's own output gradients (empirical Fisher). This needs no extra backward but measures a different curvature, so it is a cost option to test, not the method. |
| Scale: probe cost | The statistics come from ONE micro-batch per statistics step; the other micro-batches only accumulate gradients. The probe's extra backward costs about 1/3 of one micro-batch, not of the step. |
| Scale: memory | The input factor is formed in the forward hook (no activation kept for it). The output factor is consumed by tensor hooks as the probe's gradient passes each layer, so no layer-output gradient is stored. The step runs one layer at a time, so there is no fp32 copy of all gradients. |
| bf16 | Use autocast with fp32 parameters. The statistics are formed in fp32 (autocast is switched off inside the hooks), and the basis work runs in full fp32 as in kronq_main. bf16 or fp16 parameters are refused. |
| Multi-GPU (DDP) | DDP averages the gradients. The factor statistics are all-reduced, and each layer's refresh is computed on rank `layer % world` and broadcast (bases, eigenvalues, the new period). Works with `no_sync` micro-batches. |

## Usage

```python
opt = KronQLM(model, lr=lr_peak, lr_adam=lr_adam_peak, loss='ce', weight_decay=wd_per_step, stats_every=1)

# one batch per step
logits = model(x)
opt.backward(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()), logits)
opt.step(lr_t); opt.zero_grad()

# micro-batches (gradient accumulation): the statistics come from the first one
for i, (x, y) in enumerate(micro):
    with opt.capture(i == 0):
        logits = model(x)
    opt.backward(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()) / len(micro), logits)
opt.step(lr_t); opt.zero_grad()
```

`opt.u_stats()` gives the fraction of learned log-weights at the upper and lower clamp, and the median |u|, per layer. Log it
in long runs. `opt.state_bytes(split=True)` gives the optimizer state in bytes.

## Tests (`test_kronq_lm.py`, CPU, about 2 minutes on a login node)

All pass, 2026-10-14:

1. **Equivalence with kronq_main** on an MLP with squared loss, with and without decay:
   - First 8 steps (two refreshes): loss agrees to 1e-7, weights to 2e-6, identical refresh schedule.
   - After 60 steps: loss within 0.8 %.
   - The two forwards differ only by rounding. Later, a near-zero bf16 momentum entry can take the other sign in one run.
2. **Tiny GPT** (2 layers, width 64), cross entropy, Markov-chain tokens:

   | Run | Final loss |
   |---|---|
   | AdamW | 0.742 |
   | KronQ, Fisher probe | 0.733 |
   | KronQ, empirical Fisher | 0.733 |
   | KronQ, bf16 autocast (40 steps) | 1.461 |

   This checks that the code runs; the task is too easy to rank the optimizers.
3. **DDP**, 2 ranks, 2 micro-batches per step with statistics from one: parameters, bases and refresh periods are bit-identical
   across ranks.
4. **Streaming and micro-batches:**
   - The streamed statistics equal those from all layer-output gradients at once (difference 0).
   - Capture on micro-batch 1 of 4 gives exactly the statistics of that micro-batch alone.
   - 60 steps x 4 micro-batches train (loss 3.81 -> 0.87).

## Cost per step (P = KronQ parameters, N = tokens per step, M = micro-batches per step, b = block)

| Part | Cost | Comment |
|---|---|---|
| Forward + backward | 6NP | the baseline |
| Probe | about 2NP/M per statistics step | the only part that grows with the token count. About 1/(3M) of the step at `stats_every=1`; under 1 % for M >= 32, or for M >= 8 with `stats_every=4`. `probe='emp'`: 0. |
| Statistics | 2b x (inputs + outputs) per token, on one micro-batch | small; `stats_tokens` subsamples rows |
| Rotations | about 8bP per step, independent of N | fp32 (no TF32); about 0.2 % at 4M tokens per step |
| Refresh | about 4 x dim x b^2 per side, every 5-1000 steps, spread across ranks | small in FLOPs. Each refresh synchronises the GPU a few times per layer, which is latency, not FLOPs. Batching same-shape layers would remove most of it (not done). |
| All-reducing the statistics | the input and output factors (dim x b floats per side per layer) per statistics step | about 0.17 bytes per parameter. At 70B that is about 8 % of a bf16 gradient all-reduce if done every step, under 1 % at `stats_every=10`. A reduce-scatter (each rank keeping only its own blocks) would cut it further (not done). |

## Memory

Optimizer state per KronQ parameter (computed from the shapes, blocks of 256, GPT-style block: q/k/v d -> 3d, o d -> d, MLP
d -> 4d -> d):

| Part | Bytes per parameter |
|---|---|
| Momentum, learned log-weight, previous push (bf16 each) | 6 |
| Factors, bases, eigenvalues (fp32), about 10.7 b/d | d = 768: 3.6 · d = 2048: 1.3 · d = 4096: 0.67 · d = 8192: 0.33 |
| Total | d = 768: 9.6 · d = 2048: 7.3 · d = 4096: 6.7 · d = 8192: 6.3 |

For comparison:

| Optimizer | Bytes per parameter |
|---|---|
| AdamW (fp32 moments) | 8 |
| Muon (momentum) | 2-4 |
| SOAP with full factors | about 8 for Adam's moments, plus about 30 for its four d x d-sized factors and bases. Implementations cap the factor size, so in practice it is less. |

So above d = 2048, KronQ's state is below AdamW's. Replacing the stored previous push (2 bytes) by its sign (1 bit) together with
the current weight would bring it to about 4.4 bytes. That is a small change to the method, so it is not made here.

Transient memory: one layer's fp32 gradient in the basis and its update at a time (the largest layer, a few copies). On the
statistics micro-batch, the probe needs that micro-batch's activations until its backward is done, as an ordinary backward
does; nothing else is kept. With activation checkpointing the probe would redo the recomputation, so keep that one micro-batch
uncheckpointed, or accept the extra forward.

## Not done yet

- A training script with real data and the baselines (AdamW, SOAP, Muon) on the same model.
- A GPU profile of time per step and peak memory.
- A batched refresh across same-shape layers, a reduce-scatter for the statistics, ZeRO/FSDP sharding, a multi-node NCCL test,
  and the torch.compile path on GPU (the tests run with compile disabled).
