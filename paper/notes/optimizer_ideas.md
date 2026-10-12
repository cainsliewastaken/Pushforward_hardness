# Optimizer ideas deferred for later (KronSign / Kronecker-eigenbasis optimizer line)

Written 2026-10-12. Ideas that are sound but were not pursued yet, with the reason.

## 1. Fold the basis rotations into neighbouring linear maps (architecture-specific)

**Idea.** In the reparametrised form each linear layer is computed as
z = U_S W~ U_A^T x~ : the input is rotated by U_A, multiplied by the trained matrix W~ and rotated back by U_S.
When two linear maps follow each other with no elementwise nonlinearity in between, the rotation back of the first
(U_S of layer l) and the rotation in of the second (U_A of layer l+1) are applied to the same vector one after the other.
Their product can be precomputed once per basis refresh (one d x d matrix), or absorbed into one of the trained
matrices, so the per-step cost of the rotations between the two maps disappears.

**Where it applies.** Places with linear-linear compositions:
- inside attention: the value projection followed by the output projection (per head, the attention weights act on the
  token axis, not the feature axis, so they commute with feature-space rotations), and the query/key projections whose
  product enters only through q k^T;
- residual streams with linear read/write maps (pre-norm transformers: careful, LayerNorm does not commute with rotations);
- linear bottlenecks and low-rank factorised layers (W = B A).

**Why deferred.** It depends on the architecture (it needs linear-linear adjacency and care with norms and
nonlinearities), so it violates the current rule that the method must not assume structure in the data or the
architecture. Revisit when optimising for transformers specifically.

**What to check first.** That the composed rotation is exact (it is, in exact arithmetic, because the rotations are
orthogonal and nothing nonlinear sits between them), the saving in wall time on a small transformer, and how it interacts
with per-layer basis refresh schedules (the two layers' refreshes would need to be synchronised).

## 2. Fast structured rotations for very wide layers

Products of Householder reflections or butterfly matrices apply in O(d log d) instead of O(d^2). An approximation to
the exact eigenbasis; only worth it for layers tens of thousands wide (e.g. large vocabulary embeddings), where the
d^2 factor memory and d^3 refresh become the limit. Alternative to blocking, which hurt accuracy 1.5-2x at width 512.
