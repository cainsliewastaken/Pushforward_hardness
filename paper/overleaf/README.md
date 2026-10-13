# paper/overleaf: the current paper sections (2026-10-12)

Overleaf project: https://www.overleaf.com/project/6ab704f85ea5c8567e1108c9

These are the files uploaded to Overleaf; they are the current version of the theory
(the chained expansion with its infinite-series proofs).

- `held_rough_error_lean.tex`  : the "held rough error" section with the cuts applied
                                 (closure assumption, two-sided band, reduced ignition model removed; see PAPER_OUTLINE.md)
- `chain_section.tex`          : "Training through time: the chained expansion and the limits of prediction"
- `chain_appendix.tex`         : truncation / error propagation and the n = 200 verifications
- `held_rough_error_original.tex` : the section as received, for diff
- `lean_sections_preview.pdf`  : pdflatex build of the three files together (texlive/2024), no errors, no undefined references
- `PAPER_OUTLINE.md`           : what to keep / replace / cut, and what is open before submission
- (review 2026-10-12)          : corrections applied to the three tex files (leakage bound as a block norm, cutoff-bound inputs as run
                                 quantities, sign condition as a per-stage hypothesis, Thm 36 first order only, radius bound unproved,
                                 Adam's dP term, Cor. depth (ii) a trajectory bound); then, after the user's correction, the held-rough
                                 section rewritten around the Krylov functional (the roughness function of Part I), same flow; details in `../notes/THEORY.md` (REVISED / CORRECTION marks,
                                 section 21) and the experiments they require in `../notes/EXPERIMENTS.md`

In the Overleaf main file: replace `\input{held_rough_error.tex}` by `\input{held_rough_error_lean.tex}`;
add `\input{chain_section.tex}` right after it; add `\input{chain_appendix.tex}` after `\appendix`.
Theorem environments needed (amsthm): theorem, proposition, corollary, lemma, definition, assumption, remark.
`chain_section.tex` references only these labels of the held-rough section: sec:held-rough, prop:rough-energy,
def:cutoff, prop:cutoff-floor.

Context: `../notes/THEORY.md` (all statements with proofs and status labels; sections 8-20 are the chain),
`../notes/WRITEUP.md` (experimental record; every experiment is n = 200 and must be redone before quoting numbers).
