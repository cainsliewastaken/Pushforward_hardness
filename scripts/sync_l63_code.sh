#!/bin/bash
# Catch the repo up with $SCRATCH/l63_held_rough (the live working copy of the L63 sessions that
# predate this repository). Copies every script in code/ that is newer than, or missing from, its
# repo copy (chain/ or optimizer/; new files go to chain/ unless listed in OPT below), and every
# report/ file newer than its copy under paper/overleaf or paper/notes. Then re-applies the path
# patch. Shows what it did; run `git status` / `git diff` afterwards and commit.
#   usage: scripts/sync_l63_code.sh            (dry run: only prints what would be copied)
#          scripts/sync_l63_code.sh --apply
set -euo pipefail
R="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
S="${L63_ROOT:-$SCRATCH/l63_held_rough}"
APPLY=${1:-}
# optimizer-line scripts (everything else lands in chain/)
OPT=" segflow segflow_rough segflow_l63 segflow_generic_test segflow_generic_thresholds segflow_summary minibatch_segflow kadam kadam_front kronsign kronsign_v1 kronsign_v3_carrybug kronsign_pre_reparam kronsign_copy_of_otherinstance_1408 kronroot muon_roughness muon_field adam_l63 adam_ablation train_ks train_ks_v2 train_ks_v3 train_ks_pre_kronroot speed_sweep block_sweep block_sweep_v1 tune tune2 tune2_v1 mkfield kr_sweep momentum_modes runlist run_all front_summary spectral_power feature_check width_scaling band_precond_field supply_field supply_step gamma_init lazy_q mode_curvature mode_curvature_path q_summary front_summary3 trgn kronq kronq2 kronq3h q_grid_summary horizon_q mode_stats "
n=0
for f in "$S"/code/*; do
  [ -f "$f" ] || continue
  b=$(basename "$f"); stem=${b%.*}
  case "$b" in *.pyc|*.tmp|*.swp|*~) continue;; esac   # editor / in-progress files of the live session
  dst="$R/chain/$b"; [ -e "$R/optimizer/$b" ] && dst="$R/optimizer/$b"
  if [ ! -e "$dst" ]; then
    case "$OPT" in *" $stem "*) dst="$R/optimizer/$b";; esac
    echo "NEW     $b -> ${dst#$R/}"
  elif [ "$f" -nt "$dst" ] && ! cmp -s "$f" "$dst"; then
    # ignore files whose only difference is the path patch
    # count (not grep -q): with pipefail, grep -q's early exit SIGPIPEs diff and silently skipped newer files
    nd=$(diff "$f" "$dst" | grep -E '^[<>]' | grep -cvE 'L63_ROOT|sys\.path|readlink|l63_held_rough' || true)
    if [ "$nd" -gt 0 ]; then
      echo "NEWER   $b -> ${dst#$R/}"
    else continue; fi
  else continue; fi
  n=$((n+1)); [ "$APPLY" = "--apply" ] && cp -p "$f" "$dst"
done
for f in "$S"/report/*; do
  b=$(basename "$f"); dst=""
  for d in paper/overleaf paper/notes; do [ -e "$R/$d/$b" ] && dst="$R/$d/$b"; done
  [ -z "$dst" ] && { case "$b" in *.zip) continue;; esac; dst="$R/paper/notes/$b"; echo "NEW     report/$b -> ${dst#$R/}"; }
  if [ -e "$dst" ] && { [ ! "$f" -nt "$dst" ] || cmp -s "$f" "$dst"; }; then continue; fi
  [ -e "$dst" ] && echo "NEWER   report/$b -> ${dst#$R/}"
  n=$((n+1)); [ "$APPLY" = "--apply" ] && cp -p "$f" "$dst"
done
echo "$n file(s) to sync"
if [ "$APPLY" = "--apply" ] && [ "$n" -gt 0 ]; then
  shifter --image=nersc/pytorch:26.01.01 python -I "$R/scripts/patch_paths.py" "$R" 2>&1 | grep -v FutureWarning
  rm -rf "$R"/chain/__pycache__ "$R"/optimizer/__pycache__
fi
