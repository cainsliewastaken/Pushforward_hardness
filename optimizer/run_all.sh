#!/bin/bash
# All eight proposed changes on one 4-GPU interactive allocation.  Usage: run_all.sh <jobid>
# Streams (one GPU each; CUDA_VISIBLE_DEVICES is exported inside the container, see the srun-overlap note):
#   GPU0: (1) blocked + (2) identity sweep, per step / per FLOP / per second;  then (7) statistics window;  then (6) wall-time options
#   GPU1: (3) tuned SOAP baseline, p=1 then p=0;  then (4) blocked rank-1 (kronv_b64) p=0
#   GPU2: (4) rank-1 second moment (kronv) p=1, p=0;  then kronv_b64 p=1
#   GPU3: (5) convolution support: periodic CNN, Adam vs Kronecker-Adam vs kronv, p=1 then p=0
# Then (8) DDP on all 4 GPUs.  Each stream writes a .done file; run_all.done marks the end.
J=$1; C=$(dirname "$(readlink -f "$0")"); R=$SCRATCH/l63_held_rough/runs/kadam/all8; mkdir -p $R
D=$SCRATCH/l63_held_rough/runs/kadam
IMG=nersc/pytorch:26.01.01
step() { local g=$1; shift; srun --jobid=$J --overlap -N 1 --ntasks=1 --gpus=4 shifter --image=$IMG bash -c "export CUDA_VISIBLE_DEVICES=$g; cd $C && $*"; }
P1="--data $D/field20k_p1.npz --widths 512,512 --act gelu --batch 100 --standardize none --seed 1"
P0="--data $D/field20k_p0.npz --widths 512,512 --act gelu --batch 100 --standardize none --seed 1"
KS1="--opt kronsign --rule adam --lr 1.9e-2 --b2 0.999"      # tuned Kronecker-Adam, p = 1
KS0="--opt kronsign --rule adam --lr 3.8e-3 --b2 0.99"       # tuned Kronecker-Adam, p = 0
AD1="--opt adam --lr 3.8e-3 --b2 0.99"                       # tuned Adam, p = 1

( step 0 "python -u block_sweep.py $R/block 1 0" > $R/block.log 2>&1
  step 0 "python -u runlist.py $R/statswin '$P1 --steps 6000 $KS1' bs0.8 '--beta_stats 0.8' bs0.95 '--beta_stats 0.95' bs0.99 '--beta_stats 0.99' bs0.8_b64 '--beta_stats 0.8 --block 64' bs0.99_b64 '--beta_stats 0.99 --block 64'" > $R/statswin.log 2>&1
  step 0 "python -u runlist.py $R/statswin '$P0 --steps 6000 $KS0' p0_bs0.8 '--beta_stats 0.8' p0_bs0.95 '--beta_stats 0.95' p0_bs0.99 '--beta_stats 0.99'" >> $R/statswin.log 2>&1
  step 0 "python -u runlist.py $R/walltime '$P1 --steps 1500' \
     adam_fp64 '$AD1' adam_fp32 '$AD1 --float32 1' adam_fp32_tf32 '$AD1 --float32 1 --tf32 1' \
     ks_fp64 '$KS1' ks_fp32 '$KS1 --float32 1' ks_fp32_tf32 '$KS1 --float32 1 --tf32 1' \
     ks_fp32_rotbf16 '$KS1 --float32 1 --rot_dtype bfloat16' ks_fp32_eigbatch '$KS1 --float32 1 --eig_batch 1' \
     ks_fp32_eigadapt '$KS1 --float32 1 --eig_adapt 1' ks_fp32_eigh '$KS1 --float32 1 --eig eigh' \
     ks_fp32_b64 '$KS1 --float32 1 --block 64' ks_fp32_b64_eigbatch '$KS1 --float32 1 --block 64 --eig_batch 1' \
     ks_fp32_b64_all '$KS1 --float32 1 --block 64 --eig_batch 1 --eig_adapt 1 --last_S identity' \
     ks_fp64_b64_all '$KS1 --block 64 --eig_batch 1 --eig_adapt 1 --last_S identity' \
     kronv_fp32_b64_all '--opt kronsign --rule kronv --lr 1.9e-2 --b2 0.999 --float32 1 --block 64 --eig_batch 1 --eig_adapt 1 --last_S identity' \
     prof_ks_fp32 '$KS1 --float32 1 --profile 1' prof_ks_fp32_b64 '$KS1 --float32 1 --block 64 --profile 1' \
     prof_ks_fp32_b64_all '$KS1 --float32 1 --block 64 --eig_batch 1 --eig_adapt 1 --last_S identity --profile 1' \
     prof_kronv_fp32_b64_all '--opt kronsign --rule kronv --lr 1.9e-2 --b2 0.999 --float32 1 --block 64 --eig_batch 1 --eig_adapt 1 --last_S identity --profile 1'" > $R/walltime.log 2>&1
  touch $R/gpu0.done ) &
( for p in 1 0; do step 1 "python -u tune2.py field_p${p}_soap $R/tune"; done > $R/soap.log 2>&1
  step 1 "python -u tune2.py field_p0_kronv_b64 $R/tune" > $R/kronv_b64_p0.log 2>&1; touch $R/gpu1.done ) &
( for p in 1 0; do step 2 "python -u tune2.py field_p${p}_kronv $R/tune"; done > $R/kronv.log 2>&1
  step 2 "python -u tune2.py field_p1_kronv_b64 $R/tune" > $R/kronv_b64_p1.log 2>&1; touch $R/gpu2.done ) &
( for p in 1 0; do for m in adam ks_adam kronv; do step 3 "python -u tune2.py cnn_p${p}_${m} $R/tune"; done; done > $R/cnn.log 2>&1; touch $R/gpu3.done ) &
wait
# (8) DDP: 4 ranks x batch 100 against 1 rank x batch 400, Kronecker-Adam and Adam, p = 1, 1500 steps
DD="--data $D/field20k_p1.npz --widths 512,512 --act gelu --standardize none --seed 1 --steps 1500 --float32 1 --block 64 --eig_batch 1 --last_S identity"
srun --jobid=$J --overlap -N 1 --ntasks-per-node=1 --gpus-per-node=4 shifter --image=$IMG bash -c "cd $C && \
  CUDA_VISIBLE_DEVICES=0 python -u train_ks.py $DD --batch 400 --opt kronsign --rule adam --lr 1.9e-2 --b2 0.999 --out $R/ddp_1gpu_ks.jsonl ; \
  CUDA_VISIBLE_DEVICES=0 python -u train_ks.py $DD --batch 400 --opt adam --lr 3.8e-3 --b2 0.99 --out $R/ddp_1gpu_adam.jsonl ; \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 train_ks.py $DD --ddp 1 --batch 100 --opt kronsign --rule adam --lr 1.9e-2 --b2 0.999 --out $R/ddp_4gpu_ks.jsonl ; \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 train_ks.py $DD --ddp 1 --batch 100 --opt adam --lr 3.8e-3 --b2 0.99 --out $R/ddp_4gpu_adam.jsonl" > $R/ddp.log 2>&1
echo "exit=$?" > $R/run_all.done
