#!/bin/bash
# Waits for a Slurm job to leave the queue, then prints its state and the end of its log.
JOB="$1"
while squeue -h -j "$JOB" 2>/dev/null | grep -q .; do sleep 20; done
sacct -X -j "$JOB" --format=JobID,JobName,State,Elapsed,Start,End -n
echo "----- log -----"
grep -v -e FutureWarning -e pynvml "$SCRATCH/pf_logs/soft_bounds-$JOB.out" | tail -n 120
