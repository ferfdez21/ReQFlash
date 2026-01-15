#!/bin/bash

#SBATCH --job-name=reqflash_inference_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # Increased to 8 CPUs (32 Cores / 4 Jobs = 8)
#SBATCH --mem=32G
#SBATCH --array=0-3               # Reduced to 4 Jobs to match QOS Limit (4 GPUs max)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Usage: sbatch run_timesteps_sweep_slurm.sh <path_to_checkpoint_file>

# 1. Validate Arguments
CKPT_PATH="$1"
if [ -z "$CKPT_PATH" ]; then
    echo "Usage: sbatch $0 <path_to_checkpoint_file>"
    exit 1
fi

# 2. Setup Environment
eval "$(conda shell.bash hook)"
conda activate reqflash

# 3. Define Timesteps (8 total)
TIMESTEPS=(10 20 50 100 200 300 400 500)

# 4. Determine which timesteps to run for this job.
# We have a limit of 4 Concurrent GPUs (QOSMaxGRESPerUser).
# We have 8 Timesteps to run.
# We pack 2 Timesteps per Job.
# Job 0: Indices 0, 1
# Job 1: Indices 2, 3
# ...
START_IDX=$((SLURM_ARRAY_TASK_ID * 2))

echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Processing indices starting at: $START_IDX"

# Loop twice (for the two timesteps assigned to this job)
for i in {0..1}; do
    IDX=$((START_IDX + i))
    
    # Check if index is valid (just in case)
    if [ $IDX -ge ${#TIMESTEPS[@]} ]; then
        break
    fi

    T_VAL=${TIMESTEPS[$IDX]}
    
    echo "------------------------------------------------------------------"
    echo "Running Inference for Timestep: $T_VAL"
    echo "------------------------------------------------------------------"

    PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
        -cn inference_unconditional \
        inference.interpolant.sampling.num_timesteps=$T_VAL \
        inference.ckpt_path="$CKPT_PATH" \
        inference.samples.min_length=60 \
        inference.samples.max_length=128 \
        inference.samples.samples_per_length=10 \
        inference.samples.seq_per_sample=8

done

echo "Finished packed job for timesteps."
