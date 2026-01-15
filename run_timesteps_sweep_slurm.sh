#!/bin/bash

#SBATCH --job-name=reqflash_inference_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # 4 CPUs per task (32 Total CPUs / 8 Jobs = 4)
#SBATCH --mem=32G
#SBATCH --array=0-7               # 8 Jobs (Indices 0-7), one for each timestep
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

# 3. Define Timesteps
TIMESTEPS=(10 20 50 100 200 300 400 500)

# Check for SLURM Array ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID not set. Please submit with sbatch."
    exit 1
fi

# 4. Get the specific timestep for THIS job
T_VAL=${TIMESTEPS[$SLURM_ARRAY_TASK_ID]}

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "Target Timestep: $T_VAL"
echo "Checkpoint: $CKPT_PATH"

# 5. Run Inference
# Optimized parameters:
# - Lengths: 60 to 128
# - Samples per length: 10
# - Sequences per sample: 8
PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
    -cn inference_unconditional \
    inference.interpolant.sampling.num_timesteps=$T_VAL \
    inference.ckpt_path="$CKPT_PATH" \
    inference.samples.min_length=60 \
    inference.samples.max_length=128 \
    inference.samples.samples_per_length=10 \
    inference.samples.seq_per_sample=8

echo "Finished inference for num_timesteps=$T_VAL"
