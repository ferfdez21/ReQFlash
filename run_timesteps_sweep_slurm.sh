#!/bin/bash

#SBATCH --job-name=reqflash_sweep_packed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # 4 CPUs
#SBATCH --mem=32G
#SBATCH --array=0-7               # Only 8 Jobs! (One per Timestep)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Usage: sbatch run_timesteps_sweep_slurm.sh <directory_containing_checkpoints>

# 1. Validate Arguments
CKPT_DIR="$1"
if [ -z "$CKPT_DIR" ]; then
    echo "Usage: sbatch $0 <directory_containing_checkpoints>"
    exit 1
fi

# 2. Setup Environment
eval "$(conda shell.bash hook)"
conda activate reqflash

# 3. Discovery: Find all .ckpt files
CKPTS=($(ls $CKPT_DIR/*.ckpt))
NUM_CKPTS=${#CKPTS[@]}

if [ "$NUM_CKPTS" -eq 0 ]; then
    echo "No .ckpt files found in $CKPT_DIR"
    exit 1
fi

# 4. Define Timesteps (8 total)
TIMESTEPS=(10 20 50 100 200 300 400 500)

# 5. Determine Task
# Each of the 8 jobs takes ONE timestep and processes ALL checkpoints for that timestep.
# This reduces the job count from 80 to 8, fixing the QOSMaxSubmitJobPerUserLimit error.
T_VAL=${TIMESTEPS[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Assigned Timestep: $T_VAL"
echo "Found $NUM_CKPTS checkpoints to process."

# 6. Loop over all checkpoints
count=1
for CKPT in "${CKPTS[@]}"; do
    echo "------------------------------------------------------------------"
    echo "Processing Checkpoint ($count/$NUM_CKPTS): $CKPT"
    echo "------------------------------------------------------------------"
    
    PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
        -cn inference_unconditional \
        inference.interpolant.sampling.num_timesteps=$T_VAL \
        inference.ckpt_path="$CKPT"
        
    count=$((count + 1))
done

echo "All checkpoints processed for timestep $T_VAL."
