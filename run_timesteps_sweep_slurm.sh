#!/bin/bash

#SBATCH --job-name=reqflash_inference_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # 4 CPUs per task to allow packing 8 tasks on 32-core node
#SBATCH --mem=32G
#SBATCH --array=0-79              # 80 Total Tasks (10 Checkpoints * 8 Timesteps)
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Usage: sbatch run_slurm_sweep.sh <directory_containing_checkpoints>

# 1. Validate Arguments
CKPT_DIR="$1"
if [ -z "$CKPT_DIR" ]; then
    echo "Usage: sbatch $0 <directory_containing_checkpoints>"
    exit 1
fi

# 2. Setup Environment
eval "$(conda shell.bash hook)"
conda activate reqflash

# 3. Discovery: Find all .ckpt files in the directory
# We assume there are exactly 10, but this handles any number dynamically.
CKPTS=($(ls $CKPT_DIR/*.ckpt))
NUM_CKPTS=${#CKPTS[@]}

if [ "$NUM_CKPTS" -eq 0 ]; then
    echo "No .ckpt files found in $CKPT_DIR"
    exit 1
fi

# 4. Define Timesteps (8 total)
TIMESTEPS=(10 20 50 100 200 300 400 500)
NUM_TIMESTEPS=${#TIMESTEPS[@]}

# 5. Map SLURM_ARRAY_TASK_ID to (Checkpoint, Timestep) pair
# SLURM_ARRAY_TASK_ID ranges from 0 to 79.
# We divide by 8 to get the Checkpoint Index (0-9).
# We modulo 8 to get the Timestep Index (0-7).
CKPT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TIMESTEPS))
TIME_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TIMESTEPS))

# Safety check for array bounds
if [ $CKPT_IDX -ge $NUM_CKPTS ]; then
    echo "Array ID $SLURM_ARRAY_TASK_ID is out of bounds for $NUM_CKPTS checkpoints."
    exit 0
fi

# Select the specific configuration for this task
CURRENT_CKPT=${CKPTS[$CKPT_IDX]}
CURRENT_TIMESTEP=${TIMESTEPS[$TIME_IDX]}

echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Processing Checkpoint ($((CKPT_IDX+1))/$NUM_CKPTS): $CURRENT_CKPT"
echo "Processing Timestep ($((TIME_IDX+1))/$NUM_TIMESTEPS): $CURRENT_TIMESTEP"

# 6. Run Inference
PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
    -cn inference_unconditional \
    inference.interpolant.sampling.num_timesteps=$CURRENT_TIMESTEP \
    inference.ckpt_path="$CURRENT_CKPT"

echo "Job Finished Successfully"
