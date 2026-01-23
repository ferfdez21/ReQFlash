#!/bin/bash

#SBATCH --job-name=reqflash_inference_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/inference_large_%j.out
#SBATCH --error=logs/inference_large_%j.err

# Usage: sbatch run_inference_large_gpu.sh <path_to_checkpoint_file>

# 1. Validate Arguments
CKPT_PATH="$1"

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: sbatch $0 <path_to_checkpoint_file>"
    exit 1
fi

# 2. Setup Environment
eval "$(conda shell.bash hook)"
conda activate reqflash

echo "Running inference on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT_PATH"
# echo "Timesteps: 500"
# echo "Batch Size: 10 (Parallel sampling for all 10 samples per length)"

# 3. Run Inference
# processing 10 samples per length in a single batch
PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
    -cn inference_unconditional \
    inference.interpolant.sampling.num_timesteps=500 \
    inference.ckpt_path="$CKPT_PATH" \
    inference.samples.min_length=60 \
    inference.samples.max_length=128 \
    inference.samples.samples_per_length=10 \
    inference.samples.seq_per_sample=8 \
    inference.samples.batch_size=10

echo "Finished inference."
