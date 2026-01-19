#!/bin/bash

#SBATCH --job-name=reqflash_metrics
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/metrics_%j.out
#SBATCH --error=logs/metrics_%j.err

# Usage: sbatch run_checkpoint_analysis.sh <inference_output_folder>

# 1. Validate Arguments
INFERENCE_DIR="$1"

if [ -z "$INFERENCE_DIR" ]; then
    echo "Usage: sbatch $0 <inference_output_folder>"
    exit 1
fi

if [ ! -d "$INFERENCE_DIR" ]; then
    echo "Error: Directory '$INFERENCE_DIR' does not exist."
    exit 1
fi

# 2. Setup Environment
eval "$(conda shell.bash hook)"
conda activate reqflash

echo "Running Metrics Analysis on: $INFERENCE_DIR"
echo "Running on $(hostname)"

# 3. Run Metrics Calculation
PYTHONPATH=. python analysis/all_metric_calculation.py \
    --inference_dir "$INFERENCE_DIR" \
    --type qflow

echo "Metrics calculation finished."
