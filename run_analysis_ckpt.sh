#!/bin/bash
# Script to run metric analysis for a single inference output folder

if [ -z "$1" ]; then
    echo "Usage: $0 <inference_dir> [dataset_dir]"
    exit 1
fi

BASE_DIR="$(pwd)"
INFERENCE_DIR="$1"
SCRIPT_PATH="${BASE_DIR}/analysis/run_foldseek_parallel.sh"
DATASET_DIR="$2"

echo "Processing $INFERENCE_DIR"

if [ ! -d "$INFERENCE_DIR" ]; then
    echo "Error: Directory $INFERENCE_DIR does not exist."
    exit 1
fi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Check config file (Metrics.txt depends on it)
if [ ! -f "$INFERENCE_DIR/config.yaml" ]; then
   echo "Error: config file not found in $INFERENCE_DIR."
   exit 1
fi

if [ -n "$DATASET_DIR" ]; then
    echo "Running full analysis (including Foldseek)..."
    PYTHONPATH=. python analysis/all_metric_calculation.py \
        --inference_dir "$INFERENCE_DIR" \
        --script_path "$SCRIPT_PATH" \
        --dataset_dir "$DATASET_DIR" \
        --type qflow
else
    echo "Running analysis (Designability only, no Foldseek)..."
    PYTHONPATH=. python analysis/all_metric_calculation.py \
        --inference_dir "$INFERENCE_DIR" \
        --type qflow
fi

echo "Finished analysis for $INFERENCE_DIR"
