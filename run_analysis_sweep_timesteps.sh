#!/bin/bash
# Script to run metric analysis for all timesteps for ReQFlash

if [ -z "$1" ]; then
    echo "Usage: $0 <inference_dir_template> [dataset_dir]"
    exit 1
fi

TIMESTEPS=(10 20 50 100 200 300 400 500)
BASE_DIR="$(pwd)"
INF_DIR_TEMPLATE="$1"
SCRIPT_PATH="${BASE_DIR}/analysis/run_foldseek_parallel.sh"
DATASET_DIR="$2"

if [ -z "$DATASET_DIR" ]; then
    echo "Starting designability-only timesteps analysis sweep for ReQFlash (no Foldseek)..."
else
    echo "Starting full timesteps analysis sweep (including Foldseek) for ReQFlash..."
fi

# Loop through each timestep
for t in "${TIMESTEPS[@]}"; do
    CURRENT_INF_DIR="${INF_DIR_TEMPLATE}/${t}_steps"
    echo "Processing $CURRENT_INF_DIR"
    
    if [ ! -d "$CURRENT_INF_DIR" ]; then
        echo "Warning: Directory $CURRENT_INF_DIR does not exist. Skipping."
        continue
    fi

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    if [ -n "$DATASET_DIR" ]; then
        PYTHONPATH=. python analysis/all_metric_calculation.py \
            --inference_dir "$CURRENT_INF_DIR" \
            --script_path "$SCRIPT_PATH" \
            --dataset_dir "$DATASET_DIR"
    else
        PYTHONPATH=. python analysis/all_metric_calculation.py \
            --inference_dir "$CURRENT_INF_DIR"
    fi

    echo "Finished analysis for $t steps"
    echo "---------------------------------------------------"
done

echo "ReQFlash timesteps analysis sweep completed."
