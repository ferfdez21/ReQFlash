#!/bin/bash
# Script to run metric analysis for all timesteps for ReQFlash

TIMESTEPS=(10 20 50 100 200 300 400 500)
BASE_DIR="/home/ffernandez/Desktop/code/ReQFlash"
# Path pattern observed in previous steps
INF_DIR_TEMPLATE="${BASE_DIR}/inference_outputs/reqflash_train_scope_base/2026-01-09_17-23-20/epoch=176-step=325149/unconditional/inference_outputs/qflash_analysis"
SCRIPT_PATH="${BASE_DIR}/analysis/run_foldseek_parallel.sh"
# Foldseek database directory confirmed in previous turn
DATASET_DIR="/home/ffernandez/FoldSeek_PDB_Database"

echo "Starting analysis sweep for ReQFlash..."

for t in "${TIMESTEPS[@]}"; do
    CURRENT_INF_DIR="${INF_DIR_TEMPLATE}/${t}_steps"
    echo "Processing $CURRENT_INF_DIR"
    
    if [ ! -d "$CURRENT_INF_DIR" ]; then
        echo "Warning: Directory $CURRENT_INF_DIR does not exist. Skipping."
        continue
    fi

    # Using the python executable from the reqflash environment to ensure dependencies
    PYTHONPATH=. /home/ffernandez/.conda/envs/reqflash/bin/python analysis/all_metric_calculation.py \
        --inference_dir "$CURRENT_INF_DIR" \
        --script_path "$SCRIPT_PATH" \
        --dataset_dir "$DATASET_DIR"

    echo "Finished analysis for $t steps"
    echo "---------------------------------------------------"
done

echo "ReQFlash analysis sweep completed."
