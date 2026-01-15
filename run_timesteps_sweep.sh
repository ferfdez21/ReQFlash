#!/bin/bash
# Script to run unconditional inference sweep over different timesteps for ReQFlash


# Check for checkpoint argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_checkpoint>"
    exit 1
fi

CKPT_PATH="$1"

# Array of timesteps to test
TIMESTEPS=(10 20 50 100 200 300 400 500)

echo "Starting timestep sweep for ReQFlash..."

# Loop through each timestep
for t in "${TIMESTEPS[@]}"; do
    echo "=================================================================="
    echo "Running inference with num_timesteps=$t"
    echo "=================================================================="
    
    # Run the inference command
    # Overwriting inference.interpolant.sampling.num_timesteps via Hydra override
    PYTHONPATH=. python -W ignore experiments/inference_se3_flows.py \
        -cn inference_unconditional \
        inference.interpolant.sampling.num_timesteps=$t \
        inference.ckpt_path=\"$CKPT_PATH\"
    
    echo "Finished inference for num_timesteps=$t"
    echo ""
done

echo "Sweep completed."
