# FlashIPA Integration Walkthrough

## Overview
This document summarizes the integration of `FlashIPA` into `ReQFlash`.

## Changes
1.  **Dependencies**:
    *   Installed `nvidia-cuda-nvcc-cu11==11.7.99` to provide CUDA 11.7 compatible `nvcc`.
    *   Installed `flash-attn` (v2.3.5) pre-built wheel for CUDA 11.7, PyTorch 2.0, Python 3.10.
    *   Installed `beartype` and `jaxtyping` required by `FlashIPA`.

2.  **Configuration**:
    *   Updated `configs/_model.yaml` to include `use_flash_ipa` flag (default `False`) and FlashIPA-specific parameters in `ipa` and `edge_features` sections.

3.  **Model Refactoring**:
    *   Modified `models/flow_model.py` to conditionally initialize and use `FlashIPA` components when `use_flash_ipa=True`.
    *   Updated `models/flash_ipa/rigid.py` to handle quaternion inputs in `create_rigid` (previously only supported rotation matrices).
    *   Fixed imports in `models/flash_ipa/*.py` to use absolute imports (`models.flash_ipa...`) to ensure compatibility with the project structure.

## Verification
*   Created a verification script `verify_flash_ipa_integration.py` that instantiates `FlowModel` with `use_flash_ipa=True`.
*   Ran a forward pass with dummy data on CUDA.
*   Confirmed successful execution and correct output shapes.

## Usage
To use FlashIPA, run inference with the flag:
```bash
python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional model.use_flash_ipa=True
```
(Note: Ensure checkpoints are available for full inference).
