import torch
import os
import sys

ckpt_path = "/home/ffernandez/Desktop/code/ReQFlash/ckpts/qflash_scope/epoch=176-step=325149.ckpt"
print(f"Loading {ckpt_path}...")
try:
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print("Keys in checkpoint:", checkpoint.keys())
    if 'callbacks' in checkpoint:
        print("Keys in callbacks:", checkpoint['callbacks'].keys())
        # Check ModelCheckpoint callback which might store monitor values
        for key in checkpoint['callbacks']:
            if 'ModelCheckpoint' in key:
                 print(f"Callback {key}:", checkpoint['callbacks'][key])
    
    # Check loops or other state
    if 'loops' in checkpoint:
        print("Loops key present (often contains loop state)")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
