import sys
import os

# Add local path to sys.path
sys.path.append(os.getcwd())

try:
    from pytorch_lightning import Trainer
    from models.flow_module import FlowModule
    from experiments import utils as eu
    from data.protein_dataloader import ProteinData
    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
