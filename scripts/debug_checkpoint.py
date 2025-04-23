#!/usr/bin/env python3
"""
Debug script to analyze checkpoint contents.
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add project root to path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.models.rna_folding_model import RNAFoldingModel

def analyze_checkpoint(checkpoint_path):
    """Analyze contents of a checkpoint file."""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Print the keys in the checkpoint
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    
    # Check model_state_dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print(f"model_state_dict type: {type(model_state_dict)}")
        print(f"model_state_dict keys: {list(model_state_dict.keys())}")
        
        # Check if dummy is the only key (corrupted checkpoint)
        if list(model_state_dict.keys()) == ['dummy']:
            print("WARNING: Corrupted checkpoint! Only contains a dummy key.")
    else:
        print("No model_state_dict found in checkpoint")
    
    # Check state_dict (alternative name)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"state_dict type: {type(state_dict)}")
        print(f"state_dict keys: {list(state_dict.keys())[:5]}...")
    
    # Check model config
    model_config = None
    if 'args' in checkpoint:
        model_config = checkpoint['args']
        print(f"Model config from 'args': {type(model_config)}")
    elif 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"Model config from 'model_config': {type(model_config)}")
    
    # Try to create a fresh model
    if model_config:
        try:
            print("Trying to create a fresh model...")
            model = RNAFoldingModel(model_config)
            print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
            
            # The state_dict of this fresh model should have all the expected keys
            fresh_state_dict = model.state_dict()
            print(f"Fresh model state_dict keys: {list(fresh_state_dict.keys())[:5]}...")
            print(f"Total keys in fresh model: {len(fresh_state_dict)}")
            
            # Compare with checkpoint
            if 'model_state_dict' in checkpoint:
                missing_keys = [k for k in fresh_state_dict.keys() if k not in checkpoint['model_state_dict']]
                unexpected_keys = [k for k in checkpoint['model_state_dict'].keys() if k not in fresh_state_dict]
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                print(f"Unexpected keys list: {unexpected_keys}")
        except Exception as e:
            print(f"Error creating model: {str(e)}")

def main():
    """Main function."""
    # Check various model checkpoints
    checkpoints = [
        "results/final_model/run_20250423-072601/checkpoints/best_model.pt",
        "results/final_model/run_20250423-072601/checkpoints/checkpoint_epoch_30.pt",
        "results/production_run_1/run_20250423-072209/checkpoints/best_model.pt",
    ]
    
    for checkpoint_path in checkpoints:
        if os.path.exists(checkpoint_path):
            analyze_checkpoint(checkpoint_path)
            print("-" * 80)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    main()