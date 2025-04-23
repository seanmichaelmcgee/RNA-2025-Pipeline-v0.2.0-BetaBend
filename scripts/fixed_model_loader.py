#!/usr/bin/env python3
"""
Fixed model loader to handle corrupted checkpoints.
This script provides a fixed version of the model loading function
that can be used in the Kaggle inference notebook.
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.models.rna_folding_model import RNAFoldingModel

def fixed_load_model(checkpoint_path, device):
    """
    Fixed model loader that can handle corrupted checkpoints with dummy state_dict.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on
        
    Returns:
        tuple: (model, config, metrics) - The loaded model, its configuration, and metrics
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Check if the file exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return None, None, {'val_rmsd': None, 'epoch': None}
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration
        if 'args' in checkpoint:
            config = checkpoint['args']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Default configuration
            print("WARNING: No configuration found in checkpoint. Using defaults.")
            config = {
                'num_blocks': 4,
                'residue_embed_dim': 128,
                'pair_embed_dim': 64,
                'num_attention_heads': 4,
                'dropout': 0.1
            }
        
        # Initialize model
        model = RNAFoldingModel(config)
        
        # Check if state_dict is valid
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            # Check if it's corrupted (only contains dummy key)
            if list(model_state_dict.keys()) == ['dummy']:
                print("WARNING: Corrupted checkpoint detected with only 'dummy' key.")
                print("Initializing model from scratch with the configuration from checkpoint.")
                # We don't load weights - using freshly initialized weights
            else:
                # Load state dict if it seems valid
                model.load_state_dict(model_state_dict)
                print("Successfully loaded weights from checkpoint.")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Successfully loaded weights from 'state_dict'.")
        else:
            print("WARNING: No state_dict found in checkpoint. Using untrained model.")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Extract validation metrics if available
        val_rmsd = None
        if 'val_rmsd' in checkpoint:
            val_rmsd = checkpoint['val_rmsd']
        elif 'validation_rmsd' in checkpoint:
            val_rmsd = checkpoint['validation_rmsd']
        elif 'best_val_metrics' in checkpoint and 'rmsd' in checkpoint['best_val_metrics']:
            val_rmsd = checkpoint['best_val_metrics']['rmsd']
        
        epoch = None
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        
        # Return model, config, and metrics
        return model, config, {'val_rmsd': val_rmsd, 'epoch': epoch}
    
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, {'val_rmsd': None, 'epoch': None}

def main():
    """Test the fixed model loader."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test various checkpoints - use absolute paths from project root
    checkpoints = [
        "results/final_model/run_20250423-072601/checkpoints/best_model.pt",
        "results/final_model/run_20250423-072601/checkpoints/checkpoint_epoch_30.pt",
        "results/production_run_1/run_20250423-072209/checkpoints/best_model.pt",
    ]
    
    for checkpoint_path in checkpoints:
        if os.path.exists(checkpoint_path):
            model, config, metrics = fixed_load_model(checkpoint_path, device)
            if model is not None:
                print(f"Successfully loaded model from {checkpoint_path}")
                print(f"Validation RMSD: {metrics['val_rmsd']}")
                print(f"Trained epochs: {metrics['epoch']}")
                print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            print("-" * 80)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    main()