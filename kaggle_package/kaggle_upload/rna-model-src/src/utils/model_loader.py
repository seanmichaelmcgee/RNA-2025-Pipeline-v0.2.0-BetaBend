"""
RNA Model Loader

Handles loading model checkpoints with special handling for corrupted checkpoints.
Provides a clean interface for the inference notebook to load models consistently.
"""

import os
import sys
import torch
from pathlib import Path

# Import project modules (assuming src is in path)
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

# Helper function to test a model's basic functionality
def test_model_inference(model, device, seq_length=50):
    """
    Test if a model can perform basic inference.
    
    Args:
        model: The RNAFoldingModel to test
        device: The device to run on
        seq_length: Length of test sequence
    
    Returns:
        bool: True if model can perform inference, False otherwise
    """
    try:
        # Create random input tensors
        batch_size = 1
        embed_dim = model.residue_embed_dim
        
        # Create dummy input
        seq_embed = torch.randn(batch_size, seq_length, embed_dim, device=device)
        padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
        
        # Run forward pass with mixed precision for memory efficiency
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                outputs = model(seq_embed, padding_mask)
                
        # Check output shape (coordinates should be [batch, seq_len, 4, 3])
        if 'coordinates' in outputs and outputs['coordinates'].shape == (batch_size, seq_length, 4, 3):
            return True
        else:
            print(f"Unexpected output shape: {outputs['coordinates'].shape}")
            return False
            
    except Exception as e:
        print(f"Error during test inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return False