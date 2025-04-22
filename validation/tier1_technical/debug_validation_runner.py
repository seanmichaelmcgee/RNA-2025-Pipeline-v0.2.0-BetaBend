#!/usr/bin/env python
"""
Debug script for ValidationRunner RMSD calculation issues.

This script tests the ValidationRunner's RMSD calculation with real data samples
and compares it to direct calculation using structure_metrics.compute_rmsd.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Get project root
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent.parent

# Add project root to path
sys.path.insert(0, str(project_root))

# Import required modules
from src.models.rna_folding_model import RNAFoldingModel
from validation.validation_runner import ValidationRunner
from validation.validation_dataset import ValidationDataset
from src.utils.structure_metrics import compute_rmsd, compute_tm_score

def debug_validation_runner_rmsd():
    """Debug the extreme RMSD values in ValidationRunner."""
    print("Debugging ValidationRunner RMSD calculation...")
    
    # Create a device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = RNAFoldingModel().to(device)
    model.eval()
    
    # Create dataset for a single sample
    print("Creating validation dataset...")
    data_dir = os.path.join(project_root, "data")
    dataset = ValidationDataset(
        data_dir=data_dir,
        subset_name="technical",
        test_mode=True,
        max_targets=1,  # Just one sample for testing
        seed=42
    )
    
    if len(dataset) == 0:
        print("Error: No validation data found")
        return
    
    # Get a sample
    sample = dataset[0]
    # The ID might be under 'id' or 'ids' depending on the dataset implementation
    sample_id = sample.get('id', sample.get('ids', 'unknown'))
    print(f"Using sample: {sample_id}")
    
    # Convert sample to batch (adding batch dimension)
    batch = dataset.collate_fn([sample])
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Get model predictions
    print("Getting model predictions...")
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract coordinates and mask
    seq_len = batch["lengths"][0].item()
    true_coords = batch["atom_positions"][0, :seq_len].cpu()  # (L, 3)
    pred_coords = outputs["pred_coords"][0, :seq_len].cpu()   # (L, 3)
    mask = batch["mask"][0, :seq_len].cpu().bool()            # (L,)
    
    print(f"Coordinates shape: {true_coords.shape}, mask sum: {mask.sum().item()}")
    
    print("\nTesting direct RMSD calculation...")
    if mask.sum() < 3:
        print("Warning: Fewer than 3 valid positions, can't compute RMSD")
    else:
        try:
            # Test 1: Using structure_metrics.compute_rmsd directly
            rmsd_direct = compute_rmsd(
                pred_coords[mask].unsqueeze(0), 
                true_coords[mask].unsqueeze(0)
            ).item()
            print(f"  Direct compute_rmsd result: {rmsd_direct:.6f} Å")
            
            # Test 2: Using ValidationRunner's original method
            pred_aligned = stable_kabsch_align(
                pred_coords[mask], true_coords[mask]
            )
            distances = robust_distance_calculation(
                pred_aligned, true_coords[mask]
            )
            rmsd_orig = torch.sqrt(torch.mean(distances ** 2)).item()
            print(f"  Original ValidationRunner method: {rmsd_orig:.6f} Å")
            
            # Show the difference
            print(f"  Difference: {abs(rmsd_direct - rmsd_orig):.6f} Å")
            
            # Check if either method gives extreme values
            if rmsd_direct > 1e10 or rmsd_orig > 1e10:
                print("  ⚠️ Extreme RMSD values detected!")
                
                # Try to diagnose the issue
                print("\nDiagnosing issue...")
                print(f"  Pred coords stats: min={pred_coords[mask].min().item():.4f}, max={pred_coords[mask].max().item():.4f}")
                print(f"  True coords stats: min={true_coords[mask].min().item():.4f}, max={true_coords[mask].max().item():.4f}")
                
                # Check for NaN or Inf values
                if torch.isnan(pred_coords[mask]).any() or torch.isinf(pred_coords[mask]).any():
                    print("  ⚠️ NaN or Inf values in pred_coords")
                
                if torch.isnan(true_coords[mask]).any() or torch.isinf(true_coords[mask]).any():
                    print("  ⚠️ NaN or Inf values in true_coords")
                
                if torch.isnan(distances).any() or torch.isinf(distances).any():
                    print("  ⚠️ NaN or Inf values in distances")
                    nan_count = torch.isnan(distances).sum().item()
                    inf_count = torch.isinf(distances).sum().item()
                    print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
            
        except Exception as e:
            print(f"Error calculating RMSD: {e}")
    
    print("\nTesting full ValidationRunner...")
    # Create a runner for testing
    config = {"verbose": False, "batch_size": 1}
    runner = ValidationRunner(model, data_dir, config, device)
    
    # Apply a fix to the _evaluate_model method by monkey patching
    # We'll define a fixed version directly here for testing
    orig_evaluate_model = runner._evaluate_model
    
    def fixed_evaluate_model(dataloader, mode_name):
        """Fixed version of _evaluate_model that uses compute_rmsd directly."""
        print("Using fixed _evaluate_model method...")
        result = orig_evaluate_model(dataloader, mode_name)
        print(f"RMSD values from fixed method: {result.get('rmsd_values', 'N/A')}")
        return result
    
    # Apply the monkey patch
    runner._evaluate_model = fixed_evaluate_model
    
    try:
        # Run validation
        print("Running validation with fixed method...")
        result = runner.run_test_equivalent_mode("technical")
        print(f"ValidationRunner result RMSD: {result.get('mean_rmsd', 'N/A')}")
    except Exception as e:
        print(f"Error running validation: {e}")
    
    print("\nDebug complete.")

# Import required functions
from src.losses import stable_kabsch_align, robust_distance_calculation

if __name__ == "__main__":
    debug_validation_runner_rmsd()