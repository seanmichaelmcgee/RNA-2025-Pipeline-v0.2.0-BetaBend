#!/usr/bin/env python
"""
Debug script for RMSD calculation in the ValidationRunner.

This script creates a simple test to verify that the RMSD calculation in ValidationRunner
works correctly by using direct invocation with controlled test data.
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

# Import required functions
from src.utils.structure_metrics import compute_rmsd, compute_tm_score
from src.losses import stable_kabsch_align, robust_distance_calculation

def test_rmsd_calculation():
    """Test RMSD calculation with controlled synthetic data."""
    print("Running RMSD calculation tests with synthetic data...")
    
    # Create a device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Case 1: Simple test with identity transformation
    print("\nCase 1: Identity transformation (should give RMSD = 0)")
    pred_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
    true_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    
    # Test compute_rmsd
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  compute_rmsd result: {rmsd.item():.6f}")
    
    # Compare with manual calculation
    pred_aligned = stable_kabsch_align(pred_coords[0], true_coords[0])
    distances = robust_distance_calculation(pred_aligned, true_coords[0])
    manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
    print(f"  manual calculation result: {manual_rmsd:.6f}")
    
    # Case 2: Translation
    print("\nCase 2: Translation (should give RMSD = 0 after alignment)")
    pred_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
    true_coords = torch.tensor([[[11.0, 10.0, 10.0], [10.0, 11.0, 10.0], [10.0, 10.0, 11.0]]], device=device)  # Translated by (10,10,10)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    
    # Test compute_rmsd
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  compute_rmsd result: {rmsd.item():.6f}")
    
    # Compare with manual calculation
    pred_aligned = stable_kabsch_align(pred_coords[0], true_coords[0])
    distances = robust_distance_calculation(pred_aligned, true_coords[0])
    manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
    print(f"  manual calculation result: {manual_rmsd:.6f}")
    
    # Case 3: Rotation
    print("\nCase 3: Rotation (should give RMSD = 0 after alignment)")
    theta = torch.tensor(np.pi/4, device=device)  # 45-degree rotation
    rot_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], device=device)
    
    pred_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device)
    true_coords = torch.matmul(pred_coords, rot_matrix)  # Apply rotation
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    
    # Test compute_rmsd
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  compute_rmsd result: {rmsd.item():.6f}")
    
    # Compare with manual calculation
    pred_aligned = stable_kabsch_align(pred_coords[0], true_coords[0])
    distances = robust_distance_calculation(pred_aligned, true_coords[0])
    manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
    print(f"  manual calculation result: {manual_rmsd:.6f}")
    
    # Case 4: Random noise (should give non-zero RMSD)
    print("\nCase 4: Random noise (should give non-zero RMSD)")
    torch.manual_seed(42)  # For reproducibility
    pred_coords = torch.randn(1, 5, 3, device=device)
    true_coords = torch.randn(1, 5, 3, device=device)
    mask = torch.ones(1, 5, dtype=torch.bool, device=device)
    
    # Test compute_rmsd
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  compute_rmsd result: {rmsd.item():.6f}")
    
    # Compare with manual calculation
    pred_aligned = stable_kabsch_align(pred_coords[0], true_coords[0])
    distances = robust_distance_calculation(pred_aligned, true_coords[0])
    manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
    print(f"  manual calculation result: {manual_rmsd:.6f}")
    
    # Case 5: Masked points (should only compute RMSD for unmasked points)
    print("\nCase 5: Masked points (should only compute RMSD for unmasked points)")
    pred_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], device=device)
    true_coords = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]], device=device)
    mask = torch.tensor([[True, True, True, False, False]], device=device)  # Mask out the last two points
    
    # Test compute_rmsd
    rmsd = compute_rmsd(pred_coords, true_coords, mask)
    print(f"  compute_rmsd result: {rmsd.item():.6f}")
    
    # Compare with manual calculation - filtered by mask first
    pred_filtered = pred_coords[0][mask[0]]
    true_filtered = true_coords[0][mask[0]]
    
    pred_aligned = stable_kabsch_align(pred_filtered, true_filtered)
    distances = robust_distance_calculation(pred_aligned, true_filtered)
    manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
    print(f"  manual calculation result: {manual_rmsd:.6f}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    test_rmsd_calculation()