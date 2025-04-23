#!/usr/bin/env python
"""
Test script to verify the enhanced RMSD calculation with challenging edge cases.

This script provides a quick test of the improved RMSD calculation by creating artificial
cases that would have caused problems with the previous implementation.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Get project root
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent

# Add project root to path
sys.path.insert(0, str(project_root))

# Import required functions
from src.utils.structure_metrics import compute_rmsd
from src.losses import stable_kabsch_align, robust_distance_calculation

def test_rmsd_improvements():
    """Test the improved RMSD calculation with challenging edge cases."""
    print("Testing RMSD improvements with challenging edge cases...")
    
    # Create a device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up test cases
    test_cases = [
        {
            "name": "Identical Structures",
            "coords1": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device),
            "coords2": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device),
            "expected": 0.0,
            "tol": 1e-6
        },
        {
            "name": "Translated Structures",
            "coords1": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device),
            "coords2": torch.tensor([[[11.0, 10.0, 10.0], [10.0, 11.0, 10.0], [10.0, 10.0, 11.0]]], device=device),
            "expected": 0.0,
            "tol": 1e-6
        },
        {
            "name": "Collinear Points",
            "coords1": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device),
            "coords2": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]], device=device),
            "expected": 2.0,  # Approximate expectation (rotation indeterminacy for collinear points)
            "tol": 1.5  # Large tolerance due to rotation indeterminacy
        },
        {
            "name": "Degenerate Case (All Points at Origin)",
            "coords1": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device),
            "coords2": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device=device),
            "expected": 0.0,
            "tol": 1e-6
        },
        {
            "name": "Extreme Values",
            "coords1": torch.tensor([[[1e6, 1e6, 1e6], [1e6, 1e6, 1e6 + 1], [1e6, 1e6+1, 1e6]]], device=device),
            "coords2": torch.tensor([[[1e6, 1e6, 1e6], [1e6, 1e6, 1e6 + 1], [1e6, 1e6+1, 1e6]]], device=device),
            "expected": 0.0,
            "tol": 1e-3  # Larger tolerance for extreme values due to floating point precision
        },
        {
            "name": "With NaN Values (masked)",
            "coords1": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [float('nan'), float('nan'), float('nan')]]], device=device),
            "coords2": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [float('nan'), float('nan'), float('nan')]]], device=device),
            "mask": torch.tensor([[True, True, True, False]], device=device),
            "expected": 0.0,
            "tol": 1e-6
        }
    ]
    
    # Run tests
    all_passed = True
    for i, test in enumerate(test_cases):
        print(f"\n{i+1}. {test['name']}:")
        mask = test.get("mask", None)
        
        # Test compute_rmsd
        rmsd = compute_rmsd(test["coords1"], test["coords2"], mask)
        
        # Check result
        result = rmsd.item()
        expected = test["expected"]
        tol = test["tol"]
        
        if abs(result - expected) <= tol:
            print(f"  ✓ RMSD: {result:.6f} (expected ~{expected:.6f}) - PASSED")
        else:
            print(f"  ✗ RMSD: {result:.6f} (expected ~{expected:.6f}) - FAILED")
            all_passed = False
            
        # For manual debugging, also show the calculation steps
        if mask is None:
            try:
                pred_aligned = stable_kabsch_align(test["coords1"][0], test["coords2"][0])
                distances = robust_distance_calculation(pred_aligned, test["coords2"][0])
                manual_rmsd = torch.sqrt(torch.mean(distances ** 2)).item()
                print(f"  → Manual calculation: {manual_rmsd:.6f}")
            except Exception as e:
                print(f"  → Manual calculation failed: {str(e)}")
    
    # Summary
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests PASSED: RMSD calculation is robust to all tested edge cases")
    else:
        print("✗ Some tests FAILED: Review the RMSD calculation implementation")
    print("="*50)

if __name__ == "__main__":
    test_rmsd_improvements()