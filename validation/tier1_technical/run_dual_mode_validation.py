#!/usr/bin/env python
"""
Run dual-mode validation for RNA 3D folding model.

This script demonstrates the dual-mode validation capability to quantify the impact
of feature availability differences between training and testing environments.
"""

import os
import sys
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Run dual-mode validation for RNA 3D folding model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint", default=None)
    parser.add_argument("--subset", type=str, default="technical", 
                        choices=["technical", "scientific", "comprehensive"],
                        help="Validation subset to use")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (not recommended)")
    parser.add_argument("--rna-ids", type=str, nargs='+', 
                        help="Specific RNA IDs to validate", default=None)
    args = parser.parse_args()
    
    # Get project root
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent.parent
    
    # Add project root to path
    sys.path.insert(0, str(project_root))
    
    # Add validation directory to path
    validation_dir = os.path.join(project_root, "validation")
    if validation_dir not in sys.path:
        sys.path.insert(0, validation_dir)
    
    # Import required modules
    from src.models.rna_folding_model import RNAFoldingModel
    from validation.validation_runner import ValidationRunner
    
    # Determine device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine data directory
    if args.data_dir is None:
        data_dir = os.path.join(project_root, "data")
    else:
        data_dir = args.data_dir
    print(f"Using data directory: {data_dir}")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(current_dir, "results")
    else:
        output_dir = args.output_dir
    print(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration
    config = {
        "batch_size": args.batch_size,
        "results_dir": output_dir,
        "verbose": True,
    }
    
    # Add RNA IDs to configuration if provided
    if args.rna_ids:
        print(f"Filtering validation to specific RNA IDs: {', '.join(args.rna_ids)}")
        config["target_ids"] = args.rna_ids
    
    try:
        # Create default model
        if args.checkpoint is None:
            print("No checkpoint provided, initializing default model")
            model = RNAFoldingModel()
        else:
            # Load model from checkpoint
            print(f"Loading model from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "model_state_dict" in checkpoint:
                model = RNAFoldingModel()
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                print("Checkpoint doesn't contain model_state_dict, trying direct load")
                model = RNAFoldingModel()
                model.load_state_dict(checkpoint)
        
        # Run validation
        runner = ValidationRunner(model, data_dir, config, device)
        
        # Import utility functions for debugging
        from src.utils.structure_metrics import compute_rmsd, compute_tm_score
        from src.losses import stable_kabsch_align, robust_distance_calculation
        
        # Verify that structure_metrics module is being used correctly
        print("Verifying structure metrics imports...")
        print(f"  compute_rmsd imported from: {compute_rmsd.__module__}")
        print(f"  compute_tm_score imported from: {compute_tm_score.__module__}")
        
        # Run validation in both modes
        print(f"\n{'='*50}\nRUNNING DUAL-MODE VALIDATION\n{'='*50}\n")
        results = runner.run_validation(args.subset, run_both_modes=True)
        
        # Log summary of results
        print(f"\n{'='*50}\nDUAL-MODE VALIDATION SUMMARY\n{'='*50}\n")
        
        # Display test mode results
        test_mode = results["test_mode"]
        if "error" not in test_mode:
            print("\nTest-Equivalent Mode (No Dihedral Features):")
            print(f"  Mean RMSD: {test_mode.get('mean_rmsd', 'N/A'):.4f} Å")
            if "mean_tm_score" in test_mode:
                print(f"  Mean TM-score: {test_mode.get('mean_tm_score', 'N/A'):.4f}")
        else:
            print(f"\nTest-Equivalent Mode: Error - {test_mode['error']}")
        
        # Display train mode results
        train_mode = results["train_mode"]
        if "error" not in train_mode:
            print("\nTraining-Equivalent Mode (With Dihedral Features):")
            print(f"  Mean RMSD: {train_mode.get('mean_rmsd', 'N/A'):.4f} Å")
            if "mean_tm_score" in train_mode:
                print(f"  Mean TM-score: {train_mode.get('mean_tm_score', 'N/A'):.4f}")
        else:
            print(f"\nTraining-Equivalent Mode: Error - {train_mode['error']}")
        
        # Display analysis if available
        if "analysis" in results and "conclusion" in results["analysis"]:
            conclusion = results["analysis"]["conclusion"]
            print("\nImpact Analysis:")
            print(f"  Overall Impact: {conclusion['overall_impact']}")
            print(f"  Severity: {conclusion['severity']}")
            print(f"  Recommendation: {conclusion['recommendation']}")
        
        print(f"\nDetailed results and visualizations saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error running validation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())