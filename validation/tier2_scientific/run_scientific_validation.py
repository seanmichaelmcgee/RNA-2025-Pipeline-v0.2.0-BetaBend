#!/usr/bin/env python
"""
Run scientific validation for RNA 3D folding model.

This script runs the scientific validation (Tier 2) with enhanced RNA-specific
metrics to evaluate structure prediction quality.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run scientific validation for RNA 3D folding model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint", default=None)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--subset_size", type=int, default=10, help="Number of sequences to use")
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
        "max_targets": args.subset_size,
        "save_results": True,
        "image_format": "png"
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
        
        # Run validation in both modes
        print(f"\n{'='*50}\nRUNNING SCIENTIFIC VALIDATION\n{'='*50}\n")
        results = runner.run_validation("scientific", run_both_modes=True)
        
        # Log summary of results
        print(f"\n{'='*50}\nSCIENTIFIC VALIDATION SUMMARY\n{'='*50}\n")
        
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
        
        # Calculate performance gap
        if "mean_rmsd" in test_mode and "mean_rmsd" in train_mode:
            test_rmsd = test_mode["mean_rmsd"]
            train_rmsd = train_mode["mean_rmsd"]
            abs_gap = test_rmsd - train_rmsd
            rel_gap = (abs_gap / train_rmsd) * 100 if train_rmsd > 0 else 0
            
            print(f"\nPerformance Gap:")
            print(f"  Absolute: {abs_gap:.4f} Å")
            print(f"  Relative: {rel_gap:.1f}%")
            
            # Add scientific interpretation
            if rel_gap > 30:
                print("\nInterpretation: CRITICAL impact of dihedral features")
                print("Dihedral features are essential for accurate structure prediction.")
            elif rel_gap > 15:
                print("\nInterpretation: HIGH impact of dihedral features")
                print("Dihedral features significantly improve structure prediction quality.")
            elif rel_gap > 5:
                print("\nInterpretation: MEDIUM impact of dihedral features")
                print("Dihedral features provide moderate improvements in prediction quality.")
            else:
                print("\nInterpretation: LOW impact of dihedral features")
                print("Current model architecture is relatively robust to missing dihedral features.")
        
        # Display analysis if available
        if "analysis" in results and "conclusion" in results["analysis"]:
            conclusion = results["analysis"]["conclusion"]
            print("\nFeature Impact Analysis:")
            print(f"  Overall Impact: {conclusion['overall_impact']}")
            print(f"  Severity: {conclusion['severity']}")
            print(f"  Recommendation: {conclusion['recommendation']}")
        
        # Export results to markdown
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"validation_scientific_results_{timestamp}.md")
            
            with open(output_path, 'w') as f:
                f.write("# RNA 3D Folding Model - Scientific Validation Report\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add performance metrics
                if "test_mode" in results and "train_mode" in results:
                    test_rmsd = results["test_mode"].get("mean_rmsd")
                    train_rmsd = results["train_mode"].get("mean_rmsd")
                    
                    if test_rmsd is not None and train_rmsd is not None:
                        abs_gap = test_rmsd - train_rmsd
                        rel_gap = (abs_gap / train_rmsd) * 100 if train_rmsd > 0 else 0
                        
                        f.write("## Performance Metrics\n\n")
                        f.write("| Metric | Test Mode | Train Mode | Difference |\n")
                        f.write("|--------|-----------|------------|------------|\n")
                        f.write(f"| RMSD (Å) | {test_rmsd:.4f} | {train_rmsd:.4f} | {abs_gap:.4f} ({rel_gap:.1f}%) |\n")
                        
                        # Add TM-score if available
                        if "mean_tm_score" in results["test_mode"] and "mean_tm_score" in results["train_mode"]:
                            test_tm = results["test_mode"].get("mean_tm_score")
                            train_tm = results["train_mode"].get("mean_tm_score")
                            
                            if test_tm is not None and train_tm is not None:
                                tm_abs_gap = train_tm - test_tm
                                tm_rel_gap = (tm_abs_gap / test_tm) * 100 if test_tm > 0 else 0
                                f.write(f"| TM-score | {test_tm:.4f} | {train_tm:.4f} | {tm_abs_gap:.4f} ({tm_rel_gap:.1f}%) |\n")
                                
                        f.write("\n")
                    
                    # Add analysis conclusion if available
                    if "analysis" in results and "conclusion" in results["analysis"]:
                        conclusion = results["analysis"]["conclusion"]
                        f.write("## Impact Analysis\n\n")
                        f.write(f"- **Overall Impact:** {conclusion['overall_impact']}\n")
                        f.write(f"- **Severity:** {conclusion['severity']}\n")
                        f.write(f"- **Recommendation:** {conclusion['recommendation']}\n\n")
                
                # Add visualizations
                f.write("## Visualizations\n\n")
                f.write("### Performance Comparison\n\n")
                f.write(f"![Mode Comparison](mode_comparison_scientific.png)\n\n")
                f.write(f"![Metrics Comparison](metrics_comparison_scientific.png)\n\n")
                
            print(f"\nScientific validation results exported to {output_path}")
        except Exception as e:
            print(f"\nFailed to export results: {e}")
        
        print(f"\nDetailed results and visualizations saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error running validation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())