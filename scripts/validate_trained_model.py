#!/usr/bin/env python3
"""
RNA 3D Structure Prediction Validation Script for Trained Models

This script runs validation on a trained model, calculating RMSD and other metrics.
It's useful for monitoring training progress or evaluating models after training.

Usage:
    python scripts/validate_trained_model.py --checkpoint path/to/model.pt --sequences_csv data/sequences.csv 
                                           --labels_csv data/labels.csv --features_dir data/features/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to Python path so we can import modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data_loading import create_data_loader
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_combined_loss
from src.utils.structure_metrics import compute_rmsd, compute_tm_score, compute_per_residue_rmsd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RNA Model Validation for Trained Models')
    
    # Data paths
    parser.add_argument('--sequences_csv', type=str, required=True,
                        help='Path to RNA sequences CSV file')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to structure labels CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing feature files (thermo, dihedral, mi)')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Directory to save validation results')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu). If None, use cuda if available.')
    
    # Validation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate and save visualization plots')
    
    # Dataset filtering
    parser.add_argument('--max_seq_length', type=int, default=200,
                        help='Maximum sequence length to consider')
    parser.add_argument('--min_seq_length', type=int, default=10,
                        help='Minimum sequence length to consider')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to validate')
    
    return parser.parse_args()


def load_trained_model(checkpoint_path: str, device: torch.device) -> RNAFoldingModel:
    """
    Load the RNA folding model from a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        device: Device to load the model on
    
    Returns:
        The loaded model
    """
    # Load checkpoint
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint if available
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = checkpoint["model_config"]
        logger.info("Using model configuration from checkpoint")
    else:
        # Default configuration as fallback
        logger.warning("No model configuration found in checkpoint, using default")
        config = {
            "num_blocks": 4,
            "residue_embed_dim": 128,
            "pair_embed_dim": 64,
            "num_attention_heads": 4,
            "dropout": 0.1,
            "ffn_dim": 512,
            "max_relative_position": 32,
            "seq_embed_dim": 16,
            "num_embeddings": 5,  # A, C, G, U, padding
            "padding_idx": 4,
            "ipa_dim": 64,
            "max_len": 500,
            "use_conservation": False
        }
    
    # Create model with the determined config
    model = RNAFoldingModel(config).to(device)
    
    # Load state dict
    try:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("Successfully loaded model weights from checkpoint")
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            logger.info("Successfully loaded model weights from checkpoint")
        else:
            # Assume checkpoint is just the state_dict
            model.load_state_dict(checkpoint)
            logger.info("Successfully loaded model weights directly from checkpoint")
    except RuntimeError as e:
        logger.error(f"Error loading model weights: {e}")
        raise RuntimeError("Failed to load model weights from checkpoint")
    
    # Extract training epoch and validation metrics if available
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            logger.info(f"Model was trained for {checkpoint['epoch']+1} epochs")
        if "val_metrics" in checkpoint:
            logger.info(f"Validation metrics from checkpoint: {checkpoint['val_metrics']}")
    
    return model


def generate_visualizations(results: Dict, output_dir: str, save_format: str = "png"):
    """
    Generate and save visualization plots.
    
    Args:
        results: Dictionary of validation results
        output_dir: Directory to save plots
        save_format: Image format for saved plots
    """
    # Create RMSD histogram
    if "rmsd_values" in results and results["rmsd_values"]:
        # Filter out NaN values
        valid_rmsd_values = [v for v in results["rmsd_values"] if not np.isnan(v)]
        
        if valid_rmsd_values:  # Only plot if we have valid values
            plt.figure(figsize=(10, 6))
            plt.hist(valid_rmsd_values, bins=min(20, len(valid_rmsd_values)), alpha=0.7)
            plt.axvline(results["mean_rmsd"], color='red', linestyle='--', 
                      label=f'Mean: {results["mean_rmsd"]:.2f} Å')
            plt.axvline(results["median_rmsd"], color='green', linestyle='--', 
                      label=f'Median: {results["median_rmsd"]:.2f} Å')
            plt.title('RMSD Distribution')
            plt.xlabel('RMSD (Å)')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"rmsd_distribution.{save_format}"), dpi=300)
            plt.close()
        else:
            logger.warning("No valid RMSD values for histogram plot")
    
    # Create RMSD vs Sequence Length scatter plot
    if "per_target_results" in results:
        seq_lengths = []
        rmsd_values = []
        
        # Filter out NaN values
        for result in results["per_target_results"].values():
            if not np.isnan(result.get("rmsd", np.nan)):
                seq_lengths.append(result["length"])
                rmsd_values.append(result["rmsd"])
        
        if seq_lengths and rmsd_values:  # Only plot if we have valid data
            plt.figure(figsize=(10, 6))
            plt.scatter(seq_lengths, rmsd_values, alpha=0.7)
            
            # Only fit a line if we have enough data points
            if len(seq_lengths) >= 2:
                try:
                    z = np.polyfit(seq_lengths, rmsd_values, 1)
                    p = np.poly1d(z)
                    plt.plot(seq_lengths, p(seq_lengths), "r--", alpha=0.7)
                except Exception as e:
                    logger.warning(f"Could not fit trend line: {e}")
            
            plt.title('RMSD vs Sequence Length')
            plt.xlabel('Sequence Length')
            plt.ylabel('RMSD (Å)')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"rmsd_vs_length.{save_format}"), dpi=300)
            plt.close()
        else:
            logger.warning("No valid data for RMSD vs Length plot")
    
    # Create TM-score histogram
    if "tm_scores" in results and results["tm_scores"]:
        # Filter out NaN values
        valid_tm_scores = [v for v in results["tm_scores"] if not np.isnan(v)]
        
        if valid_tm_scores:  # Only plot if we have valid values
            plt.figure(figsize=(10, 6))
            plt.hist(valid_tm_scores, bins=min(20, len(valid_tm_scores)), alpha=0.7)
            
            # Only show mean line if mean is not NaN
            if not np.isnan(results.get("mean_tm_score", np.nan)):
                plt.axvline(results["mean_tm_score"], color='red', linestyle='--', 
                          label=f'Mean: {results["mean_tm_score"]:.4f}')
                plt.legend()
            
            plt.title('TM-Score Distribution')
            plt.xlabel('TM-Score')
            plt.ylabel('Count')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"tm_score_distribution.{save_format}"), dpi=300)
            plt.close()
        else:
            logger.warning("No valid TM-score values for histogram plot")


def validate_model(model, data_loader, device, args):
    """
    Run validation on a trained model.
    
    Args:
        model: The trained RNA folding model
        data_loader: Validation data loader
        device: Device to run validation on
        args: Command line arguments
        
    Returns:
        Dictionary with validation results
    """
    model.eval()  # Set model to evaluation mode
    
    # Results containers
    all_losses = []
    all_rmsd_values = []
    all_tm_scores = []
    per_target_results = {}
    
    # Loss weights (default values, can be adjusted if needed)
    loss_weights = {
        "fape": 1.0,
        "confidence": 0.1,
        "angle": 0.5
    }
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Validating")):
            # Move batch to device
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            # Run forward pass
            outputs = model(batch_on_device)
            
            # Compute combined loss
            loss, loss_components = compute_combined_loss(outputs, batch_on_device, loss_weights)
            
            # Extract batch information
            batch_size = len(batch["target_ids"])
            
            # Calculate RMSD and TM-score
            pred_coords = outputs["pred_coords"]
            true_coords = batch_on_device["coordinates"]
            mask = batch_on_device["mask"]
            
            batch_rmsd = compute_rmsd(pred_coords, true_coords, mask)
            batch_tm_score = compute_tm_score(pred_coords, true_coords, mask)
            
            # Process each sample in the batch
            for i in range(batch_size):
                target_id = batch["target_ids"][i]
                seq_len = batch_on_device["lengths"][i].item()
                
                # Record loss, RMSD, and TM-score
                all_losses.append(loss.item())
                all_rmsd_values.append(batch_rmsd[i].item())
                all_tm_scores.append(batch_tm_score[i].item())
                
                # Store per-target results
                per_target_results[target_id] = {
                    "length": seq_len,
                    "rmsd": batch_rmsd[i].item(),
                    "tm_score": batch_tm_score[i].item(),
                    "loss": loss.item(),
                    "fape_loss": loss_components["fape"].item(),
                    "confidence_loss": loss_components["confidence"].item(),
                    "angle_loss": loss_components["angle"].item()
                }
    
    # Calculate aggregate statistics
    results = {
        "num_samples": len(all_rmsd_values),
        "mean_loss": np.mean(all_losses),
        "mean_rmsd": np.mean(all_rmsd_values),
        "median_rmsd": np.median(all_rmsd_values),
        "min_rmsd": np.min(all_rmsd_values),
        "max_rmsd": np.max(all_rmsd_values),
        "mean_tm_score": np.mean(all_tm_scores),
        "rmsd_values": all_rmsd_values,
        "tm_scores": all_tm_scores,
        "per_target_results": per_target_results
    }
    
    return results


def main():
    """Main function to run the validation process."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    try:
        model = load_trained_model(args.checkpoint, device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Filter sequences by length and limit sample count
    import pandas as pd
    import random
    
    # Read and filter sequences based on length
    sequences_df = pd.read_csv(args.sequences_csv)
    sequences_df['length'] = sequences_df['sequence'].apply(len)
    
    filtered_df = sequences_df[(sequences_df['length'] >= args.min_seq_length) & 
                              (sequences_df['length'] <= args.max_seq_length)]
    
    logger.info(f"Filtered from {len(sequences_df)} to {len(filtered_df)} sequences with length between {args.min_seq_length} and {args.max_seq_length}")
    
    # Limit samples if max_samples is specified
    if args.max_samples is not None and len(filtered_df) > args.max_samples:
        logger.info(f"Limiting dataset to {args.max_samples} random samples for validation")
        filtered_df = filtered_df.sample(args.max_samples, random_state=42)
    
    # Custom split function that only includes the selected target IDs
    def val_split_fn(df):
        return df[df['target_id'].isin(filtered_df['target_id'])]
    
    # Create validation data loader
    val_loader = create_data_loader(
        sequences_csv_path=args.sequences_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        split_fn=val_split_fn,
        require_features=True
    )
    
    logger.info(f"Created validation data loader with {len(val_loader.dataset)} samples")
    
    # Run validation
    results = validate_model(model, val_loader, device, args)
    
    # Log results
    logger.info("\nValidation Results:")
    logger.info(f"Number of samples: {results['num_samples']}")
    logger.info(f"Mean loss: {results['mean_loss']:.4f}")
    logger.info(f"Mean RMSD: {results['mean_rmsd']:.4f} Å")
    logger.info(f"Median RMSD: {results['median_rmsd']:.4f} Å")
    logger.info(f"Min/Max RMSD: {results['min_rmsd']:.4f}/{results['max_rmsd']:.4f} Å")
    logger.info(f"Mean TM-score: {results['mean_tm_score']:.4f}")
    
    # Generate visualizations
    if args.generate_plots:
        generate_visualizations(results, args.output_dir)
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, "validation_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy values to native Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_to_json_serializable(results), f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate success/failure determination
    if results["mean_rmsd"] < 15.0:  # This threshold can be adjusted
        logger.info("\n✅ VALIDATION PASSED: Model produces reasonable coordinates")
    else:
        logger.info("\n❌ VALIDATION FAILED: RMSD values are too high")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())