#!/usr/bin/env python3
"""
Validation Script for RNA 3D Structure Prediction Model

This script performs a tiered validation of the RNA folding model:
1. Tier 1 (Fast Technical Validation): Basic model functionality and test set metrics
2. Tier 2 (Scientific Validation): More comprehensive evaluation on challenging cases
3. Tier 3 (Comprehensive Validation): Full dataset evaluation and comparison metrics

Usage:
    python scripts/validate_model.py --tier 1 --checkpoint path/to/model.pt
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import model components
from src.models.rna_folding_model import RNAFoldingModel
from src.utils.structure_metrics import compute_structure_metrics, compute_per_residue_rmsd
from src.data_loading import RNADataset, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate RNA 3D Structure Prediction Model")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Validation tier level (1=Fast Technical, 2=Scientific, 3=Comprehensive)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint file (.pt)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory with validation data")
    parser.add_argument("--output_dir", type=str, default="validation_results",
                        help="Directory to save validation results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda', 'cpu'). If None, use CUDA if available.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to evaluate (for quick testing)")
    return parser.parse_args()


def load_model(checkpoint_path: Optional[str], device: torch.device) -> RNAFoldingModel:
    """
    Load model from a checkpoint or create a new model with default parameters.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint has 'model_config' or 'state_dict'
        if "model_config" in checkpoint:
            model = RNAFoldingModel(checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume it's just a state dict
            model = RNAFoldingModel().to(device)
            model.load_state_dict(checkpoint)
    else:
        logger.info("Creating new model with default parameters")
        # Default model configuration
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
            "use_conservation": False,
        }
        model = RNAFoldingModel(config).to(device)
    
    return model


def create_dataloader(args) -> DataLoader:
    """
    Create a DataLoader for validation data.
    
    Args:
        args: Command-line arguments
        
    Returns:
        DataLoader for validation data
    """
    # For Tier 1, use a small test set
    if args.tier == 1:
        dataset = RNADataset(
            data_dir=args.data_dir,
            split="test",
            max_length=200,  # Limit sequence length for fast testing
        )
    # For Tier 2, use a curated set of challenging cases
    elif args.tier == 2:
        dataset = RNADataset(
            data_dir=args.data_dir,
            split="validation",
            max_length=400,
        )
    # For Tier 3, use the full validation set
    else:
        dataset = RNADataset(
            data_dir=args.data_dir,
            split="validation",
            max_length=None,  # No length limit
        )
    
    # Limit number of samples for quick testing if specified
    if args.num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.num_samples, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)} sequences")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() or 1),
    )
    
    return dataloader


def evaluate_model(model: RNAFoldingModel, dataloader: DataLoader, 
                  device: torch.device, tier: int) -> Dict:
    """
    Evaluate model on validation data.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for validation data
        device: Device to run evaluation on
        tier: Validation tier level
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    
    # Track metrics
    results = {
        "rmsd": [],
        "tm_score": [],
        "per_sequence": {},
        "timing": [],
    }
    
    # Additional metrics for higher tiers
    if tier >= 2:
        results["per_residue_rmsd"] = []
        results["per_length_rmsd"] = {
            "short": [],   # < 50 nucleotides
            "medium": [],  # 50-150 nucleotides
            "long": [],    # > 150 nucleotides
        }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"Processing batch {batch_idx+1}/{len(dataloader)}")
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Record inference time
            start_time = time.time()
            outputs = model(batch)
            inference_time = time.time() - start_time
            results["timing"].append(inference_time)
            
            # Get structure metrics
            metrics = compute_structure_metrics(
                outputs["pred_coords"], 
                batch["coordinates"], 
                batch["mask"]
            )
            
            # Add metrics to results
            results["rmsd"].extend(metrics["rmsd"].cpu().numpy().tolist())
            results["tm_score"].extend(metrics["tm_score"].cpu().numpy().tolist())
            
            # Store per-sequence results
            for i, seq_id in enumerate(batch["target_ids"]):
                seq_len = batch["lengths"][i].item()
                results["per_sequence"][seq_id] = {
                    "rmsd": metrics["rmsd"][i].item(),
                    "tm_score": metrics["tm_score"][i].item(),
                    "length": seq_len,
                    "inference_time": inference_time / len(batch["target_ids"]),
                }
                
                # Group by sequence length for tier 2+
                if tier >= 2:
                    if seq_len < 50:
                        length_category = "short"
                    elif seq_len <= 150:
                        length_category = "medium"
                    else:
                        length_category = "long"
                    results["per_length_rmsd"][length_category].append(metrics["rmsd"][i].item())
            
            # Calculate per-residue RMSD for tier 2+
            if tier >= 2:
                per_res_rmsd = compute_per_residue_rmsd(
                    outputs["pred_coords"], 
                    batch["coordinates"], 
                    batch["mask"],
                    window_size=3  # Use window for smoother results
                )
                results["per_residue_rmsd"].append(per_res_rmsd.cpu().numpy())
    
    # Calculate summary statistics
    results["mean_rmsd"] = np.mean(results["rmsd"])
    results["median_rmsd"] = np.median(results["rmsd"])
    results["mean_tm_score"] = np.mean(results["tm_score"])
    results["mean_inference_time"] = np.mean(results["timing"]) / args.batch_size
    
    # Additional statistics for higher tiers
    if tier >= 2:
        results["per_length_stats"] = {
            category: {
                "mean_rmsd": np.mean(values) if values else float('nan'),
                "count": len(values)
            }
            for category, values in results["per_length_rmsd"].items()
        }
    
    return results


def save_results(results: Dict, args):
    """
    Save validation results to output directory.
    
    Args:
        results: Dictionary of validation results
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save summary metrics to JSON file
    import json
    summary = {
        "tier": args.tier,
        "mean_rmsd": results["mean_rmsd"],
        "median_rmsd": results["median_rmsd"],
        "mean_tm_score": results["mean_tm_score"],
        "mean_inference_time": results["mean_inference_time"],
        "num_sequences": len(results["rmsd"]),
    }
    
    # Add tier-specific metrics
    if args.tier >= 2:
        summary["per_length_stats"] = results["per_length_stats"]
    
    with open(os.path.join(args.output_dir, f"tier{args.tier}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save per-sequence results to CSV file
    import csv
    with open(os.path.join(args.output_dir, f"tier{args.tier}_per_sequence.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "length", "rmsd", "tm_score", "inference_time"])
        for seq_id, seq_results in results["per_sequence"].items():
            writer.writerow([
                seq_id,
                seq_results["length"],
                seq_results["rmsd"],
                seq_results["tm_score"],
                seq_results["inference_time"],
            ])
    
    # Create visualizations for tier 2+
    if args.tier >= 2:
        create_visualizations(results, args)


def create_visualizations(results: Dict, args):
    """
    Create visualizations of validation results.
    
    Args:
        results: Dictionary of validation results
        args: Command-line arguments
    """
    # RMSD histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results["rmsd"], bins=20, alpha=0.7)
    plt.axvline(results["mean_rmsd"], color='r', linestyle='--', 
                label=f'Mean RMSD: {results["mean_rmsd"]:.2f}Å')
    plt.axvline(results["median_rmsd"], color='g', linestyle='--', 
                label=f'Median RMSD: {results["median_rmsd"]:.2f}Å')
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RMSD Values')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"tier{args.tier}_rmsd_histogram.png"))
    plt.close()
    
    # TM-score histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results["tm_score"], bins=20, alpha=0.7)
    plt.axvline(results["mean_tm_score"], color='r', linestyle='--', 
                label=f'Mean TM-score: {results["mean_tm_score"]:.2f}')
    plt.xlabel('TM-score')
    plt.ylabel('Frequency')
    plt.title('Distribution of TM-score Values')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"tier{args.tier}_tm_score_histogram.png"))
    plt.close()
    
    # RMSD vs sequence length scatter plot
    lengths = [seq_results["length"] for seq_results in results["per_sequence"].values()]
    rmsds = [seq_results["rmsd"] for seq_results in results["per_sequence"].values()]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, rmsds, alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD vs Sequence Length')
    plt.grid(alpha=0.3)
    
    # Add trend line
    if len(lengths) > 1:
        z = np.polyfit(lengths, rmsds, 1)
        p = np.poly1d(z)
        plt.plot(sorted(lengths), p(sorted(lengths)), "r--", 
                label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"tier{args.tier}_rmsd_vs_length.png"))
    plt.close()
    
    # Bar chart of RMSD by sequence length category
    if args.tier >= 2:
        categories = list(results["per_length_stats"].keys())
        means = [stats["mean_rmsd"] for stats in results["per_length_stats"].values()]
        counts = [stats["count"] for stats in results["per_length_stats"].values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, means, alpha=0.7)
        
        # Add count labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={counts[i]}', ha='center', va='bottom')
        
        plt.xlabel('Sequence Length Category')
        plt.ylabel('Mean RMSD (Å)')
        plt.title('Mean RMSD by Sequence Length Category')
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"tier{args.tier}_rmsd_by_category.png"))
        plt.close()


def main(args):
    """
    Main function for model validation.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Running Tier {args.tier} validation")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create dataloader
    dataloader = create_dataloader(args)
    
    # Evaluate model
    results = evaluate_model(model, dataloader, device, args.tier)
    
    # Display summary
    logger.info(f"Validation Results (Tier {args.tier}):")
    logger.info(f"  Mean RMSD: {results['mean_rmsd']:.4f} Å")
    logger.info(f"  Median RMSD: {results['median_rmsd']:.4f} Å")
    logger.info(f"  Mean TM-score: {results['mean_tm_score']:.4f}")
    logger.info(f"  Mean inference time: {results['mean_inference_time']*1000:.2f} ms/sequence")
    
    if args.tier >= 2:
        logger.info("  Results by sequence length:")
        for category, stats in results["per_length_stats"].items():
            logger.info(f"    {category} ({stats['count']} sequences): "
                       f"Mean RMSD = {stats['mean_rmsd']:.4f} Å")
    
    # Save results
    save_results(results, args)
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)