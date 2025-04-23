#!/usr/bin/env python3
"""
Test script to compare RMSD calculation using C1' atoms vs. all atoms
for RNA-Puzzles structures.

This script:
1. Extracts C1' atoms from PDB files using our pipeline's approach
2. Computes RMSD using our implementation
3. Compares with published RMSD values and other atom selection strategies
"""

import os
import sys
import csv
import json
import logging
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent.parent))

# Import our RMSD calculation
from src.utils.structure_metrics import compute_rmsd, compute_per_residue_rmsd
from validation.rmsd_benchmark.scripts.pdb_parser import (
    parse_pdb_coordinates, 
    extract_phosphate_backbone,
    extract_c4_prime_backbone,
    extract_all_heavy_atoms,
    convert_to_tensor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract_c1_prime_backbone(pdb_file, chain_id=None):
    """
    Extract C1' backbone coordinates from RNA structure.
    
    Args:
        pdb_file: Path to PDB file
        chain_id: Chain identifier to extract (e.g., "A")
    
    Returns:
        Dictionary with backbone atom coordinates and metadata
    """
    return parse_pdb_coordinates(pdb_file, atom_types=["C1'"], chain_id=chain_id)

def load_published_rmsd_values(csv_file):
    """
    Load published RMSD values from CSV file.
    
    Args:
        csv_file: Path to CSV file with RMSD values
        
    Returns:
        Dictionary mapping puzzle IDs to RMSD values
    """
    rmsd_data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = row['puzzle_id']
            rmsd_data[puzzle_id] = {
                'pdb_id': row['pdb_id'],
                'name': row['name'],
                'best_rmsd': float(row['best_rmsd']),
                'median_rmsd': float(row['median_rmsd']),
                'worst_rmsd': float(row['worst_rmsd']),
                'rmsd_type': row['rmsd_type'],
                'source': row['source']
            }
    return rmsd_data

def compare_atom_selection_strategies(pdb_dir, puzzle_id, pdb_id):
    """
    Compare RMSD calculations using different atom selection strategies.
    
    Args:
        pdb_dir: Directory with PDB files
        puzzle_id: Puzzle ID
        pdb_id: PDB ID of reference structure
        
    Returns:
        Dictionary with RMSD results for different atom selections
    """
    # Define paths
    reference_file = os.path.join(pdb_dir, 'reference', f"{pdb_id}.pdb")
    prediction_dir = os.path.join(pdb_dir, 'predictions', f"puzzle{puzzle_id}")
    
    if not os.path.exists(reference_file):
        logging.error(f"Reference file not found: {reference_file}")
        return None
        
    if not os.path.exists(prediction_dir):
        logging.error(f"Prediction directory not found: {prediction_dir}")
        return None
        
    # Find model files
    model_files = list(Path(prediction_dir).glob("*.pdb"))
    if not model_files:
        logging.warning(f"No model files found for Puzzle {puzzle_id}")
        return None
        
    logging.info(f"Processing {len(model_files)} models for Puzzle {puzzle_id} ({pdb_id})")
    
    # Define atom extractors
    atom_extractors = {
        "P": extract_phosphate_backbone,
        "C1'": extract_c1_prime_backbone,
        "C4'": extract_c4_prime_backbone,
        "all_heavy": extract_all_heavy_atoms
    }
    
    # Extract reference coordinates for each atom type
    reference_coords = {}
    for atom_type, extractor in atom_extractors.items():
        try:
            ref_data = extractor(reference_file)
            reference_coords[atom_type] = {
                "coords": torch.tensor(ref_data["coords"], dtype=torch.float32),
                "count": len(ref_data["coords"]),
                "residue_ids": ref_data["residue_ids"]
            }
            logging.info(f"  Reference has {reference_coords[atom_type]['count']} {atom_type} atoms")
        except Exception as e:
            logging.error(f"Error extracting {atom_type} atoms from reference: {e}")
            reference_coords[atom_type] = None
    
    # Calculate RMSD for each model and atom type
    results = {atom_type: [] for atom_type in atom_extractors.keys()}
    
    for model_file in model_files:
        model_name = model_file.name
        
        for atom_type, extractor in atom_extractors.items():
            if reference_coords[atom_type] is None:
                continue
                
            try:
                # Extract model coordinates
                model_data = extractor(str(model_file))
                model_coords = torch.tensor(model_data["coords"], dtype=torch.float32)
                
                # Handle sequence length mismatches
                ref_coords = reference_coords[atom_type]["coords"]
                seq_len = min(len(ref_coords), len(model_coords))
                
                if seq_len < 3:  # Need at least 3 atoms for RMSD
                    logging.warning(f"Too few {atom_type} atoms for model {model_name}")
                    continue
                    
                # Calculate RMSD with our implementation
                rmsd = compute_rmsd(
                    model_coords[:seq_len], 
                    ref_coords[:seq_len], 
                    aligned=True
                ).item()
                
                results[atom_type].append({
                    "model": model_name,
                    "rmsd": rmsd,
                    "atom_count": seq_len
                })
                
            except Exception as e:
                logging.error(f"Error calculating RMSD for {atom_type} atoms in {model_name}: {e}")
    
    # Calculate statistics for each atom type
    statistics = {}
    for atom_type, rmsd_values in results.items():
        if not rmsd_values:
            statistics[atom_type] = None
            continue
            
        # Sort by RMSD
        rmsd_values.sort(key=lambda x: x["rmsd"])
        
        # Extract RMSD values
        rmsd_list = [entry["rmsd"] for entry in rmsd_values]
        
        statistics[atom_type] = {
            "best_rmsd": rmsd_values[0],
            "median_rmsd": rmsd_values[len(rmsd_values)//2],
            "worst_rmsd": rmsd_values[-1],
            "mean_rmsd": sum(rmsd_list) / len(rmsd_list),
            "std_rmsd": float(np.std(rmsd_list)),
            "count": len(rmsd_values)
        }
        
        logging.info(f"  {atom_type} atoms: {len(rmsd_values)} models processed")
        logging.info(f"    Best RMSD: {statistics[atom_type]['best_rmsd']['rmsd']:.2f} Å")
        logging.info(f"    Median RMSD: {statistics[atom_type]['median_rmsd']['rmsd']:.2f} Å")
        logging.info(f"    Worst RMSD: {statistics[atom_type]['worst_rmsd']['rmsd']:.2f} Å")
    
    return {
        "puzzle_id": puzzle_id,
        "pdb_id": pdb_id,
        "reference_atoms": {k: v["count"] if v else 0 for k, v in reference_coords.items()},
        "results": results,
        "statistics": statistics
    }

def compare_with_published_values(results, published_values):
    """
    Compare calculated RMSD values with published values.
    
    Args:
        results: Dictionary with RMSD results
        published_values: Dictionary with published RMSD values
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for puzzle_id, puzzle_results in results.items():
        if puzzle_id not in published_values:
            logging.warning(f"No published values for Puzzle {puzzle_id}")
            continue
            
        published = published_values[puzzle_id]
        
        atom_comparisons = {}
        for atom_type, stats in puzzle_results["statistics"].items():
            if stats is None:
                continue
                
            # Calculate differences
            best_diff = abs(stats["best_rmsd"]["rmsd"] - published["best_rmsd"])
            best_rel_diff = best_diff / published["best_rmsd"] if published["best_rmsd"] > 0 else float('nan')
            
            median_diff = abs(stats["median_rmsd"]["rmsd"] - published["median_rmsd"])
            median_rel_diff = median_diff / published["median_rmsd"] if published["median_rmsd"] > 0 else float('nan')
            
            worst_diff = abs(stats["worst_rmsd"]["rmsd"] - published["worst_rmsd"])
            worst_rel_diff = worst_diff / published["worst_rmsd"] if published["worst_rmsd"] > 0 else float('nan')
            
            atom_comparisons[atom_type] = {
                "best": {
                    "calculated": stats["best_rmsd"]["rmsd"],
                    "published": published["best_rmsd"],
                    "abs_diff": best_diff,
                    "rel_diff": best_rel_diff
                },
                "median": {
                    "calculated": stats["median_rmsd"]["rmsd"],
                    "published": published["median_rmsd"],
                    "abs_diff": median_diff,
                    "rel_diff": median_rel_diff
                },
                "worst": {
                    "calculated": stats["worst_rmsd"]["rmsd"],
                    "published": published["worst_rmsd"],
                    "abs_diff": worst_diff,
                    "rel_diff": worst_rel_diff
                }
            }
            
        comparison[puzzle_id] = {
            "pdb_id": puzzle_results["pdb_id"],
            "atom_types": atom_comparisons,
            "published_type": published["rmsd_type"],
            "published_source": published["source"]
        }
    
    return comparison

def generate_comparison_report(comparison, output_dir):
    """
    Generate a report comparing different atom selection strategies.
    
    Args:
        comparison: Dictionary with comparison results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = os.path.join(output_dir, f"atom_selection_comparison_{timestamp}.md")
    
    with open(report_file, "w") as f:
        f.write("# RMSD Atom Selection Strategy Comparison\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report compares RMSD calculations using different atom selection strategies:\n\n")
        f.write("- **P atoms**: Phosphate backbone atoms\n")
        f.write("- **C1' atoms**: C1-prime sugar atoms (used in our pipeline)\n")
        f.write("- **C4' atoms**: C4-prime sugar atoms\n")
        f.write("- **All heavy atoms**: All non-hydrogen atoms\n\n")
        
        f.write("The goal is to determine whether our C1'-based RMSD calculation is comparable with published values and with other atom selection strategies.\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Puzzle | PDB ID | Atom Type | Best RMSD (Å) | Published Best (Å) | Diff (Å) | Rel Diff |\n")
        f.write("|--------|--------|-----------|---------------|-------------------|----------|----------|\n")
        
        for puzzle_id, puzzle_comparison in comparison.items():
            for atom_type, atom_comparison in puzzle_comparison["atom_types"].items():
                best = atom_comparison["best"]
                f.write(f"| {puzzle_id} | {puzzle_comparison['pdb_id']} | {atom_type} | ")
                f.write(f"{best['calculated']:.2f} | {best['published']:.2f} | ")
                f.write(f"{best['abs_diff']:.2f} | {best['rel_diff']:.1%} |\n")
        
        # Detailed results by puzzle
        f.write("\n## Detailed Results by Puzzle\n\n")
        
        for puzzle_id, puzzle_comparison in comparison.items():
            f.write(f"### Puzzle {puzzle_id} ({puzzle_comparison['pdb_id']})\n\n")
            f.write(f"Published RMSD type: {puzzle_comparison['published_type']}\n")
            f.write(f"Source: {puzzle_comparison['published_source']}\n\n")
            
            # Create a table for this puzzle
            f.write("| Metric | Published | P atoms | C1' atoms | C4' atoms | All heavy atoms |\n")
            f.write("|--------|-----------|---------|-----------|-----------|----------------|\n")
            
            metrics = ["best", "median", "worst"]
            for metric in metrics:
                f.write(f"| {metric.capitalize()} RMSD (Å) | ")
                
                # Published value
                published_value = 0
                for atom_type in puzzle_comparison["atom_types"]:
                    published_value = puzzle_comparison["atom_types"][atom_type][metric]["published"]
                    break
                f.write(f"{published_value:.2f} | ")
                
                # Values for each atom type
                for atom_type in ["P", "C1'", "C4'", "all_heavy"]:
                    if atom_type in puzzle_comparison["atom_types"]:
                        value = puzzle_comparison["atom_types"][atom_type][metric]["calculated"]
                        f.write(f"{value:.2f} | ")
                    else:
                        f.write("N/A | ")
                
                f.write("\n")
            
            # Add a row for relative differences
            f.write("| Rel. diff (best) | - | ")
            for atom_type in ["P", "C1'", "C4'", "all_heavy"]:
                if atom_type in puzzle_comparison["atom_types"]:
                    rel_diff = puzzle_comparison["atom_types"][atom_type]["best"]["rel_diff"]
                    f.write(f"{rel_diff:.1%} | ")
                else:
                    f.write("N/A | ")
            
            f.write("\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Calculate average relative differences
        avg_rel_diffs = {}
        for atom_type in ["P", "C1'", "C4'", "all_heavy"]:
            rel_diffs = []
            for puzzle_comp in comparison.values():
                if atom_type in puzzle_comp["atom_types"]:
                    rel_diff = puzzle_comp["atom_types"][atom_type]["best"]["rel_diff"]
                    if not np.isnan(rel_diff):
                        rel_diffs.append(rel_diff)
            
            if rel_diffs:
                avg_rel_diffs[atom_type] = sum(rel_diffs) / len(rel_diffs)
            else:
                avg_rel_diffs[atom_type] = float('nan')
        
        f.write("Average relative differences from published values:\n\n")
        for atom_type, avg_diff in avg_rel_diffs.items():
            if not np.isnan(avg_diff):
                f.write(f"- **{atom_type}**: {avg_diff:.1%}\n")
            else:
                f.write(f"- **{atom_type}**: N/A\n")
        
        f.write("\n")
        
        # Determine best atom type
        best_atom_type = min(
            [at for at in avg_rel_diffs if not np.isnan(avg_rel_diffs[at])],
            key=lambda at: avg_rel_diffs[at],
            default=None
        )
        
        if best_atom_type:
            f.write(f"The **{best_atom_type}** atom selection strategy produces RMSD values closest to published results, with an average relative difference of {avg_rel_diffs[best_atom_type]:.1%}.\n\n")
        
        # C1' atoms conclusion
        if "C1'" in avg_rel_diffs and not np.isnan(avg_rel_diffs["C1'"]):
            if avg_rel_diffs["C1'"] < 0.15:  # Less than 15% difference
                f.write("The C1' atom selection used in our pipeline shows good agreement with published values, validating our approach.\n")
            else:
                f.write("The C1' atom selection used in our pipeline shows significant differences from published values. Consider adjusting or validating the approach further.\n")
                
    logging.info(f"Comparison report saved to {report_file}")
    return report_file

def generate_comparison_plots(results, comparison, output_dir):
    """
    Generate plots comparing different atom selection strategies.
    
    Args:
        results: Dictionary with RMSD results
        comparison: Dictionary with comparison results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Best RMSD by atom type
    plt.figure(figsize=(12, 8))
    
    atom_types = ["P", "C1'", "C4'", "all_heavy"]
    puzzle_ids = sorted(results.keys())
    
    bar_width = 0.2
    index = np.arange(len(puzzle_ids))
    
    for i, atom_type in enumerate(atom_types):
        best_rmsds = []
        for puzzle_id in puzzle_ids:
            if atom_type in results[puzzle_id]["statistics"] and results[puzzle_id]["statistics"][atom_type]:
                best_rmsds.append(results[puzzle_id]["statistics"][atom_type]["best_rmsd"]["rmsd"])
            else:
                best_rmsds.append(0)
        
        plt.bar(index + i*bar_width, best_rmsds, bar_width, label=atom_type)
    
    # Add published values
    published_best = []
    for puzzle_id in puzzle_ids:
        found = False
        for atom_type in atom_types:
            if puzzle_id in comparison and atom_type in comparison[puzzle_id]["atom_types"]:
                published_best.append(comparison[puzzle_id]["atom_types"][atom_type]["best"]["published"])
                found = True
                break
        if not found:
            published_best.append(0)
    
    plt.bar(index + len(atom_types)*bar_width, published_best, bar_width, label="Published")
    
    plt.xlabel('Puzzle ID')
    plt.ylabel('Best RMSD (Å)')
    plt.title('Best RMSD by Atom Selection Strategy')
    plt.xticks(index + bar_width*2, puzzle_ids)
    plt.legend()
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "best_rmsd_by_atom_type.png")
    plt.savefig(plot_file)
    plt.close()
    
    # Plot 2: Relative difference from published values
    plt.figure(figsize=(12, 8))
    
    for i, atom_type in enumerate(atom_types):
        rel_diffs = []
        for puzzle_id in puzzle_ids:
            if (puzzle_id in comparison and 
                atom_type in comparison[puzzle_id]["atom_types"]):
                rel_diff = comparison[puzzle_id]["atom_types"][atom_type]["best"]["rel_diff"]
                rel_diffs.append(rel_diff * 100)  # Convert to percentage
            else:
                rel_diffs.append(0)
        
        plt.bar(index + i*bar_width, rel_diffs, bar_width, label=atom_type)
    
    plt.xlabel('Puzzle ID')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Difference from Published Values by Atom Type')
    plt.xticks(index + bar_width*1.5, puzzle_ids)
    plt.axhline(y=10, color='r', linestyle='--', label='10% threshold')
    plt.legend()
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "rel_diff_by_atom_type.png")
    plt.savefig(plot_file)
    plt.close()
    
    # Plot 3: Comparison scatter plot
    plt.figure(figsize=(10, 8))
    
    for atom_type in atom_types:
        calculated = []
        published = []
        labels = []
        
        for puzzle_id in puzzle_ids:
            if (puzzle_id in comparison and 
                atom_type in comparison[puzzle_id]["atom_types"]):
                calculated.append(comparison[puzzle_id]["atom_types"][atom_type]["best"]["calculated"])
                published.append(comparison[puzzle_id]["atom_types"][atom_type]["best"]["published"])
                labels.append(f"Puzzle {puzzle_id}")
        
        if calculated and published:
            plt.scatter(published, calculated, label=atom_type, alpha=0.7, s=100)
            
            # Add labels to points
            for i, label in enumerate(labels):
                plt.annotate(label, (published[i], calculated[i]), fontsize=8)
    
    # Add diagonal line
    max_val = max([max(max(comparison[p]["atom_types"][at]["best"]["calculated"], 
                         comparison[p]["atom_types"][at]["best"]["published"]) 
                     for at in comparison[p]["atom_types"]) 
                 for p in comparison]) * 1.1
    
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Published RMSD (Å)')
    plt.ylabel('Calculated RMSD (Å)')
    plt.title('Calculated vs Published RMSD Values')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "calculated_vs_published.png")
    plt.savefig(plot_file)
    plt.close()
    
    logging.info(f"Comparison plots saved to {output_dir}")
    return [
        os.path.join(output_dir, "best_rmsd_by_atom_type.png"),
        os.path.join(output_dir, "rel_diff_by_atom_type.png"),
        os.path.join(output_dir, "calculated_vs_published.png")
    ]

def load_validated_pairs(json_file):
    """
    Load validated model-reference pairs from JSON file.
    
    Args:
        json_file: Path to JSON file with validated pairs
        
    Returns:
        Dictionary with validated pairs information
    """
    if not os.path.exists(json_file):
        logging.error(f"Validated pairs file not found: {json_file}")
        return {}
        
    with open(json_file, 'r') as f:
        pairs = json.load(f)
    
    return pairs

def main():
    """Main function for atom selection comparison."""
    parser = argparse.ArgumentParser(description="Compare RMSD calculations with different atom selection strategies")
    
    parser.add_argument("--benchmark-dir", type=str, default=None,
                        help="Root directory for benchmark data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--puzzles", type=str, default="all",
                        help="Comma-separated list of puzzle IDs to process, or 'all'")
    parser.add_argument("--verified-only", action="store_true",
                        help="Use only verified model-reference pairs")
    
    args = parser.parse_args()
    
    # Set benchmark directory
    if args.benchmark_dir is None:
        # Try to find relative to this script
        script_dir = Path(__file__).absolute().parent
        benchmark_dir = script_dir.parent
    else:
        benchmark_dir = Path(args.benchmark_dir)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = benchmark_dir / "results" / "atom_selection"
    else:
        output_dir = Path(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load published RMSD values
    published_file = benchmark_dir / "published_rmsd" / "rmsd_reference_values.csv"
    if not published_file.exists():
        logging.error(f"Published RMSD file not found: {published_file}")
        return 1
    
    published_values = load_published_rmsd_values(str(published_file))
    logging.info(f"Loaded published RMSD values for {len(published_values)} puzzles")
    
    # If using verified pairs only, load the validated pairs
    validated_pairs = {}
    if args.verified_only:
        validated_pairs_file = benchmark_dir / "validated_pairs.json"
        if validated_pairs_file.exists():
            validated_pairs = load_validated_pairs(validated_pairs_file)
            logging.info(f"Loaded {len(validated_pairs)} validated model-reference pairs")
            
            # Use only puzzles with validated pairs
            puzzle_ids = list(validated_pairs.keys())
            logging.info(f"Using only verified puzzles: {', '.join(puzzle_ids)}")
        else:
            logging.warning(f"Validated pairs file not found: {validated_pairs_file}")
            logging.warning("Proceeding with all puzzles instead")
            puzzle_ids = list(published_values.keys())
    else:
        # Determine which puzzles to process
        if args.puzzles.lower() == "all":
            puzzle_ids = list(published_values.keys())
        else:
            puzzle_ids = args.puzzles.split(",")
    
    logging.info(f"Processing {len(puzzle_ids)} puzzles: {', '.join(puzzle_ids)}")
    
    # Run comparison for each puzzle
    results = {}
    for puzzle_id in puzzle_ids:
        if puzzle_id not in published_values:
            logging.warning(f"No published values for Puzzle {puzzle_id}, skipping")
            continue
            
        pdb_id = published_values[puzzle_id]["pdb_id"]
        puzzle_results = compare_atom_selection_strategies(str(benchmark_dir), puzzle_id, pdb_id)
        
        if puzzle_results:
            results[puzzle_id] = puzzle_results
            
    if not results:
        logging.error("No valid results obtained")
        return 1
        
    # Compare with published values
    comparison = compare_with_published_values(results, published_values)
    
    # Generate report and plots
    report_file = generate_comparison_report(comparison, str(output_dir))
    plot_files = generate_comparison_plots(results, comparison, str(output_dir))
    
    # Save raw results
    results_file = os.path.join(output_dir, "atom_selection_results.json")
    with open(results_file, "w") as f:
        # Convert non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        json.dump(results, f, default=json_serialize, indent=2)
    
    logging.info(f"Raw results saved to {results_file}")
    logging.info(f"Atom selection comparison complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())