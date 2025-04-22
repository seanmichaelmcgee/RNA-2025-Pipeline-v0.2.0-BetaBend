"""
RNA family classification and analysis module.

This module provides functionality for classifying RNA sequences into families
and analyzing model performance across different RNA types.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict


def classify_rna_family(rna_id: str) -> str:
    """
    Classify an RNA sequence ID into its family.
    
    Args:
        rna_id: RNA target ID
        
    Returns:
        RNA family name (e.g., tRNA, riboswitch, etc.)
    """
    # This is a placeholder implementation
    # In a real implementation, this would use a database lookup or sequence analysis
    
    rna_id = rna_id.lower()
    
    if "trna" in rna_id:
        return "tRNA"
    elif "rrna" in rna_id or "ribosom" in rna_id:
        return "rRNA"
    elif "rnase" in rna_id:
        return "ribozyme"
    elif "switch" in rna_id:
        return "riboswitch"
    elif "srp" in rna_id:
        return "SRP RNA"
    else:
        # Extract first character as ID prefix (e.g., R1107 -> "R")
        prefix = rna_id[0] if rna_id else ""
        
        # Map common prefixes to families (hypothetical mapping)
        prefix_map = {
            "r": "ribosomal",
            "t": "tRNA",
            "s": "snRNA",
            "p": "pre-mRNA",
            "m": "mRNA",
            "y": "snoRNA"
        }
        
        return prefix_map.get(prefix.lower(), "unknown")


def group_results_by_family(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group validation results by RNA family.
    
    Args:
        results: Validation results dictionary
        
    Returns:
        Dictionary mapping RNA families to their results
    """
    # This is a placeholder implementation
    # In a real implementation, this would extract results for each RNA
    # and group them based on family classification
    
    # Get list of RNAs
    if "test_mode" in results:
        target_ids = results["test_mode"].get("target_ids", [])
    elif "train_mode" in results:
        target_ids = results["train_mode"].get("target_ids", [])
    else:
        return {}
    
    # Group by family
    family_results = defaultdict(list)
    
    for rna_id in target_ids:
        family = classify_rna_family(rna_id)
        family_results[family].append(rna_id)
    
    # Placeholder for family-specific metrics
    grouped_metrics = {}
    
    for family, ids in family_results.items():
        # In a real implementation, this would compute metrics for each family
        # based on the results for each RNA in that family
        
        # Placeholder metrics
        grouped_metrics[family] = {
            "rmsd": np.random.uniform(2.5, 6.0),  # Random value between 2.5 and 6.0
            "tm_score": np.random.uniform(0.4, 0.8),  # Random value between 0.4 and 0.8
            "count": len(ids),
            "ids": ids
        }
    
    return grouped_metrics


def analyze_family_performance(
    results: Dict[str, Any], 
    min_family_size: int = 3
) -> Dict[str, Any]:
    """
    Analyze model performance across RNA families.
    
    Args:
        results: Validation results dictionary
        min_family_size: Minimum number of RNAs required for family analysis
        
    Returns:
        Dictionary with family performance analysis
    """
    # Group results by family
    family_results = group_results_by_family(results)
    
    # Filter families with too few examples
    filtered_families = {
        family: metrics for family, metrics in family_results.items()
        if metrics["count"] >= min_family_size
    }
    
    # Calculate summary statistics
    all_families = list(family_results.keys())
    filtered_family_list = list(filtered_families.keys())
    
    # Compute summary data (placeholders)
    best_family = max(filtered_families.items(), key=lambda x: x[1]["tm_score"])[0] if filtered_families else "N/A"
    worst_family = min(filtered_families.items(), key=lambda x: x[1]["tm_score"])[0] if filtered_families else "N/A"
    
    # Create analysis results
    analysis = {
        "families": all_families,
        "analyzed_families": filtered_family_list,
        "performance_by_family": family_results,
        "best_family": best_family,
        "worst_family": worst_family,
        "family_statistics": {
            "total_families": len(all_families),
            "analyzed_families": len(filtered_family_list),
            "family_sizes": {family: metrics["count"] for family, metrics in family_results.items()}
        }
    }
    
    return analysis


def generate_family_plots(
    analysis: Dict[str, Any],
    output_path: str,
    plot_format: str = "png"
) -> List[str]:
    """
    Generate plots visualizing RNA family performance.
    
    Args:
        analysis: Family performance analysis dictionary
        output_path: Directory to save plots
        plot_format: File format for plots (png, jpg, svg, pdf)
        
    Returns:
        List of paths to generated plot files
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Extract data for plotting
    families = []
    rmsd_values = []
    tm_values = []
    
    for family, metrics in analysis["performance_by_family"].items():
        families.append(family)
        rmsd_values.append(metrics["rmsd"])
        tm_values.append(metrics["tm_score"])
    
    # Plot RMSD by family
    plt.figure(figsize=(10, 6))
    sns.barplot(x=families, y=rmsd_values)
    plt.title("RMSD by RNA Family")
    plt.xlabel("RNA Family")
    plt.ylabel("RMSD (Å)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    rmsd_plot_path = os.path.join(output_path, f"rmsd_by_family.{plot_format}")
    plt.savefig(rmsd_plot_path)
    plt.close()
    
    # Plot TM-score by family
    plt.figure(figsize=(10, 6))
    sns.barplot(x=families, y=tm_values)
    plt.title("TM-score by RNA Family")
    plt.xlabel("RNA Family")
    plt.ylabel("TM-score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    tm_plot_path = os.path.join(output_path, f"tm_score_by_family.{plot_format}")
    plt.savefig(tm_plot_path)
    plt.close()
    
    # Combined plot showing both metrics
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(families))
    width = 0.35
    
    ax1 = plt.subplot(111)
    bars1 = ax1.bar(x - width/2, rmsd_values, width, label='RMSD (Å)', color='#1f77b4')
    ax1.set_xlabel('RNA Family')
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(families, rotation=45)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, tm_values, width, label='TM-score', color='#ff7f0e')
    ax2.set_ylabel('TM-score')
    
    # Add a title and legend
    plt.title('RNA Family Performance Metrics')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    combined_plot_path = os.path.join(output_path, f"rna_family_performance.{plot_format}")
    plt.savefig(combined_plot_path)
    plt.close()
    
    return [rmsd_plot_path, tm_plot_path, combined_plot_path]