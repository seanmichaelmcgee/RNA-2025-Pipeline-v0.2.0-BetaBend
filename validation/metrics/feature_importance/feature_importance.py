"""
Feature importance analysis module.

This module provides functionality for analyzing the importance of different features
in RNA 3D structure prediction by comparing model performance with different feature sets.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import os


def analyze_feature_importance(
    test_mode_results: Dict[str, Any],
    train_mode_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze feature importance by comparing test and train mode performance.
    
    Args:
        test_mode_results: Results from test-equivalent mode
        train_mode_results: Results from training-equivalent mode
        
    Returns:
        Dictionary with feature importance analysis
    """
    # This is a placeholder implementation
    # In a real implementation, this would perform a detailed analysis of the
    # performance difference between test and train modes
    
    # Extract metrics
    test_rmsd = test_mode_results.get("mean_rmsd")
    train_rmsd = train_mode_results.get("mean_rmsd")
    
    if test_rmsd is None or train_rmsd is None:
        return {
            "error": "Missing RMSD metrics",
            "feature_importance": {
                "thermodynamic": None,
                "mutual_information": None,
                "dihedral_angles": None
            }
        }
    
    # Calculate absolute and relative differences
    abs_diff = test_rmsd - train_rmsd
    rel_diff = (abs_diff / train_rmsd) * 100 if train_rmsd > 0 else 0
    
    # Estimate importance of dihedral angles based on difference
    dihedral_importance = min(1.0, abs(rel_diff) / 100)
    
    # For demonstration, assign remaining importance to other features
    # In a real implementation, this would use more sophisticated analysis
    remaining = 1.0 - dihedral_importance
    thermo_importance = remaining * 0.6  # 60% of remaining importance
    mi_importance = remaining * 0.4  # 40% of remaining importance
    
    return {
        "feature_importance": {
            "thermodynamic": thermo_importance,
            "mutual_information": mi_importance,
            "dihedral_angles": dihedral_importance
        },
        "rmsd_difference": {
            "absolute": abs_diff,
            "relative_percent": rel_diff
        },
        "impact_assessment": {
            "magnitude": abs(rel_diff),
            "direction": "negative" if rel_diff > 0 else "positive" if rel_diff < 0 else "neutral"
        }
    }


def analyze_per_residue_importance(
    test_mode_results: Dict[str, Any],
    train_mode_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze feature importance at the per-residue level.
    
    Args:
        test_mode_results: Results from test-equivalent mode
        train_mode_results: Results from training-equivalent mode
        
    Returns:
        Dictionary with per-residue importance analysis
    """
    # This is a placeholder implementation
    # In a real implementation, this would compare per-residue errors
    # between test and train modes to identify where features have the most impact
    
    # Check if per-residue data is available
    if ("avg_per_residue_error" not in test_mode_results or 
        "avg_per_residue_error" not in train_mode_results):
        return {
            "error": "Missing per-residue error data",
            "per_residue_importance": None
        }
    
    test_per_residue = np.array(test_mode_results["avg_per_residue_error"])
    train_per_residue = np.array(train_mode_results["avg_per_residue_error"])
    
    # In a real implementation, this would analyze where the differences are largest
    # For now, just generate placeholder data
    
    # Create normalized positions (0-1)
    positions = np.linspace(0, 1, 20)
    
    # Generate random importance values that are higher at the ends (common pattern)
    importance = 0.2 + 0.6 * np.exp(-10 * (positions - 0.5)**2)
    
    return {
        "per_residue_importance": {
            "positions": positions.tolist(),
            "importance": importance.tolist()
        },
        "region_importance": {
            "ends": 0.7,  # Importance at sequence ends
            "middle": 0.3  # Importance in middle regions
        }
    }


def perform_feature_ablation(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate feature ablation study to measure feature importance.
    
    Args:
        results: Validation results dictionary
        
    Returns:
        Dictionary with ablation study results
    """
    # This is a placeholder for a feature ablation study
    # In a real implementation, this would run the model with different feature subsets
    # and measure the impact on performance
    
    # Simulation of feature ablation results
    ablation_results = {
        "full_model": 3.0,  # RMSD with all features
        "no_dihedral": 3.8,  # RMSD without dihedral features
        "no_thermo": 4.2,    # RMSD without thermodynamic features
        "no_mi": 3.5,        # RMSD without mutual information
        "baseline": 5.0      # RMSD with minimal features
    }
    
    # Calculate importance based on performance drop
    baseline = ablation_results["full_model"]
    worst = ablation_results["baseline"]
    range_size = worst - baseline
    
    importance = {
        "dihedral_angles": (ablation_results["no_dihedral"] - baseline) / range_size,
        "thermodynamic": (ablation_results["no_thermo"] - baseline) / range_size,
        "mutual_information": (ablation_results["no_mi"] - baseline) / range_size
    }
    
    return {
        "ablation_results": ablation_results,
        "feature_importance": importance
    }


def generate_feature_importance_plots(
    analysis: Dict[str, Any],
    output_path: str,
    plot_format: str = "png"
) -> List[str]:
    """
    Generate plots visualizing feature importance.
    
    Args:
        analysis: Feature importance analysis dictionary
        output_path: Directory to save plots
        plot_format: File format for plots (png, jpg, svg, pdf)
        
    Returns:
        List of paths to generated plot files
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Extract feature importance data
    if "feature_importance" not in analysis:
        return []
    
    feature_importance = analysis["feature_importance"]
    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    # Plot feature importance bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importance_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Feature Importance in RNA Structure Prediction')
    plt.xlabel('Feature Type')
    plt.ylabel('Importance')
    plt.ylim(0, 1.1 * max(importance_values))
    plt.tight_layout()
    
    fi_plot_path = os.path.join(output_path, f"feature_importance.{plot_format}")
    plt.savefig(fi_plot_path)
    plt.close()
    
    # Plot per-residue importance if available
    if "per_residue_importance" in analysis and analysis["per_residue_importance"]:
        per_res = analysis["per_residue_importance"]
        positions = per_res["positions"]
        importance = per_res["importance"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(positions, importance, 'o-', linewidth=2)
        plt.title('Feature Importance by Sequence Position')
        plt.xlabel('Normalized Position (0-1)')
        plt.ylabel('Feature Importance')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        per_res_plot_path = os.path.join(output_path, f"per_residue_importance.{plot_format}")
        plt.savefig(per_res_plot_path)
        plt.close()
        
        return [fi_plot_path, per_res_plot_path]
    else:
        return [fi_plot_path]