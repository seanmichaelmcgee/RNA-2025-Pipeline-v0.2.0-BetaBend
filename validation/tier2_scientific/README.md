# Tier 2: Scientific Validation

## Purpose

This directory contains the scientific validation framework (Tier 2) for the RNA 3D folding model. The purpose of Tier 2 validation is to evaluate the model from a scientific perspective, focusing on RNA-specific metrics and structure quality analysis.

## Overview

The scientific validation extends the technical validation (Tier 1) with:

1. RNA-specific structure quality metrics
2. Feature importance analysis
3. RNA family-specific performance evaluation
4. Secondary structure assessment
5. More comprehensive dataset (10-15 sequences)

## Dual-Mode Validation Framework

The validation runs in two modes to quantify the impact of feature availability differences:

- **Test-Equivalent Mode**: Uses only features available at competition test time (thermodynamic features, mutual information matrices)
- **Training-Equivalent Mode**: Uses all features available during training (including pseudo-dihedral angles)

## Scientific Validation Approach

Tier 2 validation extends the technical validation with scientific rigor:

1. **RNA Family Analysis**
   - Group targets by RNA families
   - Assess performance across different RNA types
   - Identify strengths and weaknesses by RNA class

2. **Secondary Structure Assessment**
   - Evaluate base-pairing prediction accuracy
   - Measure base-pair distance errors
   - Analyze stacking interactions

3. **Advanced Structure Metrics**
   - Extended TM-score analysis with scientific interpretation
   - Local distance difference test (lDDT)
   - Global distance test (GDT)
   - Structure consistency analysis

4. **Feature Importance Analysis**
   - Quantify the impact of different feature types
   - Correlate feature quality with prediction accuracy
   - Identify most informative features

## Running Tier 2 Validation

### Using the Run Script

The easiest way to run scientific validation is using the provided script:

```bash
# Navigate to the tier2_scientific directory
cd validation/tier2_scientific

# Run with default settings
python run_scientific_validation.py

# Run with a specific model checkpoint
python run_scientific_validation.py --checkpoint /path/to/checkpoint.pt

# Run with specific RNA families or IDs
python run_scientific_validation.py --rna-ids R1107 R1108 R1156
```

Optional parameters:
- `--checkpoint PATH`: Path to model checkpoint
- `--data_dir PATH`: Path to data directory
- `--output_dir PATH`: Directory for output files
- `--batch_size INT`: Batch size for validation (default: 2)
- `--subset_size INT`: Number of sequences to use (default: 10)
- `--cpu`: Force CPU usage (not recommended)
- `--rna-ids ID1 ID2...`: Specific RNA IDs to validate

### Using the Notebook Directly

You can also run the validation interactively using the Jupyter notebook:

```bash
jupyter notebook validation_scientific.ipynb
```

## Expected Outputs

The validation process generates the following outputs in the `results/` directory:

1. `scientific_validation_results_[timestamp].json`: Detailed metrics and validation results
2. `scientific_validation_report_[timestamp].md`: Markdown report with structured scientific analysis
3. Visualizations:
   - `rna_family_performance.png`: Performance breakdown by RNA family
   - `secondary_structure_accuracy.png`: Base-pairing prediction accuracy
   - `motif_analysis.png`: Analysis of structural motif prediction quality
   - `feature_importance.png`: Feature importance visualization
   - `per_residue_quality.png`: Per-residue quality analysis
   - `structure_comparison_[id].png`: Structure comparisons for selected targets

## Integration with Other Tiers

- **Tier 1 (Technical)**: Provides basic technical validation with smaller dataset
- **Tier 2 (Scientific)**: This tier - focuses on scientific quality with medium dataset
- **Tier 3 (Comprehensive)**: Full validation with large dataset and extensive metrics

## Time Expectations

Tier 2 validation is expected to complete in 15-30 minutes on the target hardware (RTX 4070 Ti).
