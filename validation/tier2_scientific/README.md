# RNA 3D Folding Model - Scientific Validation (Tier 2)

This directory contains the implementation of Tier 2 validation for the RNA 3D folding model. The focus is on scientific evaluation of model predictions and comprehensive performance analysis.

## Overview

The Tier 2 validation provides a more in-depth analysis of the model's scientific performance compared to Tier 1 technical validation. It uses a larger subset of data, performs more detailed analysis, and generates comprehensive visualizations and reports.

## Characteristics

- **Runtime**: Medium (15-30 minutes)
- **Data**: Uses substantial subset of data (10-15 sequences)
- **Focus**: Scientific metrics, prediction quality, and detailed performance analysis
- **Mock Data**: Does not allow mock data (requires complete feature sets)
- **Required Features**: Dihedral and thermodynamic features

## Components

- `validation_scientific.ipynb`: Main validation notebook
- `results/`: Directory for validation results and visualizations
  - `validation_results.json`: Comprehensive validation results
  - `sequence_length_analysis.png`: Analysis of performance by sequence length
  - `rmsd_distribution.png`: Distribution of RMSD values
  - `per_residue_plots/`: Individual per-residue RMSD plots for each sequence
  - `predictions/`: Saved model predictions for further analysis

## Usage

1. Ensure you have activated the RNA 3D folding environment:
   ```
   conda activate rna-3d-folding
   ```

2. Launch the Jupyter notebook:
   ```
   jupyter notebook validation_scientific.ipynb
   ```

3. Run all cells to execute the validation pipeline

## Key Metrics

The Tier 2 validation evaluates the following metrics:

- **Structure Quality**:
  - RMSD (Root Mean Square Deviation)
  - TM-score (Template Modeling score)
  - Per-residue RMSD

- **Performance Analysis**:
  - Sequence length correlation with performance
  - Detailed distribution of per-residue error
  - Structure comparison visualization

- **Computational Performance**:
  - Detailed inference time measurement
  - Memory usage analysis
  - Throughput (samples/second and residues/second)

## Success Criteria

Tier 2 validation is considered successful if:

1. Mean TM-score > 0.4 across the diverse set
2. Base-pairing accuracy > 70%
3. Stacking interactions are reasonably preserved
4. Long-range interactions are detected
5. Small RNA motifs are correctly modeled
6. Model performs consistently across different RNA types

## Implementation Status

✅ **Implemented**

The Tier 2 validation notebook has been created with comprehensive scientific metrics, enhanced visualizations, and detailed reporting. It leverages the tier-specific dataset implementation to ensure appropriate feature verification and validation requirements.

## Notes

- Tier 2 validation requires complete feature sets and does not use mock data
- Results can be compared with Tier 1 validation to understand model behavior
- For even more comprehensive evaluation, see Tier 3 validation