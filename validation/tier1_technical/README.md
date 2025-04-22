# Tier 1: Technical Validation

## Purpose

This directory contains the scripts and notebooks for Tier 1 (Technical) validation of the RNA 3D folding model. The purpose of Tier 1 validation is to quickly verify that the model is functioning correctly from a technical perspective, without focusing on detailed scientific accuracy.

## Components

- `validation_technical.ipynb`: Jupyter notebook for interactive technical validation
- `run_validation.py`: Script to execute the validation notebook from the command line
- `run_dual_mode_validation.py`: Script for comparing test-equivalent and training-equivalent modes
- `debug_rmsd_calculation.py`: Utility for debugging RMSD calculation issues
- `results/`: Directory where validation results are stored

### Data Components

- **Target Selection**: 
  - Targets are loaded from CSV files with priority: validation_sequences.csv → train_sequences.csv → test_sequences.csv
  - Use `--rna-ids` parameter to filter validation to specific RNA sequences

- **Feature Loading**:
  - Test-equivalent mode: Only thermodynamic and MI features
  - Training-equivalent mode: Thermodynamic, MI, and dihedral features

## Dual-Mode Validation Framework

The core innovation in Tier 1 validation is the dual-mode validation framework that addresses the critical feature availability mismatch between training and testing:

- **Test-Equivalent Mode**: Uses only features available at competition test time (thermodynamic and MI matrices, NO dihedral angles)
- **Training-Equivalent Mode**: Uses all features available during training (including pseudo-dihedral angles)

This approach allows quantification of the performance gap caused by missing features and provides scientific insights to guide model architecture decisions.

## Validation Approach

Tier 1 validation follows a comprehensive approach:

1. **Shape and Configuration Check**
   - Load a small subset of data (3-5 sequences)
   - Verify input and output tensor shapes
   - Check model configuration

2. **Gradient Flow Verification**
   - Verify gradient flow through the model
   - Check loss values and parameter gradients
   - Identify any components missing gradients

3. **Structure Metrics Evaluation**
   - Calculate RMSD (Root Mean Square Deviation) with Kabsch alignment
   - Calculate TM-score (Template Modeling score)
   - Compute per-residue RMSD for detailed analysis

4. **Mode Comparison Analysis**
   - Compare test-equivalent and training-equivalent mode performance
   - Quantify the impact of missing features
   - Classify impact severity (MAJOR, MODERATE, MINOR, NEGLIGIBLE)
   - Provide recommendations based on impact

5. **Performance Benchmarking**
   - Measure inference time
   - Track memory usage
   - Evaluate model efficiency

## Success Criteria

Tier 1 validation is considered successful if:

1. The model loads without errors
2. Forward and backward passes complete successfully
3. Output shapes match expectations
4. Loss values are finite and reasonable
5. All critical components have gradients
6. Structure metrics can be calculated without numerical instability
7. No NaN values or extreme values are observed
8. Mode comparison produces meaningful insights

## Running Tier 1 Validation

### Using the Dual-Mode Validation Script

```bash
# Navigate to the tier1_technical directory
cd validation/tier1_technical

# Run with default settings
python run_dual_mode_validation.py

# Run with a specific model checkpoint
python run_dual_mode_validation.py --checkpoint /path/to/checkpoint.pt

# Run for specific RNA IDs
python run_dual_mode_validation.py --rna-ids R1107 R1108 R1156

# Get help on available options
python run_dual_mode_validation.py --help
```

Available options:
- `--checkpoint`: Path to model checkpoint (default: None, uses random weights)
- `--subset`: Validation subset (default: "technical")
- `--data_dir`: Path to data directory (default: auto-detected)
- `--output_dir`: Directory for output files (default: "./results")
- `--batch_size`: Batch size for validation (default: 4)
- `--cpu`: Force CPU usage (not recommended)
- `--rna-ids`: Specific RNA IDs to validate (default: None, uses random subset)

### Using the Notebook Directly

1. Activate the RNA folding environment
2. Launch Jupyter Notebook
3. Open `validation_technical.ipynb`
4. Update the checkpoint path if needed
5. Run all cells

## Validation Outputs

The validation process generates the following outputs in the `results/` directory:

1. `validation_results_technical_[timestamp].json`: Detailed metrics and validation results
2. `validation_report_technical_[timestamp].md`: Markdown report with structured results
3. Visualizations:
   - `mode_comparison_technical.png`: Comparison of test vs. training modes
   - `metrics_comparison_technical.png`: Bar chart of key metrics
   - `rmsd_dist_test_equivalent.png`: RMSD distribution for test mode
   - `rmsd_dist_training_equivalent.png`: RMSD distribution for train mode
   - `per_residue_error_test_equivalent.png`: Per-residue error for test mode
   - `per_residue_error_training_equivalent.png`: Per-residue error for train mode

## Understanding Results

### Key Metrics
- **Mean RMSD**: Average root-mean-square deviation in Angstroms (lower is better)
- **Median RMSD**: Median RMSD value, more robust to outliers
- **TM-score**: Template modeling score (higher is better)
- **Impact Analysis**: Classification of feature impact (NEGATIVE/POSITIVE/NEUTRAL)
- **Severity**: Impact severity (MAJOR/MODERATE/MINOR/NEGLIGIBLE)

### Problematic Sample Tracking
The validation framework now includes robust tracking of problematic samples with diagnostic details:
- Extreme coordinate values detection
- Numerical instability identification
- Per-sample diagnostic information

## Recent Updates (2025-04-21)
- Fixed extreme RMSD values caused by numerical instability
- Added proper coordinate validation and sanity checks
- Enhanced error reporting with problematic sample tracking
- Added detailed diagnostics for validation failures
- Improved visualization of mode comparisons

## Time Expectations
Tier 1 validation should complete in less than 5 minutes on the target hardware (RTX 4070 Ti).