# Tier 1: Technical Validation

## Purpose

This directory contains the scripts and notebooks for Tier 1 (Technical) validation of the RNA 3D folding model. The purpose of Tier 1 validation is to quickly verify that the model is functioning correctly from a technical perspective, without focusing on detailed scientific accuracy.

## Components

- `validation_technical.ipynb`: Jupyter notebook for interactive technical validation
- `run_validation.py`: Script to execute the validation notebook from the command line
- `results/`: Directory where validation results are stored

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
   - Calculate RMSD (Root Mean Square Deviation)
   - Calculate TM-score (Template Modeling score)
   - Compute per-residue RMSD for detailed analysis

4. **Performance Benchmarking**
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
6. Structure metrics can be calculated
7. No NaN values or numerical instabilities are observed

## Running Tier 1 Validation

### Using the Run Script

```bash
# Navigate to the tier1_technical directory
cd validation/tier1_technical

# Run with default settings
./run_validation.py

# Run with a specific model checkpoint
./run_validation.py --checkpoint /path/to/checkpoint.pt

# Get help on available options
./run_validation.py --help
```

### Using the Notebook Directly

1. Activate the RNA folding environment
2. Launch Jupyter Notebook
3. Open `validation_technical.ipynb`
4. Update the checkpoint path if needed
5. Run all cells

## Validation Outputs

The validation process generates the following outputs in the `results/` directory:

1. `validation_results.json`: Detailed metrics and validation results
2. `validation_results.html`: HTML report of the executed notebook
3. `per_residue_rmsd.png`: Visualization of per-residue RMSD

## Time Expectations

Tier 1 validation should complete in less than 5 minutes on the target hardware (RTX 4070 Ti).