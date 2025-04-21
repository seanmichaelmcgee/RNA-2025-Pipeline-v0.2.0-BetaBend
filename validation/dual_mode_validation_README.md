# Dual-Mode Validation Framework

This guide explains how to use the dual-mode validation framework to assess the impact of feature availability differences between training and testing.

## Overview

The RNA 3D folding model faces a critical challenge: during training, three feature types are available (thermodynamic features, MI matrices, pseudo-dihedral angles), but at test time, only two are available (thermodynamic and MI matrices). This mismatch creates potential issues for model performance at test time.

The dual-mode validation framework addresses this by running validation in two parallel modes:

1. **Test-Equivalent Mode:** Uses only features available during testing (no pseudo-dihedral angles)
2. **Training-Equivalent Mode:** Uses all features available during training (including pseudo-dihedral angles)

This allows us to quantify the impact of missing pseudo-dihedral angles at test time.

## Components

The dual-mode validation framework consists of these core components:

1. **NPZFeatureLoader:** Loads features from NPZ files with test-mode filtering
   - Loads target IDs from CSV files with priority: validation_sequences.csv → train_sequences.csv → test_sequences.csv
   - Filters features based on test/train mode to match competition conditions
2. **CSVCoordinateLoader:** Loads RNA coordinates from CSV files
   - Handles multiple coordinate sets per residue (up to 40)
   - Automatically selects the most complete coordinate set
3. **ValidationDataset:** PyTorch dataset that combines both loaders with dual-mode support
   - Supports filtering to specific RNA IDs via `--rna-ids` parameter
4. **ValidationRunner:** Executes validation in both modes and analyzes differences
   - Generates comparison metrics between test-equivalent and training-equivalent modes

## Usage

### Running From Command Line

You can use the `run_dual_mode_validation.py` script to run dual-mode validation:

```bash
# Navigate to the tier1_technical directory
cd validation/tier1_technical

# Run with default settings
./run_dual_mode_validation.py

# Specify a model checkpoint
./run_dual_mode_validation.py --checkpoint /path/to/model/checkpoint.pt

# Use a different validation subset
./run_dual_mode_validation.py --subset scientific

# Control batch size
./run_dual_mode_validation.py --batch_size 2

# Force CPU usage
./run_dual_mode_validation.py --cpu
```

### Using in Code

You can also use the ValidationRunner directly in your code:

```python
from validation.validation_runner import ValidationRunner
from src.models.rna_folding_model import RNAFoldingModel

# Initialize model
model = RNAFoldingModel()

# Create validation runner
runner = ValidationRunner(
    model=model,
    data_dir="/path/to/data",
    config={
        "batch_size": 4,
        "results_dir": "/path/to/results",
    }
)

# Run validation in both modes
results = runner.run_validation(
    subset_name="technical",
    run_both_modes=True
)

# Or run only test-equivalent mode
test_results = runner.run_test_equivalent_mode("technical")

# Or run only training-equivalent mode
train_results = runner.run_training_equivalent_mode("technical")
```

## Interpreting Results

The dual-mode validation produces several key metrics for comparison:

1. **RMSD Difference:** Measures the absolute and relative difference in RMSD between modes
2. **TM-Score Difference:** Measures the absolute and relative difference in TM-scores
3. **Position-Specific Analysis:** Identifies where in the RNA sequence the impact is largest
4. **Overall Impact Analysis:** Classifies the impact as MAJOR, MODERATE, MINOR, or NEGLIGIBLE

The framework also generates visualizations to help understand the differences:

1. **Per-Residue Error Comparison:** Shows how error profiles differ between modes
2. **Performance Gap Visualization:** Highlights regions where the model's performance degrades
3. **Metric Comparison Charts:** Provides bar charts comparing key metrics between modes

## Understanding the Output

The ValidationRunner produces a comprehensive output with the following components:

1. **JSON Results File:** Contains all numeric data and analysis
2. **Markdown Report:** Human-readable summary of findings
3. **Visualizations:** PNG/PDF files showing performance comparisons
4. **Console Output:** Key findings and recommendations

## Validation Tiers

The dual-mode validation works with all three validation tiers:

1. **Technical (Tier 1):** Fast, small subset (3-5 sequences)
2. **Scientific (Tier 2):** Medium-sized subset (15 sequences)
3. **Comprehensive (Tier 3):** Large subset (30+ sequences)

For initial testing and debugging, use the "technical" tier. For reliable scientific conclusions, use the "scientific" or "comprehensive" tiers.

## Implementation Status

This dual-mode validation framework is fully implemented and ready for use. The components include:

- ✅ NPZFeatureLoader
- ✅ CSVCoordinateLoader
- ✅ ValidationDataset
- ✅ ValidationRunner
- ✅ Tier 1 Technical Integration
- ⬜ Tier 2 Scientific Integration
- ⬜ Tier 3 Comprehensive Integration