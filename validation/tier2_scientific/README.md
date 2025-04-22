# Scientific Validation (Tier 2)

This directory contains the scientific validation framework (Tier 2) for the RNA 3D folding model. The scientific validation focuses on in-depth analysis of structural prediction quality from a scientific perspective.

## Overview

The scientific validation extends the technical validation (Tier 1) with:

1. RNA-specific structure quality metrics
2. Feature importance analysis
3. RNA family-specific performance evaluation
4. More comprehensive dataset (10-15 sequences)

## Dual-Mode Validation

The validation runs in two modes to quantify the impact of feature availability differences:

- **Test-Equivalent Mode**: Uses only features available at competition test time (thermodynamic features, mutual information matrices)
- **Training-Equivalent Mode**: Uses all features available during training (including pseudo-dihedral angles)

## Running the Validation

### Using the Script

The easiest way to run scientific validation is using the provided script:

```bash
python run_scientific_validation.py
```

Optional parameters:
- `--checkpoint PATH`: Path to model checkpoint
- `--data_dir PATH`: Path to data directory
- `--output_dir PATH`: Directory for output files
- `--batch_size INT`: Batch size for validation (default: 2)
- `--subset_size INT`: Number of sequences to use (default: 10)
- `--cpu`: Force CPU usage (not recommended)
- `--rna-ids ID1 ID2...`: Specific RNA IDs to validate

### Using the Notebook

You can also run the validation interactively using the Jupyter notebook:

```bash
jupyter notebook validation_scientific.ipynb
```

## Output and Results

The validation produces:

1. Performance metrics for both modes (RMSD, TM-score)
2. Analysis of feature importance
3. Visualizations comparing test and train modes
4. Scientific recommendations for model improvements

Results are saved in the `results/` directory, including JSON data files, visualizations, and a comprehensive markdown report.

## Integration with Other Tiers

- **Tier 1 (Technical)**: Provides basic technical validation with smaller dataset
- **Tier 2 (Scientific)**: This tier - focuses on scientific quality with medium dataset
- **Tier 3 (Comprehensive)**: Full validation with large dataset and extensive metrics
EOF < /dev/null
