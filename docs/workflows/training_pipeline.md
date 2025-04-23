# RNA 3D Structure Prediction Training Pipeline

This document outlines the comprehensive training and analysis pipeline implemented for the RNA 3D structure prediction model.

## Overview

The training pipeline provides a complete workflow for:
- Running production training with resource monitoring
- Tracking GPU utilization and performance metrics
- Generating detailed reports of training progress
- Analyzing model performance through interactive notebooks
- Validating model outputs in both training and testing modes

## Key Components

### Production Training Script

The `scripts/run_production_training.sh` script provides a robust entry point for model training with:

- **Comprehensive Configuration**: Command-line options for all training parameters
- **Resource Management**: Proper directory structure and cleanup procedures
- **Monitoring**: Integrated GPU utilization tracking
- **Reproducibility**: Configuration recording and seed management
- **Reporting**: Automated report generation after training

```bash
# Example usage:
./scripts/run_production_training.sh \
  --output_dir results/production_runs \
  --batch_size 32 \
  --num_epochs 50 \
  --lr 0.0005 \
  --fape_weight 1.0 \
  --confidence_weight 0.2 \
  --angle_weight 0.5
```

### GPU Monitoring

The `scripts/monitor_gpu.py` script provides real-time monitoring of:

- GPU utilization percentage
- Memory usage
- Temperature
- Power consumption

```bash
# Example usage:
python scripts/monitor_gpu.py \
  --output_dir results/training_run \
  --interval 10 \
  --gpu_ids 0,1
```

### Training Report Generation

The `scripts/generate_training_report.py` script creates comprehensive reports with:

- Training progress visualization
- Loss curves analysis
- Validation metric tracking
- GPU utilization statistics
- Model performance metrics (RMSD distribution)
- Checkpoint comparison

```bash
# Example usage:
python scripts/generate_training_report.py \
  --training_dir results/production_run_20250423 \
  --output_format pdf
```

### Interactive Analysis Notebook

The `notebooks/production_run_analysis.ipynb` notebook provides:

- Interactive visualization of training metrics
- RMSD distribution analysis
- GPU utilization plotting
- Structure visualization
- Feature importance analysis
- Checkpoint comparison
- Performance recommendations

## Dual-Mode Validation Framework

The validation framework ensures model performance in both training and testing conditions:

- **Training Mode**: Uses all available features (thermodynamic, MI, dihedral angles)
- **Testing Mode**: Excludes dihedral angles to simulate competition conditions
- **Comparison**: Analyzes performance difference between modes

The framework includes:

- NPZFeatureLoader with test-mode filtering
- ValidationDataset with dual-mode support
- ValidationRunner for side-by-side comparison
- Improved RMSD calculation with MDAnalysis integration

## Temporal Cutoff Enforcement

The pipeline strictly enforces the May 2022 (2022-05-27) temporal cutoff to prevent data leakage:

- Feature filtering based on generation date
- Sequence filtering based on temporal_cutoff field
- Runtime configuration of cutoff dates

## Auxiliary Training with Dihedral Angles

The model uses dihedral angles as an auxiliary training signal:

- Angle prediction loss during training
- Model architecture independent of angles for inference
- Combined loss function with configurable weighting

## Improved Evaluation Metrics

The pipeline includes stable implementations of structure quality metrics:

- RMSD (Root Mean Square Deviation)
- TM-score (Template Modeling Score)
- Per-residue error analysis
- Atom selection strategies

## Usage Workflow

1. **Run Production Training**:
   ```bash
   ./scripts/run_production_training.sh --output_dir results/run1
   ```

2. **Generate Training Report**:
   ```bash
   python scripts/generate_training_report.py --training_dir results/run1
   ```

3. **Analyze Results Interactively**:
   - Open `notebooks/production_run_analysis.ipynb`
   - Set `training_dir` to your training directory
   - Execute cells to analyze results

4. **Run Dual-Mode Validation**:
   ```bash
   ./validation/run_dual_mode_validation.sh results/run1/checkpoints/best_model.pt
   ```

5. **Review Validation Reports**:
   - Technical validation: `validation/tier1_technical/results`
   - Scientific validation: `validation/tier2_scientific/results`

## Next Steps

- Run baseline model training
- Perform hyperparameter optimization
- Implement ensemble methods for checkpoints
- Expand scientific validation metrics
- Integrate feature importance analysis