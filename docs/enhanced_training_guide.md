# Enhanced RNA 3D Structure Training Pipeline

This document provides instructions for using the enhanced RNA 3D structure training pipeline with memory optimization, curriculum learning, and advanced checkpointing capabilities.

## Overview

The enhanced training pipeline builds upon the base training system with several key improvements:

1. **Memory Optimization**
   - Mixed precision training (FP16) for reduced memory usage
   - Gradient checkpointing to minimize memory footprint
   - Memory monitoring with emergency checkpointing on high memory usage

2. **Curriculum Learning**
   - Gradual increase in sequence length during training
   - Automatic stage advancement based on loss improvement
   - Optional adaptive batch sizing based on sequence length
   - Enhanced error handling for dataset filtering

3. **Advanced Checkpointing**
   - Regular epoch-based checkpoint saving
   - Best model tracking based on validation metrics
   - Step-based checkpointing for large datasets
   - Checkpoint-based training resumption

4. **Robust Error Handling**
   - Defensive programming for cross-component interfaces
   - Safe dataset analysis for sequence lengths
   - Graceful degradation when curriculum stages can't be applied

## Quick Start

The simplest way to train a model is to use one of the example scripts in `scripts/examples/`:

```bash
# Basic training with memory optimization
./scripts/examples/train_basic.sh

# Training with curriculum learning
./scripts/examples/train_curriculum.sh

# Debug training with simplified model
./scripts/examples/train_debug.sh

# Resume training from a checkpoint 
# (edit the CHECKPOINT_PATH variable in the script first)
./scripts/examples/train_resume.sh
```

## Manual Training

You can also run the training script directly with custom parameters:

```bash
# Activate mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Run training with custom parameters
python3 scripts/train_enhanced_model_fixed.py \
    --mixed_precision \
    --gradient_checkpointing \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_len 200 \
    --save_interval_epochs 5 \
    --epochs 50 \
    --output_dir results/my_model
```

## Command-line Arguments

The training script supports numerous command-line arguments for configuration:

### Data Parameters
- `--train_csv`: Path to training sequences CSV (default: 'data/raw/train_sequences.csv')
- `--labels_csv`: Path to training labels CSV with 3D coordinates (default: 'data/raw/train_labels.csv')
- `--features_dir`: Path to processed features directory (default: 'data/processed/')
- `--val_split`: Validation split ratio (default: 0.1)
- `--temporal_cutoff`: Temporal cutoff date for features (default: '2022-05-01')
- `--max_seq_len`: Maximum sequence length for training (default: 300)

### Model Architecture
- `--num_blocks`: Number of transformer blocks (default: 6)
- `--residue_embed_dim`: Residue embedding dimension (default: 192)
- `--pair_embed_dim`: Pair embedding dimension (default: 64)
- `--num_heads`: Number of attention heads (default: 8)
- `--ff_dim`: Feed-forward dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `--batch_size`: Training batch size (default: 8)
- `--grad_accum_steps`: Number of gradient accumulation steps (default: 2)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0005)
- `--weight_decay`: Weight decay (default: 1e-5)
- `--patience`: Patience for early stopping (default: 10)
- `--scheduler`: LR scheduler type ('plateau', 'cosine', 'none') (default: 'cosine')

### Loss Weights
- `--fape_weight`: Weight for FAPE loss (default: 1.0)
- `--confidence_weight`: Weight for confidence loss (default: 0.1)
- `--angle_weight`: Weight for angle loss (default: 0.5)

### Memory Optimization
- `--mixed_precision`: Enable mixed precision training with autocast
- `--gradient_checkpointing`: Enable gradient checkpointing to reduce memory usage
- `--memory_fraction_warning`: Fraction of GPU memory usage to trigger warnings (default: 0.85)
- `--memory_fraction_critical`: Fraction of GPU memory usage to trigger emergency actions (default: 0.92)

### Curriculum Learning
- `--curriculum_learning`: Enable curriculum learning by sequence length
- `--curriculum_stages`: Sequence length stages for curriculum learning (default: [100, 150, 200, 250, 300])
- `--epochs_per_stage`: Minimum epochs per curriculum stage (default: 5)
- `--batch_adaptive`: Dynamically adapt batch size based on sequence length

### Checkpointing and Save Options
- `--save_interval_epochs`: Save checkpoint every N epochs (default: 5)
- `--save_interval_steps`: Save checkpoint every N steps (optional)
- `--max_checkpoints`: Maximum number of checkpoints to keep (default: 3)
- `--output_dir`: Output directory for saving models and logs (default: 'results/enhanced_model')
- `--resume`: Path to checkpoint to resume training from (optional)
- `--resume_reset_optimizer`: Reset optimizer when resuming training
- `--resume_reset_scheduler`: Reset learning rate scheduler when resuming training
- `--resume_reset_curriculum`: Reset curriculum stage when resuming training
- `--validate_checkpoints`: Run full validation on best checkpoints

### Debug Options
- `--debug`: Enable debug mode with small dataset
- `--debug_samples`: Number of samples to use in debug mode (default: 20)
- `--profile`: Profile one training step and exit

## Output Structure

The training script creates an output directory with the following structure:

```
results/enhanced_model/run_YYYYMMDD-HHMMSS/
├── checkpoints/
│   ├── best_model.pt          # Best model based on validation metrics
│   ├── checkpoint_epoch_N.pt  # Regular epoch checkpoints
│   └── emergency_TIMESTAMP.pt # Emergency checkpoints (if memory pressure detected)
├── logs/
│   └── training.log           # Training log file
├── config.json                # Configuration parameters
├── training_log.csv           # Training metrics in CSV format
├── training_metrics.png       # Plot of training metrics
├── loss_curves.png            # Plot of loss curves
├── rmsd_over_training.png     # Plot of RMSD over training
├── curriculum_progress.png    # Plot with curriculum transitions (if enabled)
└── gpu_memory_usage.png       # Plot of GPU memory usage over time
```

## Troubleshooting

If you encounter issues with the training pipeline, here are some common problems and solutions:

### Import Errors
- **Module not found errors**: Make sure you're running the script from the project root directory or use absolute paths.
- **Relative import errors**: The fixed training script (`train_enhanced_model_fixed.py`) addresses common import issues by using direct imports and absolute paths.

### Memory Issues
- **CUDA out of memory**: Try using a smaller batch size, enable gradient checkpointing, or use mixed precision training.
- **Memory spikes during validation**: Set a smaller validation batch size using `--batch_size` and ensure `--gradient_checkpointing` is enabled.

### Curriculum Learning Issues
- **"No valid lengths found in dataset"**: The enhanced analyzer should handle this robustly now, but verify your dataset has accessible length information.
- **Curriculum not advancing stages**: Check that you have enough samples at each sequence length stage (min_sequences_per_stage parameter).

### Training Instability
- **NaN losses**: Check the learning rate; it might be too high. Try starting with a smaller learning rate like 0.0001.
- **Exploding gradients**: Enable gradient clipping with `--max_grad_norm 1.0`.
- **Model not learning**: Check your loss weights to ensure they're properly balanced.

### Cross-Component Interface Issues

If you see errors related to cross-component interfaces, particularly between the training script and the curriculum learning components:

1. Look for "Error during dataset length analysis" or "Error during curriculum filtering" warnings in the logs.
2. Check that the dataset items have a consistent structure with expected fields.
3. Use the debugging features by adding `--debug` and `--debug_samples 10` to test with a small dataset first.
4. Inspect the dataset manually to ensure it has the expected fields by adding temporary logging in the data loading code.

## Example Workflow

For a complete training workflow:

1. **Quick Testing**
   ```bash
   # Test with a small dataset
   ./scripts/examples/train_debug.sh
   ```

2. **Basic Training**
   ```bash
   # Train with standard parameters
   ./scripts/examples/train_basic.sh
   ```

3. **Advanced Training**
   ```bash
   # Train with curriculum learning for better results
   ./scripts/examples/train_curriculum.sh
   ```

4. **Resuming Training**
   ```bash
   # Edit the checkpoint path in the script
   # Then resume training from the checkpoint
   ./scripts/examples/train_resume.sh
   ```