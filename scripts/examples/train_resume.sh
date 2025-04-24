#!/bin/bash
# Resume training from a checkpoint for RNA 3D structure prediction

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Go to project root
cd $(dirname "$0")/../..

# This script assumes you have a previous training run and want to resume
# Replace the path below with the actual path to your checkpoint
CHECKPOINT_PATH="results/enhanced_model/run_YYYYMMDD-HHMMSS/checkpoints/best_model.pt"

# Run the enhanced training script with:
# - Resume from a checkpoint
# - Continue with the same settings
# - Extend training for more epochs

# Use the fixed version of the training script
python3 scripts/train_enhanced_model_fixed.py \
    --resume ${CHECKPOINT_PATH} \
    --mixed_precision \
    --gradient_checkpointing \
    --epochs 100 \
    --save_interval_epochs 5 \
    --output_dir results/enhanced_model_continued \
    --memory_fraction_warning 0.8 \
    --memory_fraction_critical 0.9