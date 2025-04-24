#!/bin/bash
# Basic training run for RNA 3D structure prediction with memory optimization

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Go to project root
cd $(dirname "$0")/../..

# Run the enhanced training script with:
# - Mixed precision for memory efficiency
# - Gradient checkpointing to reduce memory usage
# - Regular checkpointing every 5 epochs
# - Maximum sequence length of 200 nucleotides

# Use the fixed version of the training script
python3 scripts/train_enhanced_model_fixed.py \
    --mixed_precision \
    --gradient_checkpointing \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --max_seq_len 200 \
    --save_interval_epochs 5 \
    --epochs 50 \
    --output_dir results/enhanced_model \
    --memory_fraction_warning 0.8 \
    --memory_fraction_critical 0.9