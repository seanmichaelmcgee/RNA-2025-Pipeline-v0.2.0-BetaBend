#!/bin/bash
# Debug training run for RNA 3D structure prediction

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Go to project root
cd $(dirname "$0")/../..

# Run the enhanced training script with:
# - Debug mode with small dataset (10 samples)
# - Mixed precision for memory efficiency
# - Simplified model architecture
# - Few epochs for quick testing

# Use the fixed version of the training script
python3 scripts/train_enhanced_model_fixed.py \
    --debug \
    --debug_samples 10 \
    --mixed_precision \
    --num_blocks 3 \
    --residue_embed_dim 128 \
    --pair_embed_dim 32 \
    --num_heads 4 \
    --ff_dim 256 \
    --batch_size 2 \
    --epochs 3 \
    --output_dir results/debug_model