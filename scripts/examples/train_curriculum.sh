#!/bin/bash
# Curriculum learning training run for RNA 3D structure prediction

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Go to project root
cd $(dirname "$0")/../..

# Run the enhanced training script with:
# - Curriculum learning (gradually increasing sequence length)
# - Mixed precision for memory efficiency
# - Gradient checkpointing to reduce memory usage
# - Adaptive batch sizing based on sequence length
# - Increased epochs for comprehensive training

# Use the fixed version of the training script
python3 scripts/train_enhanced_model_fixed.py \
    --mixed_precision \
    --gradient_checkpointing \
    --curriculum_learning \
    --curriculum_stages 100 150 200 250 300 \
    --epochs_per_stage 5 \
    --batch_adaptive \
    --batch_size 16 \
    --epochs 100 \
    --save_interval_epochs 5 \
    --save_interval_steps 2000 \
    --output_dir results/enhanced_model_curriculum \
    --memory_fraction_warning 0.8 \
    --memory_fraction_critical 0.9