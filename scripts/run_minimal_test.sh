#!/bin/bash
# Run a minimal test of the training pipeline with fixed issues
set -e

# Create output directory
OUTPUT_DIR="results/minimal_test_fixed"
mkdir -p $OUTPUT_DIR

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

echo "Running minimal test with fixed curriculum and loss computation modules..."

# Run a minimal version of the training (1 epoch, debug mode)
python scripts/train_enhanced_model_fixed.py \
  --debug \
  --debug_samples 20 \
  --train_csv data/raw/train_sequences.csv \
  --labels_csv data/raw/train_labels.csv \
  --features_dir data/processed/ \
  --output_dir $OUTPUT_DIR \
  --batch_size 4 \
  --epochs 1 \
  --curriculum_learning \
  --curriculum_stages 50 75 100 \
  --mixed_precision \
  --patience 999 \
  --gradient_checkpointing

echo "Minimal test completed - check logs for any remaining errors"