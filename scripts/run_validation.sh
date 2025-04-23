#!/bin/bash
# Run validation on a trained model

# Set default paths (adjust as needed for your environment)
SEQUENCES_CSV="data/raw/train_sequences.csv"
LABELS_CSV="data/raw/train_labels.csv"
FEATURES_DIR="data/processed"
OUTPUT_DIR="validation_results"
CHECKPOINT="training_results/checkpoints/best_model.pt"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run validation
python scripts/validate_trained_model.py \
  --checkpoint $CHECKPOINT \
  --sequences_csv $SEQUENCES_CSV \
  --labels_csv $LABELS_CSV \
  --features_dir $FEATURES_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size 8 \
  --generate_plots \
  "$@"  # Pass any additional arguments to the script