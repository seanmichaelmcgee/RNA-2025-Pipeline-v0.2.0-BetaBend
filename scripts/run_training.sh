#!/bin/bash
# Run training script with common parameters

# Set up paths (adjust as needed for your environment)
SEQUENCES_CSV="data/raw/train_sequences.csv"
LABELS_CSV="data/raw/train_labels.csv"
FEATURES_DIR="data/processed"
OUTPUT_DIR="training_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python scripts/train.py \
  --sequences_csv $SEQUENCES_CSV \
  --labels_csv $LABELS_CSV \
  --features_dir $FEATURES_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size 8 \
  --num_epochs 100 \
  --lr 0.001 \
  --max_seq_length 200 \
  --min_seq_length 10 \
  --val_fraction 0.1 \
  --val_frequency 1 \
  --patience 10 \
  --fape_weight 1.0 \
  --confidence_weight 0.1 \
  --angle_weight 0.5 \
  "$@"  # Pass any additional arguments to the script