#!/bin/bash
# Production-ready training script for RNA 3D structure prediction
# This script is optimized for producing a high-quality model with reasonable training time

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Go to project root
cd $(dirname "$0")/../..

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_DIR="results/production_model_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Log training parameters for reproducibility
echo "Starting production training at $(date)" | tee ${OUTPUT_DIR}/training_log.txt
echo "Output directory: ${OUTPUT_DIR}" | tee -a ${OUTPUT_DIR}/training_log.txt

# Make dataset analyzer executable
chmod +x scripts/fix_dataset_analyzer.py

# Run the production-ready training with optimized parameters:
# - Curriculum learning to gradually increase sequence length
# - Mixed precision for memory efficiency
# - Gradient checkpointing to reduce memory usage
# - Reduced model size for faster training (5 blocks instead of 6)
# - Moderate batch size with gradient accumulation
# - 50 epochs with early stopping (patience 7)
# - Regular checkpointing every 3 epochs
python3 scripts/train_enhanced_model_fixed.py \
    --mixed_precision \
    --gradient_checkpointing \
    --curriculum_learning \
    --curriculum_stages 80 120 160 200 250 300 \
    --epochs_per_stage 3 \
    --batch_adaptive \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --num_blocks 5 \
    --residue_embed_dim 160 \
    --pair_embed_dim 64 \
    --num_heads 8 \
    --ff_dim 384 \
    --epochs 50 \
    --patience 7 \
    --scheduler cosine \
    --lr 0.0005 \
    --fape_weight 1.0 \
    --confidence_weight 0.1 \
    --angle_weight 0.5 \
    --save_interval_epochs 3 \
    --memory_fraction_warning 0.85 \
    --memory_fraction_critical 0.92 \
    --output_dir $OUTPUT_DIR \
    --val_split 0.15 \
    2>&1 | tee -a ${OUTPUT_DIR}/training_log.txt

# Log completion
echo "Training completed at $(date)" | tee -a ${OUTPUT_DIR}/training_log.txt
echo "Final model saved to ${OUTPUT_DIR}/checkpoints/best_model.pt" | tee -a ${OUTPUT_DIR}/training_log.txt

# Optional: run quick validation on the final model
# Uncomment if you want to run validation immediately after training
# python3 scripts/validate_model.py --model_path ${OUTPUT_DIR}/checkpoints/best_model.pt --output_dir ${OUTPUT_DIR}/validation