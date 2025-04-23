#!/bin/bash
#
# Production Training Script for RNA 3D Structure Prediction Model
#
# This script runs a comprehensive training job with:
# - Proper output directory structure
# - GPU monitoring
# - Checkpoint saving
# - Training configuration tracking

set -e  # Exit on error

# ===== Configuration =====
# Default parameters
DATA_DIR="data"
OUTPUT_DIR=""
BATCH_SIZE=16
NUM_EPOCHS=50
LR=0.0005
MAX_SEQ_LENGTH=300
MIN_SEQ_LENGTH=10
VAL_FRACTION=0.15
VAL_FREQUENCY=1
PATIENCE=15
FAPE_WEIGHT=1.0
CONFIDENCE_WEIGHT=0.2
ANGLE_WEIGHT=0.5
MONITOR_GPU=true
MONITOR_INTERVAL=10
USE_MIXED_PRECISION=true
NUM_WORKERS=4
SEED=42
GPU_IDS="0"  # Use all available GPUs by default

# ===== Helper Functions =====
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --data_dir DIR             Path to data directory [default: $DATA_DIR]"
    echo "  --output_dir DIR           Path to output directory [required]"
    echo "  --batch_size N             Batch size for training [default: $BATCH_SIZE]"
    echo "  --num_epochs N             Number of training epochs [default: $NUM_EPOCHS]"
    echo "  --lr FLOAT                 Learning rate [default: $LR]"
    echo "  --max_seq_length N         Maximum sequence length [default: $MAX_SEQ_LENGTH]"
    echo "  --min_seq_length N         Minimum sequence length [default: $MIN_SEQ_LENGTH]"
    echo "  --val_fraction FLOAT       Fraction of data for validation [default: $VAL_FRACTION]"
    echo "  --val_frequency N          Validation frequency in epochs [default: $VAL_FREQUENCY]"
    echo "  --patience N               Early stopping patience [default: $PATIENCE]"
    echo "  --fape_weight FLOAT        Weight for FAPE loss [default: $FAPE_WEIGHT]"
    echo "  --confidence_weight FLOAT  Weight for confidence loss [default: $CONFIDENCE_WEIGHT]"
    echo "  --angle_weight FLOAT       Weight for angle loss [default: $ANGLE_WEIGHT]"
    echo "  --no_monitor_gpu           Disable GPU monitoring"
    echo "  --monitor_interval N       GPU monitoring interval in seconds [default: $MONITOR_INTERVAL]"
    echo "  --no_mixed_precision       Disable mixed precision training"
    echo "  --num_workers N            DataLoader num_workers [default: $NUM_WORKERS]"
    echo "  --seed N                   Random seed [default: $SEED]"
    echo "  --gpu_ids IDS              Comma-separated list of GPU IDs to use (e.g., '0,1') [default: use all]"
    echo "  --help                     Show this message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --min_seq_length)
            MIN_SEQ_LENGTH="$2"
            shift 2
            ;;
        --val_fraction)
            VAL_FRACTION="$2"
            shift 2
            ;;
        --val_frequency)
            VAL_FREQUENCY="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --fape_weight)
            FAPE_WEIGHT="$2"
            shift 2
            ;;
        --confidence_weight)
            CONFIDENCE_WEIGHT="$2"
            shift 2
            ;;
        --angle_weight)
            ANGLE_WEIGHT="$2"
            shift 2
            ;;
        --no_monitor_gpu)
            MONITOR_GPU=false
            shift
            ;;
        --monitor_interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        --no_mixed_precision)
            USE_MIXED_PRECISION=false
            shift
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output_dir is required"
    usage
fi

# Create timestamp for run ID
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_ID="run_${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_ID}"

# Create directory structure
mkdir -p "${OUTPUT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}/metrics"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/predictions"

# Save configuration
CONFIG_FILE="${OUTPUT_DIR}/config.json"
cat > "$CONFIG_FILE" <<EOF
{
  "run_id": "${RUN_ID}",
  "timestamp": "${TIMESTAMP}",
  "data_dir": "${DATA_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "batch_size": ${BATCH_SIZE},
  "num_epochs": ${NUM_EPOCHS},
  "learning_rate": ${LR},
  "max_seq_length": ${MAX_SEQ_LENGTH},
  "min_seq_length": ${MIN_SEQ_LENGTH},
  "val_fraction": ${VAL_FRACTION},
  "val_frequency": ${VAL_FREQUENCY},
  "patience": ${PATIENCE},
  "fape_weight": ${FAPE_WEIGHT},
  "confidence_weight": ${CONFIDENCE_WEIGHT},
  "angle_weight": ${ANGLE_WEIGHT},
  "mixed_precision": ${USE_MIXED_PRECISION},
  "num_workers": ${NUM_WORKERS},
  "seed": ${SEED},
  "gpu_ids": "${GPU_IDS}"
}
EOF

echo "Starting training run ${RUN_ID}"
echo "Configuration saved to ${CONFIG_FILE}"

# Set up GPU monitoring if enabled
GPU_MONITOR_PID=""
if [ "$MONITOR_GPU" = true ]; then
    echo "Starting GPU monitoring..."
    python scripts/monitor_gpu.py \
        --output_dir "${OUTPUT_DIR}" \
        --interval "${MONITOR_INTERVAL}" \
        --gpu_ids "${GPU_IDS}" \
        > "${OUTPUT_DIR}/logs/gpu_monitor.log" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "GPU monitoring started (PID: ${GPU_MONITOR_PID})"
fi

# Set CUDA_VISIBLE_DEVICES if specific GPUs are requested
if [ "$GPU_IDS" != "0" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    echo "Using GPUs: ${GPU_IDS}"
fi

# Function to cleanup on exit
cleanup() {
    # Stop GPU monitoring if running
    if [ -n "$GPU_MONITOR_PID" ]; then
        echo "Stopping GPU monitoring (PID: ${GPU_MONITOR_PID})..."
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
    
    echo "Training completed or interrupted. See logs at ${OUTPUT_DIR}/logs"
}

# Register cleanup function
trap cleanup EXIT

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Training started at ${START_TIME}" > "${OUTPUT_DIR}/logs/training_timeline.txt"

# Run the training
echo "Starting model training..."
python -u src/train.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --min_seq_length ${MIN_SEQ_LENGTH} \
    --val_fraction ${VAL_FRACTION} \
    --val_frequency ${VAL_FREQUENCY} \
    --patience ${PATIENCE} \
    --fape_weight ${FAPE_WEIGHT} \
    --confidence_weight ${CONFIDENCE_WEIGHT} \
    --angle_weight ${ANGLE_WEIGHT} \
    --mixed_precision ${USE_MIXED_PRECISION} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    2>&1 | tee "${OUTPUT_DIR}/logs/training.log"

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Training ended at ${END_TIME}" >> "${OUTPUT_DIR}/logs/training_timeline.txt"

# Generate training report
echo "Generating training report..."
python scripts/generate_training_report.py \
    --training_dir "${OUTPUT_DIR}" \
    --output_format "pdf" \
    > "${OUTPUT_DIR}/logs/report_generation.log" 2>&1

echo "Training complete. Results saved to ${OUTPUT_DIR}"
echo "To visualize the results, open the notebook:"
echo "  notebooks/production_run_analysis.ipynb"
echo ""
echo "To explore specific run metrics, use the command:"
echo "  python scripts/generate_training_report.py --training_dir ${OUTPUT_DIR}"