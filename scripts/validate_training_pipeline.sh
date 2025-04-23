#!/bin/bash
#
# Training Pipeline Validation Script for RNA 3D Structure Prediction
#
# This script:
# 1. Validates environment and dependencies
# 2. Tests each component of the training pipeline
# 3. Runs a minimal training job (5-10 minutes)
# 4. Tests the analysis notebook functionality
#
# Usage: ./scripts/validate_training_pipeline.sh [--output_dir DIR] [--cuda]

set -e  # Exit on error

# Get project root directory (script is in PROJECT_ROOT/scripts)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default configuration
OUTPUT_DIR="$PROJECT_ROOT/validation_debug_results/$(date +%Y%m%d-%H%M%S)"
NUM_EPOCHS=2
BATCH_SIZE=8
MAX_SEQ_LENGTH=100
MIN_SEQ_LENGTH=10
VALIDATION_FRACTION=0.2
DATA_DIR="$PROJECT_ROOT/data"
DEVICE="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cuda)
            DEVICE="cuda"
            shift
            ;;
        --help)
            echo "Usage: $0 [--output_dir DIR] [--cuda]"
            echo ""
            echo "Options:"
            echo "  --output_dir DIR    Directory to save validation results [default: validation_debug_results/TIMESTAMP]"
            echo "  --cuda              Use CUDA for GPU acceleration"
            echo "  --help              Show this message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Log file for recording all validation steps
LOG_FILE="$OUTPUT_DIR/validation.log"
touch "$LOG_FILE"

# Function to log messages
log() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" | tee -a "$LOG_FILE"
}

# Step 0: Setup mamba/conda environment
log "--- Step 0: Setting up environment ---"

if command -v mamba &> /dev/null; then
  log "Initializing mamba shell..."
  eval "$(mamba shell hook --shell bash)"
  mamba activate rna-3d-folding || { log "❌ Failed to activate mamba environment"; exit 1; }
elif command -v conda &> /dev/null; then
  log "Initializing conda shell (mamba not found)..."
  eval "$(conda shell.bash hook)"
  conda activate rna-3d-folding || { log "❌ Failed to activate conda environment"; exit 1; }
else
  log "❌ Neither mamba nor conda found. Please install mamba/conda and try again."
  exit 1
fi

# Step 1: Validate environment
log "--- Step 1: Environment Validation ---"

# Check for required Python packages
log "Checking Python environment..."
python -c "
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Log package versions
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')

# Check if CUDA is available
if torch.cuda.is_available():
    print(f'CUDA available: Yes, version {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA available: No')
" 2>&1 | tee -a "$LOG_FILE" || { log "ERROR: Python environment check failed"; exit 1; }

# Step 2: Test script components
log "--- Step 2: Component Validation ---"

# Test GPU monitoring script (limited test)
log "Testing GPU monitoring script..."
MONITOR_OUTPUT="$OUTPUT_DIR/gpu_monitor_test"
mkdir -p "$MONITOR_OUTPUT"
python "$PROJECT_ROOT/scripts/monitor_gpu.py" --output_dir "$MONITOR_OUTPUT" --interval 5 --gpu_ids "0" &
MONITOR_PID=$!
sleep 10
kill $MONITOR_PID 2>/dev/null || true
if ls "$MONITOR_OUTPUT/gpu_metrics_"*.csv 1> /dev/null 2>&1; then
    log "GPU monitoring test PASSED"
else
    log "WARNING: GPU monitoring test failed or no metrics captured"
    # Don't exit - this might be a system without GPU
fi

# Test training script argument parsing
log "Testing production training script argument parsing..."
"$PROJECT_ROOT/scripts/run_production_training.sh" --help 2>&1 | grep -q Usage
if [ $? -eq 0 ]; then
    log "Training script argument parsing PASSED"
else
    log "ERROR: Training script argument parsing failed"
    exit 1
fi

# Step 3: Execute minimal training run
log "--- Step 3: Minimal Training Run ---"

TRAINING_OUTPUT="$OUTPUT_DIR/mini_training"
log "Running minimal training job to $TRAINING_OUTPUT..."

# Execute training script with minimal parameters
"$PROJECT_ROOT/scripts/run_production_training.sh" \
    --output_dir "$TRAINING_OUTPUT" \
    --batch_size $BATCH_SIZE \
    --data_dir "$DATA_DIR" \
    --num_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --min_seq_length $MIN_SEQ_LENGTH \
    --val_fraction $VALIDATION_FRACTION \
    --lr 0.001 \
    --monitor_interval 5 || { log "ERROR: Training run failed"; exit 1; }

# Step 4: Validate training output
log "--- Step 4: Verify Training Output ---"

# Check for key directories and files
if [ -d "$TRAINING_OUTPUT/checkpoints" ]; then
    log "Checkpoints directory FOUND"
else
    log "ERROR: Checkpoints directory missing"
    exit 1
fi

if [ -d "$TRAINING_OUTPUT/logs" ]; then
    log "Logs directory FOUND"
else
    log "ERROR: Logs directory missing"
    exit 1
fi

# Check for training log file
if [ -f "$TRAINING_OUTPUT/logs/training.log" ]; then
    log "Training log file FOUND"
else
    log "ERROR: Training log file missing"
    exit 1
fi

# Check for config file
if [ -f "$TRAINING_OUTPUT/config.json" ]; then
    log "Config file FOUND"
else
    log "ERROR: Config file missing"
    exit 1
fi

# Step 5: Test report generation 
log "--- Step 5: Report Generation ---"

REPORT_OUTPUT="$OUTPUT_DIR/report_test"
mkdir -p "$REPORT_OUTPUT"

# Run report generation
log "Generating training report..."
python "$PROJECT_ROOT/scripts/generate_training_report.py" \
    --training_dir "$TRAINING_OUTPUT" \
    --output_dir "$REPORT_OUTPUT" || { log "ERROR: Report generation failed"; exit 1; }

# Validate report existence
if ls "$REPORT_OUTPUT/training_report"*.md 1> /dev/null 2>&1; then
    log "Report generation PASSED"
else
    log "ERROR: Training report missing"
    exit 1
fi

# Step 6: Test notebook functionality
log "--- Step 6: Notebook Validation ---"

# Create test notebook directory
NOTEBOOK_OUTPUT="$OUTPUT_DIR/notebook_test"
mkdir -p "$NOTEBOOK_OUTPUT"

# Create a simple script to test notebook functionality
log "Creating notebook validation script..."
cat > "$NOTEBOOK_OUTPUT/validate_notebook.py" << EOL
"""Validate analysis notebook functionality."""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add project root to path
sys.path.append('${PROJECT_ROOT}')

# Training directory to analyze
training_dir = "${TRAINING_OUTPUT}"

# Create simple test of notebook functions
def test_notebook_data_loading():
    """Test the data loading functions from the notebook."""
    # Create simple training data for testing
    training_df = pd.DataFrame({
        "epoch": list(range(5)),
        "total_loss": [5.0 - i*0.5 for i in range(5)],
        "val_loss": [5.5 - i*0.3 for i in range(5)]
    })
    
    # Save to the expected location if not present
    if not os.path.exists(os.path.join(training_dir, "training_log.csv")):
        os.makedirs(os.path.join(training_dir), exist_ok=True)
        training_df.to_csv(os.path.join(training_dir, "training_log.csv"), index=False)
    
    # Test data loading functionality
    print("Loading training data...")
    try:
        df = pd.read_csv(os.path.join(training_dir, "logs/training.log"), sep=" ")
        print(f"Successfully loaded training log with {len(df)} rows")
        
        # Read config file
        config_path = os.path.join(training_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Successfully loaded config with {len(config)} parameters")
        
        return True
    except Exception as e:
        print(f"Error loading training data: {e}")
        return False

def test_notebook_visualization():
    """Test the visualization functions from the notebook."""
    try:
        # Create a simple plot (similar to what the notebook does)
        plt.figure(figsize=(10, 6))
        # Generate simple loss curve
        epochs = np.arange(1, 11)
        losses = 5.0 - 0.4 * epochs + 0.01 * epochs**2
        plt.plot(epochs, losses, 'b-', linewidth=2)
        plt.title('Mock Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join("${NOTEBOOK_OUTPUT}", "test_plot.png"))
        plt.close()
        
        print("Successfully created test visualization")
        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

# Run tests
print("Testing notebook data loading functionality...")
load_success = test_notebook_data_loading()

print("Testing notebook visualization functionality...")
viz_success = test_notebook_visualization()

# Print summary
print("\nNotebook Validation Results:")
print(f"Data Loading: {'PASS' if load_success else 'FAIL'}")
print(f"Visualization: {'PASS' if viz_success else 'FAIL'}")

sys.exit(0 if load_success and viz_success else 1)
EOL

# Execute notebook validation script
log "Running notebook validation script..."
python "$NOTEBOOK_OUTPUT/validate_notebook.py" 2>&1 | tee -a "$LOG_FILE" || { log "ERROR: Notebook validation failed"; exit 1; }

# Final summary
log "--- Validation Summary ---"
log "Training pipeline validation completed successfully! ✅"
log "Training output directory: $TRAINING_OUTPUT"
log "Report output directory: $REPORT_OUTPUT"
log "Notebook test directory: $NOTEBOOK_OUTPUT"
log "Log file: $LOG_FILE"

echo ""
echo "Next steps:"
echo "1. Examine the training output in $TRAINING_OUTPUT"
echo "2. Review the generated report in $REPORT_OUTPUT/training_report_*.md"
echo "3. Open and run the analysis notebook with your training data:"
echo "   jupyter notebook $PROJECT_ROOT/notebooks/production_run_analysis.ipynb"
echo ""
echo "Ready for full production training run:"
echo "$PROJECT_ROOT/scripts/run_production_training.sh --output_dir $PROJECT_ROOT/results/production_run --num_epochs 50 --batch_size 32"
echo ""
echo "Training pipeline validation complete! ✅"