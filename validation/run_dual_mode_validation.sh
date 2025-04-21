#!/bin/bash
# Run dual-mode validation for RNA structure prediction

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default parameters
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_DIR="${SCRIPT_DIR}/tier1_technical/results"
SUBSET="technical"  # Default subset (technical, scientific, comprehensive)
RNA_IDS=("R1107")  # Example RNA ID - empty means automatic detection

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data-dir)
      DATA_DIR="$2"
      shift
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --subset)
      SUBSET="$2"
      shift
      shift
      ;;
    --rna-ids)
      shift
      RNA_IDS=()
      while [[ $# -gt 0 && ! $1 == --* ]]; do
        RNA_IDS+=("$1")
        shift
      done
      ;;
    --cpu)
      CPU_FLAG="--cpu"
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build RNA IDs argument string if provided
RNA_IDS_ARG=""
if [[ ${#RNA_IDS[@]} -gt 0 ]]; then
  for id in "${RNA_IDS[@]}"; do
    RNA_IDS_ARG="$RNA_IDS_ARG --rna-ids $id"
  done
fi

# Build checkpoint argument if provided
CHECKPOINT_ARG=""
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT_ARG="--checkpoint $CHECKPOINT"
fi

# Print summary of parameters
echo "Running dual-mode validation..."
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Validation subset: $SUBSET"
if [[ ${#RNA_IDS[@]} -gt 0 ]]; then
  echo "RNA IDs: ${RNA_IDS[*]}"
else
  echo "RNA IDs: Auto-detection enabled"
fi
if [[ -n "$CPU_FLAG" ]]; then
  echo "Device: CPU (forced)"
else
  echo "Device: Auto-detection (prefers GPU if available)"
fi
if [[ -n "$CHECKPOINT" ]]; then
  echo "Checkpoint: $CHECKPOINT"
else
  echo "Checkpoint: None (using randomly initialized model)"
fi

# Run validation
echo -e "\nStarting validation script..."

# Run the script with the activated environment
python "${SCRIPT_DIR}/tier1_technical/run_dual_mode_validation.py" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  $RNA_IDS_ARG \
  $CHECKPOINT_ARG \
  $CPU_FLAG

# Check if execution was successful
if [ $? -eq 0 ]; then
  echo -e "\nValidation complete. Results saved to $OUTPUT_DIR"
  # If visualizations were generated, show path
  if [ -f "$OUTPUT_DIR/per_residue_rmsd.png" ]; then
    echo "Visualizations are available in $OUTPUT_DIR"
  fi
else
  echo -e "\nError: Validation script exited with errors. Check the output above for details."
fi