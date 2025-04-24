#!/usr/bin/env bash
# Test script to reformat a submission file

set -e  # Exit on error

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_SUBMISSION="$SCRIPT_DIR/../submissions/submission_final_model_20250423-212450.csv"
TEST_SEQUENCES="$SCRIPT_DIR/../data/raw/test_sequences.csv"
OUTPUT_PATH="$SCRIPT_DIR/../submissions/submission_kaggle_format.csv"

# Make script executable if needed
chmod +x "$SCRIPT_DIR/reformat_submission.py"

# Run the reformatting script with verbose output
echo "Reformatting submission file..."

# Use python3 explicitly (some environments default to python2)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

$PYTHON_CMD "$SCRIPT_DIR/reformat_submission.py" \
  --input "$INPUT_SUBMISSION" \
  --sequences "$TEST_SEQUENCES" \
  --output "$OUTPUT_PATH" \
  --verbose

# Check if the output file was created
if [ -f "$OUTPUT_PATH" ]; then
  echo "Success! Reformatted submission saved to: $OUTPUT_PATH"
  
  # Show a preview of the first few lines
  echo "Preview of reformatted submission:"
  head -n 5 "$OUTPUT_PATH"
  
  # Count the number of rows in the output
  ROWS=$(wc -l < "$OUTPUT_PATH")
  echo "Total rows in reformatted submission: $((ROWS - 1))"  # Subtract 1 for header
else
  echo "Error: Failed to create reformatted submission at $OUTPUT_PATH"
  exit 1
fi

echo "Done!"