#!/bin/bash
# Script to export results from the validation notebook

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the Python export script
EXPORT_SCRIPT="${SCRIPT_DIR}/../export_notebook_results.py"

# Path to the validation notebook
NOTEBOOK="${SCRIPT_DIR}/validation_technical.ipynb"

# Output path
RESULTS_DIR="${SCRIPT_DIR}/results"
OUTPUT="${RESULTS_DIR}/validation_technical_results.md"

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Export the results
python "${EXPORT_SCRIPT}" "${NOTEBOOK}" "${OUTPUT}"

echo "Results exported to ${OUTPUT}"