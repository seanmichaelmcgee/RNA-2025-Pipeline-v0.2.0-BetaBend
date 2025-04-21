# Structure Evaluation Metrics Handoff

## Component Overview

This document details the structure evaluation metrics implemented for the RNA 3D folding model. These metrics are used to assess the quality of predicted RNA structures by comparing them to reference structures. The metrics include RMSD (Root-Mean-Square Deviation) and TM-score (Template-Modeling score), along with per-residue analysis capabilities.

## Implementation Details

### Location

- **Module**: `src/utils/structure_metrics.py`
- **Tests**: `tests/test_structure_metrics.py`
- **Validation Script**: `scripts/validate_model.py`

### Dependencies

- PyTorch for tensor operations
- NumPy for mathematical operations
- Matplotlib for visualization (in the validation script)
- `src.losses.stable_kabsch_align` for optimal structure alignment

### Core Functions

1. **`compute_rmsd`**: Calculates RMSD between predicted and true coordinates
2. **`compute_tm_score`**: Calculates TM-score between predicted and true coordinates
3. **`compute_structure_metrics`**: Convenience function to calculate multiple metrics at once
4. **`compute_per_residue_rmsd`**: Calculates per-residue RMSD for detailed analysis

### Validation Strategy

The validation script implements a tiered approach to model evaluation:

1. **Tier 1 (Fast Technical Validation)**: Basic model functionality on a small test set
2. **Tier 2 (Scientific Validation)**: Comprehensive evaluation on challenging cases
3. **Tier 3 (Comprehensive Validation)**: Full dataset evaluation and comparison metrics

## Usage Examples

### Basic RMSD Calculation

```python
from src.utils.structure_metrics import compute_rmsd

# Calculate RMSD between predicted and true coordinates
rmsd = compute_rmsd(
    pred_coords,  # shape (batch_size, seq_len, 3)
    true_coords,  # shape (batch_size, seq_len, 3)
    mask,         # shape (batch_size, seq_len), optional
    aligned=True  # Whether to optimally align structures before calculation
)
```

### TM-score Calculation

```python
from src.utils.structure_metrics import compute_tm_score

# Calculate TM-score between predicted and true coordinates
tm_score = compute_tm_score(
    pred_coords,  # shape (batch_size, seq_len, 3)
    true_coords,  # shape (batch_size, seq_len, 3)
    mask,         # shape (batch_size, seq_len), optional
)
```

### Multiple Metrics at Once

```python
from src.utils.structure_metrics import compute_structure_metrics

# Calculate multiple metrics at once
metrics = compute_structure_metrics(
    pred_coords,  # shape (batch_size, seq_len, 3)
    true_coords,  # shape (batch_size, seq_len, 3)
    mask,         # shape (batch_size, seq_len), optional
    metrics=["rmsd", "tm_score"]  # Metrics to compute
)

# Access individual metrics
rmsd = metrics["rmsd"]
tm_score = metrics["tm_score"]
```

### Per-Residue Analysis

```python
from src.utils.structure_metrics import compute_per_residue_rmsd

# Calculate per-residue RMSD
per_residue_rmsd = compute_per_residue_rmsd(
    pred_coords,   # shape (batch_size, seq_len, 3)
    true_coords,   # shape (batch_size, seq_len, 3)
    mask,          # shape (batch_size, seq_len), optional
    aligned=True,  # Whether to optimally align structures before calculation
    window_size=1  # Size of window for local RMSD calculation
)
```

### Running Validation

```bash
# Run Tier 1 validation (fast technical validation)
python scripts/validate_model.py --tier 1 --checkpoint path/to/model.pt

# Run Tier 2 validation (scientific validation)
python scripts/validate_model.py --tier 2 --checkpoint path/to/model.pt

# Run Tier 3 validation (comprehensive validation)
python scripts/validate_model.py --tier 3 --checkpoint path/to/model.pt
```

## Advanced Features

### Shape and Device Compatibility

- Handles different tensor shapes, including both 3D and 4D coordinate tensors
- Works on both CPU and CUDA devices
- Supports batched calculations for efficiency
- Handles masked inputs for sequences of different lengths

### Numerical Stability

- Uses epsilon values to ensure numerical stability
- Special case handling for identical structures
- Robust calculation of distances to avoid NaN values
- Proper error handling for edge cases

### Optimal Structure Alignment

- Uses the Kabsch algorithm for optimal structural alignment
- Reuses the stable implementation from `src.losses.stable_kabsch_align`
- Handles degenerate cases gracefully

## Testing

Comprehensive tests are provided in `tests/test_structure_metrics.py`, covering:

- Identical structures (expect near-zero RMSD)
- Shifted structures (with and without alignment)
- Rotated structures (with and without alignment)
- Masked inputs
- Batched calculations
- Different input shapes (3D and 4D)
- Device compatibility (CPU and CUDA)

## Known Limitations

1. The current implementation uses a simplified TM-score calculation that assumes C1' atoms for all residues. For protein structures, TM-score typically uses Cα atoms.

2. Performance on very large structures (>1000 residues) may be slower due to the Kabsch alignment algorithm.

3. The per-residue RMSD calculation with non-unit window size may yield unexpected results near the sequence boundaries.

## Future Improvements

1. Add more advanced structure comparison metrics (e.g., GDT-TS, lDDT)
2. Optimize performance for very large structures
3. Add support for multi-model evaluation (e.g., for ensemble predictions)
4. Integrate with visualization tools for better interpretation of results
5. Add support for evaluating specific structural elements (e.g., helices, loops)

## Integration with Model Training

These metrics are designed to be used in the following scenarios:

1. **Model Evaluation**: Assessing model performance on test sets
2. **Training Monitoring**: Tracking structural quality during training
3. **Hyperparameter Tuning**: Comparing models with different hyperparameters
4. **Ensemble Selection**: Choosing the best models for ensemble prediction
5. **Validation Reporting**: Generating reports for model validation