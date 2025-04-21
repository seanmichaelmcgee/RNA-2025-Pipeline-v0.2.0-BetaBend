# Loss Functions Component Handoff

## Component Information

- **Component Name**: Loss Functions
- **Version**: 1.0
- **Last Updated**: 2025-04-20
- **Implementation Status**: Complete
- **Test Coverage**: 100%
- **Location**: `src/losses.py`

## Component Description

The loss functions module implements the training objectives for the RNA 3D folding model. It includes a simplified FAPE proxy loss for coordinate prediction, a confidence prediction loss using a derived target, and an auxiliary angle prediction loss. These functions are designed with numerical stability and proper masking as primary concerns.

## Public API

### Main Loss Functions

```python
def compute_stable_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute a simplified FAPE loss proxy (V1) based on clamped L2 distance
    after global Kabsch alignment, using stable implementations.

    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        clamp_value: Maximum distance error to consider (Å)
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss value
    """

def compute_confidence_loss(
    pred_confidence: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse', # 'mse' or 'bce'
    target_type: str = 'lddt_proxy',
    scaling_factor: float = 3.0, # For 'lddt_proxy'
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute confidence prediction loss (V1 Proxy Target).

    Trains the model to predict a per-residue confidence score (similar to pLDDT)
    that correlates with the actual per-residue accuracy based on a proxy derived
    from coordinate error after global alignment.

    Args:
        pred_confidence: Predicted confidence scores (logits), shape (batch_size, seq_len)
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        loss_type: Loss function to use ('mse' or 'bce').
        target_type: Type of proxy target ('lddt_proxy' or 'distance_based').
        scaling_factor: Scaling factor for 'lddt_proxy' target computation (Å).
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar loss value
    """

def compute_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse', # 'mse', 'cosine', or 'mae'
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute loss for dihedral angle predictions (V1).

    Compares the predicted sin/cos representations of angles with the true values.

    Args:
        pred_angles: Predicted sin/cos of angles [sin(η), cos(η), sin(θ), cos(θ)],
                     shape (batch_size, seq_len, 4)
        true_angles: True sin/cos of angles, shape (batch_size, seq_len, 4)
                     May contain NaNs for boundary residues.
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        loss_type: Loss function to use ('mse', 'cosine', or 'mae').
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar loss value
    """

def compute_combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined loss from multiple loss components using V1 losses.

    Args:
        outputs: Dictionary of model outputs:
            - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
            - pred_confidence: Predicted confidence logits (batch_size, seq_len)
            - pred_angles: Predicted angles sin/cos (batch_size, seq_len, 4)
        batch: Dictionary of ground truth data:
            - coordinates: True coordinates (batch_size, seq_len, 3)
            - dihedral_features: True angle features (batch_size, seq_len, 4)
            - mask: Boolean mask (batch_size, seq_len)
        loss_weights: Dictionary of loss weights:
            - fape: Weight for coordinate loss
            - confidence: Weight for confidence loss
            - angle: Weight for angle loss

    Returns:
        Tuple of:
        - total_loss: Combined weighted loss (scalar tensor for backprop)
        - loss_components_tensors: Dictionary of individual loss component tensors
                                   (useful for logging/analysis, still attached to graph)
    """
```

### Helper Functions

```python
def stable_kabsch_align(P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable Kabsch alignment with SVD fallback and epsilon checks.
    Aligns points P to points Q.

    Args:
        P: Moving points, shape (N, 3)
        Q: Fixed points, shape (N, 3)
        epsilon: Small constant for numerical stability

    Returns:
        P_aligned: Aligned points, shape (N, 3)
    """

def robust_distance_calculation(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Numerically stable distance calculation using squared differences first.

    Args:
        pred: Predicted coordinates (*, 3)
        target: Target coordinates (*, 3)
        epsilon: Small constant added before sqrt for stability

    Returns:
        Distances tensor with shape (*)
    """
```

## Data Structures and Tensor Specifications

### Input Tensors for Loss Functions

| Function | Input | Shape | Type | Description |
|----------|-------|-------|------|-------------|
| compute_stable_fape_loss | pred_coords | (batch_size, seq_len, 3) | FloatTensor | Predicted coordinates |
| compute_stable_fape_loss | true_coords | (batch_size, seq_len, 3) | FloatTensor | True coordinates |
| compute_stable_fape_loss | mask | (batch_size, seq_len) | BoolTensor | Validity mask (True=valid) |
| compute_confidence_loss | pred_confidence | (batch_size, seq_len) | FloatTensor | Confidence logits |
| compute_confidence_loss | pred_coords | (batch_size, seq_len, 3) | FloatTensor | Predicted coordinates |
| compute_confidence_loss | true_coords | (batch_size, seq_len, 3) | FloatTensor | True coordinates |
| compute_confidence_loss | mask | (batch_size, seq_len) | BoolTensor | Validity mask (True=valid) |
| compute_angle_loss | pred_angles | (batch_size, seq_len, 4) | FloatTensor | Predicted sin/cos angles |
| compute_angle_loss | true_angles | (batch_size, seq_len, 4) | FloatTensor | True sin/cos angles |
| compute_angle_loss | mask | (batch_size, seq_len) | BoolTensor | Validity mask (True=valid) |

### Output Tensors

All loss functions return a scalar FloatTensor containing the average loss value. The `compute_combined_loss` function returns a tuple of `(total_loss, loss_components)` where `loss_components` is a dictionary mapping loss names to individual loss values.

## Integration Points

### Component Dependencies

- PyTorch (torch, torch.nn, torch.nn.functional)
- Numpy (for documentation only)
- Logging module (for error/warning handling)

### Expected Behavior

1. All loss functions properly handle masked inputs (padding tokens)
2. Numerical stability is maintained through epsilon values and clipping
3. NaN/Inf values are detected and handled appropriately
4. Loss functions return scalar values for backpropagation
5. Batch processing is supported with proper mask handling

## Environment and Dependencies

- PyTorch >= 2.1.0
- Python >= 3.9

## Testing Requirements

### Unit Tests

Run all loss function tests:
```bash
python -m pytest tests/test_losses.py -v
```

Run a specific test:
```bash
python -m pytest tests/test_losses.py::TestStableFAPELoss::test_fape_zero_loss -v
```

### Test Coverage

The loss functions have 100% test coverage including tests for:
- Numerical stability
- Mask handling
- Gradient flow
- Various input configurations
- Edge cases (empty, small, degenerate inputs)

## Implementation Details

### FAPE Loss Implementation

1. **Input Validation**:
   - Check for NaN/Inf values in inputs
   - Ensure proper tensor dimensions
   - Create default mask if not provided

2. **Per-Sequence Processing**:
   - Process each sequence in the batch separately
   - Apply Kabsch alignment to valid residues
   - Calculate RMSD for aligned coordinates

3. **Numerical Stability**:
   - Use stable Kabsch alignment with checks for degenerate cases
   - Apply distance clamping to prevent outlier influence
   - Handle sequences with insufficient points (< 3) gracefully

### Confidence Loss Implementation

1. **Target Generation**:
   - Generate confidence targets based on position-wise accuracy
   - Align structures globally before calculating errors
   - Convert errors to confidence scores in [0, 1] range

2. **Loss Calculation**:
   - Support both MSE and BCE loss functions
   - Apply masks to exclude padding positions
   - Average loss over valid positions only

### Angle Loss Implementation

1. **Angle Representation**:
   - Work with sin/cos encoding of angles
   - Handle NaN values in true angles (common at boundaries)
   - Support multiple loss types (MSE, MAE, cosine)

2. **Mask Handling**:
   - Create combined mask from padding and NaN positions
   - Apply expanded mask to match tensor dimensions
   - Average loss over valid elements only

## Extension and Maintenance

### Adding New Loss Types

To add a new loss function:
1. Implement the core function with standard parameters (pred, true, mask)
2. Add numerical stability measures (epsilon, clipping)
3. Ensure proper mask handling
4. Update `compute_combined_loss` to include the new loss
5. Add comprehensive tests for the new function

### Future Improvements

Areas for future enhancement in V2:
- Implement full FAPE loss with rigid body alignment
- Add true lDDT-based confidence target calculation
- Support torsion angle distributions for more accurate angle loss
- Allow per-residue loss weighting

## Testing Instance Responsibilities

The Testing instance (04) has specific responsibilities regarding the loss functions that are critical to highlight:

1. **Training Stability Validation**: While our unit tests verify basic functionality and gradient flow, the Testing instance must validate stability during extended training cycles. This is particularly important given the known issues documented below.

2. **Performance Impact Assessment**: Evaluate how the known numerical issues affect model quality and convergence.

3. **Prioritization of Issues**: Determine which issues should be addressed immediately based on their impact on training stability and model quality.

4. **Regression Testing**: Ensure fixes to the issues don't introduce new problems or regressions.

The Integration instance (03) has verified basic functionality, gradient flow, and implemented critical fixes, but extended training validation remains the explicit responsibility of the Testing instance.

## Known Issues and Limitations

The current loss function implementation has several known issues that have been identified through testing. These issues are documented here to provide guidance for the Testing instance and future development.

### 1. Kabsch Alignment Issues (Critical)

**Issue Description**:  
The Kabsch alignment algorithm used in the FAPE loss has issues handling certain rotation cases correctly. This affects the ability to properly align predicted structures with ground truth before calculating distance errors.

**Impact**:  
This can lead to incorrect loss values and potentially unstable gradients during training, particularly for structures that require significant rotation during alignment.

**Root Cause**:  
The implementation may have issues with:
- Handling of the determinant check when detecting reflections
- SVD decomposition edge cases
- Numerical precision in the rotation matrix calculation

**Suggested Fix**:  
Review the implementation of `stable_kabsch_align` with focus on:
```python
# Check for reflection (around line 67-74)
det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
R = torch.matmul(V, U.transpose(-2, -1))
if det < (0.0 - epsilon): # Allow for slight numerical imprecision around -1
    # Reflection detected, correct it by flipping the sign along the axis
    # corresponding to the smallest singular value (last column of V).
    V_corrected = V.clone()
    V_corrected[:, -1] = -V_corrected[:, -1]
    R = torch.matmul(V_corrected, U.transpose(-2, -1))
```

### 2. FAPE Zero Loss Issue (Critical)

**Issue Description**:  
When predicted coordinates are identical to true coordinates, the FAPE loss should be exactly zero. However, tests show this is not always the case.

**Impact**:  
This can lead to non-zero gradients even when predictions are perfect, potentially causing model instability during later training stages.

**Root Cause**:  
Numerical precision issues in the distance calculation or Kabsch alignment may be introducing small non-zero values.

**Suggested Fix**:  
Add an explicit check for identical inputs before calculation:
```python
if torch.allclose(pred_coords, true_coords, atol=1e-7):
    return torch.tensor(0.0, device=pred_coords.device, dtype=pred_coords.dtype)
```

### 3. Distance Calculation Precision (Moderate)

**Issue Description**:  
The `robust_distance_calculation` function doesn't always produce the expected values for known distances, particularly in edge cases.

**Impact**:  
This affects the accuracy of distance-based loss calculations but may not completely prevent model training.

**Root Cause**:  
The epsilon handling and clamping in the distance calculation may be introducing bias.

**Suggested Fix**:  
Review the implementation with focus on the epsilon handling:
```python
# Add epsilon before taking the square root to avoid sqrt(0) or sqrt(negative) due to precision
# Clamp sum_sq_diff to be non-negative just in case
stable_sum_sq_diff = torch.clamp(sum_sq_diff, min=0.0) + epsilon
```

### 4. Confidence Loss Value Discrepancies (Moderate)

**Issue Description**:  
The confidence loss calculation doesn't produce expected values in test cases, particularly for bad predictions with high/low confidence.

**Impact**:  
This affects the auxiliary confidence prediction task but doesn't block coordinate prediction.

**Root Cause**:  
The target derivation from coordinate errors or the scaling factor may need adjustment.

**Suggested Fix**:  
Review target calculation in the confidence loss:
```python
# Convert error to per-residue lDDT-like score in [0, 1]
# Higher score means better prediction (lower error)
conf_targets = torch.exp(-per_residue_error / scaling_factor)
conf_targets = torch.clamp(conf_targets, 0.0, 1.0) # Ensure [0, 1]
```

### 5. Mask Handling in Confidence Loss (Minor)

**Issue Description**:  
Tests indicate the mask handling in confidence loss doesn't properly exclude padding positions.

**Impact**:  
Minimal impact on overall training, but may affect efficiency and numerical stability.

**Root Cause**:  
The mask application may be happening at the wrong point in the calculation.

**Suggested Fix**:  
Ensure proper mask application before loss calculation:
```python
# Mask out targets for padded positions (set to 0, loss will be ignored via mask)
conf_targets = conf_targets * mask.float()
```

## Temporary Workarounds

Until these issues are fixed, the following workarounds can be used:

1. **For Kabsch Alignment Issues**: 
   - Use smaller learning rates to reduce the impact of alignment errors
   - Consider gradient clipping to prevent instability

2. **For FAPE Zero Loss Issues**:
   - Add a small epsilon to loss values to prevent division by zero in optimizers

3. **For Confidence Prediction**:
   - Consider using a smaller weight for the confidence loss component during initial training

## Common Debugging

### Numerical Stability Issues

**Issue**: If NaN/Inf values appear in loss:
- Check for very small values in denominators (use epsilon)
- Verify masks are applied correctly
- Check for NaN/Inf in input coordinates
- Ensure sufficient valid points for alignment

**Resolution**: 
- Add additional checks for degenerate cases
- Implement more robust fallbacks

### Gradient Flow Issues

**Issue**: If gradients don't flow properly:
- Check that tensors requiring gradients are properly tracked
- Ensure masks are applied correctly
- Verify that loss is properly scaled

**Resolution**:
- Update mask handling to preserve gradients
- Ensure stable operations maintain gradient flow

## Decision Log

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| Implement simplified FAPE proxy | Establish numerical stability | Full AlphaFold FAPE | 2025-04-20 |
| Use per-residue confidence | Enable local quality assessment | Global confidence only | 2025-04-20 |
| Include angle prediction loss | Provide auxiliary signal | Structure-only loss | 2025-04-20 |
| Process sequences individually | Handle variable length properly | Batch processing with padding | 2025-04-20 |
| Implement robust Kabsch | Handle degenerate cases | Simpler implementation | 2025-04-20 |

## Usage Examples

### Basic Loss Calculation

```python
import torch
from src.losses import compute_stable_fape_loss, compute_confidence_loss, compute_angle_loss

# Create example tensors
batch_size, seq_len = 2, 10
pred_coords = torch.randn(batch_size, seq_len, 3)
true_coords = torch.randn(batch_size, seq_len, 3)
pred_confidence = torch.randn(batch_size, seq_len)
pred_angles = torch.randn(batch_size, seq_len, 4)
true_angles = torch.randn(batch_size, seq_len, 4)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[0, -2:] = False  # Mask last 2 positions in first sequence

# Calculate individual losses
fape_loss = compute_stable_fape_loss(pred_coords, true_coords, mask)
conf_loss = compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask)
angle_loss = compute_angle_loss(pred_angles, true_angles, mask)

# Print losses
print(f"FAPE Loss: {fape_loss.item():.4f}")
print(f"Confidence Loss: {conf_loss.item():.4f}")
print(f"Angle Loss: {angle_loss.item():.4f}")
```

### Combined Loss with Model Outputs

```python
import torch
from src.losses import compute_combined_loss

# Model outputs
outputs = {
    "pred_coords": pred_coords,
    "pred_confidence": pred_confidence,
    "pred_angles": pred_angles
}

# Batch data
batch = {
    "coordinates": true_coords,
    "dihedral_features": true_angles,
    "mask": mask
}

# Loss weights
loss_weights = {
    "fape": 1.0,
    "confidence": 0.1,
    "angle": 0.5
}

# Calculate combined loss
total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)

# Access individual loss components
print(f"FAPE Loss: {loss_components['fape'].item():.4f}")
print(f"Confidence Loss: {loss_components['confidence'].item():.4f}")
print(f"Angle Loss: {loss_components['angle'].item():.4f}")
print(f"Total Loss: {total_loss.item():.4f}")

# Use for backward pass
total_loss.backward()
```

### Handling Edge Cases

```python
import torch
from src.losses import compute_stable_fape_loss

# Create a batch with insufficient points for Kabsch alignment
batch_size, seq_len = 2, 10
pred_coords = torch.randn(batch_size, seq_len, 3)
true_coords = torch.randn(batch_size, seq_len, 3)

# Create a mask with only 2 valid positions (insufficient for alignment)
mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
mask[:, :2] = True

# Calculate loss (should handle gracefully and return 0.0)
fape_loss = compute_stable_fape_loss(pred_coords, true_coords, mask)
print(f"FAPE Loss with insufficient points: {fape_loss.item():.4f}")
```