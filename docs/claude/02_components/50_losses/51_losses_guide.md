# Losses Implementation Guide

## Component Overview

The losses component is essential for training the RNA 3D folding model, providing the optimization objectives that guide the learning process. This module implements three key loss functions: a coordinate loss (simplified FAPE proxy for V1), a confidence prediction loss, and an auxiliary angle prediction loss. Together, these losses enable the model to learn accurate 3D structure prediction, estimate prediction confidence, and leverage auxiliary structural information.

In the RNA 3D folding pipeline, the losses are computed based on the model's outputs and ground truth data, and their weighted combination forms the overall training objective. The design of appropriate loss functions is critical for effective learning, as they define what constitutes a "good" prediction and guide the gradient-based optimization.

## Requirements Reference

From the Product Requirements Document (`4_Product_Requirements_V1.md`), the losses component must satisfy the following requirements:

- **LF-01**: Implement `compute_fape_loss` function (**simplified proxy** for V1: clamped L2 distance between predicted and true coordinates).
- **LF-02**: Implement `compute_confidence_loss` function (**proxy**: MSE or BCEWithLogitsLoss vs. derived lDDT proxy target).
- **LF-03**: Implement `compute_angle_loss` function (auxiliary loss comparing predicted vs. true sin/cos angle features).
- **LF-04**: All loss functions must correctly ignore contributions from padded sequence positions using the input mask.

## Technical Background

### Coordinate Loss: FAPE and Simplified Proxy

The Frame-Aligned Point Error (FAPE) is a coordinate-based loss that measures structural accuracy while being invariant to global rotations and translations. The full FAPE implementation:

1. Aligns local coordinate frames for each residue
2. Measures distances between corresponding points in these aligned frames
3. Is invariant to global rotations and translations of the entire structure

For V1, we implement a simplified proxy that:
1. Performs global alignment of predicted and true structures (using Kabsch algorithm)
2. Computes L2 distance between corresponding atoms after alignment
3. Clamps large distance errors to improve robustness
4. Averages over valid (non-padded) positions

This simplified approach provides a reasonable approximation while being easier to implement for the initial version.

### Confidence Loss

The confidence loss guides the model to predict its own structural accuracy. The model predicts a per-residue confidence score (similar to AlphaFold's pLDDT), which is trained to match a derived target based on the true prediction accuracy.

For V1, we:
1. Compute a proxy for local structural accuracy (based on coordinate error)
2. Train the model's confidence head to predict this accuracy measure
3. Use MSE or BCE loss between predicted confidence and derived target

This enables the model to provide uncertainty estimates alongside its predictions.

### Angle Prediction Loss

The angle prediction loss is an auxiliary task that helps the model learn from the pseudo-dihedral angles present in the ground truth data. These angles (η and θ) capture the backbone geometry and provide additional structural information.

The key aspects are:
1. The model predicts sin/cos representations of angles to avoid discontinuity at 0/360°
2. The loss compares predicted sin/cos values with true sin/cos values
3. NaN values in true angles (e.g., at sequence boundaries) are properly handled

This auxiliary task helps guide the model's learning even when coordinate prediction is challenging.

## Interfaces

### Input Interface

The loss functions take the following inputs:

```python
# For compute_fape_loss (coordinate loss)
pred_coords: torch.Tensor  # Shape: (batch_size, seq_len, 3), predicted C1' coordinates
true_coords: torch.Tensor  # Shape: (batch_size, seq_len, 3), ground truth C1' coordinates
mask: torch.Tensor         # Shape: (batch_size, seq_len), boolean mask (True for valid positions)
clamp_value: float = 10.0  # Maximum distance error to consider

# For compute_confidence_loss
pred_confidence: torch.Tensor  # Shape: (batch_size, seq_len), predicted confidence scores
pred_coords: torch.Tensor      # Shape: (batch_size, seq_len, 3), predicted coordinates
true_coords: torch.Tensor      # Shape: (batch_size, seq_len, 3), ground truth coordinates
mask: torch.Tensor             # Shape: (batch_size, seq_len), boolean mask

# For compute_angle_loss
pred_angles: torch.Tensor  # Shape: (batch_size, seq_len, 4), predicted sin/cos of angles
true_angles: torch.Tensor  # Shape: (batch_size, seq_len, 4), true sin/cos of angles 
mask: torch.Tensor         # Shape: (batch_size, seq_len), boolean mask
```

### Output Interface

Each loss function returns a scalar loss value:

```python
# All loss functions
loss: torch.Tensor  # Scalar loss value, shape: ()
```

## Implementation Steps

### 1. Implement Coordinate Loss (FAPE Proxy)

```python
def compute_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0
) -> torch.Tensor:
    """
    Compute a simplified FAPE loss proxy based on clamped L2 distance
    after global Kabsch alignment.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        clamp_value: Maximum distance error to consider
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_coords.shape
    
    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_coords.device)
    
    # Initialize loss
    total_loss = 0.0
    
    # Process each sequence in the batch separately for Kabsch alignment
    for b in range(batch_size):
        # Extract valid coordinates for this sequence
        valid_mask = mask[b]
        if not valid_mask.any():
            continue  # Skip if no valid positions
            
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Perform Kabsch alignment
        p_aligned = kabsch_align(p_valid, t_valid)
        
        # Calculate clamped L2 distance
        dist = torch.norm(p_aligned - t_valid, dim=1)
        clamped_dist = torch.clamp(dist, max=clamp_value)
        
        # Average over valid positions
        seq_loss = clamped_dist.mean()
        total_loss += seq_loss
    
    # Average over batch
    loss = total_loss / batch_size
    return loss


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Align points P to points Q using Kabsch algorithm.
    
    Args:
        P: Moving points, shape (N, 3)
        Q: Fixed points, shape (N, 3)
        
    Returns:
        P_aligned: Aligned points, shape (N, 3)
    """
    # Center the points
    p_mean = P.mean(dim=0, keepdim=True)
    q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - p_mean
    Q_centered = Q - q_mean
    
    # Compute covariance matrix
    C = torch.matmul(P_centered.transpose(-2, -1), Q_centered)
    
    # Compute optimal rotation using SVD
    U, _, Vt = torch.linalg.svd(C)
    V = Vt.transpose(-2, -1)
    
    # Ensure proper rotation (no reflection)
    det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
    if det < 0:
        V[:, 2] = -V[:, 2]
    
    # Compute rotation matrix
    R = torch.matmul(V, U.transpose(-2, -1))
    
    # Apply rotation and translation
    P_aligned = torch.matmul(P_centered, R) + q_mean
    
    return P_aligned
```

### 2. Implement Confidence Loss

```python
def compute_confidence_loss(
    pred_confidence: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute confidence prediction loss.
    
    Train the model to predict a per-residue confidence score (similar to pLDDT)
    that correlates with the actual per-residue accuracy.
    
    Args:
        pred_confidence: Predicted confidence scores, shape (batch_size, seq_len)
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len = pred_confidence.shape
    
    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_confidence.device)
    
    # Calculate residue-wise error (as proxy for lDDT)
    with torch.no_grad():
        # Compute per-residue coordinate error
        # For simplicity, we're using raw distance without Kabsch alignment here
        # A more accurate target would use a proper lDDT implementation
        coord_error = torch.norm(pred_coords - true_coords, dim=2)
        
        # Convert to per-residue lDDT-like score in [0, 1]
        # Higher is better (1.0 = perfect prediction)
        lddt_proxy = torch.exp(-coord_error / 3.0)  # Simple exponential proxy
        
        # Ensure values are in [0, 1]
        lddt_proxy = torch.clamp(lddt_proxy, 0.0, 1.0)
        
        # Create confidence targets
        # 1 = high confidence (low error), 0 = low confidence (high error)
        conf_targets = lddt_proxy
    
    # Apply sigmoid to predicted logits (if output is logits)
    # If your model already outputs in [0,1] range, comment this out
    pred_probs = torch.sigmoid(pred_confidence)
    
    # Calculate MSE loss
    squared_error = (pred_probs - conf_targets) ** 2
    
    # Apply mask and average
    masked_se = squared_error * mask.float()
    loss = masked_se.sum() / (mask.sum() + 1e-8)
    
    return loss
```

### 3. Implement Angle Prediction Loss

```python
def compute_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute loss for dihedral angle predictions.
    
    Compare the predicted sin/cos representations of angles with the true values.
    
    Args:
        pred_angles: Predicted sin/cos of angles [sin(η), cos(η), sin(θ), cos(θ)], 
                    shape (batch_size, seq_len, 4)
        true_angles: True sin/cos of angles, shape (batch_size, seq_len, 4)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_angles.shape
    
    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_angles.device)
    
    # Handle NaNs in true angles (typically at boundaries)
    angle_mask = mask.clone()
    if torch.isnan(true_angles).any():
        # Create mask for non-NaN angles
        nan_mask = ~torch.isnan(true_angles).any(dim=2)  # (batch_size, seq_len)
        angle_mask = angle_mask & nan_mask
    
    # Expand mask to match angle dimensions
    expanded_mask = angle_mask.unsqueeze(-1).expand_as(pred_angles)  # (batch_size, seq_len, 4)
    
    # Replace NaNs with zeros in true angles
    true_angles_clean = torch.nan_to_num(true_angles, nan=0.0)
    
    # Calculate mean squared error
    squared_error = (pred_angles - true_angles_clean) ** 2
    
    # Apply mask and calculate mean
    masked_se = squared_error * expanded_mask.float()
    total_elements = expanded_mask.sum() + 1e-8
    mse = masked_se.sum() / total_elements
    
    return mse
```

### 4. Implement Combined Loss

```python
def compute_combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined loss from multiple loss components.
    
    Args:
        outputs: Dictionary of model outputs:
            - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
            - pred_confidence: Predicted confidence (batch_size, seq_len)
            - pred_angles: Predicted angles (batch_size, seq_len, 4)
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
        - total_loss: Combined weighted loss (scalar)
        - loss_components: Dictionary of individual loss components
    """
    # Extract model outputs
    pred_coords = outputs['pred_coords']
    pred_confidence = outputs['pred_confidence']
    pred_angles = outputs['pred_angles']
    
    # Extract ground truth
    true_coords = batch['coordinates']
    true_angles = batch['dihedral_features']
    mask = batch['mask']
    
    # Compute individual losses
    fape_loss = compute_fape_loss(pred_coords, true_coords, mask)
    confidence_loss = compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask)
    angle_loss = compute_angle_loss(pred_angles, true_angles, mask)
    
    # Combine losses using weights
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * confidence_loss +
        loss_weights['angle'] * angle_loss
    )
    
    # Return total loss and individual components
    loss_components = {
        'fape': fape_loss,
        'confidence': confidence_loss,
        'angle': angle_loss,
        'total': total_loss
    }
    
    return total_loss, loss_components
```

## Critical Aspects

### 1. Proper Masking Implementation

Correct masking is critical for handling variable-length sequences. All loss functions must properly ignore contributions from padding positions:

- **Why it matters**: Without proper masking, padding positions would contribute to the loss, potentially dominating and destabilizing training.
- **Implementation pattern**: Apply the mask when averaging losses, using `masked_tensor.sum() / (mask.sum() + epsilon)` to handle case where mask is all zeros.
- **Mask expansion**: For feature dimensions, expand the mask to match the shape of the tensor being masked (e.g., `mask.unsqueeze(-1).expand(...)`).
- **Mask convention**: In our implementation, `True` indicates valid positions, `False` indicates padding.

### 2. Numerical Stability

Several numerical stability considerations are important:

- Add a small epsilon (e.g., `1e-8`) when dividing by potentially zero values.
- Use `torch.clamp()` to restrict values to valid ranges when generating targets or computing losses.
- Handle NaN values in dihedral angle data using `torch.nan_to_num()` and proper masking.
- Use `torch.norm(..., dim=X)` instead of manual square root of sum of squares for better numerical properties.

### 3. FAPE Implementation Considerations

The simplified V1 FAPE proxy has important limitations to be aware of:

- **Kabsch alignment** is performed per-sequence in the batch. For very large sequences, this could be a computational bottleneck.
- **Global alignment** means this loss isn't truly local like the full FAPE will be in future versions.
- **Clamping** of large distances is essential to avoid domination by outliers, but the specific `clamp_value` needs tuning.
- **Correct gradient flow**: Ensure the alignment doesn't break gradient computation. Our implementation preserves gradients.

### 4. Confidence Loss Target Generation

The confidence prediction target is a key design choice:

- We use a simple exponential decay function to convert distance error to a confidence score in [0, 1].
- In a more advanced implementation, proper lDDT calculation would be used.
- Ensure that `torch.no_grad()` is used when computing targets to avoid unnecessary gradient computation.
- The loss can be MSE or BCE depending on the confidence head output interpretation.

### 5. Loss Weight Balancing

Balancing multiple loss components is critical for effective training:

- Start with relatively high weight on the coordinate loss (e.g., 1.0).
- Use lower weights for auxiliary losses (e.g., 0.1-0.5).
- Monitor individual loss components during training to ensure one isn't dominating.
- Consider implementing a dynamic weighting schedule if necessary.

Default weights to start with:
```python
loss_weights = {
    'fape': 1.0,       # Primary structure prediction loss
    'confidence': 0.1, # Secondary importance
    'angle': 0.5       # Auxiliary supervision
}
```

## Testing Requirements

### Unit Tests

1. **FAPE Loss Tests**:
   - Test scalar output (shape `()`) 
   - Verify non-negative values
   - Confirm that masked positions don't contribute
   - Test with all positions masked
   - Verify alignment correctness
   - Test with clamping and without clamping

2. **Confidence Loss Tests**:
   - Test scalar output
   - Verify that perfect predictions get low loss, imperfect predictions get higher loss
   - Test with different mask patterns
   - Verify target generation produces values in [0, 1]

3. **Angle Loss Tests**:
   - Test handling of NaN values in true angles
   - Verify masking works correctly
   - Test with various angle patterns (e.g., all zeros, random values)

4. **Combined Loss Tests**:
   - Verify weight application
   - Test with various combinations of loss weights
   - Check gradient flow through all loss components

### Integration Tests

1. Test integration with model outputs to ensure shape compatibility
2. Verify loss computation in an end-to-end training step
3. Test with realistic data from the data loader

## Example Usage

Here's an example of integrating the loss functions with the training loop:

```python
def train_step(model, batch, optimizer, loss_weights):
    """
    Perform a single training step.
    
    Args:
        model: RNA folding model
        batch: Dictionary of input data
        optimizer: PyTorch optimizer
        loss_weights: Dictionary of loss weights
        
    Returns:
        Dictionary of loss values
    """
    # Set model to training mode
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(batch)
    
    # Compute loss
    total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)
    
    # Backward pass
    total_loss.backward()
    
    # Clip gradients (optional)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    optimizer.step()
    
    # Return loss values
    return {k: v.item() for k, v in loss_components.items()}
```

And within a training epoch:

```python
def train_epoch(model, dataloader, optimizer, loss_weights):
    """
    Train for one epoch.
    
    Args:
        model: RNA folding model
        dataloader: PyTorch DataLoader
        optimizer: PyTorch optimizer
        loss_weights: Dictionary of loss weights
        
    Returns:
        Average loss values
    """
    # Initialize loss accumulators
    epoch_losses = {
        'fape': 0.0,
        'confidence': 0.0,
        'angle': 0.0,
        'total': 0.0
    }
    
    # Iterate over batches
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to appropriate device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # Perform training step
        step_losses = train_step(model, batch, optimizer, loss_weights)
        
        # Accumulate losses
        for k in epoch_losses:
            epoch_losses[k] += step_losses[k]
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: "
                  f"FAPE={step_losses['fape']:.4f}, "
                  f"Conf={step_losses['confidence']:.4f}, "
                  f"Angle={step_losses['angle']:.4f}, "
                  f"Total={step_losses['total']:.4f}")
    
    # Average losses over batches
    for k in epoch_losses:
        epoch_losses[k] /= len(dataloader)
    
    return epoch_losses
```

## Related Documentation

- **Architecture Specification**: `docs/3_Architecture_Specification.md` - See "Loss Functions" section
- **Product Requirements**: `docs/4_Product_Requirements_V1.md` - Requirements LF-01 to LF-04
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md` - For loss function best practices

## Next Steps

1. Implement the loss functions in `src/losses.py`
2. Write unit tests in `tests/test_losses.py` to verify functionality
3. Integrate with the main training loop
4. Plan for future improvements, such as:
   - Full FAPE implementation with local reference frames
   - More accurate lDDT calculation for confidence targets
   - Exploring alternative loss functions or weighting schemes

Note that for future versions beyond V1, a more sophisticated FAPE implementation and better confidence targets will be developed, building on this foundation.
