Okay, let's generate the focused `52_losses_examples.md` file. This version will concentrate on demonstrating how to *use* the V1 loss functions defined in `src/losses.py`, including common patterns and potential issues, aligning with the structure of other example files in the project.

# START OF FILE: docs/claude/components/50_losses/52_losses_examples.md

# Losses Component Examples (V1)

This document provides practical examples for using the V1 loss functions implemented in `src/losses.py` for the RNA 3D folding project. It covers basic usage, integration into training workflows, common patterns, and troubleshooting tips related to the simplified FAPE proxy, confidence proxy loss, and auxiliary angle loss.

## Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict

# Assume loss functions are imported from src/losses.py
from src.losses import (
    compute_stable_fape_loss,
    compute_confidence_loss,
    compute_angle_loss,
    compute_combined_loss
)

# Assume a placeholder model exists for context
class MockRNAFoldingModel(nn.Module):
    def __init__(self, residue_dim=64):
        super().__init__()
        self.residue_dim = residue_dim
        # Dummy heads
        self.coord_head = nn.Linear(residue_dim, 3)
        self.conf_head = nn.Linear(residue_dim, 1)
        self.angle_head = nn.Linear(residue_dim, 4)

    def forward(self, batch):
        # Simplified forward pass for demonstration
        residue_repr = torch.rand(batch['mask'].shape[0], batch['mask'].shape[1], self.residue_dim, device=batch['mask'].device)
        pred_coords = self.coord_head(residue_repr)
        pred_confidence = self.conf_head(residue_repr).squeeze(-1)
        pred_angles = self.angle_head(residue_repr)
        
        # Apply mask to outputs (important!)
        mask = batch['mask']
        pred_coords = pred_coords * mask.unsqueeze(-1).float()
        pred_confidence = pred_confidence * mask.float()
        pred_angles = pred_angles * mask.unsqueeze(-1).float()
        
        return {
            'pred_coords': pred_coords,
            'pred_confidence': pred_confidence,
            'pred_angles': pred_angles
        }
```

## Basic Usage Examples

These examples show how to call individual loss functions with sample data.

### 1. Coordinate Loss (FAPE Proxy - V1)

The V1 coordinate loss uses `compute_stable_fape_loss`, which calculates a clamped L2 distance after global Kabsch alignment.

```python
# --- Example Data ---
batch_size, seq_len = 2, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Predicted coordinates
pred_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10

# Ground truth coordinates
true_coords = pred_coords + torch.randn_like(pred_coords) * 1.5 # Add some noise

# Mask (mask out last 2 positions in first sequence)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
mask[0, -2:] = False

# --- Compute Loss ---
# Using default clamp value (10.0 Å)
fape_loss_default = compute_stable_fape_loss(pred_coords, true_coords, mask)
print(f"FAPE Loss (clamp=10.0): {fape_loss_default.item():.4f}")

# Using a different clamp value
fape_loss_clamp5 = compute_stable_fape_loss(pred_coords, true_coords, mask, clamp_value=5.0)
print(f"FAPE Loss (clamp=5.0): {fape_loss_clamp5.item():.4f}")

# Using no clamping
fape_loss_noclamp = compute_stable_fape_loss(pred_coords, true_coords, mask, clamp_value=float('inf'))
print(f"FAPE Loss (no clamp): {fape_loss_noclamp.item():.4f}")

# Note: clamp_value prevents large errors from dominating the loss gradient.
```

### 2. Confidence Loss (V1 Proxy Target)

The V1 confidence loss uses `compute_confidence_loss`, comparing predicted confidence logits against a target derived from coordinate error.

```python
# --- Example Data ---
# Use coordinates from previous example
# Predicted confidence (logits)
pred_confidence = torch.randn(batch_size, seq_len, device=device) # Example logits

# --- Compute Loss ---
# Using MSE loss with default lDDT proxy target
conf_loss_mse = compute_confidence_loss(
    pred_confidence, pred_coords, true_coords, mask, loss_type='mse'
)
print(f"Confidence Loss (MSE, lDDT_proxy): {conf_loss_mse.item():.4f}")

# Using BCE loss with default lDDT proxy target
conf_loss_bce = compute_confidence_loss(
    pred_confidence, pred_coords, true_coords, mask, loss_type='bce'
)
print(f"Confidence Loss (BCE, lDDT_proxy): {conf_loss_bce.item():.4f}")

# Using distance-based target
conf_loss_dist = compute_confidence_loss(
    pred_confidence, pred_coords, true_coords, mask, target_type='distance_based'
)
print(f"Confidence Loss (MSE, distance_based): {conf_loss_dist.item():.4f}")

# Note: The choice between 'mse' and 'bce' depends on whether you prefer
# regression or classification interpretation for the confidence score.
# BCEWithLogitsLoss is often preferred for probabilities.
```

### 3. Angle Loss (V1)

The auxiliary angle loss uses `compute_angle_loss` to compare predicted sin/cos representations with ground truth.

```python
# --- Example Data ---
# Predicted angles [sin(eta), cos(eta), sin(theta), cos(theta)]
pred_angles = torch.randn(batch_size, seq_len, 4, device=device)
# Normalize predicted vectors to be on unit circle (optional but good practice)
pred_angles[:, :, 0:2] = F.normalize(pred_angles[:, :, 0:2], p=2, dim=2)
pred_angles[:, :, 2:4] = F.normalize(pred_angles[:, :, 2:4], p=2, dim=2)

# Ground truth angles (sin/cos form)
# Note: Real true_angles often have NaNs at boundaries
angles_rad = torch.rand(batch_size, seq_len, 2, device=device) * 2 * math.pi
true_angles = torch.cat([
    torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
    torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])
], dim=2)
# Introduce some NaNs
true_angles[0, 0, :] = float('nan')
true_angles[1, -1, :] = float('nan')

# --- Compute Loss ---
# Using MSE loss
angle_loss_mse = compute_angle_loss(pred_angles, true_angles, mask, loss_type='mse')
print(f"Angle Loss (MSE): {angle_loss_mse.item():.4f}")

# Using Cosine similarity loss
angle_loss_cos = compute_angle_loss(pred_angles, true_angles, mask, loss_type='cosine')
print(f"Angle Loss (Cosine): {angle_loss_cos.item():.4f}")

# Using MAE loss
angle_loss_mae = compute_angle_loss(pred_angles, true_angles, mask, loss_type='mae')
print(f"Angle Loss (MAE): {angle_loss_mae.item():.4f}")

# Note: 'cosine' loss directly measures angular difference, while 'mse'/'mae'
# compare the sin/cos components. The function handles NaNs in true_angles.
```

### 4. Combined Loss

The `compute_combined_loss` function takes model outputs, batch data, and loss weights to calculate the final training objective.

```python
# --- Example Data ---
# Assume 'outputs' dict from model and 'batch' dict from dataloader exist
# outputs = {'pred_coords': pred_coords, 'pred_confidence': pred_confidence, 'pred_angles': pred_angles}
# batch = {'coordinates': true_coords, 'dihedral_features': true_angles, 'mask': mask}

# Define loss weights
loss_weights = {
    'fape': 1.0,       # Primary structure loss
    'confidence': 0.1, # Auxiliary confidence prediction
    'angle': 0.5       # Auxiliary angle prediction
}

# --- Compute Combined Loss ---
total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)

print(f"Combined Loss: {total_loss.item():.4f}")
print("Individual Components (weighted contributions):")
for name, component_loss in loss_components.items():
    weight = loss_weights.get(name, 1.0) # Default weight 1 if not specified
    weighted_contribution = weight * component_loss.item()
    print(f"  {name}: {component_loss.item():.4f} (weighted: {weighted_contribution:.4f})")

# Note: total_loss is the tensor used for backpropagation.
# loss_components contains the individual (unweighted) tensor components.
```

## Integration Examples

### Using Losses in a Training Step

Here's how to integrate the combined loss into a typical training step:

```python
def train_step(model, batch, optimizer, loss_weights, device, grad_clip=1.0):
    """Performs a single training step."""
    model.train() # Set model to training mode

    # Move batch to device
    batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(batch_on_device)

    # Compute combined loss
    # compute_combined_loss returns the total loss tensor and a dict of component tensors
    total_loss, loss_component_tensors = compute_combined_loss(
        outputs, batch_on_device, loss_weights
    )

    # Check for NaN/Inf loss before backward
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
         print(f"WARNING: NaN/Inf loss detected ({total_loss.item()}). Skipping step.")
         return None # Skip this step

    # Backward pass
    total_loss.backward()

    # Optional: Gradient Clipping
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    # Optimizer step
    optimizer.step()

    # Return detached loss values (floats) for logging
    loss_values_float = {k: v.item() for k, v in loss_component_tensors.items()}
    loss_values_float['total'] = total_loss.item()

    return loss_values_float

# --- Example usage within a training loop ---
# model = MockRNAFoldingModel(config['model']).to(device)
# optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
# loss_weights = config['training']['loss_weights']
# dataloader = create_data_loader(...) # Create your dataloader

# for batch in dataloader:
#     loss_dict = train_step(model, batch, optimizer, loss_weights, device)
#     if loss_dict:
#         # Log loss_dict['total'], loss_dict['fape'], etc.
#         print(f"Batch Loss: {loss_dict['total']:.4f}")
```

## Common Patterns and Best Practices

### 1. Masking is Crucial
- Always pass the correct boolean mask (`True` for valid positions) to all loss functions.
- The loss functions internally handle applying the mask to ignore contributions from padded regions. Double-check that the mask shape (`B, L`) is correct.

### 2. Loss Weight Tuning
- The `loss_weights` dictionary is critical for balancing the contributions of different objectives.
- **Starting Point**: Give the primary structure loss (`fape`) the highest weight (e.g., 1.0). Auxiliary losses (`confidence`, `angle`) should have lower weights (e.g., 0.1-0.5).
- **Monitoring**: Track the individual loss components during training (returned by `compute_combined_loss`). If one loss dominates or becomes stagnant, adjust its weight.
- **Curriculum Learning (Advanced)**: Consider starting with simpler losses weighted higher (e.g., angle) and gradually increasing the weight of the coordinate loss (`fape`) as training progresses.

### 3. Numerical Stability
- The provided V1 loss functions (`compute_stable_fape_loss`, etc.) incorporate stability measures (epsilon, stable Kabsch).
- Be mindful of potential NaN/Inf values in model outputs, especially coordinates, before passing them to loss functions. Add checks in your training loop.
- Using `torch.clamp` in `compute_stable_fape_loss` helps prevent exploding gradients from very large coordinate errors.

### 4. V1 Proxy Limitations
- Remember that `compute_stable_fape_loss` is a *proxy* for the full FAPE loss. It uses global alignment, while true FAPE uses local frames.
- `compute_confidence_loss` uses a *proxy* target derived from coordinate error, not a true lDDT calculation (which is planned for V2+). Its accuracy depends on how well the error-to-confidence mapping (`lddt_proxy` or `distance_based`) reflects actual structural quality.

### 5. Choosing Loss Types
- **Angle Loss**: `cosine` loss directly measures the angular difference and might be more interpretable than `mse` or `mae` on sin/cos components. Experiment to see which performs best.
- **Confidence Loss**: `bce` (Binary Cross-Entropy with Logits) often provides better calibrated probability predictions compared to `mse`.

## Common Errors and Solutions

### 1. NaN/Infinity Loss
- **Cause**: Numerical instability (division by zero, large coordinate errors, SVD failure in Kabsch), NaN values in inputs (e.g., model outputs, true angles).
- **Solution**:
    - Use the `stable` loss versions provided (`compute_stable_fape_loss`).
    - Add epsilon values to denominators (e.g., in averaging).
    - Check model outputs for NaNs/Infs *before* loss calculation.
    - Ensure `true_angles` NaN handling is correct in `compute_angle_loss`.
    - Use gradient clipping in the training loop.
    - Reduce learning rate if gradients explode.
    - Check model parameter initialization (`_init_weights`).

### 2. Zero Loss
- **Cause**: Predictions might perfectly match targets (unlikely early in training), or the batch might contain only masked positions.
- **Solution**:
    - Verify if the batch was entirely masked (check `mask.sum()`).
    - Inspect predictions and targets to see if they are unexpectedly identical.
    - Ensure loss functions are correctly implemented (e.g., averaging logic).

### 3. Gradients Not Flowing
- **Cause**: Loss tensor detached from computation graph; `torch.no_grad()` used incorrectly; operations within loss function break gradients (e.g., converting to NumPy prematurely).
- **Solution**:
    - Ensure the final `total_loss` returned by `compute_combined_loss` requires grad (`total_loss.requires_grad` should be True).
    - Verify that target calculations (like lDDT proxy) are correctly wrapped in `with torch.no_grad()`.
    - Check that helper functions like `stable_kabsch_align` maintain gradient flow for the *predicted* coordinates.
    - Use `loss.backward()` and check `param.grad` for key model parameters.

### 4. Loss Domination
- **Cause**: One loss component (e.g., FAPE) has a much larger magnitude than others, overwhelming the total loss and gradients.
- **Solution**:
    - Adjust `loss_weights`. Decrease the weight of the dominating loss or increase others.
    - Monitor individual loss components during training using a `LossTracker` or similar logging.
    - Consider loss normalization techniques (though simple weighting is standard).

### 5. Masking Errors
- **Cause**: Incorrect mask shape, mask applied incorrectly, mask values inverted.
- **Solution**:
    - Verify mask shape is `(B, L)`.
    - Check mask application logic within loss functions (e.g., `loss * mask.float()`).
    - Ensure mask is expanded correctly for higher-dimensional tensors (`mask.unsqueeze(-1)`).
    - Confirm `True` represents valid positions and `False` represents padding.

## Conclusion

These examples provide a practical guide for using the V1 loss functions in the RNA 3D folding project. Remember to use the provided functions (`compute_stable_fape_loss`, `compute_confidence_loss`, `compute_angle_loss`, `compute_combined_loss`) from `src/losses.py` in your training scripts. Pay close attention to proper masking, loss weighting, and numerical stability to ensure effective model training.
