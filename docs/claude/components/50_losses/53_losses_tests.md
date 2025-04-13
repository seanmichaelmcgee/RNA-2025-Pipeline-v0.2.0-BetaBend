Okay, generating the testing guide for the `src/losses.py` module, following the structure and principles established in the previous guides. This will provide a clear roadmap for testing the V1 loss functions.

```markdown
# Losses Testing Guide

This document outlines the testing strategy and implementation for the loss function component (`src/losses.py`) of the RNA 3D folding pipeline. Comprehensive testing ensures that loss calculations are correct, numerically stable, handle masking properly, and integrate correctly with model outputs.

## Testing Objectives

1.  **Validate Calculation Correctness**: Verify each loss function computes the correct value for known inputs.
2.  **Ensure Masking Works**: Confirm that padded sequence positions are correctly excluded from loss calculations.
3.  **Test Gradient Flow**: Ensure gradients flow back correctly from the loss to the relevant model outputs (coordinates, confidence scores, angles).
4.  **Verify Numerical Stability**: Test robustness against potential issues like NaNs, infinities, or large/small input values.
5.  **Check Device Compatibility**: Confirm correct operation on both CPU and CUDA devices.
6.  **Validate Combined Loss**: Ensure weighted combination of losses is calculated correctly.
7.  **Test Edge Cases**: Verify behavior with empty batches, fully masked sequences, or sequences below minimum length thresholds.

## Test Structure

Create a comprehensive test file `tests/test_losses.py` with these major test groups:

1.  Helper Function Tests (`kabsch_align`, `stable_kabsch_align`, `robust_distance_calculation`)
2.  FAPE Proxy Loss Tests (`compute_fape_loss` / `compute_stable_fape_loss`)
3.  Confidence Proxy Loss Tests (`compute_confidence_loss`)
4.  Angle Loss Tests (`compute_angle_loss`)
5.  Combined Loss Tests (`compute_combined_loss`)
6.  Integration Tests (Using mock model outputs)
7.  Device Compatibility Tests
8.  Edge Case Tests

## 1. Helper Function Tests

### 1.1 Test `kabsch_align` / `stable_kabsch_align`

```python
import pytest
import torch
import numpy as np
from src.losses import kabsch_align, stable_kabsch_align # Assuming these are importable

# --- Fixtures for Kabsch Tests ---
@pytest.fixture
def points_a():
    """Reference points (N, 3)."""
    return torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=torch.float32)

@pytest.fixture
def points_b_identity(points_a):
    """Points identical to A."""
    return points_a.clone()

@pytest.fixture
def points_b_translated(points_a):
    """Points translated from A."""
    return points_a + torch.tensor([10.0, -5.0, 2.0])

@pytest.fixture
def points_b_rotated(points_a):
    """Points rotated from A (e.g., 90 degrees around Z)."""
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float32)
    center = points_a.mean(dim=0, keepdim=True)
    return torch.matmul(points_a - center, R) + center

@pytest.fixture
def points_b_reflected(points_a):
    """Points reflected from A (e.g., across XY plane)."""
    R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], dtype=torch.float32)
    center = points_a.mean(dim=0, keepdim=True)
    return torch.matmul(points_a - center, R) + center

# --- Kabsch Alignment Tests ---
@pytest.mark.parametrize("align_func", [kabsch_align, stable_kabsch_align])
def test_kabsch_identity(align_func, points_a, points_b_identity):
    """Test aligning identical point sets."""
    aligned_b = align_func(points_b_identity, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)

@pytest.mark.parametrize("align_func", [kabsch_align, stable_kabsch_align])
def test_kabsch_translation(align_func, points_a, points_b_translated):
    """Test alignment is invariant to translation."""
    aligned_b = align_func(points_b_translated, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)

@pytest.mark.parametrize("align_func", [kabsch_align, stable_kabsch_align])
def test_kabsch_rotation(align_func, points_a, points_b_rotated):
    """Test alignment correctly handles rotation."""
    aligned_b = align_func(points_b_rotated, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)

# --- Stable Kabsch Specific Tests ---
def test_stable_kabsch_reflection(points_a, points_b_reflected):
    """Test stable Kabsch correctly handles reflections."""
    # Standard Kabsch might fail or produce reflection if not handled
    # stable_kabsch_align should detect and correct the reflection in R
    aligned_b = stable_kabsch_align(points_b_reflected, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)

def test_stable_kabsch_degenerate_coincident(points_a):
    """Test stable Kabsch with coincident points."""
    points_coincident = torch.zeros_like(points_a) # All points at origin
    aligned = stable_kabsch_align(points_coincident, points_a)
    # Should align centers, which means output is just the center of points_a
    center_a = points_a.mean(dim=0, keepdim=True).expand_as(points_a)
    assert torch.allclose(aligned, center_a, atol=1e-6)

def test_stable_kabsch_degenerate_collinear():
    """Test stable Kabsch with collinear points."""
    points_collinear = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]], dtype=torch.float32)
    points_collinear_rotated = torch.matmul(points_collinear, torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    aligned = stable_kabsch_align(points_collinear_rotated, points_collinear)
    # Alignment should still work for collinear points
    assert torch.allclose(aligned, points_collinear, atol=1e-6)

```

### 1.2 Test `robust_distance_calculation`

```python
from src.losses import robust_distance_calculation

def test_robust_distance_calculation():
    """Test the stable distance calculation."""
    coords1 = torch.tensor([[0., 0., 0.], [3., 0., 0.]])
    coords2 = torch.tensor([[0., 0., 0.], [0., 4., 0.]])

    # Distance between identical points
    dist_zero = robust_distance_calculation(coords1[0], coords1[0])
    assert torch.isclose(dist_zero, torch.tensor(0.0), atol=1e-7)

    # Known distance
    dist_known = robust_distance_calculation(coords1[1], coords2[1]) # Dist between (3,0,0) and (0,4,0) is 5
    assert torch.isclose(dist_known, torch.tensor(5.0), atol=1e-7)

    # Test with epsilon for stability near zero
    small_diff = torch.tensor([1e-9, 0., 0.])
    dist_small = robust_distance_calculation(coords1[0], coords1[0] + small_diff, epsilon=1e-12)
    assert dist_small > 0 # Should not be exactly zero due to epsilon
```

## 2. FAPE Proxy Loss Tests (`compute_stable_fape_loss`)

```python
import pytest
import torch
from src.losses import compute_stable_fape_loss

@pytest.fixture
def fape_test_data(batch_size=2, seq_len=10, device='cpu'):
    """Fixture for FAPE test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    pred_coords = true_coords.clone() + torch.randn_like(true_coords) * 0.5 # Small noise
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False # Mask last 2 in first sequence
    return {'pred_coords': pred_coords, 'true_coords': true_coords, 'mask': mask}

def test_fape_zero_loss(fape_test_data):
    """Test zero loss for identical coordinates."""
    loss = compute_stable_fape_loss(
        pred_coords=fape_test_data['true_coords'], # Use true as pred
        true_coords=fape_test_data['true_coords'],
        mask=fape_test_data['mask']
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_fape_positive_loss(fape_test_data):
    """Test positive loss for non-identical coordinates."""
    loss = compute_stable_fape_loss(
        pred_coords=fape_test_data['pred_coords'],
        true_coords=fape_test_data['true_coords'],
        mask=fape_test_data['mask']
    )
    assert loss > 0.0

def test_fape_masking(fape_test_data):
    """Test that masked positions are correctly ignored."""
    pred = fape_test_data['pred_coords']
    true = fape_test_data['true_coords']
    mask = fape_test_data['mask'] # Mask [0, -2:] is False

    # Calculate loss with mask
    loss_masked = compute_stable_fape_loss(pred, true, mask)

    # Calculate loss only for sequence 1 (unmasked part)
    loss_seq0_unmasked = compute_stable_fape_loss(pred[0:1, :-2], true[0:1, :-2], mask[0:1, :-2])
    # Calculate loss only for sequence 2 (fully unmasked)
    loss_seq1_full = compute_stable_fape_loss(pred[1:2], true[1:2], mask[1:2])

    # Expected average loss
    # Note: Need to handle cases where sequences might be skipped if len < 3
    # Assuming sequences are long enough here.
    expected_loss = (loss_seq0_unmasked + loss_seq1_full) / 2.0

    assert torch.isclose(loss_masked, expected_loss, atol=1e-6)

def test_fape_clamping(fape_test_data):
    """Test the effect of the clamp_value."""
    pred = fape_test_data['pred_coords'].clone()
    true = fape_test_data['true_coords']
    mask = fape_test_data['mask']

    # Introduce a large error > 10
    pred[1, 0] += torch.tensor([50.0, 0.0, 0.0], device=pred.device)

    loss_clamp10 = compute_stable_fape_loss(pred, true, mask, clamp_value=10.0)
    loss_clamp1 = compute_stable_fape_loss(pred, true, mask, clamp_value=1.0)
    loss_clamp_inf = compute_stable_fape_loss(pred, true, mask, clamp_value=float('inf'))

    assert loss_clamp1 < loss_clamp10
    assert loss_clamp10 < loss_clamp_inf

def test_fape_gradient_flow(fape_test_data):
    """Test gradient flow through FAPE loss."""
    pred = fape_test_data['pred_coords'].clone().requires_grad_(True)
    true = fape_test_data['true_coords']
    mask = fape_test_data['mask']

    loss = compute_stable_fape_loss(pred, true, mask)
    assert loss.requires_grad
    loss.backward()

    assert pred.grad is not None
    # Gradient should be non-zero where mask is True
    assert torch.any(pred.grad[mask] != 0)
    # Gradient should be zero where mask is False
    if (~mask).any():
         assert torch.all(pred.grad[~mask] == 0)

def test_fape_insufficient_points(fape_test_data):
    """Test FAPE loss when sequences have < 3 valid points."""
    pred = fape_test_data['pred_coords']
    true = fape_test_data['true_coords']
    mask_few = torch.zeros_like(fape_test_data['mask'])
    mask_few[:, :2] = True # Only 2 valid points per sequence

    loss = compute_stable_fape_loss(pred, true, mask_few)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_fape_numerical_stability(fape_test_data):
    """Test FAPE loss with potentially unstable inputs."""
    pred = fape_test_data['pred_coords'].clone()
    true = fape_test_data['true_coords']
    mask = fape_test_data['mask']

    # Add NaN to prediction
    pred_nan = pred.clone()
    pred_nan[0, 0, 0] = float('nan')
    # Stable FAPE should handle NaNs by replacing them
    loss_nan = compute_stable_fape_loss(pred_nan, true, mask)
    assert not torch.isnan(loss_nan)
    assert not torch.isinf(loss_nan)

    # Test with zero variance points (coincident) - should use stable Kabsch fallback
    pred_coincident = torch.zeros_like(pred)
    true_coincident = torch.ones_like(true) * 5.0
    loss_coincident = compute_stable_fape_loss(pred_coincident, true_coincident, mask)
    assert not torch.isnan(loss_coincident)
    assert not torch.isinf(loss_coincident)
    # Loss should be approx sqrt( (5-0)^2 * 3 ) = sqrt(75) ~ 8.66, clamped to 10
    assert torch.isclose(loss_coincident, torch.tensor(10.0), atol=1e-4) # Check if it's clamped
```

## 3. Confidence Proxy Loss Tests (`compute_confidence_loss`)

```python
import pytest
import torch
from src.losses import compute_confidence_loss

@pytest.fixture
def conf_test_data(batch_size=2, seq_len=10, device='cpu'):
    """Fixture for Confidence loss test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    # Perfect predictions
    pred_coords_perfect = true_coords.clone()
    # Bad predictions (large errors -> low target confidence)
    pred_coords_bad = true_coords.clone() + 5.0 # ~5A error -> target ~exp(-5/3) ~ 0.18
    # Predicted confidence (logits)
    pred_conf_high = torch.full((batch_size, seq_len), 5.0, device=device) # ~1 prob
    pred_conf_low = torch.full((batch_size, seq_len), -5.0, device=device) # ~0 prob
    pred_conf_medium = torch.zeros((batch_size, seq_len), device=device) # 0.5 prob
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -1] = False # Mask last pos in seq 0
    return {'true_coords': true_coords,
            'pred_coords_perfect': pred_coords_perfect,
            'pred_coords_bad': pred_coords_bad,
            'pred_conf_high': pred_conf_high,
            'pred_conf_low': pred_conf_low,
            'pred_conf_medium': pred_conf_medium,
            'mask': mask}

def test_conf_perfect_pred_high_conf(conf_test_data):
    """Perfect prediction (target ~1), high predicted confidence (~1) -> low loss."""
    loss = compute_confidence_loss(
        pred_confidence=conf_test_data['pred_conf_high'],
        pred_coords=conf_test_data['pred_coords_perfect'],
        true_coords=conf_test_data['true_coords'],
        mask=conf_test_data['mask']
    )
    assert loss < 0.1

def test_conf_perfect_pred_low_conf(conf_test_data):
    """Perfect prediction (target ~1), low predicted confidence (~0) -> high loss."""
    loss = compute_confidence_loss(
        pred_confidence=conf_test_data['pred_conf_low'],
        pred_coords=conf_test_data['pred_coords_perfect'],
        true_coords=conf_test_data['true_coords'],
        mask=conf_test_data['mask']
    )
    # Expected loss ~ (0-1)^2 = 1.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=0.1)

def test_conf_bad_pred_high_conf(conf_test_data):
    """Bad prediction (target ~0.18), high predicted confidence (~1) -> high loss."""
    loss = compute_confidence_loss(
        pred_confidence=conf_test_data['pred_conf_high'],
        pred_coords=conf_test_data['pred_coords_bad'],
        true_coords=conf_test_data['true_coords'],
        mask=conf_test_data['mask']
    )
    # Expected loss ~ (1 - 0.18)^2 ~ 0.67
    assert torch.isclose(loss, torch.tensor(0.67), atol=0.1)

def test_conf_bad_pred_low_conf(conf_test_data):
    """Bad prediction (target ~0.18), low predicted confidence (~0) -> low loss."""
    loss = compute_confidence_loss(
        pred_confidence=conf_test_data['pred_conf_low'],
        pred_coords=conf_test_data['pred_coords_bad'],
        true_coords=conf_test_data['true_coords'],
        mask=conf_test_data['mask']
    )
    # Expected loss ~ (0 - 0.18)^2 ~ 0.03
    assert torch.isclose(loss, torch.tensor(0.03), atol=0.1)

def test_conf_masking(conf_test_data):
    """Test confidence loss masking."""
    pred_conf = conf_test_data['pred_conf_medium']
    pred_coords = conf_test_data['pred_coords_perfect']
    true_coords = conf_test_data['true_coords']
    mask = conf_test_data['mask'] # Last pos in seq 0 masked

    loss_masked = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask)

    # Compute loss without masking last element of seq 0
    mask_full_seq0 = mask.clone()
    mask_full_seq0[0, -1] = True
    loss_full_seq0 = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask_full_seq0)

    # Loss with mask should be different if the masked element would have contributed
    # Contribution of last element: (sigmoid(0) - target)^2 = (0.5 - ~1)^2 = ~0.25
    # Check if losses differ (exact value depends on other points)
    assert not torch.isclose(loss_masked, loss_full_seq0)

def test_conf_gradient_flow(conf_test_data):
    """Test gradient flow for confidence loss."""
    pred_conf = conf_test_data['pred_conf_medium'].clone().requires_grad_(True)
    # NOTE: Gradients should NOT flow back through pred_coords for target calculation
    # as it's done within torch.no_grad() context in the loss function.
    pred_coords = conf_test_data['pred_coords_perfect'].clone()
    true_coords = conf_test_data['true_coords']
    mask = conf_test_data['mask']

    loss = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask)
    assert loss.requires_grad

    loss.backward()

    # Gradients should exist for pred_confidence
    assert pred_conf.grad is not None
    assert torch.any(pred_conf.grad[mask] != 0) # Check unmasked grads
    if (~mask).any():
         assert torch.all(pred_conf.grad[~mask] == 0) # Check masked grads
```

## 4. Angle Loss Tests (`compute_angle_loss`)

```python
import pytest
import torch
import math
from src.losses import compute_angle_loss

@pytest.fixture
def angle_test_data(batch_size=2, seq_len=10, device='cpu'):
    """Fixture for Angle loss test data."""
    # Generate angles theta, eta in radians
    angles_rad = torch.rand(batch_size, seq_len, 2, device=device) * 2 * math.pi
    # Convert to sin/cos pairs [sin(eta), cos(eta), sin(theta), cos(theta)]
    true_angles = torch.cat([
        torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
        torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])
    ], dim=2) # Shape (B, N, 4)

    pred_angles_perfect = true_angles.clone()
    pred_angles_opposite = -true_angles.clone() # Opposite direction sin/cos
    pred_angles_random = torch.randn_like(true_angles)

    true_angles_with_nan = true_angles.clone()
    true_angles_with_nan[0, 0, :] = float('nan') # NaN at start of seq 0
    true_angles_with_nan[1, -1, :] = float('nan') # NaN at end of seq 1

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False # Mask last 2 pos in seq 0

    return {'true_angles': true_angles,
            'pred_angles_perfect': pred_angles_perfect,
            'pred_angles_opposite': pred_angles_opposite,
            'pred_angles_random': pred_angles_random,
            'true_angles_with_nan': true_angles_with_nan,
            'mask': mask}

@pytest.mark.parametrize("loss_type", ['mse', 'mae', 'cosine'])
def test_angle_perfect_pred(angle_test_data, loss_type):
    """Test zero loss for perfect angle predictions."""
    loss = compute_angle_loss(
        pred_angles=angle_test_data['pred_angles_perfect'],
        true_angles=angle_test_data['true_angles'],
        mask=angle_test_data['mask'],
        loss_type=loss_type
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

def test_angle_opposite_pred(angle_test_data):
    """Test loss values for opposite angle predictions."""
    # MSE Loss: Expected ~ 2.0
    loss_mse = compute_angle_loss(
        pred_angles=angle_test_data['pred_angles_opposite'],
        true_angles=angle_test_data['true_angles'],
        mask=angle_test_data['mask'],
        loss_type='mse'
    )
    assert torch.isclose(loss_mse, torch.tensor(2.0), atol=0.1)

    # MAE Loss: Expected > 0.5
    loss_mae = compute_angle_loss(
        pred_angles=angle_test_data['pred_angles_opposite'],
        true_angles=angle_test_data['true_angles'],
        mask=angle_test_data['mask'],
        loss_type='mae'
    )
    assert loss_mae > 0.5

    # Cosine Loss: Expected = 2.0 (1 - (-1))
    loss_cos = compute_angle_loss(
        pred_angles=angle_test_data['pred_angles_opposite'],
        true_angles=angle_test_data['true_angles'],
        mask=angle_test_data['mask'],
        loss_type='cosine'
    )
    assert torch.isclose(loss_cos, torch.tensor(2.0), atol=1e-5)

def test_angle_nan_handling(angle_test_data):
    """Test that NaNs in true angles are handled correctly."""
    pred = angle_test_data['pred_angles_random']
    true_nan = angle_test_data['true_angles_with_nan']
    mask = angle_test_data['mask']

    # Loss should be computed ignoring NaNs
    loss = compute_angle_loss(pred, true_nan, mask, loss_type='mse')
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Compute loss manually excluding NaN positions
    valid_mask = ~torch.isnan(true_nan).any(dim=2) & mask
    num_valid = valid_mask.sum()
    if num_valid > 0:
        pred_valid = pred[valid_mask]
        true_valid = torch.nan_to_num(true_nan[valid_mask], nan=0.0) # Replace NaNs just in case
        expected_loss = F.mse_loss(pred_valid, true_valid, reduction='mean')
        # Loss is averaged over features too, so divide by 4
        # expected_loss = F.mse_loss(pred_valid.view(-1), true_valid.view(-1), reduction='mean')
        # Need careful calculation based on implementation average
        # Let's check relative difference with a version where NaNs are NOT excluded by the function
        # but manually masked AFTER calculation:
        raw_mse = (pred - torch.nan_to_num(true_nan))**2
        manual_masked_loss = (raw_mse * valid_mask.unsqueeze(-1)).sum() / (valid_mask.sum() * 4 + 1e-8)
        assert torch.isclose(loss, manual_masked_loss, atol=1e-6)

def test_angle_masking(angle_test_data):
    """Test masking behavior in angle loss."""
    pred = angle_test_data['pred_angles_random']
    true = angle_test_data['true_angles']
    mask = angle_test_data['mask'] # Last 2 pos in seq 0 masked

    loss_masked = compute_angle_loss(pred, true, mask, loss_type='mse')

    # Compute loss without masking last 2 elements of seq 0
    mask_full_seq0 = mask.clone()
    mask_full_seq0[0, -2:] = True
    loss_full_seq0 = compute_angle_loss(pred, true, mask_full_seq0, loss_type='mse')

    # Loss with mask should be different if masked elements would have contributed
    assert not torch.isclose(loss_masked, loss_full_seq0)

def test_angle_gradient_flow(angle_test_data):
    """Test gradient flow for angle loss."""
    pred = angle_test_data['pred_angles_random'].clone().requires_grad_(True)
    true = angle_test_data['true_angles']
    mask = angle_test_data['mask']

    loss = compute_angle_loss(pred, true, mask, loss_type='mse')
    assert loss.requires_grad

    loss.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad[mask] != 0)
    if (~mask).any():
        assert torch.all(pred.grad[~mask] == 0)

```

## 5. Combined Loss Tests (`compute_combined_loss`)

```python
import pytest
import torch
from src.losses import compute_combined_loss, compute_stable_fape_loss, compute_confidence_loss, compute_angle_loss

@pytest.fixture
def combined_loss_data(batch_size=2, seq_len=10, device='cpu'):
    """Fixture for combined loss test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    pred_coords = true_coords.clone() + torch.randn_like(true_coords) * 0.5
    pred_conf = torch.randn(batch_size, seq_len, device=device)
    # Generate valid sin/cos pairs for angles
    angles_rad = torch.rand(batch_size, seq_len, 2, device=device) * 2 * math.pi
    true_angles = torch.cat([torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
                            torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])], dim=2)
    pred_angles = true_angles.clone() + torch.randn_like(true_angles) * 0.1
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    outputs = {'pred_coords': pred_coords, 'pred_confidence': pred_conf, 'pred_angles': pred_angles}
    batch = {'coordinates': true_coords, 'dihedral_features': true_angles, 'mask': mask}
    return {'outputs': outputs, 'batch': batch}

def test_combined_loss_calculation(combined_loss_data):
    """Test the weighted sum calculation."""
    outputs = combined_loss_data['outputs']
    batch = combined_loss_data['batch']
    weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

    # Calculate individual losses manually
    fape_val = compute_stable_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
    conf_val = compute_confidence_loss(outputs['pred_confidence'], outputs['pred_coords'], batch['coordinates'], batch['mask'])
    angle_val = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])

    expected_total = weights['fape'] * fape_val + weights['confidence'] * conf_val + weights['angle'] * angle_val

    # Calculate using combined function
    total_loss, loss_components = compute_combined_loss(outputs, batch, weights)

    assert torch.isclose(total_loss, expected_total, atol=1e-6)
    assert torch.isclose(loss_components['fape'], fape_val, atol=1e-6)
    assert torch.isclose(loss_components['confidence'], conf_val, atol=1e-6)
    assert torch.isclose(loss_components['angle'], angle_val, atol=1e-6)

def test_combined_loss_weighting(combined_loss_data):
    """Test the effect of different weights."""
    outputs = combined_loss_data['outputs']
    batch = combined_loss_data['batch']

    weights1 = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}
    weights0 = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0}
    weights_fape_only = {'fape': 1.0, 'confidence': 0.0, 'angle': 0.0}

    total1, comps1 = compute_combined_loss(outputs, batch, weights1)
    total0, comps0 = compute_combined_loss(outputs, batch, weights0) # Should be 0
    total_fape, comps_fape = compute_combined_loss(outputs, batch, weights_fape_only)

    assert torch.isclose(total0, torch.tensor(0.0))
    assert torch.isclose(total_fape, comps_fape['fape']) # Total loss should equal FAPE loss
    assert total1 > total_fape # Adding other components should increase total loss (assuming components > 0)

def test_combined_loss_gradient_flow(combined_loss_data):
    """Test gradients flow from combined loss back to inputs."""
    outputs = combined_loss_data['outputs']
    batch = combined_loss_data['batch']
    weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

    # Make outputs require grad
    outputs['pred_coords'].requires_grad_(True)
    outputs['pred_confidence'].requires_grad_(True)
    outputs['pred_angles'].requires_grad_(True)

    total_loss, _ = compute_combined_loss(outputs, batch, weights)
    assert total_loss.requires_grad

    total_loss.backward()

    # Check gradients exist for all output heads
    assert outputs['pred_coords'].grad is not None
    assert outputs['pred_confidence'].grad is not None
    assert outputs['pred_angles'].grad is not None

    # Check gradients are non-zero (assuming non-perfect predictions)
    assert torch.any(outputs['pred_coords'].grad != 0)
    assert torch.any(outputs['pred_confidence'].grad != 0)
    assert torch.any(outputs['pred_angles'].grad != 0)

```

## 6. Integration Tests

```python
# Minimal Model for Integration Test
class MockRNAFoldingModel(torch.nn.Module):
    def __init__(self, seq_len=10, embed_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.linear = torch.nn.Linear(embed_dim, embed_dim) # Dummy layer
        self.coord_head = torch.nn.Parameter(torch.randn(seq_len, 3))
        self.conf_head = torch.nn.Parameter(torch.randn(seq_len))
        self.angle_head = torch.nn.Parameter(torch.randn(seq_len, 4))

    def forward(self, batch):
        # Simple forward that ignores input and returns parameters
        batch_size = batch['mask'].shape[0]
        return {
            'pred_coords': self.coord_head.unsqueeze(0).expand(batch_size, -1, -1),
            'pred_confidence': self.conf_head.unsqueeze(0).expand(batch_size, -1),
            'pred_angles': self.angle_head.unsqueeze(0).expand(batch_size, -1, -1)
        }

def test_loss_integration_with_model(combined_loss_data):
    """Test computing loss using mock model outputs."""
    mock_model = MockRNAFoldingModel(seq_len=combined_loss_data['batch']['mask'].shape[1])
    outputs = mock_model(combined_loss_data['batch']) # Generate outputs
    batch = combined_loss_data['batch']
    weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

    # Should compute without errors
    total_loss, loss_components = compute_combined_loss(outputs, batch, weights)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0 # Scalar
    assert 'fape' in loss_components
    assert 'confidence' in loss_components
    assert 'angle' in loss_components

    # Test gradient flow back to model parameters
    total_loss.backward()
    for param in mock_model.parameters():
        assert param.grad is not None
```

## 7. Device Compatibility Tests

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_losses_device_compatibility(combined_loss_data):
    """Test losses work correctly on CPU and CUDA."""
    outputs_cpu = combined_loss_data['outputs']
    batch_cpu = combined_loss_data['batch']
    weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

    # --- CPU Calculation ---
    total_loss_cpu, comps_cpu = compute_combined_loss(outputs_cpu, batch_cpu, weights)

    # --- CUDA Calculation ---
    device = torch.device('cuda')
    outputs_cuda = {k: v.to(device) for k, v in outputs_cpu.items()}
    batch_cuda = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch_cpu.items()}

    total_loss_cuda, comps_cuda = compute_combined_loss(outputs_cuda, batch_cuda, weights)

    # --- Compare Results ---
    assert torch.isclose(total_loss_cpu, total_loss_cuda.cpu(), atol=1e-5)
    assert torch.isclose(comps_cpu['fape'], comps_cuda['fape'].cpu(), atol=1e-5)
    assert torch.isclose(comps_cpu['confidence'], comps_cuda['confidence'].cpu(), atol=1e-5)
    assert torch.isclose(comps_cpu['angle'], comps_cuda['angle'].cpu(), atol=1e-5)

```

## 8. Edge Case Tests

```python
def test_losses_all_masked(combined_loss_data):
    """Test loss calculation when the entire batch is masked."""
    outputs = combined_loss_data['outputs']
    batch = combined_loss_data['batch']
    weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

    # Create all-False mask
    mask_all_false = torch.zeros_like(batch['mask'])
    batch['mask'] = mask_all_false

    # Compute losses - they should all be zero
    total_loss, loss_components = compute_combined_loss(outputs, batch, weights)

    assert torch.isclose(total_loss, torch.tensor(0.0))
    assert torch.isclose(loss_components['fape'], torch.tensor(0.0))
    assert torch.isclose(loss_components['confidence'], torch.tensor(0.0))
    assert torch.isclose(loss_components['angle'], torch.tensor(0.0))

# Add tests for empty batches if the DataLoader allows them,
# although typically DataLoader raises an error for empty batches.
# def test_losses_empty_batch(): ...
```

## Running the Tests

Execute the tests using `pytest`:

```bash
# Run all loss tests
pytest -xvs tests/test_losses.py

# Run specific test groups
pytest -xvs tests/test_losses.py::test_fape_masking
pytest -xvs tests/test_losses.py::test_conf_gradient_flow
pytest -xvs tests/test_losses.py::TestAngleLoss # Run all tests in the class
```

## Common Test Failures and Remediation

| Failure Pattern           | Likely Cause                                           | Remediation                                                            |
| :------------------------ | :----------------------------------------------------- | :--------------------------------------------------------------------- |
| Loss is NaN/Inf           | Division by zero (e.g., mask sum is zero), SVD failure | Add epsilon to denominators, use stable Kabsch, handle NaNs in inputs. |
| Incorrect Loss Value      | Error in calculation logic, incorrect target calc      | Debug loss calculation step-by-step, verify formulas.                  |
| Masking Not Working       | Mask not applied correctly before reduction            | Ensure mask is broadcasted and multiplied before `sum()` or `mean()`.    |
| Zero Gradients            | Loss detached from graph, `torch.no_grad()` misused    | Check computation graph, ensure inputs require grad, review `no_grad`. |
| Device Mismatch Errors    | Tensors on different devices                           | Ensure all inputs and module parameters are on the same target device. |
| Incorrect Combined Loss   | Error in weighting or summing components             | Verify weight application and summation logic.                         |
| Angle Loss NaN            | Input `true_angles` has NaNs not handled             | Ensure `compute_angle_loss` checks for and masks NaNs in `true_angles`.|

## Test Coverage Goals

Aim for >95% code coverage for the `src/losses.py` module, ensuring:

1.  All loss functions (`fape`, `confidence`, `angle`, `combined`) are tested.
2.  Helper functions (`kabsch`, `stable_kabsch`, `robust_distance`) are tested.
3.  Masking logic is verified in all relevant loss functions.
4.  Gradient flow is confirmed for all functions contributing to the final loss.
5.  Numerical stability and edge cases (NaNs, empty masks, insufficient points) are covered.
6.  Different `loss_type` options (e.g., in `compute_angle_loss`) are tested.

## Next Steps

After implementing and verifying the loss components:

1.  Fix any identified issues based on test results.
2.  Integrate these loss functions into the main training script (`scripts/train.py`).
3.  Use the `LossTracker` utility for monitoring during training.
4.  Consider implementing and testing the V2+ advanced loss functions (full FAPE, lDDT) as the project progresses.
```
