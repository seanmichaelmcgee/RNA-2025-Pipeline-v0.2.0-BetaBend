# Loss Function Test Failure Analysis

## Overview

This document provides a detailed analysis of the failing tests in the loss functions implementation (`src/losses.py`). Based on test execution, 16 out of 76 tests are failing. This analysis groups failures by component and root cause to inform our enhancement approach.

## Test Failures by Component

### 1. Kabsch Alignment Issues

#### Failing Tests:
- `test_kabsch_rotation[cpu]`
- `test_kabsch_rotation[cuda]`
- `test_stable_kabsch_degenerate_collinear[cpu]`
- `test_stable_kabsch_degenerate_collinear[cuda]`

#### Root Cause Analysis:
The Kabsch algorithm is failing to properly handle rotation cases. The assertion failures indicate that after alignment, the rotated points do not match the target points within the specified tolerance. For collinear cases, the algorithm appears to be unable to handle the degenerate case correctly.

Specific issues:
- The rotation matrix calculation may not be handling certain edge cases correctly
- The sign handling in the determinant check for reflections may be incorrect
- The algorithm may not be properly handling the collinear case where the rank of the covariance matrix is reduced

#### Impact:
This is a **critical issue** as Kabsch alignment is fundamental to the FAPE loss calculation. Incorrect alignment will directly affect gradient flow and model training, potentially causing instability or preventing convergence.

### 2. Distance Calculation Issues

#### Failing Tests:
- `test_robust_distance_calculation[cpu]`
- `test_robust_distance_calculation[cuda]`

#### Root Cause Analysis:
The robust distance calculation function is not producing expected results in certain test cases. The assertions check specific distance values that should result from known input coordinates, but the calculated distances don't match the expected values.

Specific issues:
- Epsilon handling may be introducing unexpected bias in distance calculations
- The clamping of values to non-negative may be affecting precision
- The square root operation may have numerical stability issues

#### Impact:
This is a **moderate issue** as it affects the accuracy of distance calculations, which are used in multiple loss functions. However, it may not completely prevent gradient flow if the errors are consistent.

### 3. FAPE Loss Issues

#### Failing Tests:
- `test_fape_zero_loss[cpu]`
- `test_fape_zero_loss[cuda]`
- `test_fape_numerical_stability[cpu]`
- `test_fape_numerical_stability[cuda]`

#### Root Cause Analysis:
The FAPE loss function is not correctly handling zero loss cases (identical points) and has issues with numerical stability in certain edge cases.

Specific issues:
- Zero loss test: When pred_coords equals true_coords, the loss should be exactly zero, but it's not
- Numerical stability: Test with NaN inputs or coincident points is not producing expected results, suggesting the fallback mechanisms are not working correctly

#### Impact:
This is a **critical issue** as FAPE loss is the primary training signal for coordinate prediction. Incorrect handling of zero loss cases may cause unnecessary gradients even when predictions are perfect.

### 4. Confidence Loss Issues

#### Failing Tests:
- `test_conf_bad_pred_high_conf[cpu]`
- `test_conf_bad_pred_high_conf[cuda]`
- `test_conf_bad_pred_low_conf[cpu]`
- `test_conf_bad_pred_low_conf[cuda]`
- `test_conf_masking[cpu]`
- `test_conf_masking[cuda]`

#### Root Cause Analysis:
The confidence loss function is not producing expected values for certain test cases, particularly when predictions are poor but confidence is high or low.

Specific issues:
- The confidence targets may not be properly derived from coordinate errors
- The scaling of confidence values may be incorrect
- The masking mechanism is not properly excluding padded positions from the loss calculation

#### Impact:
This is a **moderate issue** as confidence prediction is an auxiliary objective. While important for model quality assessment, it doesn't directly impact coordinate prediction and gradient flow through the main prediction task.

## Summary of Critical vs. Non-Critical Issues

### Critical Issues (Blocking Model Usage):
1. **Kabsch Rotation Handling**: Directly impacts coordinate alignment and primary loss calculation
2. **FAPE Zero-Loss Handling**: Prevents proper optimization when predictions are good

### Moderate Issues (Affecting Model Quality):
1. **Distance Calculation Precision**: Affects accuracy but may not block training
2. **Confidence Loss Calculation**: Auxiliary objective with moderate impact on training

### Minor Issues (Future Improvements):
1. **Collinear Case Handling**: Edge case that may rarely occur in practice
2. **Mask Propagation in Confidence Loss**: Affects batch efficiency but not core functionality

## Recommended Focus Areas

Based on this analysis, we should prioritize fixes in the following order:

1. **Kabsch Rotation Handling**: Fix the core alignment algorithm to ensure proper rotational alignment
2. **FAPE Zero-Loss Handling**: Ensure zero loss for identical coordinates
3. **Test Clarity Improvements**: Add clear documentation to all tests indicating expected behavior

The moderate and minor issues should be thoroughly documented for the Testing instance but may not require immediate fixes as they don't block basic model functionality.

## Detailed Test Analysis by File Location

### `tests/test_losses.py::TestKabschAlign`

```python
def test_kabsch_rotation(self, points_a, points_b_rotated):
    aligned_b = stable_kabsch_align(points_b_rotated, points_a)
    assert torch.allclose(aligned_b, points_a, atol=1e-6)
```
**Issue**: Aligned points don't match target points after rotation.

### `tests/test_losses.py::TestRobustDistance`

```python
def test_robust_distance_calculation(self, device):
    coords1 = torch.tensor([[0., 0., 0.], [3., 0., 0.]], device=device)
    coords2 = torch.tensor([[0., 0., 0.], [0., 4., 0.]], device=device)
    
    dist_zero = robust_distance_calculation(coords1[0], coords1[0])
    assert torch.isclose(dist_zero, torch.tensor(0.0, device=device), atol=1e-7)
    
    dist_known = robust_distance_calculation(coords1[1], coords2[1])
    assert torch.isclose(dist_known, torch.tensor(5.0, device=device), atol=1e-7)
```
**Issue**: Calculated distances don't match expected values.

### `tests/test_losses.py::TestStableFAPELoss`

```python
def test_fape_zero_loss(self, fape_test_data):
    loss = compute_stable_fape_loss(
        pred_coords=fape_test_data['true_coords'],
        true_coords=fape_test_data['true_coords'],
        mask=fape_test_data['mask']
    )
    assert torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6)
```
**Issue**: Loss is not zero when coordinates are identical.

### `tests/test_losses.py::TestConfidenceLoss`

```python
def test_conf_bad_pred_high_conf(self, conf_test_data):
    loss = compute_confidence_loss(
        pred_confidence=conf_test_data['pred_conf_high'],
        pred_coords=conf_test_data['pred_coords_bad'],
        true_coords=conf_test_data['true_coords'],
        mask=conf_test_data['mask']
    )
    # Target for 5A error: exp(-5/3) = 0.188. Loss = (sigmoid(5) - 0.188)^2 = (0.993 - 0.188)^2 = 0.805^2 = 0.648
    assert torch.isclose(loss, torch.tensor(0.648, device=loss.device), atol=0.1)
```
**Issue**: Calculated confidence loss doesn't match expected value based on error distance.