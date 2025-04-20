# START OF FILE: tests/test_losses.py
"""
Unit tests for the loss functions defined in src/losses.py.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import math
from src.losses import (
    stable_kabsch_align,
    robust_distance_calculation,
    compute_stable_fape_loss,
    compute_confidence_loss,
    compute_angle_loss,
    compute_combined_loss
)

# --- Fixtures ---

@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Fixture to provide CPU and CUDA devices if available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

@pytest.fixture
def points_a(device):
    """Reference points (N, 3)."""
    return torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=torch.float32, device=device)

@pytest.fixture
def points_b_identity(points_a):
    """Points identical to A."""
    return points_a.clone()

@pytest.fixture
def points_b_translated(points_a, device):
    """Points translated from A."""
    return points_a + torch.tensor([10.0, -5.0, 2.0], device=device)

@pytest.fixture
def points_b_rotated(points_a, device):
    """Points rotated from A (e.g., 90 degrees around Z)."""
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float32, device=device)
    center = points_a.mean(dim=0, keepdim=True)
    return torch.matmul(points_a - center, R) + center

@pytest.fixture
def points_b_reflected(points_a, device):
    """Points reflected from A (e.g., across XY plane)."""
    R = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]], dtype=torch.float32, device=device)
    center = points_a.mean(dim=0, keepdim=True)
    return torch.matmul(points_a - center, R) + center

@pytest.fixture
def fape_test_data(device, batch_size=2, seq_len=10):
    """Fixture for FAPE test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    pred_coords = true_coords.clone() + torch.randn_like(true_coords) * 0.5 # Small noise
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False # Mask last 2 in first sequence
    return {'pred_coords': pred_coords, 'true_coords': true_coords, 'mask': mask}

@pytest.fixture
def conf_test_data(device, batch_size=2, seq_len=10):
    """Fixture for Confidence loss test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    pred_coords_perfect = true_coords.clone()
    pred_coords_bad = true_coords.clone() + 5.0 # ~5A error
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

@pytest.fixture
def angle_test_data(device, batch_size=2, seq_len=10):
    """Fixture for Angle loss test data."""
    angles_rad = torch.rand(batch_size, seq_len, 2, device=device) * 2 * math.pi
    true_angles = torch.cat([
        torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
        torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])
    ], dim=2)
    pred_angles_perfect = true_angles.clone()
    pred_angles_opposite = -true_angles.clone()
    pred_angles_random = torch.randn_like(true_angles)
    true_angles_with_nan = true_angles.clone()
    true_angles_with_nan[0, 0, :] = float('nan')
    true_angles_with_nan[1, -1, :] = float('nan')
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False
    return {'true_angles': true_angles,
            'pred_angles_perfect': pred_angles_perfect,
            'pred_angles_opposite': pred_angles_opposite,
            'pred_angles_random': pred_angles_random,
            'true_angles_with_nan': true_angles_with_nan,
            'mask': mask}

@pytest.fixture
def combined_loss_data(device, batch_size=2, seq_len=10):
    """Fixture for combined loss test data."""
    true_coords = torch.randn(batch_size, seq_len, 3, device=device) * 10
    pred_coords = true_coords.clone() + torch.randn_like(true_coords) * 0.5
    pred_conf = torch.randn(batch_size, seq_len, device=device)
    angles_rad = torch.rand(batch_size, seq_len, 2, device=device) * 2 * math.pi
    true_angles = torch.cat([torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
                            torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])], dim=2)
    pred_angles = true_angles.clone() + torch.randn_like(true_angles) * 0.1
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    outputs = {'pred_coords': pred_coords, 'pred_confidence': pred_conf, 'pred_angles': pred_angles}
    batch = {'coordinates': true_coords, 'dihedral_features': true_angles, 'mask': mask}
    return {'outputs': outputs, 'batch': batch}

# --- Test Classes ---

class TestKabschAlign:
    """Tests for the stable_kabsch_align helper function."""

    def test_kabsch_identity(self, points_a, points_b_identity):
        aligned_b = stable_kabsch_align(points_b_identity, points_a)
        assert torch.allclose(aligned_b, points_a, atol=1e-6)

    def test_kabsch_translation(self, points_a, points_b_translated):
        aligned_b = stable_kabsch_align(points_b_translated, points_a)
        assert torch.allclose(aligned_b, points_a, atol=1e-6)

    def test_kabsch_rotation(self, points_a, points_b_rotated):
        aligned_b = stable_kabsch_align(points_b_rotated, points_a)
        assert torch.allclose(aligned_b, points_a, atol=1e-6)

    def test_stable_kabsch_reflection(self, points_a, points_b_reflected):
        aligned_b = stable_kabsch_align(points_b_reflected, points_a)
        assert torch.allclose(aligned_b, points_a, atol=1e-6)

    def test_stable_kabsch_degenerate_coincident(self, points_a):
        points_coincident = torch.zeros_like(points_a)
        aligned = stable_kabsch_align(points_coincident, points_a)
        center_a = points_a.mean(dim=0, keepdim=True).expand_as(points_a)
        assert torch.allclose(aligned, center_a, atol=1e-6)

    def test_stable_kabsch_degenerate_collinear(self, device):
        points_collinear = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.], [3.,0.,0.]], dtype=torch.float32, device=device)
        R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
        points_collinear_rotated = torch.matmul(points_collinear, R)
        aligned = stable_kabsch_align(points_collinear_rotated, points_collinear)
        assert torch.allclose(aligned, points_collinear, atol=1e-6)

    def test_stable_kabsch_small_input(self, device):
        P = torch.randn(2, 3, device=device) # Only 2 points
        Q = torch.randn(2, 3, device=device)
        aligned_P = stable_kabsch_align(P, Q)
        assert aligned_P.shape == P.shape
        assert not torch.isnan(aligned_P).any()

    def test_stable_kabsch_empty_input(self, device):
        P = torch.empty((0, 3), device=device)
        Q = torch.empty((0, 3), device=device)
        aligned_P = stable_kabsch_align(P, Q)
        assert aligned_P.shape == P.shape


class TestRobustDistance:
    """Tests for the robust_distance_calculation helper function."""

    def test_robust_distance_calculation(self, device):
        coords1 = torch.tensor([[0., 0., 0.], [3., 0., 0.]], device=device)
        coords2 = torch.tensor([[0., 0., 0.], [0., 4., 0.]], device=device)

        dist_zero = robust_distance_calculation(coords1[0], coords1[0])
        assert torch.isclose(dist_zero, torch.tensor(0.0, device=device), atol=1e-7)

        dist_known = robust_distance_calculation(coords1[1], coords2[1])
        assert torch.isclose(dist_known, torch.tensor(5.0, device=device), atol=1e-7)

        small_diff = torch.tensor([1e-9, 0., 0.], device=device)
        dist_small = robust_distance_calculation(coords1[0], coords1[0] + small_diff, epsilon=1e-12)
        assert dist_small > 0
        # Check if close to epsilon^0.5 when diff is smaller than epsilon
        dist_tiny = robust_distance_calculation(coords1[0], coords1[0], epsilon=1e-12)
        assert torch.isclose(dist_tiny, torch.tensor(1e-6, device=device), atol=1e-7)

    def test_robust_distance_batch(self, device):
        coords1 = torch.randn(5, 10, 3, device=device)
        coords2 = torch.randn(5, 10, 3, device=device)
        distances = robust_distance_calculation(coords1, coords2)
        assert distances.shape == (5, 10)


class TestStableFAPELoss:
    """Unit tests for compute_stable_fape_loss (V1 FAPE proxy)."""

    def test_fape_zero_loss(self, fape_test_data):
        loss = compute_stable_fape_loss(
            pred_coords=fape_test_data['true_coords'],
            true_coords=fape_test_data['true_coords'],
            mask=fape_test_data['mask']
        )
        assert torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6)

    def test_fape_positive_loss(self, fape_test_data):
        loss = compute_stable_fape_loss(
            pred_coords=fape_test_data['pred_coords'],
            true_coords=fape_test_data['true_coords'],
            mask=fape_test_data['mask']
        )
        assert loss > 0.0

    def test_fape_masking(self, fape_test_data):
        pred = fape_test_data['pred_coords']
        true = fape_test_data['true_coords']
        mask = fape_test_data['mask']

        loss_masked = compute_stable_fape_loss(pred, true, mask)

        # Ensure loss differs from unmasked calculation
        mask_full = torch.ones_like(mask)
        loss_unmasked = compute_stable_fape_loss(pred, true, mask_full)
        # Unless the masked parts happened to have zero error
        # Let's force a difference in the masked part
        pred_diff = pred.clone()
        pred_diff[0, -1] += 100.0 # Large difference in masked area
        loss_masked_diff = compute_stable_fape_loss(pred_diff, true, mask)
        loss_unmasked_diff = compute_stable_fape_loss(pred_diff, true, mask_full, clamp_value=1000.0)

        # Masked loss should not change much from original masked loss
        assert torch.isclose(loss_masked, loss_masked_diff, atol=1e-4)
        # Unmasked loss should increase significantly due to the large error
        assert loss_unmasked_diff > loss_unmasked * 10 # Heuristic check

    def test_fape_clamping(self, fape_test_data):
        pred = fape_test_data['pred_coords'].clone()
        true = fape_test_data['true_coords']
        mask = fape_test_data['mask']
        pred[1, 0] += torch.tensor([50.0, 0.0, 0.0], device=pred.device)

        loss_clamp10 = compute_stable_fape_loss(pred, true, mask, clamp_value=10.0)
        loss_clamp1 = compute_stable_fape_loss(pred, true, mask, clamp_value=1.0)
        loss_clamp_inf = compute_stable_fape_loss(pred, true, mask, clamp_value=float('inf'))

        assert loss_clamp1 < loss_clamp10
        assert loss_clamp10 < loss_clamp_inf

    def test_fape_gradient_flow(self, fape_test_data):
        pred = fape_test_data['pred_coords'].clone().requires_grad_(True)
        true = fape_test_data['true_coords']
        mask = fape_test_data['mask']

        loss = compute_stable_fape_loss(pred, true, mask)
        assert loss.requires_grad
        loss.backward()

        assert pred.grad is not None
        assert torch.any(pred.grad[mask] != 0)
        if (~mask).any():
             assert torch.all(pred.grad[~mask] == 0)

    def test_fape_insufficient_points(self, fape_test_data):
        pred = fape_test_data['pred_coords']
        true = fape_test_data['true_coords']
        mask_few = torch.zeros_like(fape_test_data['mask'])
        mask_few[:, :2] = True # Only 2 valid points per sequence

        loss = compute_stable_fape_loss(pred, true, mask_few)
        assert torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6)

    def test_fape_all_masked_batch(self, fape_test_data):
        mask_all_false = torch.zeros_like(fape_test_data['mask'])
        loss = compute_stable_fape_loss(fape_test_data['pred_coords'], fape_test_data['true_coords'], mask_all_false)
        assert torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6)

    def test_fape_numerical_stability(self, fape_test_data):
        pred = fape_test_data['pred_coords'].clone()
        true = fape_test_data['true_coords']
        mask = fape_test_data['mask']

        pred_nan = pred.clone()
        pred_nan[0, 0, 0] = float('nan')
        loss_nan = compute_stable_fape_loss(pred_nan, true, mask)
        assert not torch.isnan(loss_nan)
        assert not torch.isinf(loss_nan)

        pred_coincident = torch.zeros_like(pred)
        true_coincident = torch.ones_like(true) * 5.0
        loss_coincident = compute_stable_fape_loss(pred_coincident, true_coincident, mask)
        assert not torch.isnan(loss_coincident)
        assert not torch.isinf(loss_coincident)
        # Loss should be approx sqrt( (5-0)^2 * 3 ) = sqrt(75) ~ 8.66, clamped to 10
        # Consider the mask: seq0 has 8 valid, seq1 has 10. Total 18 points.
        # Expected loss should be 10.0 (clamped)
        assert torch.isclose(loss_coincident, torch.tensor(10.0, device=loss_coincident.device), atol=1e-4)


class TestConfidenceLoss:
    """Unit tests for compute_confidence_loss (V1 Proxy Target)."""

    def test_conf_perfect_pred_high_conf(self, conf_test_data):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_high'],
            pred_coords=conf_test_data['pred_coords_perfect'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask']
        )
        assert loss < 0.1

    def test_conf_perfect_pred_low_conf(self, conf_test_data):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_low'],
            pred_coords=conf_test_data['pred_coords_perfect'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask']
        )
        assert torch.isclose(loss, torch.tensor(1.0, device=loss.device), atol=0.1)

    def test_conf_bad_pred_high_conf(self, conf_test_data):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_high'],
            pred_coords=conf_test_data['pred_coords_bad'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask']
        )
        # Target for 5A error: exp(-5/3) = 0.188. Loss = (sigmoid(5) - 0.188)^2 = (0.993 - 0.188)^2 = 0.805^2 = 0.648
        assert torch.isclose(loss, torch.tensor(0.648, device=loss.device), atol=0.1)

    def test_conf_bad_pred_low_conf(self, conf_test_data):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_low'],
            pred_coords=conf_test_data['pred_coords_bad'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask']
        )
        # Target for 5A error: 0.188. Loss = (sigmoid(-5) - 0.188)^2 = (0.0067 - 0.188)^2 = (-0.181)^2 = 0.032
        assert torch.isclose(loss, torch.tensor(0.032, device=loss.device), atol=0.1)

    def test_conf_masking(self, conf_test_data):
        pred_conf = conf_test_data['pred_conf_medium']
        pred_coords = conf_test_data['pred_coords_perfect']
        true_coords = conf_test_data['true_coords']
        mask = conf_test_data['mask'] # Last pos in seq 0 masked

        loss_masked = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask)

        mask_full_seq0 = mask.clone()
        mask_full_seq0[0, -1] = True
        loss_full_seq0 = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask_full_seq0)

        assert not torch.isclose(loss_masked, loss_full_seq0)

    def test_conf_gradient_flow(self, conf_test_data):
        pred_conf = conf_test_data['pred_conf_medium'].clone().requires_grad_(True)
        pred_coords = conf_test_data['pred_coords_perfect'].clone()
        true_coords = conf_test_data['true_coords']
        mask = conf_test_data['mask']

        loss = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask)
        assert loss.requires_grad
        loss.backward()

        assert pred_conf.grad is not None
        assert torch.any(pred_conf.grad[mask] != 0)
        if (~mask).any():
             assert torch.all(pred_conf.grad[~mask] == 0)

    @pytest.mark.parametrize("loss_type", ['mse', 'bce'])
    def test_conf_loss_types(self, conf_test_data, loss_type):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_medium'],
            pred_coords=conf_test_data['pred_coords_bad'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask'],
            loss_type=loss_type
        )
        assert loss >= 0.0
        assert not torch.isnan(loss)

    @pytest.mark.parametrize("target_type", ['lddt_proxy', 'distance_based'])
    def test_conf_target_types(self, conf_test_data, target_type):
        loss = compute_confidence_loss(
            pred_confidence=conf_test_data['pred_conf_medium'],
            pred_coords=conf_test_data['pred_coords_bad'],
            true_coords=conf_test_data['true_coords'],
            mask=conf_test_data['mask'],
            target_type=target_type
        )
        assert loss >= 0.0
        assert not torch.isnan(loss)

class TestAngleLoss:
    """Unit tests for compute_angle_loss."""

    @pytest.mark.parametrize("loss_type", ['mse', 'mae', 'cosine'])
    def test_angle_perfect_pred(self, angle_test_data, loss_type):
        loss = compute_angle_loss(
            pred_angles=angle_test_data['pred_angles_perfect'],
            true_angles=angle_test_data['true_angles'],
            mask=angle_test_data['mask'],
            loss_type=loss_type
        )
        assert torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6)

    def test_angle_opposite_pred(self, angle_test_data):
        loss_mse = compute_angle_loss(
            pred_angles=angle_test_data['pred_angles_opposite'],
            true_angles=angle_test_data['true_angles'],
            mask=angle_test_data['mask'],
            loss_type='mse'
        )
        # Expected MSE ~ 2.0 * (avg(sin^2+cos^2)) over 2 angles = 2.0
        assert torch.isclose(loss_mse, torch.tensor(2.0, device=loss_mse.device), atol=0.1)

        loss_mae = compute_angle_loss(
            pred_angles=angle_test_data['pred_angles_opposite'],
            true_angles=angle_test_data['true_angles'],
            mask=angle_test_data['mask'],
            loss_type='mae'
        )
        # Expected MAE = avg(|-t - t|) = avg(2|t|). Avg(|sin|, |cos|) > 0.5 -> MAE > 1.0
        assert loss_mae > 1.0

        loss_cos = compute_angle_loss(
            pred_angles=angle_test_data['pred_angles_opposite'],
            true_angles=angle_test_data['true_angles'],
            mask=angle_test_data['mask'],
            loss_type='cosine'
        )
        assert torch.isclose(loss_cos, torch.tensor(2.0, device=loss_cos.device), atol=1e-5)

    def test_angle_nan_handling(self, angle_test_data):
        pred = angle_test_data['pred_angles_random']
        true_nan = angle_test_data['true_angles_with_nan']
        mask = angle_test_data['mask']

        loss = compute_angle_loss(pred, true_nan, mask, loss_type='mse')
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Ensure loss differs from calculation without NaN handling (manual check)
        valid_mask = ~torch.isnan(true_nan).any(dim=2) & mask
        num_valid_elements = valid_mask.sum() * 4
        if num_valid_elements > 0:
             manual_loss = torch.sum(((pred - torch.nan_to_num(true_nan))**2) * valid_mask.unsqueeze(-1)) / num_valid_elements
             assert torch.isclose(loss, manual_loss, atol=1e-6)

    def test_angle_masking(self, angle_test_data):
        pred = angle_test_data['pred_angles_random']
        true = angle_test_data['true_angles']
        mask = angle_test_data['mask'] # Last 2 pos in seq 0 masked

        loss_masked = compute_angle_loss(pred, true, mask, loss_type='mse')

        mask_full_seq0 = mask.clone()
        mask_full_seq0[0, -2:] = True
        loss_full_seq0 = compute_angle_loss(pred, true, mask_full_seq0, loss_type='mse')

        assert not torch.isclose(loss_masked, loss_full_seq0)

    def test_angle_gradient_flow(self, angle_test_data):
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


class TestCombinedLoss:
    """Unit tests for compute_combined_loss."""

    def test_combined_loss_calculation(self, combined_loss_data, device):
        outputs = {k: v.to(device) for k, v in combined_loss_data['outputs'].items()}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in combined_loss_data['batch'].items()}
        weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

        fape_val = compute_stable_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
        conf_val = compute_confidence_loss(outputs['pred_confidence'], outputs['pred_coords'], batch['coordinates'], batch['mask'])
        angle_val = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
        expected_total = weights['fape'] * fape_val + weights['confidence'] * conf_val + weights['angle'] * angle_val

        total_loss, loss_components = compute_combined_loss(outputs, batch, weights)

        assert torch.isclose(total_loss, expected_total, atol=1e-6)
        assert torch.isclose(loss_components['fape'], fape_val, atol=1e-6)
        assert torch.isclose(loss_components['confidence'], conf_val, atol=1e-6)
        assert torch.isclose(loss_components['angle'], angle_val, atol=1e-6)

    def test_combined_loss_weighting(self, combined_loss_data, device):
        outputs = {k: v.to(device) for k, v in combined_loss_data['outputs'].items()}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in combined_loss_data['batch'].items()}

        weights1 = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}
        weights0 = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0}
        weights_fape_only = {'fape': 1.0, 'confidence': 0.0, 'angle': 0.0}

        total1, comps1 = compute_combined_loss(outputs, batch, weights1)
        total0, comps0 = compute_combined_loss(outputs, batch, weights0)
        total_fape, comps_fape = compute_combined_loss(outputs, batch, weights_fape_only)

        assert torch.isclose(total0, torch.tensor(0.0, device=device))
        assert torch.isclose(total_fape, comps_fape['fape'])
        # Ensure total1 is roughly the weighted sum
        expected_total1 = (1.0 * comps1['fape'] + 0.1 * comps1['confidence'] + 0.5 * comps1['angle'])
        assert torch.isclose(total1, expected_total1, atol=1e-6)

    def test_combined_loss_gradient_flow(self, combined_loss_data, device):
        outputs = combined_loss_data['outputs']
        batch = combined_loss_data['batch']
        weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}

        outputs_grad = {k: v.to(device).clone().requires_grad_(True) for k, v in outputs.items()}
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        total_loss, _ = compute_combined_loss(outputs_grad, batch_device, weights)
        assert total_loss.requires_grad

        total_loss.backward()

        assert outputs_grad['pred_coords'].grad is not None
        assert outputs_grad['pred_confidence'].grad is not None
        assert outputs_grad['pred_angles'].grad is not None
        assert torch.any(outputs_grad['pred_coords'].grad != 0)
        assert torch.any(outputs_grad['pred_confidence'].grad != 0)
        assert torch.any(outputs_grad['pred_angles'].grad != 0)