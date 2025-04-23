"""
Exhaustive test suite for RMSD calculation edge cases.

This test file is focused on identifying and validating fixes for numerical 
instability issues in the RMSD calculation for RNA 3D structure prediction.
"""

import pytest
import torch
import numpy as np
from math import sqrt

from src.utils.structure_metrics import compute_rmsd
from src.losses import stable_kabsch_align, robust_distance_calculation


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Fixture to provide CPU and CUDA devices if available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


class TestRMSDEdgeCases:
    """Exhaustive tests for edge cases in RMSD calculation."""
    
    def test_identical_structures(self, device):
        """Test RMSD calculation for identical structures."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=device)
        
        rmsd = compute_rmsd(coords, coords)
        assert rmsd.item() == 0.0, "RMSD should be exactly zero for identical structures"
    
    def test_very_close_structures(self, device):
        """Test RMSD calculation for nearly identical structures."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=device)
        
        # Add small noise
        epsilon = 1e-10
        noise = torch.rand_like(coords1) * epsilon
        coords2 = coords1 + noise
        
        rmsd = compute_rmsd(coords1, coords2)
        assert rmsd.item() < epsilon * 2, f"RMSD should be very small for nearly identical structures: {rmsd.item()}"
    
    def test_only_translation(self, device):
        """Test RMSD calculation for structures differing only by translation."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=device)
        
        # Apply large translation
        translation = torch.tensor([100.0, -50.0, 25.0], device=device)
        coords2 = coords1 + translation
        
        rmsd = compute_rmsd(coords1, coords2)
        assert rmsd.item() < 1e-5, f"RMSD should be near zero for structures differing only by translation: {rmsd.item()}"
    
    def test_only_rotation(self, device):
        """Test RMSD calculation for structures differing only by rotation."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=device)
        
        # 90-degree rotation around z-axis
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # Center before rotation
        center = coords1.mean(dim=0, keepdim=True)
        coords2 = torch.matmul(coords1 - center, rotation.T) + center
        
        rmsd = compute_rmsd(coords1, coords2)
        assert rmsd.item() < 1e-5, f"RMSD should be near zero for structures differing only by rotation: {rmsd.item()}"
    
    def test_collinear_points(self, device):
        """Test RMSD calculation for collinear points."""
        # Points on a line along x-axis
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], device=device)
        
        # Points on a line along y-axis (rotated)
        coords2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0]
        ], device=device)
        
        rmsd = compute_rmsd(coords1, coords2)
        assert torch.isfinite(rmsd), "RMSD should be finite for collinear points"
        
    def test_coplanar_points(self, device):
        """Test RMSD calculation for coplanar points."""
        # Points on the xy-plane
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], device=device)
        
        # Same points rotated around x-axis (now in xz-plane)
        coords2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0]
        ], device=device)
        
        rmsd = compute_rmsd(coords1, coords2)
        assert torch.isfinite(rmsd), "RMSD should be finite for coplanar points"
    
    def test_minimal_points(self, device):
        """Test RMSD calculation with minimal number of points (3)."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=device)
        
        coords2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.1]  # Small difference in one point
        ], device=device)
        
        rmsd = compute_rmsd(coords1, coords2)
        expected_rmsd = sqrt(0.1**2 / 3)  # One point moved by 0.1 in z-direction
        assert abs(rmsd.item() - expected_rmsd) < 1e-5, f"RMSD incorrect for minimal points: {rmsd.item()} vs {expected_rmsd}"
    
    def test_extreme_coordinate_values(self, device):
        """Test RMSD calculation with extreme coordinate values."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1e9, 0.0, 0.0],  # Very large x-coordinate
            [0.0, 1.0, 0.0]
        ], device=device)
        
        # Same structure, just translated
        coords2 = coords1 + 1.0
        
        rmsd = compute_rmsd(coords1, coords2)
        assert torch.isfinite(rmsd), "RMSD should be finite even with extreme coordinates"
    
    def test_with_nan_values(self, device):
        """Test robust handling of NaN values."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], device=device)
        
        coords2 = coords1.clone()
        coords2[1, 0] = float('nan')  # Introduce NaN
        
        # Create mask to exclude NaN values
        mask = torch.tensor([True, False, True, True], device=device)
        
        rmsd = compute_rmsd(coords1, coords2, mask=mask)
        assert torch.isfinite(rmsd), "RMSD should be finite when NaNs are masked out"
    
    def test_rotation_with_reflection(self, device):
        """Test RMSD calculation with rotation including a reflection."""
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # Reflection across yz-plane
        reflection = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        coords2 = torch.matmul(coords1, reflection.T)
        
        rmsd = compute_rmsd(coords1, coords2)
        assert torch.isfinite(rmsd), "RMSD should be finite when reflection is involved"
    
    def test_single_point_structures(self, device):
        """Test stability with single-point structures (should not be allowed)."""
        coords1 = torch.tensor([[0.0, 0.0, 0.0]], device=device)
        coords2 = torch.tensor([[1.0, 0.0, 0.0]], device=device)  # Moved point
        
        with pytest.raises(Exception):
            # This should raise an error as we need at least 3 points
            compute_rmsd(coords1, coords2)
    
    def test_batched_edge_cases(self, device):
        """Test batched RMSD calculation with edge cases."""
        batch_size = 3
        
        # Create batch with different edge cases:
        # 1. Identical structures
        # 2. Collinear points
        # 3. Extreme values
        
        # Batch item 1: Identical structures
        identical_struct = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ], device=device)
        
        # Batch item 2: Collinear points
        collinear_struct1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], device=device)
        
        collinear_struct2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0]
        ], device=device)
        
        # Batch item 3: Extreme values
        extreme_struct = torch.tensor([
            [0.0, 0.0, 0.0],
            [1e9, 0.0, 0.0],
            [0.0, 1e9, 0.0],
            [0.0, 0.0, 1e9]
        ], device=device)
        
        # Create batched tensors
        pred_coords = torch.stack([
            identical_struct,
            collinear_struct1,
            extreme_struct
        ])
        
        true_coords = torch.stack([
            identical_struct,
            collinear_struct2,
            extreme_struct + 1.0  # Translated
        ])
        
        # Compute batched RMSD
        rmsd_values = compute_rmsd(pred_coords, true_coords)
        
        # Check all values
        assert rmsd_values.shape == (batch_size,), "Should return one RMSD value per batch"
        assert rmsd_values[0].item() == 0.0, "RMSD should be exactly 0 for identical structures"
        assert torch.isfinite(rmsd_values[1]), "RMSD should be finite for collinear points"
        assert torch.isfinite(rmsd_values[2]), "RMSD should be finite for extreme values"
    
    def test_svd_comparison(self, device):
        """Test different SVD implementations for stability."""
        # Create a test case known to be problematic
        coords1 = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], device=device)
        
        coords2 = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0]
        ], device=device)
        
        # Center the coordinates
        center1 = coords1.mean(dim=0, keepdim=True)
        center2 = coords2.mean(dim=0, keepdim=True)
        coords1_centered = coords1 - center1
        coords2_centered = coords2 - center2
        
        # Compute covariance matrix
        covariance = torch.matmul(coords1_centered.transpose(-2, -1), coords2_centered)
        
        # Test different SVD options
        try:
            # Full matrices SVD
            U_full, S_full, Vt_full = torch.linalg.svd(covariance, full_matrices=True)
            
            # Reduced SVD (commonly more stable)
            U, S, Vt = torch.linalg.svd(covariance, full_matrices=False)
            
            # Test our stable implementation
            coords_aligned = stable_kabsch_align(coords1, coords2)
            
            # All should succeed (not raise exceptions) and give finite results
            assert torch.isfinite(U_full).all()
            assert torch.isfinite(S_full).all()
            assert torch.isfinite(Vt_full).all()
            assert torch.isfinite(U).all()
            assert torch.isfinite(S).all()
            assert torch.isfinite(Vt).all()
            assert torch.isfinite(coords_aligned).all()
            
        except Exception as e:
            pytest.fail(f"SVD failed: {e}")
    
    def test_robust_distance_edge_cases(self, device):
        """Test edge cases for robust distance calculation."""
        # Test case 1: Identical points (zero distance)
        point1 = torch.tensor([1.0, 2.0, 3.0], device=device)
        dist_zero = robust_distance_calculation(point1, point1)
        assert dist_zero.item() == 0.0, "Distance between identical points should be exactly 0.0"
        
        # Test case 2: Very small difference
        point2 = point1 + 1e-16
        dist_tiny = robust_distance_calculation(point1, point2)
        assert dist_tiny.item() >= 0.0, "Distance should never be negative"
        
        # Test case 3: Large distance
        point3 = torch.tensor([1e6, 2e6, 3e6], device=device)
        dist_large = robust_distance_calculation(point1, point3)
        assert torch.isfinite(dist_large), "Distance calculation should be finite even for large distances"
        
        # Test case 4: With NaN
        point_nan = torch.tensor([float('nan'), 2.0, 3.0], device=device)
        dist_nan = robust_distance_calculation(point1, point_nan)
        assert not torch.isfinite(dist_nan), "Distance with NaN input should be NaN"
        
    def test_compare_with_numpy(self, device):
        """Compare our RMSD implementation with NumPy for validation."""
        # Generate random coordinates
        torch.manual_seed(42)
        coords1 = torch.randn(10, 3, device=device)
        
        # Small perturbation
        coords2 = coords1 + torch.randn(10, 3, device=device) * 0.1
        
        # Compute RMSD with our implementation
        rmsd_torch = compute_rmsd(coords1, coords2)
        
        # Convert to NumPy
        coords1_np = coords1.cpu().numpy()
        coords2_np = coords2.cpu().numpy()
        
        # Center the coordinates
        c1_mean = coords1_np.mean(axis=0)
        c2_mean = coords2_np.mean(axis=0)
        coords1_np_centered = coords1_np - c1_mean
        coords2_np_centered = coords2_np - c2_mean
        
        # Compute covariance matrix
        covariance = np.dot(coords1_np_centered.T, coords2_np_centered)
        
        # Use NumPy's SVD
        u, s, vh = np.linalg.svd(covariance)
        
        # Correct for reflections
        d = np.sign(np.linalg.det(np.dot(vh.T, u.T)))
        if d < 0:
            vh[-1, :] = -vh[-1, :]
        
        # Compute rotation matrix
        rotation = np.dot(vh.T, u.T)
        
        # Apply rotation to centered coordinates
        coords1_np_aligned = np.dot(coords1_np_centered, rotation)
        
        # Compute RMSD
        squared_diff = ((coords1_np_aligned - coords2_np_centered) ** 2).sum(axis=1)
        rmsd_np = np.sqrt(squared_diff.mean())
        
        # Compare results
        assert abs(rmsd_torch.item() - rmsd_np) < 1e-5, f"PyTorch and NumPy RMSD calculations differ: {rmsd_torch.item()} vs {rmsd_np}"