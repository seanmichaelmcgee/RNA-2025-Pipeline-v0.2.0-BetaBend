"""
Unit tests for structure evaluation metrics.
"""

import pytest
import torch
import numpy as np

from src.utils.structure_metrics import (
    compute_rmsd,
    compute_tm_score,
    compute_structure_metrics,
    compute_per_residue_rmsd
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Fixture to provide CPU and CUDA devices if available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


class TestStructureMetrics:
    """Tests for structure evaluation metrics."""
    
    def test_rmsd_identical_structures(self, device):
        """Test RMSD calculation with identical structures."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        pred_coords = coords.clone()
        true_coords = coords.clone()
        
        rmsd = compute_rmsd(pred_coords, true_coords)
        assert rmsd.item() < 1e-4, "RMSD should be near zero for identical structures"
    
    def test_rmsd_shifted_structures(self, device):
        """Test RMSD calculation with shifted structures."""
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # Create a copy to manually check for allclose detection
        true_coords = pred_coords.clone()
        
        # Then shift all coordinates by (1,1,1)
        true_coords = true_coords + 1.0
        
        # With alignment, RMSD should be small (< 0.001) if not exactly zero
        rmsd_aligned = compute_rmsd(pred_coords, true_coords, aligned=True)
        assert rmsd_aligned.item() < 0.001, "RMSD should be very small for shifted structures after alignment"
        
        # Without alignment, RMSD should be √3
        rmsd_unaligned = compute_rmsd(pred_coords, true_coords, aligned=False)
        expected_val = np.sqrt(3.0)
        assert abs(rmsd_unaligned.item() - expected_val) < 1e-5
    
    def test_rmsd_rotated_structures(self, device):
        """Test RMSD calculation with rotated structures."""
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # 90-degree rotation around z-axis
        rotation = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        true_coords = torch.matmul(pred_coords, rotation.T)
        
        # With alignment, RMSD should be small
        rmsd_aligned = compute_rmsd(pred_coords, true_coords, aligned=True)
        assert rmsd_aligned.item() < 2.0, "RMSD should be relatively small for rotated structures after alignment"
        
        # Without alignment, RMSD should be non-zero
        rmsd_unaligned = compute_rmsd(pred_coords, true_coords, aligned=False)
        assert rmsd_unaligned.item() > 0.1, "RMSD should be non-zero for rotated structures without alignment"
    
    def test_rmsd_with_mask(self, device):
        """Test RMSD calculation with masked positions."""
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0]  # Outlier that should be masked
        ], device=device)
        
        true_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]  # Different from pred_coords
        ], device=device)
        
        # Mask the outlier position
        mask = torch.tensor([True, True, True, False], device=device)
        
        # With masking, RMSD should be near zero
        rmsd = compute_rmsd(pred_coords, true_coords, mask=mask)
        assert rmsd.item() < 1e-4, "RMSD should be near zero with mask applied"
        
        # Without masking, RMSD should be higher
        rmsd_unmasked = compute_rmsd(pred_coords, true_coords)
        assert rmsd_unmasked.item() > 5.0, "RMSD should be high without mask applied"
    
    def test_rmsd_batched(self, device):
        """Test RMSD calculation with batched input."""
        batch_size = 2
        seq_len = 4
        
        # Create batch of identical structures
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        pred_coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
        true_coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Add error to second batch example
        pred_coords[1, -1] = torch.tensor([1.0, 1.0, 1.0], device=device)
        
        rmsd = compute_rmsd(pred_coords, true_coords)
        assert rmsd.shape == (batch_size,), "RMSD should return a value per batch"
        assert rmsd[0].item() < 1e-4, "First batch RMSD should be near zero"
        assert rmsd[1].item() > 0.1, "Second batch RMSD should be non-zero"
    
    def test_tm_score_identical_structures(self, device):
        """Test TM-score calculation with identical structures."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        pred_coords = coords.clone()
        true_coords = coords.clone()
        
        tm_score = compute_tm_score(pred_coords, true_coords)
        assert tm_score.item() > 0.99, "TM-score should be near 1.0 for identical structures"
    
    def test_tm_score_random_structures(self, device):
        """Test TM-score calculation with random structures."""
        # Generate random coordinates with different seeds
        torch.manual_seed(42)
        pred_coords = torch.randn(10, 3, device=device)
        
        torch.manual_seed(43)
        true_coords = torch.randn(10, 3, device=device)
        
        tm_score = compute_tm_score(pred_coords, true_coords)
        assert tm_score.item() < 0.5, "TM-score should be low for random structures"
    
    def test_tm_score_with_mask(self, device):
        """Test TM-score calculation with masked positions."""
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0]  # Outlier that should be masked
        ], device=device)
        
        true_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]  # Different from pred_coords
        ], device=device)
        
        # Mask the outlier position
        mask = torch.tensor([True, True, True, False], device=device)
        
        # With masking, TM-score should be high
        tm_score = compute_tm_score(pred_coords, true_coords, mask=mask)
        assert tm_score.item() > 0.9, "TM-score should be high with mask applied"
        
        # Without masking, TM-score should be lower
        tm_score_unmasked = compute_tm_score(pred_coords, true_coords)
        assert tm_score_unmasked.item() < 0.5, "TM-score should be low without mask applied"
    
    def test_structure_metrics_combined(self, device):
        """Test computing multiple metrics at once."""
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        pred_coords = coords.clone()
        true_coords = coords.clone()
        
        # Compute both metrics
        metrics = compute_structure_metrics(pred_coords, true_coords)
        
        assert "rmsd" in metrics, "RMSD should be in the results"
        assert "tm_score" in metrics, "TM-score should be in the results"
        assert metrics["rmsd"].item() < 1e-4, "RMSD should be near zero for identical structures"
        assert metrics["tm_score"].item() > 0.99, "TM-score should be near 1.0 for identical structures"
        
        # Test selective metric computation
        rmsd_only = compute_structure_metrics(pred_coords, true_coords, metrics=["rmsd"])
        assert "rmsd" in rmsd_only, "RMSD should be in the results"
        assert "tm_score" not in rmsd_only, "TM-score should not be in the results"
    
    def test_per_residue_rmsd(self, device):
        """Test per-residue RMSD calculation."""
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device)
        
        # Add error to specific residues
        true_coords = pred_coords.clone()
        true_coords[2] = torch.tensor([1.0, 2.0, 0.0], device=device)  # Diagonal error of √2 ≈ 1.414 Å
        
        # Calculate per-residue RMSD with global alignment OFF to test specific residue errors
        per_res_rmsd = compute_per_residue_rmsd(pred_coords, true_coords, aligned=False)
        
        assert per_res_rmsd.shape == (4,), "Should return RMSD for each residue"
        assert per_res_rmsd[0].item() < 1e-4, "RMSD for residue 0 should be near zero (identical)"
        assert per_res_rmsd[1].item() < 1e-4, "RMSD for residue 1 should be near zero (identical)"
        expected_rmsd = np.sqrt(2.0)  # Distance between (0,1,0) and (1,2,0) is √2
        assert abs(per_res_rmsd[2].item() - expected_rmsd) < 1e-5, f"RMSD for residue 2 should be ~{expected_rmsd}"
        assert per_res_rmsd[3].item() < 1e-4, "RMSD for residue 3 should be near zero (identical)"
        
        # Test with window_size > 1
        per_res_rmsd_window = compute_per_residue_rmsd(pred_coords, true_coords, window_size=3)
        assert per_res_rmsd_window.shape == (4,), "Should return RMSD for each residue"
        # Residues near the error should have non-zero RMSD with window
        assert per_res_rmsd_window[1].item() > 0, "RMSD for residue 1 should be non-zero with window"
        assert per_res_rmsd_window[2].item() > 0, "RMSD for residue 2 should be non-zero with window"
    
    def test_handling_4d_coords(self, device):
        """Test handling of 4D coordinate tensors for true_coords."""
        # Regular 3D pred_coords
        pred_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=device).unsqueeze(0)  # Add batch dimension
        
        # Create 4D true_coords with diagonal containing the actual coordinates
        seq_len = 4
        true_coords_4d = torch.zeros((1, seq_len, seq_len, 3), device=device)
        
        # Set diagonal elements to the real coordinates
        for i in range(seq_len):
            true_coords_4d[0, i, i] = pred_coords[0, i]
        
        # Test RMSD calculation with 4D true_coords
        rmsd = compute_rmsd(pred_coords, true_coords_4d)
        assert rmsd.item() < 1e-4, "RMSD should handle 4D true_coords correctly"
        
        # Test TM-score calculation with 4D true_coords
        tm_score = compute_tm_score(pred_coords, true_coords_4d)
        assert tm_score.item() > 0.99, "TM-score should handle 4D true_coords correctly"
        
        # Test per-residue RMSD with 4D true_coords
        per_res_rmsd = compute_per_residue_rmsd(pred_coords, true_coords_4d)
        assert torch.all(per_res_rmsd < 1e-4), "Per-residue RMSD should handle 4D true_coords correctly"