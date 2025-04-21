"""
Unit tests for the RNA folding model.
"""

import pytest
import torch
import numpy as np

from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_combined_loss


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Fixture to provide CPU and CUDA devices if available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    return {
        "num_blocks": 2,
        "residue_embed_dim": 64,
        "pair_embed_dim": 32,
        "num_attention_heads": 2,
        "dropout": 0.1,
        "ffn_dim": 128,
        "max_relative_position": 16,
        "seq_embed_dim": 16,
        "num_embeddings": 5,
        "padding_idx": 4,
        "ipa_dim": 32,
        "max_len": 100,
        "use_conservation": False,
    }


@pytest.fixture
def batch_data(device, batch_size=2, seq_len=10):
    """Fixture for creating a batch of test data."""
    # Create a batch of random data
    sequence_int = torch.randint(0, 4, (batch_size, seq_len), device=device)
    dihedral_features = torch.randn(batch_size, seq_len, 4, device=device)
    pairing_probs = torch.rand(batch_size, seq_len, seq_len, device=device)
    positional_entropy = torch.rand(batch_size, seq_len, device=device)
    accessibility = torch.rand(batch_size, seq_len, device=device)
    coupling_matrix = torch.rand(batch_size, seq_len, seq_len, device=device)
    coordinates = torch.randn(batch_size, seq_len, 3, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False  # Mask last 2 positions in first sequence
    
    return {
        "sequence_int": sequence_int,
        "dihedral_features": dihedral_features,
        "pairing_probs": pairing_probs,
        "positional_entropy": positional_entropy,
        "accessibility": accessibility,
        "coupling_matrix": coupling_matrix,
        "coordinates": coordinates,
        "mask": mask,
    }


class TestRNAFoldingModel:
    """Tests for the RNAFoldingModel class."""
    
    def test_initialization(self, model_config):
        """Test model initialization."""
        model = RNAFoldingModel(model_config)
        
        assert model.num_blocks == model_config["num_blocks"]
        assert model.residue_dim == model_config["residue_embed_dim"]
        assert model.pair_dim == model_config["pair_embed_dim"]
        assert isinstance(model.embedding_module, torch.nn.Module)
        assert len(model.transformer_blocks) == model_config["num_blocks"]
        assert isinstance(model.ipa_module, torch.nn.Module)
        assert isinstance(model.confidence_head, torch.nn.Sequential)
        assert isinstance(model.angle_head, torch.nn.Sequential)
    
    def test_forward_shapes(self, model_config, batch_data, device):
        """Test that the forward pass produces tensors with the correct shapes."""
        model = RNAFoldingModel(model_config).to(device)
        
        # Move batch data to the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Check that all expected outputs are present
        assert "pred_coords" in outputs
        assert "pred_confidence" in outputs
        assert "pred_angles" in outputs
        
        # Check shapes
        batch_size, seq_len = batch["sequence_int"].shape
        assert outputs["pred_coords"].shape == (batch_size, seq_len, 3)
        assert outputs["pred_confidence"].shape == (batch_size, seq_len)
        assert outputs["pred_angles"].shape == (batch_size, seq_len, 4)
        
        # Check device (comparing device types since indices might differ)
        assert outputs["pred_coords"].device.type == device.type
        assert outputs["pred_confidence"].device.type == device.type
        assert outputs["pred_angles"].device.type == device.type
    
    def test_mask_propagation(self, model_config, batch_data, device):
        """Test that the masking is properly propagated through the model."""
        model = RNAFoldingModel(model_config).to(device)
        
        # Move batch data to the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Check that masked positions have zero values
        mask = batch["mask"]
        assert torch.all(outputs["pred_coords"][~mask] == 0)
        assert torch.all(outputs["pred_confidence"][~mask] == 0)
        assert torch.all(outputs["pred_angles"][~mask] == 0)
    
    def test_input_validation(self, model_config, batch_data, device):
        """Test that the model properly validates input."""
        model = RNAFoldingModel(model_config).to(device)
        
        # Move batch data to the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        
        # Try removing a required key
        incomplete_batch = {k: v for k, v in batch.items() if k != "sequence_int"}
        
        with pytest.raises(ValueError):
            model(incomplete_batch)
    
    def test_end_to_end(self, model_config, batch_data, device):
        """Test end-to-end forward and loss computation."""
        model = RNAFoldingModel(model_config).to(device)
        
        # Move batch data to the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss_weights = {"fape": 1.0, "confidence": 0.1, "angle": 0.5}
        total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)
        
        # Check that loss components exist
        assert "fape" in loss_components
        assert "confidence" in loss_components
        assert "angle" in loss_components
        
        # Check that all loss values are finite
        assert torch.isfinite(total_loss)
        assert torch.isfinite(loss_components["fape"])
        assert torch.isfinite(loss_components["confidence"])
        assert torch.isfinite(loss_components["angle"])
        
        # Check that total loss is the weighted sum of components
        expected_total = (
            loss_weights["fape"] * loss_components["fape"] +
            loss_weights["confidence"] * loss_components["confidence"] +
            loss_weights["angle"] * loss_components["angle"]
        )
        assert torch.isclose(total_loss, expected_total, atol=1e-6)
    
    def test_gradient_flow(self, model_config, batch_data, device):
        """Test that gradients flow properly through the model."""
        model = RNAFoldingModel(model_config).to(device)
        
        # Move batch data to the device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss_weights = {"fape": 1.0, "confidence": 0.1, "angle": 0.5}
        total_loss, _ = compute_combined_loss(outputs, batch, loss_weights)
        
        # Check that gradients flow
        total_loss.backward()
        
        # Check that embeddings have gradients
        assert model.embedding_module.residue_projection.weight.grad is not None
        assert torch.any(model.embedding_module.residue_projection.weight.grad != 0)
        
        # Check that transformer blocks have gradients
        assert model.transformer_blocks[0].residue_attention.in_proj_weight.grad is not None
        assert torch.any(model.transformer_blocks[0].residue_attention.in_proj_weight.grad != 0)
        
        # Check that IPA module has gradients
        assert model.ipa_module.coord_projection[0].weight.grad is not None
        assert torch.any(model.ipa_module.coord_projection[0].weight.grad != 0)
        
        # Check that prediction heads have gradients
        assert model.confidence_head[0].weight.grad is not None
        assert torch.any(model.confidence_head[0].weight.grad != 0)
        assert model.angle_head[0].weight.grad is not None
        assert torch.any(model.angle_head[0].weight.grad != 0)