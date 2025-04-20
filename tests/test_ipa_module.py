import pytest
import torch
import numpy as np
from src.models.ipa_module import IPAModule


class TestIPAModule:
    """Tests for the IPAModule placeholder."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64,
            'num_ipa_iterations': 1
        }
        ipa_module = IPAModule(config)
        
        # Check attributes
        assert ipa_module.residue_dim == 128
        assert ipa_module.pair_dim == 64
        assert ipa_module.ipa_dim == 64
        assert ipa_module.num_iterations == 1
        
        # Check that the projection MLP is initialized
        assert hasattr(ipa_module, 'coord_projection')
        assert isinstance(ipa_module.coord_projection, torch.nn.Sequential)
        
        # Check projection layers dimensions
        layers = list(ipa_module.coord_projection.children())
        assert layers[0].in_features == config['residue_embed_dim']
        assert layers[0].out_features == config['ipa_dim']
        assert layers[2].out_features == 3  # x, y, z coordinates
    
    def test_forward_pass(self):
        """Test the forward pass with mock inputs."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Set some positions as padding
        mask[0, -2:] = False
        
        # Forward pass
        ipa_module.eval()
        with torch.no_grad():
            coords = ipa_module(residue_repr, pair_repr, mask)
        
        # Check shape
        assert coords.shape == (batch_size, seq_len, 3)
        
        # Check mask handling
        # Masked positions should have zero coordinates
        assert torch.all(coords[0, -2:] == 0)
    
    def test_pair_repr_unused(self):
        """Test that pair_repr is genuinely unused in V1."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        
        # Create two different pair representations
        pair_repr1 = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        pair_repr2 = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Ensure they're different
        assert not torch.allclose(pair_repr1, pair_repr2)
        
        # Forward pass with both pair representations
        ipa_module.eval()
        with torch.no_grad():
            coords1 = ipa_module(residue_repr, pair_repr1)
            coords2 = ipa_module(residue_repr, pair_repr2)
        
        # Coordinates should be identical since pair_repr is unused in V1
        assert torch.allclose(coords1, coords2)
    
    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Forward pass on CPU
        ipa_module.eval()
        with torch.no_grad():
            coords_cpu = ipa_module(residue_repr, pair_repr, mask)
        
        # Check shape
        assert coords_cpu.shape == (batch_size, seq_len, 3)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            ipa_module = ipa_module.to('cuda')
            residue_repr = residue_repr.to('cuda')
            pair_repr = pair_repr.to('cuda')
            mask = mask.to('cuda')
            
            # Forward pass on GPU
            with torch.no_grad():
                coords_gpu = ipa_module(residue_repr, pair_repr, mask)
            
            # Check that output is on the correct device
            assert coords_gpu.device.type == 'cuda'
            
            # Check shape
            assert coords_gpu.shape == (batch_size, seq_len, 3)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], requires_grad=True)
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Forward pass
        coords = ipa_module(residue_repr, pair_repr, mask)
        
        # Create a dummy loss and perform backward pass
        loss = coords.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert residue_repr.grad is not None
        assert not torch.all(residue_repr.grad == 0)
    
    def test_initialize_coordinates_method(self):
        """Test the _initialize_coordinates method."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Set some positions as padding
        mask[0, -2:] = False
        
        # Call the method
        coords = ipa_module._initialize_coordinates(residue_repr, mask)
        
        # Check shape
        assert coords.shape == (batch_size, seq_len, 3)
        
        # Check mask handling
        assert torch.all(coords[0, -2:] == 0)
        
        # For V1, _initialize_coordinates should be equivalent to forward
        with torch.no_grad():
            forward_coords = ipa_module(residue_repr, None, mask)
        
        assert torch.allclose(coords, forward_coords)
    
    def test_no_mask(self):
        """Test the forward pass without providing a mask."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'ipa_dim': 64
        }
        ipa_module = IPAModule(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Forward pass without mask
        ipa_module.eval()
        with torch.no_grad():
            coords = ipa_module(residue_repr, pair_repr)
        
        # Check shape
        assert coords.shape == (batch_size, seq_len, 3)
        
        # Check that there are no zeros in the output (since no masking)
        # Note: This test might fail with extremely small probability if 
        # random initialization creates exact zeros - adjust if needed
        assert not torch.all(coords == 0)