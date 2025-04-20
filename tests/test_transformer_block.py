import pytest
import torch
import numpy as np
from src.models.transformer_block import TransformerBlock


class TestTransformerBlock:
    """Tests for the TransformerBlock module."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'ffn_dim': 512
        }
        transformer_block = TransformerBlock(config)
        
        # Check attributes
        assert transformer_block.residue_dim == 128
        assert transformer_block.pair_dim == 64
        assert transformer_block.num_heads == 4
        assert transformer_block.dropout_rate == 0.1
        assert transformer_block.ffn_dim == 512
        
        # Check that submodules are initialized
        assert hasattr(transformer_block, 'residue_attn_norm')
        assert hasattr(transformer_block, 'residue_attention')
        assert hasattr(transformer_block, 'residue_ffn')
        assert hasattr(transformer_block, 'pair_norm')
        assert hasattr(transformer_block, 'pair_update_mlp')
    
    def test_initialization_dimension_validation(self):
        """Test that initialization validates dimensions."""
        # Invalid config: residue_dim not divisible by num_heads
        config = {
            'residue_embed_dim': 127,  # Not divisible by 4
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.1
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            _ = TransformerBlock(config)
        
        # Check error message
        assert "must be divisible by" in str(excinfo.value)
    
    def test_forward_pass(self):
        """Test the forward pass with mock inputs."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'ffn_dim': 512
        }
        transformer_block = TransformerBlock(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Set some positions as padding
        mask[0, -2:] = False
        
        # Set model to eval mode to disable dropout for deterministic testing
        transformer_block.eval()
        
        # Forward pass
        with torch.no_grad():
            updated_residue_repr, updated_pair_repr = transformer_block(
                residue_repr, pair_repr, mask
            )
        
        # Check shapes
        assert updated_residue_repr.shape == residue_repr.shape
        assert updated_pair_repr.shape == pair_repr.shape
        
        # Check mask handling
        # Masked positions should still be zero
        assert torch.all(updated_residue_repr[0, -2:] == 0)
        
        # Any pair involving a masked position should be zero
        assert torch.all(updated_pair_repr[0, -2:, :] == 0)
        assert torch.all(updated_pair_repr[0, :, -2:] == 0)
    
    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'ffn_dim': 512
        }
        transformer_block = TransformerBlock(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Forward pass on CPU
        transformer_block.eval()
        with torch.no_grad():
            updated_residue_repr, updated_pair_repr = transformer_block(
                residue_repr, pair_repr, mask
            )
        
        # Check shapes
        assert updated_residue_repr.shape == residue_repr.shape
        assert updated_pair_repr.shape == pair_repr.shape
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            transformer_block = transformer_block.to('cuda')
            residue_repr = residue_repr.to('cuda')
            pair_repr = pair_repr.to('cuda')
            mask = mask.to('cuda')
            
            # Forward pass on GPU
            with torch.no_grad():
                updated_residue_repr, updated_pair_repr = transformer_block(
                    residue_repr, pair_repr, mask
                )
            
            # Check that output is on the correct device
            assert updated_residue_repr.device.type == 'cuda'
            assert updated_pair_repr.device.type == 'cuda'
            
            # Check shapes
            assert updated_residue_repr.shape == residue_repr.shape
            assert updated_pair_repr.shape == pair_repr.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.0,  # Disable dropout for deterministic testing
            'ffn_dim': 512
        }
        transformer_block = TransformerBlock(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], requires_grad=True)
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'], requires_grad=True)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Forward pass
        updated_residue_repr, updated_pair_repr = transformer_block(
            residue_repr, pair_repr, mask
        )
        
        # Create a dummy loss and perform backward pass
        loss = updated_residue_repr.sum() + updated_pair_repr.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert residue_repr.grad is not None
        assert pair_repr.grad is not None
        assert not torch.all(residue_repr.grad == 0)
        assert not torch.all(pair_repr.grad == 0)
    
    def test_without_mask(self):
        """Test the forward pass without providing a mask."""
        config = {
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'num_attention_heads': 4,
            'dropout': 0.0,  # Disable dropout for deterministic testing
            'ffn_dim': 512
        }
        transformer_block = TransformerBlock(config)
        
        # Create mock inputs
        batch_size = 2
        seq_len = 10
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Forward pass without mask
        transformer_block.eval()
        with torch.no_grad():
            updated_residue_repr, updated_pair_repr = transformer_block(
                residue_repr, pair_repr
            )
        
        # Check shapes
        assert updated_residue_repr.shape == residue_repr.shape
        assert updated_pair_repr.shape == pair_repr.shape
        
        # Check that forward pass works with different sequence lengths
        seq_len_2 = 15
        residue_repr_2 = torch.rand(batch_size, seq_len_2, config['residue_embed_dim'])
        pair_repr_2 = torch.rand(batch_size, seq_len_2, seq_len_2, config['pair_embed_dim'])
        
        with torch.no_grad():
            updated_residue_repr_2, updated_pair_repr_2 = transformer_block(
                residue_repr_2, pair_repr_2
            )
        
        # Check shapes
        assert updated_residue_repr_2.shape == residue_repr_2.shape
        assert updated_pair_repr_2.shape == pair_repr_2.shape