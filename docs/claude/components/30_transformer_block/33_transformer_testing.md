# Transformer Block Testing Guide

This document outlines the testing strategy and implementation for the transformer block component (`src/models/transformer_block.py`) of the RNA 3D folding pipeline. Thorough testing is critical to ensure this component correctly processes residue and pair representations while properly handling masking for variable-length sequences.

## Testing Objectives

1. **Validate Shape Transformations**: Verify that the transformer block correctly processes tensors with expected shapes for both residue and pair representations
2. **Ensure Mask Handling**: Confirm proper application of masking in both attention and pair update mechanisms
3. **Verify Residual Connections**: Test that residual connections are properly implemented
4. **Validate Layer Normalization**: Ensure pre-normalization is correctly applied
5. **Check Numerical Stability**: Confirm the component doesn't produce NaN or extreme values
6. **Verify Device Compatibility**: Test correct device handling for both CPU and CUDA
7. **Validate Integration**: Ensure compatibility with embedding outputs and correct input to subsequent components
8. **Test Performance**: Evaluate memory efficiency and speed with realistic sequence lengths

## Test Structure

Create a comprehensive test file `tests/test_transformer_block.py` with these major test groups:

1. Initialization Tests
2. Forward Pass Tests
3. Masking Tests
4. Shape Handling Tests
5. Integration Tests
6. Edge Case Tests
7. Performance Tests (Optional)

## 1. Initialization Tests

```python
import pytest
import torch
import torch.nn as nn
from src.models.transformer_block import TransformerBlock

def test_transformer_block_initialization():
    """Test TransformerBlock initialization with valid parameters."""
    # Valid config
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.1,
        'ffn_dim': 512
    }
    
    # Should initialize without errors
    transformer_block = TransformerBlock(config)
    
    # Check that parameters were set correctly
    assert transformer_block.residue_dim == 128
    assert transformer_block.pair_dim == 64
    assert transformer_block.num_heads == 4
    assert transformer_block.dropout_rate == 0.1
    assert transformer_block.ffn_dim == 512
    
    # Check that necessary components were initialized
    assert isinstance(transformer_block.residue_attention, nn.MultiheadAttention)
    assert isinstance(transformer_block.residue_attn_norm, nn.LayerNorm)
    assert isinstance(transformer_block.residue_ffn_norm, nn.LayerNorm)
    assert isinstance(transformer_block.pair_norm, nn.LayerNorm)
    assert isinstance(transformer_block.pair_update_mlp, nn.Sequential)

def test_transformer_block_initialization_invalid_dimensions():
    """Test TransformerBlock initialization with invalid dimensions."""
    # Invalid config: residue_embed_dim not divisible by num_attention_heads
    config = {
        'residue_embed_dim': 127,  # Not divisible by 4
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.1
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        transformer_block = TransformerBlock(config)
    
    # Check error message
    assert "residue_embed_dim (127) must be divisible by num_attention_heads (4)" in str(excinfo.value)

def test_transformer_block_initialization_defaults():
    """Test TransformerBlock initialization with default values for optional parameters."""
    # Minimal config with only required parameters
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4
    }
    
    # Should use default values for missing parameters
    transformer_block = TransformerBlock(config)
    
    # Check default values
    assert transformer_block.dropout_rate == 0.1  # Default dropout
    assert transformer_block.ffn_dim == 128 * 4   # Default ffn_dim = residue_dim * 4
```

## 2. Forward Pass Tests

```python
def test_transformer_block_forward_shapes():
    """Test that forward pass maintains correct tensor shapes."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.1
    }
    transformer_block = TransformerBlock(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # Forward pass
    out_residue_repr, out_pair_repr = transformer_block(residue_repr, pair_repr)
    
    # Check output shapes
    assert out_residue_repr.shape == residue_repr.shape
    assert out_pair_repr.shape == pair_repr.shape

def test_transformer_residue_update():
    """Test the residue update path in isolation."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    
    # Set model to eval mode to disable dropout
    transformer_block.eval()
    
    # Apply residue update only
    updated_residue = transformer_block._update_residue_repr(residue_repr)
    
    # Check shape
    assert updated_residue.shape == residue_repr.shape
    
    # Make sure the update actually changed the representation
    assert not torch.allclose(updated_residue, residue_repr)

def test_transformer_pair_update():
    """Test the pair update path in isolation."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # Set model to eval mode to disable dropout
    transformer_block.eval()
    
    # Apply pair update only
    updated_pair = transformer_block._update_pair_repr(residue_repr, pair_repr)
    
    # Check shape
    assert updated_pair.shape == pair_repr.shape
    
    # Make sure the update actually changed the representation
    assert not torch.allclose(updated_pair, pair_repr)

def test_transformer_pre_normalization():
    """Test that pre-normalization is correctly applied."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create dummy inputs with non-zero mean and non-unit variance
    batch_size = 2
    seq_len = 10
    residue_repr = torch.randn(batch_size, seq_len, config['residue_embed_dim']) * 5 + 2
    pair_repr = torch.randn(batch_size, seq_len, seq_len, config['pair_embed_dim']) * 3 - 1
    
    # Get normalized representations directly
    residue_norm = transformer_block.residue_attn_norm(residue_repr)
    pair_norm = transformer_block.pair_norm(pair_repr)
    
    # Check normalization (approximately zero mean, unit variance across feature dimension)
    residue_mean = residue_norm.mean(dim=-1)
    residue_var = residue_norm.var(dim=-1)
    assert torch.allclose(residue_mean, torch.zeros_like(residue_mean), atol=1e-6)
    assert torch.allclose(residue_var, torch.ones_like(residue_var), atol=1e-6)
    
    pair_mean = pair_norm.mean(dim=-1)
    pair_var = pair_norm.var(dim=-1)
    assert torch.allclose(pair_mean, torch.zeros_like(pair_mean), atol=1e-6)
    assert torch.allclose(pair_var, torch.ones_like(pair_var), atol=1e-6)

def test_transformer_residual_connections():
    """Test that residual connections are properly implemented."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # Set specific modules to identity to isolate residual connections
    # Store original layers
    original_attn = transformer_block.residue_attention
    original_ffn = transformer_block.residue_ffn
    original_pair_mlp = transformer_block.pair_update_mlp
    
    # Replace with identity (zero) layers
    class ZeroModule(nn.Module):
        def forward(self, *args, **kwargs):
            if len(args) > 0:
                return torch.zeros_like(args[0])
            else:
                return torch.zeros_like(kwargs['query'])
    
    transformer_block.residue_attention = ZeroModule()
    transformer_block.residue_ffn = ZeroModule()
    transformer_block.pair_update_mlp = ZeroModule()
    
    # Forward pass with zero updates
    updated_residue, updated_pair = transformer_block(residue_repr, pair_repr)
    
    # With zero updates and no dropout, the residual connection should return the inputs exactly
    assert torch.allclose(updated_residue, residue_repr)
    assert torch.allclose(updated_pair, pair_repr)
    
    # Restore original layers for cleanup
    transformer_block.residue_attention = original_attn
    transformer_block.residue_ffn = original_ffn
    transformer_block.pair_update_mlp = original_pair_mlp
```

## 3. Masking Tests

```python
def test_transformer_attention_masking():
    """Test that attention properly handles masked positions."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create inputs with padding
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # Create mask (mask out last 3 positions in first sequence)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -3:] = False
    
    # Set model to eval mode to disable dropout
    transformer_block.eval()
    
    # Forward pass with mask
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, mask)
    
    # Masked positions in first sequence should remain zero in output
    # To test this, zero out these positions in the input and compare
    masked_residue_input = residue_repr.clone()
    masked_residue_input[0, -3:] = 0
    
    # Apply forward pass with zeroed input but no mask
    residue_out_zeroed, _ = transformer_block(masked_residue_input, pair_repr)
    
    # Check if masked outputs and zeroed-input outputs match in the masked regions
    assert torch.allclose(residue_out[0, -3:], residue_out_zeroed[0, -3:])
    
    # Verify masking for pair representation
    # Masked pairs should be all positions where either i or j is masked
    assert torch.all(pair_out[0, -3:, :] == 0)  # Last 3 rows are zero
    assert torch.all(pair_out[0, :, -3:] == 0)  # Last 3 columns are zero

def test_transformer_pair_mask_creation():
    """Test the creation of 2D pair masks from 1D sequence masks."""
    # Create sequence mask
    batch_size = 2
    seq_len = 5
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Mask positions: seq1[3:5], seq2[4:5]
    mask[0, 3:] = False
    mask[1, 4:] = False
    
    # Create pair mask (in the same way the transformer block would)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
    
    # Check dimensions
    assert pair_mask.shape == (batch_size, seq_len, seq_len)
    
    # Check first sequence
    # Only positions where both i and j are valid (< 3) should be True
    for i in range(seq_len):
        for j in range(seq_len):
            if i < 3 and j < 3:
                assert pair_mask[0, i, j]
            else:
                assert not pair_mask[0, i, j]
    
    # Check second sequence
    # Only positions where both i and j are valid (< 4) should be True
    for i in range(seq_len):
        for j in range(seq_len):
            if i < 4 and j < 4:
                assert pair_mask[1, i, j]
            else:
                assert not pair_mask[1, i, j]

def test_transformer_masked_attention_weights():
    """Test that attention weights respect the mask (advanced test)."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create inputs
    batch_size = 2
    seq_len = 6
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    
    # Create mask (mask out last 2 positions in first sequence)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False
    
    # Set model to eval mode to disable dropout
    transformer_block.eval()
    
    # Apply attention with need_weights=True
    residue_norm = transformer_block.residue_attn_norm(residue_repr)
    key_padding_mask = ~mask if mask is not None else None
    _, attn_weights = transformer_block.residue_attention(
        query=residue_norm,
        key=residue_norm,
        value=residue_norm,
        key_padding_mask=key_padding_mask,
        need_weights=True
    )
    
    # Check shape of attention weights
    assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    # Verify that masked positions have zero attention weight
    # In first sequence, last 2 positions should not be attended to
    assert torch.all(attn_weights[0, :, -2:] == 0)
    
    # The sum of attention weights across the key dimension should be 1 for unmasked positions
    # and 0 for masked positions
    attn_sum = attn_weights.sum(dim=2)  # Sum across key dimension
    for b in range(batch_size):
        for i in range(seq_len):
            if mask[b, i]:
                assert torch.isclose(attn_sum[b, i], torch.tensor(1.0))
            else:
                assert torch.isclose(attn_sum[b, i], torch.tensor(0.0))
```

## 4. Shape Handling Tests

```python
def test_transformer_variable_sequence_lengths():
    """Test transformer with different sequence lengths in the same batch."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create batch with different effective sequence lengths
    batch_size = 3
    max_seq_len = 10
    residue_repr = torch.rand(batch_size, max_seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, max_seq_len, max_seq_len, config['pair_embed_dim'])
    
    # Create mask for variable lengths (seq1=10, seq2=7, seq3=5)
    mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    mask[1, 7:] = False  # seq2 has length 7
    mask[2, 5:] = False  # seq3 has length 5
    
    # Forward pass
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, mask)
    
    # Check output shapes
    assert residue_out.shape == (batch_size, max_seq_len, config['residue_embed_dim'])
    assert pair_out.shape == (batch_size, max_seq_len, max_seq_len, config['pair_embed_dim'])
    
    # Padded positions should be zero in output
    assert torch.all(residue_out[1, 7:] == 0)
    assert torch.all(residue_out[2, 5:] == 0)
    
    # Check pair masking
    assert torch.all(pair_out[1, 7:, :] == 0)
    assert torch.all(pair_out[1, :, 7:] == 0)
    assert torch.all(pair_out[2, 5:, :] == 0)
    assert torch.all(pair_out[2, :, 5:] == 0)

def test_transformer_extreme_sequence_lengths():
    """Test transformer with very short and very long sequences."""
    # Create transformer block
    config = {
        'residue_embed_dim': 64,  # Smaller dim for memory efficiency in testing
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Test very short sequence (length 1)
    batch_size = 1
    short_seq_len = 1
    short_residue = torch.rand(batch_size, short_seq_len, config['residue_embed_dim'])
    short_pair = torch.rand(batch_size, short_seq_len, short_seq_len, config['pair_embed_dim'])
    
    # Forward pass with short sequence
    short_res_out, short_pair_out = transformer_block(short_residue, short_pair)
    
    # Check shapes
    assert short_res_out.shape == short_residue.shape
    assert short_pair_out.shape == short_pair.shape
    
    # Test long sequence (if memory allows)
    try:
        long_seq_len = 200
        long_residue = torch.rand(batch_size, long_seq_len, config['residue_embed_dim'])
        long_pair = torch.rand(batch_size, long_seq_len, long_seq_len, config['pair_embed_dim'])
        
        # Forward pass with long sequence
        long_res_out, long_pair_out = transformer_block(long_residue, long_pair)
        
        # Check shapes
        assert long_res_out.shape == long_residue.shape
        assert long_pair_out.shape == long_pair.shape
    except RuntimeError as e:
        # If OOM error occurs, note that it's memory-related, not a logic error
        if "CUDA out of memory" in str(e):
            pytest.skip("Skipping long sequence test due to CUDA memory constraints")
        else:
            raise  # Re-raise if it's not a memory error

def test_transformer_pair_outer_product_shapes():
    """Test the shape handling in the pair update's outer product operation."""
    # Create transformer block
    config = {
        'residue_embed_dim': 32,
        'pair_embed_dim': 16,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create inputs of different sequence lengths
    batch_size = 2
    lengths = [3, 5]  # Test different sequence lengths
    
    for seq_len in lengths:
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Extract _update_pair_repr method to test in isolation
        updated_pair = transformer_block._update_pair_repr(residue_repr, pair_repr)
        
        # Check shape of updated pair representation
        assert updated_pair.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Manually compute expanded residue representations
        h_i = residue_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)
        h_j = residue_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Check shapes of expanded tensors
        assert h_i.shape == (batch_size, seq_len, seq_len, config['residue_embed_dim'])
        assert h_j.shape == (batch_size, seq_len, seq_len, config['residue_embed_dim'])
```

## 5. Integration Tests

```python
def test_transformer_integration_with_embeddings():
    """Test integration of transformer block with embedding outputs."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create mock embedding outputs
    batch_size = 2
    seq_len = 10
    embedding_outputs = {
        'residue_repr': torch.rand(batch_size, seq_len, config['residue_embed_dim']),
        'pair_repr': torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim']),
    }
    
    # Create mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False  # Mask last 2 positions in first sequence
    
    # Process through transformer block
    residue_out, pair_out = transformer_block(
        embedding_outputs['residue_repr'],
        embedding_outputs['pair_repr'],
        mask
    )
    
    # Check shapes match
    assert residue_out.shape == embedding_outputs['residue_repr'].shape
    assert pair_out.shape == embedding_outputs['pair_repr'].shape
    
    # Verify masking
    assert torch.all(residue_out[0, -2:] == 0)

def test_transformer_integration_with_ipa_placeholder():
    """Test integration with IPA module placeholder."""
    # Create transformer block
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create mock IPA placeholder
    class IPAPlaceholder(nn.Module):
        def __init__(self, residue_dim):
            super().__init__()
            self.coord_projection = nn.Linear(residue_dim, 3)
            
        def forward(self, residue_repr, pair_repr=None, mask=None):
            # Simple projection to coordinates
            coords = self.coord_projection(residue_repr)
            if mask is not None:
                coords = coords * mask.unsqueeze(-1)
            return coords
    
    ipa_module = IPAPlaceholder(config['residue_embed_dim'])
    
    # Create inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False  # Mask last 2 positions in first sequence
    
    # Process through transformer block
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, mask)
    
    # Process through IPA module
    coords = ipa_module(residue_out, pair_out, mask)
    
    # Check coordinate shape
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Verify masked positions have zero coordinates
    assert torch.all(coords[0, -2:] == 0)

def test_transformer_stacking():
    """Test stacking multiple transformer blocks."""
    # Create config
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    
    # Create stack of transformer blocks
    num_blocks = 3
    transformer_blocks = nn.ModuleList([
        TransformerBlock(config) for _ in range(num_blocks)
    ])
    
    # Create inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False  # Mask last 2 positions in first sequence
    
    # Process through transformer blocks sequentially
    for block in transformer_blocks:
        residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
    
    # Check output shapes
    assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
    assert pair_repr.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # Verify masking is preserved across blocks
    assert torch.all(residue_repr[0, -2:] == 0)
    assert torch.all(pair_repr[0, -2:, :] == 0)
    assert torch.all(pair_repr[0, :, -2:] == 0)
```

## 6. Edge Case Tests

```python
def test_transformer_empty_mask():
    """Test transformer with completely masked sequences."""
    # Create transformer block
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create batch with one sequence completely masked
    batch_size = 2
    seq_len = 5
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # First sequence fully masked, second sequence normal
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[1, :] = True
    
    # Forward pass with this extreme mask
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, mask)
    
    # First sequence should be all zeros
    assert torch.all(residue_out[0] == 0)
    assert torch.all(pair_out[0] == 0)
    
    # Second sequence should be processed normally
    assert not torch.all(residue_out[1] == 0)
    assert not torch.all(pair_out[1] == 0)

def test_transformer_numerical_stability():
    """Test transformer with extreme input values."""
    # Create transformer block
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create inputs with extreme values
    batch_size = 2
    seq_len = 5
    
    # Test with very large values
    large_values = 1e6
    residue_repr = torch.ones(batch_size, seq_len, config['residue_embed_dim']) * large_values
    pair_repr = torch.ones(batch_size, seq_len, seq_len, config['pair_embed_dim']) * large_values
    
    # Forward pass with large values
    residue_out_large, pair_out_large = transformer_block(residue_repr, pair_repr)
    
    # Check for NaNs or infinities
    assert not torch.isnan(residue_out_large).any()
    assert not torch.isinf(residue_out_large).any()
    assert not torch.isnan(pair_out_large).any()
    assert not torch.isinf(pair_out_large).any()
    
    # Test with very small values
    small_values = 1e-6
    residue_repr = torch.ones(batch_size, seq_len, config['residue_embed_dim']) * small_values
    pair_repr = torch.ones(batch_size, seq_len, seq_len, config['pair_embed_dim']) * small_values
    
    # Forward pass with small values
    residue_out_small, pair_out_small = transformer_block(residue_repr, pair_repr)
    
    # Check for NaNs or infinities
    assert not torch.isnan(residue_out_small).any()
    assert not torch.isinf(residue_out_small).any()
    assert not torch.isnan(pair_out_small).any()
    assert not torch.isinf(pair_out_small).any()

def test_transformer_shape_mismatch_handling():
    """Test how transformer handles shape mismatches in inputs."""
    # Create transformer block
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0
    }
    transformer_block = TransformerBlock(config)
    
    # Create mismatched inputs
    batch_size = 2
    seq_len_1 = 10
    seq_len_2 = 8  # Different sequence length
    
    residue_repr = torch.rand(batch_size, seq_len_1, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len_2, seq_len_2, config['pair_embed_dim'])  # Mismatched
    
    # This should raise an error due to incompatible shapes
    with pytest.raises(Exception) as excinfo:
        residue_out, pair_out = transformer_block(residue_repr, pair_repr)
    
    # Check the actual error that was raised - it should be related to shape
    error_msg = str(excinfo.value)
    assert "shape" in error_msg.lower() or "size" in error_msg.lower()
```

## 7. Device and Performance Tests

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_transformer_device_compatibility(device):
    """Test transformer works correctly on different devices."""
    # Skip if CUDA requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")
        
    # Create actual device
    device = torch.device(device)
    
    # Create transformer block on device
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0
    }
    transformer_block = TransformerBlock(config).to(device)
    
    # Create inputs on device
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], device=device)
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'], device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, -2:] = False
    
    # Forward pass
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, mask)
    
    # Check device of outputs
    assert residue_out.device == device
    assert pair_out.device == device
    
    # Basic shape check
    assert residue_out.shape == residue_repr.shape
    assert pair_out.shape == pair_repr.shape

@pytest.mark.performance
def test_transformer_memory_usage():
    """Test memory usage of the transformer block with different sequence lengths."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for memory testing")
    
    # Create smaller config for memory testing
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0
    }
    
    # Track memory usage for different sequence lengths
    seq_lengths = [10, 50, 100]  # Add longer if your GPU has enough memory
    batch_size = 1
    
    results = {}
    
    for seq_len in seq_lengths:
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create model and inputs
        transformer_block = TransformerBlock(config).cuda()
        residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim']).cuda()
        pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim']).cuda()
        
        # Record starting memory
        start_mem = torch.cuda.memory_allocated()
        
        # Forward pass
        residue_out, pair_out = transformer_block(residue_repr, pair_repr)
        
        # Record final and peak memory
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        
        # Record memory metrics
        results[seq_len] = {
            'start_mem_mb': start_mem / 1024**2,
            'end_mem_mb': end_mem / 1024**2,
            'peak_mem_mb': peak_mem / 1024**2,
            'used_mem_mb': (end_mem - start_mem) / 1024**2
        }
        
        print(f"Sequence length {seq_len}: Used memory {results[seq_len]['used_mem_mb']:.2f} MB, "
              f"Peak memory {results[seq_len]['peak_mem_mb']:.2f} MB")
    
    # Check if memory usage scales reasonable with sequence length
    # The pair representation should dominate with O(L²) complexity
    if len(seq_lengths) >= 2:
        ratio_seq = seq_lengths[-1] / seq_lengths[0]
        ratio_mem = results[seq_lengths[-1]]['used_mem_mb'] / results[seq_lengths[0]]['used_mem_mb']
        
        # Memory should scale roughly with sequence length squared
        # Allow some leeway in the expected scaling factor
        expected_ratio = ratio_seq**2
        tolerance = 0.5  # Allow 50% deviation from theoretical scaling
        
        print(f"Memory scaling: L ratio = {ratio_seq:.2f}, "
              f"Mem ratio = {ratio_mem:.2f}, Expected ~ {expected_ratio:.2f}")
        
        assert ratio_mem < expected_ratio * (1 + tolerance), \
            f"Memory scaling higher than expected: {ratio_mem:.2f} vs {expected_ratio:.2f}"

@pytest.mark.performance
def test_transformer_speed():
    """Test forward pass speed of the transformer block."""
    # Create config
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'num_attention_heads': 4,
        'dropout': 0.0
    }
    
    # Create transformer block
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer_block = TransformerBlock(config).to(device)
    
    # Create inputs
    batch_size = 1
    seq_len = 100
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim']).to(device)
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim']).to(device)
    
    # Warm-up pass
    _ = transformer_block(residue_repr, pair_repr)
    
    # Time multiple passes
    import time
    num_runs = 10
    
    start_time = time.time()
    for _ in range(num_runs):
        _ = transformer_block(residue_repr, pair_repr)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Make sure all operations are complete
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_runs
    
    print(f"Average forward pass time for sequence length {seq_len}: {avg_time:.4f} seconds")
    
    # No specific assertion, just informational
```

## 8. Gradient Flow Tests

```python
def test_transformer_gradient_flow():
    """Test that gradients flow correctly through the transformer block."""
    # Create transformer block
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.0  # Set to 0 for deterministic testing
    }
    transformer_block = TransformerBlock(config)
    
    # Create inputs that require gradients
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], requires_grad=True)
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'], requires_grad=True)
    
    # Forward pass
    residue_out, pair_out = transformer_block(residue_repr, pair_repr)
    
    # Create dummy loss and backpropagate
    loss = residue_out.sum() + pair_out.sum()
    loss.backward()
    
    # Check that gradients were computed for inputs
    assert residue_repr.grad is not None
    assert pair_repr.grad is not None
    
    # Check that all parameters received gradients
    params_without_grad = []
    for name, param in transformer_block.named_parameters():
        if param.grad is None:
            params_without_grad.append(name)
    
    assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"
    
    # Check that gradients are not all zeros
    all_zeros = True
    for name, param in transformer_block.named_parameters():
        if not torch.all(param.grad == 0):
            all_zeros = False
            break
    
    assert not all_zeros, "All parameter gradients are zero"
```

## Running the Tests

To run the tests, execute:

```bash
# Run all transformer tests
pytest -xvs tests/test_transformer_block.py

# Run specific test groups
pytest -xvs tests/test_transformer_block.py::test_transformer_block_initialization
pytest -xvs tests/test_transformer_block.py::test_transformer_attention_masking

# Skip performance tests (which might be slow)
pytest -xvs tests/test_transformer_block.py -k "not performance"

# Run device-specific tests
pytest -xvs tests/test_transformer_block.py::test_transformer_device_compatibility
```

## Common Test Failures and Remediation

| Failure Pattern | Likely Cause | Remediation |
|-----------------|--------------|-------------|
| Dimension not divisible by num_heads | `residue_embed_dim` not divisible by `num_attention_heads` | Ensure `residue_embed_dim` is divisible by `num_attention_heads` |
| NaNs in output tensors | Numerical instability in attention or MLP layers | Check normalization, add epsilon values, adjust initialization |
| Zero outputs with masking | Mask inversion error in attention | Remember to invert mask for PyTorch's MultiheadAttention (`key_padding_mask` is `True` for masked positions) |
| Shape mismatch | Inconsistent handling of tensor dimensions | Double-check expansion operations, verify tensor shapes at each step |
| Device mismatch | Tensors or modules on different devices | Move all tensors and modules to the same device |
| OOM errors | Large sequence lengths causing memory issues | Reduce batch size, reduce model dimensions, or use gradient checkpointing |
| Zero gradients | Missing connections in forward pass | Check residual connections, ensure correct gradient path |

## Test Coverage Goals

Aim for at least 90% code coverage for the transformer block component, ensuring:

1. All public methods and initialization logic are tested
2. Both residue and pair update paths are thoroughly tested
3. Masking logic is verified for both attention and pair updates
4. Shape handling is validated for various sequence lengths
5. Integration with adjacent components is tested
6. Edge cases are explicitly tested
7. Numerical stability and performance are evaluated

## Next Steps

After implementing and verifying the transformer block component through these tests:

1. Fix any identified issues
2. Integrate with the embeddings component (already tested)
3. Connect to the IPA module placeholder
4. Build the complete model architecture

Remember that transformer block is a critical component of the architecture, and proper testing ensures it correctly processes and updates both residue and pair representations while handling variable sequence lengths properly.
