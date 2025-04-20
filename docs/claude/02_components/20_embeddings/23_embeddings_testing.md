# Embeddings Testing Guide

This document outlines the testing strategy and implementation for the embeddings component (`src/models/embeddings.py`) of the RNA 3D folding pipeline. Thorough testing of this component is critical as it forms the foundation of the model's ability to process RNA sequence and feature data.

## Testing Objectives

1. **Validate Shape Transformations**: Verify that all embedding modules produce outputs with the expected tensor shapes
2. **Ensure Numerical Stability**: Confirm that embeddings don't produce NaN or extreme values
3. **Verify Device Compatibility**: Test that all embedding operations work correctly on both CPU and CUDA devices
4. **Validate Masking Behavior**: Ensure proper handling of padding and masking
5. **Test Integration**: Verify that embedding outputs can be properly consumed by downstream components

## Test Structure

Create a comprehensive test file `tests/test_embeddings.py` with these major test groups:

1. Module-Specific Tests
2. Integration Tests
3. Edge Case Tests
4. Performance Tests (Optional)

## 1. Module-Specific Tests

### 1.1 Test `SequenceEmbedding`

```python
import pytest
import torch
from src.models.embeddings import SequenceEmbedding

def test_sequence_embedding_shapes():
    """Test output shapes for SequenceEmbedding."""
    # Setup
    batch_size = 2
    seq_len = 10
    embedding_dim = 32
    num_embeddings = 5  # A, C, G, U, N/padding
    
    # Create input
    sequence_int = torch.randint(0, num_embeddings, (batch_size, seq_len))
    
    # Create embedding layer
    embedding = SequenceEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # Get output
    output = embedding(sequence_int)
    
    # Verify shape
    assert output.shape == (batch_size, seq_len, embedding_dim)
    
    # Test single example (batch_size=1)
    sequence_int_single = torch.randint(0, num_embeddings, (1, seq_len))
    output_single = embedding(sequence_int_single)
    assert output_single.shape == (1, seq_len, embedding_dim)
    
    # Test variable sequence length
    sequence_int_var = torch.randint(0, num_embeddings, (batch_size, seq_len + 5))
    output_var = embedding(sequence_int_var)
    assert output_var.shape == (batch_size, seq_len + 5, embedding_dim)

def test_sequence_embedding_padding():
    """Test padding behavior in SequenceEmbedding."""
    # Setup
    batch_size = 2
    seq_len = 10
    embedding_dim = 32
    num_embeddings = 5
    padding_idx = 0  # Assuming 0 is padding
    
    # Create embedding layer
    embedding = SequenceEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # Create sequence with padding
    sequence_int = torch.ones((batch_size, seq_len), dtype=torch.long)
    sequence_int[:, -3:] = padding_idx  # Set last 3 positions as padding
    
    # Get output
    output = embedding(sequence_int)
    
    # Verify padding positions have zeros
    padding_embeddings = output[:, -3:, :]
    non_padding_embeddings = output[:, :-3, :]
    
    # Padding values should be all zeros
    assert torch.all(padding_embeddings == 0)
    
    # Non-padding values should not be all zeros
    assert not torch.all(non_padding_embeddings == 0)

def test_sequence_embedding_values():
    """Test that the same nucleotide maps to the same embedding vector."""
    # Setup
    embedding_dim = 16
    num_embeddings = 5
    
    # Create two identical sequences
    seq1 = torch.tensor([[1, 2, 3, 0, 4]])
    seq2 = torch.tensor([[1, 2, 3, 0, 4]])
    
    # Create embedding
    embedding = SequenceEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # Get outputs
    out1 = embedding(seq1)
    out2 = embedding(seq2)
    
    # The outputs should be identical
    assert torch.all(out1 == out2)
    
    # Check specific positions
    # Nucleotide 1 in position 0 should have the same embedding in both outputs
    assert torch.all(out1[0, 0] == out2[0, 0])
    
    # Different nucleotides should have different embeddings
    # Compare nucleotide 1 vs 2
    assert not torch.all(out1[0, 0] == out1[0, 1])

def test_sequence_embedding_grad_flow():
    """Test gradient flow through the sequence embedding."""
    # Setup
    embedding_dim = 16
    num_embeddings = 5
    
    # Create sequence
    sequence_int = torch.tensor([[1, 2, 3, 0, 4]])
    
    # Create embedding with requires_grad=True
    embedding = SequenceEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # Get output
    output = embedding(sequence_int)
    
    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    for name, param in embedding.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0)
```

### 1.2 Test `PositionalEncoding`

```python
import math
import pytest
import torch
from src.models.embeddings import PositionalEncoding

def test_positional_encoding_shapes():
    """Test output shapes for PositionalEncoding."""
    # Setup
    embed_dim = 128
    max_len = 500
    
    # Create encoding layer
    pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
    
    # Test with different sequence lengths
    for seq_len in [10, 50, 100, 500]:
        # Get output
        output = pos_encoding(seq_len=seq_len)
        
        # Verify shape
        assert output.shape == (1, seq_len, embed_dim)
    
    # Test with tensor input to determine sequence length
    x = torch.zeros(2, 35, embed_dim)
    output = pos_encoding(x)
    assert output.shape == (1, 35, embed_dim)

def test_positional_encoding_values():
    """Test that positional encodings have expected patterns."""
    # Setup
    embed_dim = 128
    max_len = 500
    seq_len = 100
    
    # Create encoding layer
    pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
    
    # Get output
    output = pos_encoding(seq_len=seq_len)
    
    # Verify specific properties of positional encodings
    
    # 1. Values should be bounded between -1 and 1 (sin/cos)
    assert torch.all(output >= -1) and torch.all(output <= 1)
    
    # 2. No NaN values
    assert not torch.isnan(output).any()
    
    # 3. Different positions should have different encodings
    for i in range(1, seq_len):
        assert not torch.allclose(output[0, i-1], output[0, i])
    
    # 4. Different positions should have similar patterns in different dimensions
    # Check that the frequency decreases with dimension
    # Even indices (sin): Values should change more rapidly in early dimensions
    changes_dim0 = torch.abs(output[0, 1:, 0] - output[0, :-1, 0]).mean()
    changes_dim50 = torch.abs(output[0, 1:, 50] - output[0, :-1, 50]).mean()
    assert changes_dim0 > changes_dim50

def test_positional_encoding_deterministic():
    """Test that positional encodings are deterministic for the same parameters."""
    # Setup
    embed_dim = 128
    max_len = 500
    seq_len = 100
    
    # Create two encoding layers with same parameters
    pos_encoding1 = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
    pos_encoding2 = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
    
    # Get outputs
    output1 = pos_encoding1(seq_len=seq_len)
    output2 = pos_encoding2(seq_len=seq_len)
    
    # Outputs should be identical
    assert torch.allclose(output1, output2)

def test_positional_encoding_with_embeddings():
    """Test adding positional encodings to embeddings."""
    # Setup
    batch_size = 2
    seq_len = 30
    embed_dim = 128
    
    # Create encoding layer
    pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=500)
    
    # Create dummy embeddings
    embeddings = torch.rand(batch_size, seq_len, embed_dim)
    
    # Get positional encodings
    pos_enc = pos_encoding(seq_len=seq_len)
    
    # Expand to batch dimension
    pos_enc_expanded = pos_enc.expand(batch_size, -1, -1)
    
    # Add to embeddings
    embeddings_with_pos = embeddings + pos_enc_expanded
    
    # Shape should be preserved
    assert embeddings_with_pos.shape == embeddings.shape
    
    # Embeddings should be modified
    assert not torch.allclose(embeddings, embeddings_with_pos)
    
    # The difference should be exactly the positional encoding
    diff = embeddings_with_pos - embeddings
    assert torch.allclose(diff, pos_enc_expanded)
```

### 1.3 Test `RelativePositionalEncoding`

```python
import pytest
import torch
from src.models.embeddings import RelativePositionalEncoding

def test_relative_positional_encoding_shapes():
    """Test output shapes for RelativePositionalEncoding."""
    # Setup
    max_relative_position = 32
    num_units = 64
    
    # Create encoding layer
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=max_relative_position,
        num_units=num_units
    )
    
    # Test with different sequence lengths
    for seq_len in [5, 10, 50]:
        # Get output
        output = rel_pos_encoding(length=seq_len)
        
        # Verify shape
        assert output.shape == (seq_len, seq_len, num_units)

def test_relative_positional_encoding_values():
    """Test properties of relative positional encodings."""
    # Setup
    max_relative_position = 32
    num_units = 64
    seq_len = 10
    
    # Create encoding layer
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=max_relative_position,
        num_units=num_units
    )
    
    # Get output
    output = rel_pos_encoding(length=seq_len)
    
    # 1. No NaN values
    assert not torch.isnan(output).any()
    
    # 2. Check symmetric properties
    # Positions (i,j) and (j,i) should have complementary encodings
    # For distance d, positions (i,i+d) and (i+d,i) should be related
    for i in range(seq_len):
        for j in range(seq_len):
            # The relationship depends on the specific implementation
            # For sinusoidal encodings, some dimensions would be same, some opposite
            # Just check that they're not identical
            if i != j:
                assert not torch.allclose(output[i, j], output[j, i])
    
    # 3. Check that equal relative distances have similar patterns
    # e.g., positions (0,1) and (1,2) both have distance 1
    assert torch.allclose(output[0, 1], output[1, 2])
    assert torch.allclose(output[2, 5], output[3, 6])  # Both distance 3

def test_relative_positional_encoding_clipping():
    """Test clipping of relative positions beyond max_relative_position."""
    # Setup
    max_relative_position = 10  # Small value to test clipping
    num_units = 32
    seq_len = 30  # Longer than 2*max_relative_position
    
    # Create encoding layer
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=max_relative_position,
        num_units=num_units
    )
    
    # Get output
    output = rel_pos_encoding(length=seq_len)
    
    # Check that distant positions get clipped to same encoding
    # Distance 10 and 20 should map to same encoding after clipping
    assert torch.allclose(output[0, 10], output[0, 20])
    
    # Check diagonal (self-positions)
    for i in range(seq_len):
        assert torch.allclose(output[i, i], output[0, 0])

def test_relative_positional_encoding_device():
    """Test device handling in RelativePositionalEncoding."""
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")
    
    # Setup
    max_relative_position = 32
    num_units = 64
    seq_len = 10
    
    # Create encoding layer
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=max_relative_position,
        num_units=num_units
    )
    
    # Move to GPU
    rel_pos_encoding = rel_pos_encoding.cuda()
    
    # Get output, specifying device
    output = rel_pos_encoding(length=seq_len, device=torch.device('cuda'))
    
    # Check device
    assert output.device.type == 'cuda'
    
    # Can also get output and specify device with string
    output = rel_pos_encoding(length=seq_len, device='cuda')
    assert output.device.type == 'cuda'
```

### 1.4 Test Feature Projection Functions

If your embeddings module includes separate functions for projecting features (or these are part of the `RNAFoldingModel`), test them as well:

```python
import pytest
import torch
import torch.nn as nn
from src.models.embeddings import ResidueFeatureProjection, PairFeatureProjection

# If these are implemented as standalone functions, adjust accordingly
# Here we're assuming they're implemented as classes

def test_residue_feature_projection_shapes():
    """Test output shapes for residue feature projection."""
    # Setup
    batch_size = 2
    seq_len = 10
    seq_dim = 32
    residue_dim = 128
    
    config = {
        'residue_embed_dim': residue_dim,
        'seq_embed_dim': seq_dim,
        'use_conservation': True
    }
    
    # Create projection layer
    projection = ResidueFeatureProjection(config)
    
    # Create mock features
    features = {
        'sequence_embedding': torch.rand(batch_size, seq_len, seq_dim),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'conservation': torch.rand(batch_size, seq_len),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Get output
    output = projection(features)
    
    # Verify shape
    assert output.shape == (batch_size, seq_len, residue_dim)
    
    # Test without conservation
    config['use_conservation'] = False
    projection = ResidueFeatureProjection(config)
    del features['conservation']
    output = projection(features)
    assert output.shape == (batch_size, seq_len, residue_dim)

def test_residue_feature_projection_masking():
    """Test masking behavior in residue feature projection."""
    # Setup
    batch_size = 2
    seq_len = 10
    seq_dim = 32
    residue_dim = 128
    
    config = {
        'residue_embed_dim': residue_dim,
        'seq_embed_dim': seq_dim
    }
    
    # Create projection layer
    projection = ResidueFeatureProjection(config)
    
    # Create features with padding
    features = {
        'sequence_embedding': torch.rand(batch_size, seq_len, seq_dim),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Set mask to False for last 3 positions in first sequence
    features['mask'][0, -3:] = False
    
    # Get output
    output = projection(features)
    
    # Masked positions should be zero
    assert torch.all(output[0, -3:] == 0)
    
    # Non-masked positions should not be all zero
    assert not torch.all(output[0, :-3] == 0)
    assert not torch.all(output[1] == 0)

def test_pair_feature_projection_shapes():
    """Test output shapes for pair feature projection."""
    # Setup
    batch_size = 2
    seq_len = 10
    rel_pos_dim = 32
    pair_dim = 64
    
    config = {
        'pair_embed_dim': pair_dim,
        'rel_pos_dim': rel_pos_dim
    }
    
    # Create projection layer
    projection = PairFeatureProjection(config)
    
    # Create mock features
    features = {
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'relative_pos_encoding': torch.rand(seq_len, seq_len, rel_pos_dim),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Get output
    output = projection(features)
    
    # Verify shape
    assert output.shape == (batch_size, seq_len, seq_len, pair_dim)
    
    # Test with additional features
    config['use_contact_prior'] = True
    projection = PairFeatureProjection(config)
    features['contact_prior'] = torch.rand(batch_size, seq_len, seq_len)
    output = projection(features)
    assert output.shape == (batch_size, seq_len, seq_len, pair_dim)

def test_pair_feature_projection_masking():
    """Test masking behavior in pair feature projection."""
    # Setup
    batch_size = 2
    seq_len = 10
    rel_pos_dim = 32
    pair_dim = 64
    
    config = {
        'pair_embed_dim': pair_dim,
        'rel_pos_dim': rel_pos_dim
    }
    
    # Create projection layer
    projection = PairFeatureProjection(config)
    
    # Create features with padding
    features = {
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'relative_pos_encoding': torch.rand(seq_len, seq_len, rel_pos_dim),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Set mask to False for last 3 positions in first sequence
    features['mask'][0, -3:] = False
    
    # Get output
    output = projection(features)
    
    # Check 2D mask effect: 
    # - Last 3 rows should be zero (positions are masked)
    # - Last 3 columns should be zero (positions are masked)
    assert torch.all(output[0, -3:, :, :] == 0)
    assert torch.all(output[0, :, -3:, :] == 0)
    
    # Non-masked positions should not be all zero
    assert not torch.all(output[0, :-3, :-3, :] == 0)
    assert not torch.all(output[1] == 0)
```

## 2. Integration Tests

Test how embedding components work together and integrate with other model components:

```python
import pytest
import torch
from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
from src.models.embeddings import ResidueFeatureProjection, PairFeatureProjection
# Import other relevant components for integration testing
# from src.models.transformer_block import TransformerBlock

class MockTransformerBlock:
    """Mock TransformerBlock for testing embeddings integration."""
    def __init__(self, residue_dim, pair_dim):
        self.residue_dim = residue_dim
        self.pair_dim = pair_dim
    
    def __call__(self, residue_repr, pair_repr, mask=None):
        # Just return the inputs to verify shapes
        return residue_repr, pair_repr

def test_complete_embedding_pipeline():
    """Test the full embedding pipeline from raw inputs to transformer inputs."""
    # Setup
    batch_size = 2
    seq_len = 10
    
    config = {
        'seq_embed_dim': 32,
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'max_relative_position': 32,
        'rel_pos_dim': 32,
        'max_seq_len': 500
    }
    
    # Create embedding components
    sequence_embedding = SequenceEmbedding(
        num_embeddings=5,
        embedding_dim=config['seq_embed_dim']
    )
    
    positional_encoding = PositionalEncoding(
        embed_dim=config['residue_embed_dim'],
        max_len=config['max_seq_len']
    )
    
    relative_pos_encoding = RelativePositionalEncoding(
        max_relative_position=config['max_relative_position'],
        num_units=config['rel_pos_dim']
    )
    
    residue_projection = ResidueFeatureProjection(config)
    pair_projection = PairFeatureProjection(config)
    
    # Create mock transformer block
    transformer_block = MockTransformerBlock(
        residue_dim=config['residue_embed_dim'],
        pair_dim=config['pair_embed_dim']
    )
    
    # Create a batch similar to what would come from the data loader
    batch = {
        'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'conservation': torch.rand(batch_size, seq_len),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Apply embedding pipeline
    
    # 1. Get sequence embeddings
    seq_emb = sequence_embedding(batch['sequence_int'])
    assert seq_emb.shape == (batch_size, seq_len, config['seq_embed_dim'])
    
    # 2. Get positional encodings
    pos_enc = positional_encoding(seq_len=seq_len)
    assert pos_enc.shape == (1, seq_len, config['residue_embed_dim'])
    
    # 3. Get relative positional encodings
    rel_pos = relative_pos_encoding(length=seq_len)
    assert rel_pos.shape == (seq_len, seq_len, config['rel_pos_dim'])
    
    # 4. Prepare features for projection
    features = dict(batch)
    features['sequence_embedding'] = seq_emb
    features['relative_pos_encoding'] = rel_pos
    
    # 5. Project residue features
    residue_repr = residue_projection(features)
    assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
    
    # 6. Add positional encodings
    residue_repr = residue_repr + pos_enc.expand(batch_size, -1, -1)
    assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
    
    # 7. Project pair features
    pair_repr = pair_projection(features)
    assert pair_repr.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])
    
    # 8. Pass to transformer block
    residue_out, pair_out = transformer_block(residue_repr, pair_repr, batch['mask'])
    
    # Check output shapes
    assert residue_out.shape == (batch_size, seq_len, config['residue_embed_dim'])
    assert pair_out.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])

def test_embedding_pipeline_with_variable_lengths():
    """Test embedding pipeline with variable sequence lengths."""
    # Setup similar to previous test
    config = {
        'seq_embed_dim': 32,
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'rel_pos_dim': 32
    }
    
    # Create embedding components
    sequence_embedding = SequenceEmbedding(num_embeddings=5, embedding_dim=config['seq_embed_dim'])
    positional_encoding = PositionalEncoding(embed_dim=config['residue_embed_dim'])
    relative_pos_encoding = RelativePositionalEncoding(num_units=config['rel_pos_dim'])
    
    # Create two sequences of different lengths
    seq1 = torch.randint(0, 5, (1, 8))  # Length 8
    seq2 = torch.randint(0, 5, (1, 12))  # Length 12
    
    # Process both sequences
    # For seq1
    seq_emb1 = sequence_embedding(seq1)
    pos_enc1 = positional_encoding(seq_len=8)
    rel_pos1 = relative_pos_encoding(length=8)
    
    # For seq2
    seq_emb2 = sequence_embedding(seq2)
    pos_enc2 = positional_encoding(seq_len=12)
    rel_pos2 = relative_pos_encoding(length=12)
    
    # Check shapes
    assert seq_emb1.shape == (1, 8, config['seq_embed_dim'])
    assert pos_enc1.shape == (1, 8, config['residue_embed_dim'])
    assert rel_pos1.shape == (8, 8, config['rel_pos_dim'])
    
    assert seq_emb2.shape == (1, 12, config['seq_embed_dim'])
    assert pos_enc2.shape == (1, 12, config['residue_embed_dim'])
    assert rel_pos2.shape == (12, 12, config['rel_pos_dim'])
```

## 3. Edge Case Tests

Test unusual inputs and potential error conditions:

```python
import pytest
import torch
from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding

def test_sequence_embedding_edge_cases():
    """Test SequenceEmbedding with edge cases."""
    # Setup
    embedding_dim = 32
    num_embeddings = 5
    embedding = SequenceEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    # Test with empty sequence (batch_size=1, seq_len=0)
    empty_seq = torch.zeros((1, 0), dtype=torch.long)
    
    # This should either handle gracefully or raise a specific error
    # Depending on the implementation, adjust the assertion accordingly
    with pytest.raises(ValueError):
        _ = embedding(empty_seq)
    
    # Test with out-of-bounds indices
    invalid_seq = torch.tensor([[5, 6, 7]])  # Indices > num_embeddings-1
    
    # Should handle or raise appropriate error
    # PyTorch's embedding will allow this but produce meaningless results
    try:
        output = embedding(invalid_seq)
        # If it doesn't raise an error, output should at least not be NaN
        assert not torch.isnan(output).any()
    except IndexError:
        pass  # This is also acceptable behavior

def test_positional_encoding_edge_cases():
    """Test PositionalEncoding with edge cases."""
    # Setup
    embed_dim = 128
    max_len = 500
    pos_encoding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)
    
    # Test with sequence length 0
    with pytest.raises(ValueError):
        _ = pos_encoding(seq_len=0)
    
    # Test with sequence length > max_len
    with pytest.raises(ValueError):
        _ = pos_encoding(seq_len=max_len + 1)
    
    # Test with non-integer sequence length
    with pytest.raises(TypeError):
        _ = pos_encoding(seq_len=10.5)

def test_relative_positional_encoding_edge_cases():
    """Test RelativePositionalEncoding with edge cases."""
    # Setup
    max_relative_position = 32
    num_units = 64
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=max_relative_position,
        num_units=num_units
    )
    
    # Test with sequence length 0
    with pytest.raises(ValueError):
        _ = rel_pos_encoding(length=0)
    
    # Test with very long sequence
    long_seq_len = 1000  # This should work, but performance may be poor
    output = rel_pos_encoding(length=long_seq_len)
    assert output.shape == (long_seq_len, long_seq_len, num_units)
    
    # Test with negative max_relative_position during initialization
    with pytest.raises(ValueError):
        _ = RelativePositionalEncoding(max_relative_position=-5, num_units=num_units)

def test_embedding_pipeline_with_fully_masked_sequence():
    """Test embedding pipeline when sequence is fully masked."""
    # Setup
    batch_size = 2
    seq_len = 10
    
    config = {
        'seq_embed_dim': 32,
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'rel_pos_dim': 32
    }
    
    # Create embedding components
    sequence_embedding = SequenceEmbedding(num_embeddings=5, embedding_dim=config['seq_embed_dim'])
    positional_encoding = PositionalEncoding(embed_dim=config['residue_embed_dim'])
    relative_pos_encoding = RelativePositionalEncoding(num_units=config['rel_pos_dim'])
    residue_projection = ResidueFeatureProjection(config)
    pair_projection = PairFeatureProjection(config)
    
    # Create batch with one sequence fully masked
    batch = {
        'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Set mask to False for all positions in first sequence
    batch['mask'][0, :] = False
    
    # Process batch
    seq_emb = sequence_embedding(batch['sequence_int'])
    pos_enc = positional_encoding(seq_len=seq_len)
    rel_pos = relative_pos_encoding(length=seq_len)
    
    features = dict(batch)
    features['sequence_embedding'] = seq_emb
    features['relative_pos_encoding'] = rel_pos
    
    residue_repr = residue_projection(features)
    residue_repr = residue_repr + pos_enc.expand(batch_size, -1, -1)
    pair_repr = pair_projection(features)
    
    # First sequence should be all zeros after masking
    assert torch.all(residue_repr[0] == 0)
    assert torch.all(pair_repr[0] == 0)
    
    # Second sequence should not be all zeros
    assert not torch.all(residue_repr[1] == 0)
    assert not torch.all(pair_repr[1] == 0)
```

## 4. Performance Tests (Optional)

These tests can help ensure the embedding components are efficient and memory-friendly:

```python
import pytest
import torch
import time
from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding

@pytest.mark.performance
def test_embedding_memory_usage():
    """Test memory usage of embedding components."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for memory testing")
    
    # Setup
    batch_size = 16
    seq_len = 256
    embed_dim = 128
    rel_pos_dim = 32
    
    # Start with a clean GPU state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory usage
    baseline_memory = torch.cuda.memory_allocated()
    
    # Create and move components to GPU
    sequence_embedding = SequenceEmbedding(
        num_embeddings=5, 
        embedding_dim=64
    ).cuda()
    
    pos_encoding = PositionalEncoding(
        embed_dim=embed_dim,
        max_len=500
    ).cuda()
    
    rel_pos_encoding = RelativePositionalEncoding(
        max_relative_position=32,
        num_units=rel_pos_dim
    ).cuda()
    
    # Process large batch
    sequence_int = torch.randint(0, 5, (batch_size, seq_len), device='cuda')
    
    # Measure memory for sequence embedding
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    seq_emb = sequence_embedding(sequence_int)
    seq_embed_mem = torch.cuda.memory_allocated() - start_mem
    
    # Measure memory for positional encoding
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    pos_enc = pos_encoding(seq_len=seq_len)
    pos_enc = pos_enc.expand(batch_size, -1, -1)
    pos_encoding_mem = torch.cuda.memory_allocated() - start_mem
    
    # Measure memory for relative positional encoding
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    rel_pos = rel_pos_encoding(length=seq_len)
    rel_pos_mem = torch.cuda.memory_allocated() - start_mem
    
    # Log memory usage
    print(f"Sequence Embedding Memory: {seq_embed_mem / 1024**2:.2f} MB")
    print(f"Positional Encoding Memory: {pos_encoding_mem / 1024**2:.2f} MB")
    print(f"Relative Positional Encoding Memory: {rel_pos_mem / 1024**2:.2f} MB")
    
    # Assert memory usage is within reasonable limits
    # These thresholds depend on your specific requirements
    max_seq_embed_mem = batch_size * seq_len * 64 * 4 * 1.5 / 1024**2  # 1.5x theoretical size in MB
    max_rel_pos_mem = seq_len * seq_len * rel_pos_dim * 4 * 1.5 / 1024**2  # 1.5x theoretical size in MB
    
    assert seq_embed_mem / 1024**2 < max_seq_embed_mem
    assert rel_pos_mem / 1024**2 < max_rel_pos_mem

@pytest.mark.performance
def test_embedding_speed():
    """Test processing speed of embedding components."""
    # Setup
    batch_size = 16
    seq_len = 256
    embed_dim = 128
    
    # Create components
    sequence_embedding = SequenceEmbedding(num_embeddings=5, embedding_dim=64)
    pos_encoding = PositionalEncoding(embed_dim=embed_dim)
    rel_pos_encoding = RelativePositionalEncoding(num_units=32)
    
    # Create inputs
    sequence_int = torch.randint(0, 5, (batch_size, seq_len))
    
    # Measure sequence embedding time
    start_time = time.time()
    for _ in range(100):
        seq_emb = sequence_embedding(sequence_int)
    seq_embed_time = time.time() - start_time
    
    # Measure positional encoding time
    start_time = time.time()
    for _ in range(100):
        pos_enc = pos_encoding(seq_len=seq_len)
        pos_enc = pos_enc.expand(batch_size, -1, -1)
    pos_encoding_time = time.time() - start_time
    
    # Measure relative positional encoding time
    start_time = time.time()
    for _ in range(10):  # Fewer iterations as this is more expensive
        rel_pos = rel_pos_encoding(length=seq_len)
    rel_pos_time = time.time() - start_time
    
    # Log times
    print(f"Sequence Embedding Time (100 iterations): {seq_embed_time:.4f} s")
    print(f"Positional Encoding Time (100 iterations): {pos_encoding_time:.4f} s")
    print(f"Relative Positional Encoding Time (10 iterations): {rel_pos_time:.4f} s")
    
    # No strict assertions here, just logging for performance monitoring
```

## 5. Device Compatibility Tests

Ensure embeddings work correctly across devices:

```python
import pytest
import torch
from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device):
    """Test that embeddings work correctly on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available for testing")
    
    # Setup
    batch_size = 2
    seq_len = 10
    embed_dim = 128
    device = torch.device(device)
    
    # Create components
    sequence_embedding = SequenceEmbedding(
        num_embeddings=5, 
        embedding_dim=32
    ).to(device)
    
    pos_encoding = PositionalEncoding(
        embed_dim=embed_dim
    ).to(device)
    
    rel_pos_encoding = RelativePositionalEncoding(
        num_units=32
    ).to(device)
    
    # Create input
    sequence_int = torch.randint(0, 5, (batch_size, seq_len), device=device)
    
    # Process
    seq_emb = sequence_embedding(sequence_int)
    pos_enc = pos_encoding(seq_len=seq_len).to(device)
    rel_pos = rel_pos_encoding(length=seq_len, device=device)
    
    # Check device
    assert seq_emb.device == device
    assert pos_enc.device == device
    assert rel_pos.device == device
    
    # Check that we can combine them
    pos_enc_expanded = pos_enc.expand(batch_size, -1, -1)
    combined = seq_emb[:, :, :embed_dim] + pos_enc_expanded
    assert combined.device == device
```

## 6. Test Fixtures

For convenience, create fixtures for frequently used test setups:

```python
import pytest
import torch
from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
from src.models.embeddings import ResidueFeatureProjection, PairFeatureProjection

@pytest.fixture
def config():
    """Standard config for embedding tests."""
    return {
        'seq_embed_dim': 32,
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'max_relative_position': 32,
        'rel_pos_dim': 32,
        'max_seq_len': 500,
        'use_conservation': True
    }

@pytest.fixture
def embedding_components(config):
    """Create standard embedding components for tests."""
    return {
        'sequence_embedding': SequenceEmbedding(
            num_embeddings=5,
            embedding_dim=config['seq_embed_dim']
        ),
        'positional_encoding': PositionalEncoding(
            embed_dim=config['residue_embed_dim'],
            max_len=config['max_seq_len']
        ),
        'relative_pos_encoding': RelativePositionalEncoding(
            max_relative_position=config['max_relative_position'],
            num_units=config['rel_pos_dim']
        ),
        'residue_projection': ResidueFeatureProjection(config),
        'pair_projection': PairFeatureProjection(config)
    }

@pytest.fixture
def mock_batch(batch_size=2, seq_len=10):
    """Create a mock batch for embedding tests."""
    return {
        'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'conservation': torch.rand(batch_size, seq_len),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }

# Example test using fixtures
def test_with_fixtures(config, embedding_components, mock_batch):
    """Test using predefined fixtures."""
    # Extract components
    sequence_embedding = embedding_components['sequence_embedding']
    positional_encoding = embedding_components['positional_encoding']
    
    # Get batch info
    batch_size, seq_len = mock_batch['sequence_int'].shape
    
    # Apply embeddings
    seq_emb = sequence_embedding(mock_batch['sequence_int'])
    pos_enc = positional_encoding(seq_len=seq_len)
    
    # Check shapes
    assert seq_emb.shape == (batch_size, seq_len, config['seq_embed_dim'])
    assert pos_enc.shape == (1, seq_len, config['residue_embed_dim'])
```

## Running the Tests

To run the tests, use pytest from the project root:

```bash
# Run all embedding tests
pytest -xvs tests/test_embeddings.py

# Run specific test groups
pytest -xvs tests/test_embeddings.py::test_sequence_embedding_shapes
pytest -xvs tests/test_embeddings.py::test_complete_embedding_pipeline

# Skip performance tests
pytest -xvs tests/test_embeddings.py -k "not performance"

# Run tests with specific marker
pytest -xvs tests/test_embeddings.py -m "performance"
```

Add the following to `pytest.ini` to register custom markers:

```ini
[pytest]
markers =
    performance: marks tests as performance tests (may be slow)
```

## Common Test Failures and Remediation

| Failure Pattern | Likely Cause | Remediation |
|-----------------|--------------|-------------|
| Shape mismatch in projection layers | Incorrect feature dimension calculation | Verify dimensions in feature concatenation, check config parameters |
| NaN values in positional encodings | Numerical instability in sinusoidal calculation | Add numerical stability checks, decrease frequency scaling |
| Device mismatch errors | Inconsistent device management | Ensure all tensors and modules are moved to the same device |
| Out of memory with large sequences | Inefficient handling of relative positional encodings | Implement lazy computation or caching for large matrices |
| Masking not applied correctly | Incorrect mask expansion or application | Verify mask shape and proper broadcasting in all components |

## Integration with CI/CD

These tests should be incorporated into your CI/CD pipeline to ensure embedding functionality is preserved across changes. Consider implementing the following in your CI configuration:

1. Fast tests: Run basic unit tests on each commit
2. Full test suite: Run all tests including integration tests on pull requests
3. Performance tests: Run performance tests on a schedule or for releases

## Test Coverage Goals

Aim for at least 90% code coverage for the embeddings component, ensuring:

1. All public methods and classes are tested
2. All branches of conditional logic are tested
3. Edge cases and error conditions are explicitly tested
4. Device compatibility is verified

## Next Steps

After implementing and verifying the embeddings component through these tests:

1. Integrate with the transformer block component
2. Ensure the embedding outputs are correctly consumed by downstream components
3. Proceed to end-to-end model testing

Remember that the embeddings component forms the foundation of the model's ability to process RNA sequence data, so thorough testing is essential for the success of the entire pipeline.
