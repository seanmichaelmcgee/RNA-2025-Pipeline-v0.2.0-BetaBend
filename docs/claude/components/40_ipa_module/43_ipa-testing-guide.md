# IPA Module Testing Guide

This document outlines the testing strategy and implementation for the Invariant Point Attention (IPA) module (`src/models/ipa_module.py`) of the RNA 3D folding pipeline. Thorough testing is essential to ensure this component correctly generates 3D coordinates from the transformer backbone's representations while properly handling variable-length sequences.

## Testing Objectives

1. **Validate Coordinate Prediction**: Verify that the IPA module correctly projects residue representations to 3D coordinates
2. **Ensure Masking Correctness**: Confirm proper handling of masked positions for variable-length sequences
3. **Verify Interface Compatibility**: Test that the module maintains the interface needed for future V2+ implementations
4. **Validate Device Handling**: Ensure correct operation across different devices (CPU/CUDA)
5. **Check Gradient Flow**: Confirm that gradients properly flow through the module during backpropagation
6. **Test Memory Efficiency**: Evaluate performance with realistic sequence lengths
7. **Verify Integration**: Ensure compatibility with upstream transformer blocks and downstream loss computation

## Test Structure

Create a comprehensive test file `tests/test_ipa_module.py` with these major test groups:

1. Core Functionality Tests
2. Masking Tests
3. Device Compatibility Tests
4. Gradient Flow Tests
5. Integration Tests
6. V1 Placeholder-Specific Tests
7. Performance and Edge Case Tests

## 1. Core Functionality Tests

### 1.1 Initialization Tests

```python
def test_initialization(self, config):
    """Test IPAModule initialization with different configurations."""
    # Test standard initialization
    ipa_module = IPAModule(config)
    assert ipa_module.residue_dim == config['residue_embed_dim']
    assert ipa_module.pair_dim == config['pair_embed_dim']
    assert ipa_module.ipa_dim == config['ipa_dim']
    assert ipa_module.num_iterations == config['num_ipa_iterations']
    
    # Test with minimum configuration
    min_config = {'residue_embed_dim': 64}
    ipa_module = IPAModule(min_config)
    assert ipa_module.residue_dim == 64
    assert ipa_module.pair_dim == 64  # Default
    assert ipa_module.ipa_dim == 32   # Default (half of residue_dim)
    assert ipa_module.num_iterations == 1  # Default
    
    # Check that the coordinate projection MLP exists
    assert isinstance(ipa_module.coord_projection, nn.Sequential)
    assert len(ipa_module.coord_projection) == 3  # Linear, ReLU, Linear
    assert isinstance(ipa_module.coord_projection[0], nn.Linear)
    assert isinstance(ipa_module.coord_projection[1], nn.ReLU)
    assert isinstance(ipa_module.coord_projection[2], nn.Linear)
    
    # Check output dimension of final layer
    assert ipa_module.coord_projection[2].out_features == 3  # x, y, z coordinates
```

### 1.2 Forward Pass Shape Tests

```python
def test_forward_shapes(self, ipa_module, test_inputs):
    """Test forward pass produces tensors with correct shapes."""
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Test with all inputs
    coords = ipa_module(residue_repr, pair_repr, mask)
    batch_size, seq_len = residue_repr.shape[:2]
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Test without mask
    coords = ipa_module(residue_repr, pair_repr)
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Test without pair_repr (should work in V1 since pair_repr is unused)
    coords = ipa_module(residue_repr, mask=mask)
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Test with different sequence lengths
    for seq_len in [5, 20, 50]:
        batch = 1
        res_repr = torch.rand(batch, seq_len, ipa_module.residue_dim)
        coords = ipa_module(res_repr)
        assert coords.shape == (batch, seq_len, 3)
```

### 1.3 Output Value Tests

```python
def test_forward_values(self, ipa_module, test_inputs):
    """Test forward pass produces reasonable output values."""
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Forward pass
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Check for NaNs or infinities
    assert not torch.isnan(coords).any()
    assert not torch.isinf(coords).any()
    
    # Check range of values (depends on initialization but shouldn't be extreme)
    # Since we're using Xavier/Glorot initialization, values should be reasonable
    assert coords.abs().max() < 100  # arbitrary reasonable threshold
```

## 2. Masking Tests (Critical)

### 2.1 Basic Masking

```python
def test_masking(self, ipa_module, test_inputs):
    """Test that masking correctly zeroes out padded positions."""
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask'].clone()
    
    # Modify mask to have some padding
    batch_size, seq_len = mask.shape
    mask[0, -2:] = False  # Mask out last two positions in first sequence
    
    # Forward pass with mask
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Check that masked positions are zero
    assert torch.all(coords[0, -2:] == 0)
    
    # Check that non-masked positions are non-zero (with high probability)
    # Note: there's a small chance some coordinates could be zero by chance
    non_masked_coords = coords[0, :-2].reshape(-1)
    assert torch.any(non_masked_coords != 0)
    
    # Check second sequence (not masked)
    assert torch.any(coords[1] != 0)
```

### 2.2 Variable-Length Sequences

```python
def test_variable_length_sequences(self, ipa_module):
    """Test handling of variable-length sequences with proper masking."""
    batch_size = 3
    max_seq_len = 15
    
    # Create sequences of different lengths
    seq_lengths = [15, 10, 5]  # One full-length, two with padding
    
    # Create residue representations
    residue_repr = torch.rand(batch_size, max_seq_len, ipa_module.residue_dim)
    
    # Create mask based on sequence lengths
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        mask[i, :length] = True
    
    # Forward pass
    coords = ipa_module(residue_repr, mask=mask)
    
    # Check output shape
    assert coords.shape == (batch_size, max_seq_len, 3)
    
    # Verify masking for each sequence
    for i, length in enumerate(seq_lengths):
        # Masked positions should be zero
        if length < max_seq_len:
            assert torch.all(coords[i, length:] == 0)
        
        # At least some non-masked positions should be non-zero
        assert torch.any(coords[i, :length] != 0)
```

### 2.3 Complex Masking Patterns

```python
def test_complex_masking_pattern(self, ipa_module):
    """Test handling of complex masking patterns with non-contiguous masks."""
    batch_size = 2
    seq_len = 12
    
    # Create residue representations
    residue_repr = torch.rand(batch_size, seq_len, ipa_module.residue_dim)
    
    # Create a complex mask pattern
    # First sequence: mask positions 3-5 and 8-9
    # Second sequence: mask positions 1, 4, 7, 10
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 3:6] = False
    mask[0, 8:10] = False
    mask[1, 1] = False
    mask[1, 4] = False
    mask[1, 7] = False
    mask[1, 10] = False
    
    # Forward pass
    coords = ipa_module(residue_repr, mask=mask)
    
    # Check that masked positions are zero
    assert torch.all(coords[0, 3:6] == 0)
    assert torch.all(coords[0, 8:10] == 0)
    assert torch.all(coords[1, 1] == 0)
    assert torch.all(coords[1, 4] == 0)
    assert torch.all(coords[1, 7] == 0)
    assert torch.all(coords[1, 10] == 0)
    
    # Check that non-masked positions are non-zero
    # (at least some values should be non-zero)
    assert torch.any(coords[0, :3] != 0)
    assert torch.any(coords[0, 6:8] != 0)
    assert torch.any(coords[0, 10:] != 0)
    assert torch.any(coords[1, 0] != 0)
    assert torch.any(coords[1, 2:4] != 0)
    assert torch.any(coords[1, 5:7] != 0)
    assert torch.any(coords[1, 8:10] != 0)
    assert torch.any(coords[1, 11] != 0)
```

### 2.4 Full Masking (Edge Case)

```python
def test_full_masking(self, ipa_module):
    """Test extreme case where an entire sequence is masked."""
    batch_size = 2
    seq_len = 10
    
    # Create residue representations
    residue_repr = torch.rand(batch_size, seq_len, ipa_module.residue_dim)
    
    # Mask out the entire first sequence
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, :] = False
    
    # Forward pass
    coords = ipa_module(residue_repr, mask=mask)
    
    # Check that first sequence is all zeros
    assert torch.all(coords[0] == 0)
    
    # Check that second sequence has non-zero values
    assert torch.any(coords[1] != 0)
```

## 3. Device Compatibility Tests

### 3.1 CPU/CUDA Compatibility

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility(self, config, test_inputs):
    """Test compatibility with different devices (CPU/CUDA)."""
    # Create inputs on CPU
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Create module on CPU and run forward pass
    ipa_module_cpu = IPAModule(config)
    coords_cpu = ipa_module_cpu(residue_repr, pair_repr, mask)
    
    # Create module on CUDA and run forward pass
    ipa_module_cuda = IPAModule(config).cuda()
    residue_repr_cuda = residue_repr.cuda()
    pair_repr_cuda = pair_repr.cuda()
    mask_cuda = mask.cuda()
    coords_cuda = ipa_module_cuda(residue_repr_cuda, pair_repr_cuda, mask_cuda)
    
    # Check device of outputs
    assert coords_cpu.device.type == 'cpu'
    assert coords_cuda.device.type == 'cuda'
    
    # Values should be similar when moved to the same device
    # (allowing for some floating point differences)
    assert torch.allclose(coords_cpu, coords_cuda.cpu(), rtol=1e-4, atol=1e-4)
```

### 3.2 Device Consistency

```python
def test_device_consistency(self, ipa_module):
    """Test module correctly handles inputs on different devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for device consistency test")
    
    # Move module to CUDA
    ipa_module = ipa_module.cuda()
    
    # Create inputs on different devices
    batch_size, seq_len = 2, 10
    residue_repr = torch.rand(batch_size, seq_len, ipa_module.residue_dim).cuda()
    pair_repr = torch.rand(batch_size, seq_len, seq_len, ipa_module.pair_dim)  # CPU
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
    
    # This should raise an error due to device mismatch
    with pytest.raises(RuntimeError) as excinfo:
        coords = ipa_module(residue_repr, pair_repr, mask)
    
    # The error should be related to device mismatch
    assert "device" in str(excinfo.value).lower() or "cuda" in str(excinfo.value).lower()
```

## 4. Gradient Flow Tests

### 4.1 Basic Gradient Flow

```python
def test_gradient_flow(self, ipa_module, test_inputs):
    """Test that gradients flow correctly through the module."""
    residue_repr = test_inputs['residue_repr'].requires_grad_(True)
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Forward pass
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Create dummy loss and backpropagate
    loss = coords.sum()
    loss.backward()
    
    # Check that gradients were computed for residue_repr
    assert residue_repr.grad is not None
    assert torch.any(residue_repr.grad != 0)
    
    # Check that gradients exist for module parameters
    for name, param in ipa_module.named_parameters():
        assert param.grad is not None
        # Most parameters should have non-zero gradients
        assert torch.any(param.grad != 0)
```

### 4.2 Masked Gradient Flow

```python
def test_masking_gradients(self, ipa_module, test_inputs):
    """Test gradient behavior with masking."""
    residue_repr = test_inputs['residue_repr'].clone().requires_grad_(True)
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask'].clone()
    
    # Mask out some positions
    mask[0, -2:] = False
    
    # Forward pass
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Create dummy loss and backpropagate
    loss = coords.sum()
    loss.backward()
    
    # Masked positions should still have gradients, but they might be zero
    # Let's verify gradients exist and are non-zero for unmasked positions
    assert torch.any(residue_repr.grad[0, :-2] != 0)
```

## 5. Integration Tests

### 5.1 Integration with Transformer

```python
def test_integration_with_transformer(self, ipa_module, test_inputs):
    """Test integration with outputs from transformer blocks."""
    # Create a mock transformer block output
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Mock transformer would update these representations
    # Here we just simulate that by adding a constant
    mock_transformer_residue = residue_repr + 0.1
    mock_transformer_pair = pair_repr + 0.1
    
    # Process through IPA module
    coords = ipa_module(mock_transformer_residue, mock_transformer_pair, mask)
    
    # Verify basic shape correctness
    batch_size, seq_len = residue_repr.shape[:2]
    assert coords.shape == (batch_size, seq_len, 3)
```

### 5.2 Integration in Full Model

```python
def test_in_full_model(self, ipa_module, test_inputs):
    """Test usage in a full model context with confidence and angle heads."""
    # Simulate a full model pipeline
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Generate coordinates
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Mock confidence and angle prediction heads
    confidence_head = nn.Linear(ipa_module.residue_dim, 1)
    angle_head = nn.Linear(ipa_module.residue_dim, 4)
    
    # Generate auxiliary outputs
    confidence = confidence_head(residue_repr).squeeze(-1)
    angles = angle_head(residue_repr)
    
    # Apply mask
    confidence = confidence * mask.float()
    angles = angles * mask.unsqueeze(-1).float()
    
    # Check shapes
    assert coords.shape == (residue_repr.shape[0], residue_repr.shape[1], 3)
    assert confidence.shape == (residue_repr.shape[0], residue_repr.shape[1])
    assert angles.shape == (residue_repr.shape[0], residue_repr.shape[1], 4)
    
    # Check masking
    mask[0, -2:] = False  # Mask last two positions in first sequence
    masked_coords = coords * mask.unsqueeze(-1).float()
    assert torch.all(masked_coords[0, -2:] == 0)
```

## 6. V1 Placeholder-Specific Tests

### 6.1 Placeholder Nature

```python
def test_placeholder_nature(self, ipa_module, test_inputs):
    """
    Test that the V1 placeholder implementation behaves as expected.
    
    In V1, the module is just a linear projection from residue representations
    to coordinates, so pair_repr should have no effect on the output.
    """
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Forward pass with original pair_repr
    coords1 = ipa_module(residue_repr, pair_repr, mask)
    
    # Forward pass with different pair_repr
    different_pair = torch.rand_like(pair_repr)
    coords2 = ipa_module(residue_repr, different_pair, mask)
    
    # In V1, pair_repr is not used, so results should be identical
    assert torch.allclose(coords1, coords2)
    
    # Also verify a direct call to coord_projection gives the same result
    direct_coords = ipa_module.coord_projection(residue_repr)
    # Apply mask
    direct_coords = direct_coords * mask.unsqueeze(-1).float()
    
    assert torch.allclose(coords1, direct_coords)
```

### 6.2 Future Compatibility

```python
def test_future_compatibility(self, ipa_module, test_inputs):
    """
    Test that the implementation maintains interface compatibility for future versions.
    
    The V1 placeholder should accept `pair_repr` and `num_iterations` configuration
    even though they're not used in V1, to maintain API compatibility for V2+.
    """
    residue_repr = test_inputs['residue_repr']
    pair_repr = test_inputs['pair_repr']
    mask = test_inputs['mask']
    
    # Check that module has future-relevant attributes
    assert hasattr(ipa_module, 'num_iterations')
    assert hasattr(ipa_module, 'pair_dim')
    
    # V1 should accept pair_repr argument even though it's not used
    # (This was already verified in test_placeholder_nature)
    
    # Check that the module can be created with V2+ relevant config options
    v2_config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'ipa_dim': 16,
        'ipa_heads': 4,
        'num_ipa_iterations': 8
    }
    v2_compatible_module = IPAModule(v2_config)
    
    # The module should store these parameters for future use even if
    # they don't affect the V1 placeholder behavior
    assert v2_compatible_module.num_iterations == 8
```

## 7. Performance and Edge Case Tests

### 7.1 Numerical Stability

```python
def test_numerical_stability(self, ipa_module):
    """Test numerical stability with extreme input values."""
    batch_size, seq_len = 2, 10
    
    # Test with very large values
    large_residue = torch.ones(batch_size, seq_len, ipa_module.residue_dim) * 1e6
    large_pair = torch.ones(batch_size, seq_len, seq_len, ipa_module.pair_dim) * 1e6
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass with large values
    coords_large = ipa_module(large_residue, large_pair, mask)
    
    # Check for NaNs or infinities
    assert not torch.isnan(coords_large).any()
    assert not torch.isinf(coords_large).any()
    
    # Test with very small values
    small_residue = torch.ones(batch_size, seq_len, ipa_module.residue_dim) * 1e-6
    small_pair = torch.ones(batch_size, seq_len, seq_len, ipa_module.pair_dim) * 1e-6
    
    # Forward pass with small values
    coords_small = ipa_module(small_residue, small_pair, mask)
    
    # Check for NaNs or infinities
    assert not torch.isnan(coords_small).any()
    assert not torch.isinf(coords_small).any()
```

### 7.2 Memory Efficiency

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_efficiency(self, small_config):
    """Test memory efficiency with larger sequences."""
    # This test is more important for future V2+ implementations
    # For V1, it's mostly validating that the placeholder can handle larger sequences
    
    # Create a larger sequence length
    batch_size = 1
    seq_len = 500  # Adjust based on available GPU memory
    
    # Create module and inputs on CUDA
    ipa_module = IPAModule(small_config).cuda()
    residue_repr = torch.rand(batch_size, seq_len, small_config['residue_embed_dim']).cuda()
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool).cuda()
    
    # Record initial memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated()
    
    # Forward pass
    coords = ipa_module(residue_repr, mask=mask)
    
    # Record peak memory
    peak_mem = torch.cuda.max_memory_allocated()
    
    # Memory usage should be reasonable for the V1 placeholder
    # For a seq_len of 500 and small_config, it should be well under 1GB
    mem_used_mb = (peak_mem - initial_mem) / (1024 * 1024)
    assert mem_used_mb < 1000  # Should be much less for V1 placeholder
    
    # Print memory usage for information
    print(f"Memory used for sequence length {seq_len}: {mem_used_mb:.2f} MB")
```

### 7.3 Extreme Sequence Lengths

```python
def test_extreme_sequence_lengths(self, ipa_module):
    """Test with extremely short and very long sequences."""
    # Very short sequence (single residue)
    batch_size = 2
    seq_len = 1
    
    residue_repr = torch.rand(batch_size, seq_len, ipa_module.residue_dim)
    pair_repr = torch.rand(batch_size, seq_len, seq_len, ipa_module.pair_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    coords = ipa_module(residue_repr, pair_repr, mask)
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Test with a longer sequence (if memory allows)
    # This is more relevant for V2+ implementations
    # For V1, this should work fine since it's just a linear projection
    try:
        seq_len = 1000
        residue_repr = torch.rand(1, seq_len, ipa_module.residue_dim)
        coords = ipa_module(residue_repr)
        assert coords.shape == (1, seq_len, 3)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # This is an acceptable reason for failure
            print(f"Skipping long sequence test due to CUDA memory constraints")
        else:
            # Other errors should be raised
            raise
```

## Test Fixtures and Setup

For convenience, create fixtures for test setup:

```python
@pytest.fixture
def config(self):
    """Fixture for standard test configuration."""
    return {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'ipa_dim': 64,
        'num_ipa_iterations': 1  # V1 setting
    }

@pytest.fixture
def small_config(self):
    """Fixture for smaller test configuration for memory efficiency."""
    return {
        'residue_embed_dim': 32,
        'pair_embed_dim': 16,
        'ipa_dim': 16,
        'num_ipa_iterations': 1
    }

@pytest.fixture
def ipa_module(self, config):
    """Fixture for IPAModule instance."""
    return IPAModule(config)

@pytest.fixture
def test_inputs(self, config):
    """Fixture for standard test inputs."""
    batch_size, seq_len = 2, 10
    return {
        'residue_repr': torch.rand(batch_size, seq_len, config['residue_embed_dim']),
        'pair_repr': torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim']),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
```

## Running the Tests

To run the tests, execute:

```bash
# Run all IPA module tests
pytest -xvs tests/test_ipa_module.py

# Run specific test groups
pytest -xvs tests/test_ipa_module.py::TestIPAModule::test_masking
pytest -xvs tests/test_ipa_module.py::TestIPAModule::test_variable_length_sequences

# Skip performance tests (which might be slow)
pytest -xvs tests/test_ipa_module.py -k "not memory"

# Run device-specific tests
pytest -xvs tests/test_ipa_module.py::TestIPAModule::test_device_compatibility
```

## Common Test Failures and Remediation

| Failure Pattern | Likely Cause | Remediation |
|-----------------|--------------|-------------|
| Masked positions not zero | Incorrect mask application | Ensure mask is properly expanded with `mask.unsqueeze(-1)` |
| Device mismatch errors | Inputs/module on different devices | Move all tensors and module to same device |
| NaN/inf values in output | Numerical instability in projection | Check initialization, add epsilon values |
| Shape mismatch errors | Inconsistent handling of batch/sequence dimensions | Double-check broadcasting and expansion operations |
| OOM errors with large sequences | Memory-inefficient implementations | Use smaller dimensions, implement gradient checkpointing |
| Zero gradients | Missing connections in forward pass | Check projection implementation and gradient path |

## Test Coverage Goals

Aim for at least 90% code coverage for the IPA module component, ensuring:

1. All public methods are tested
2. Both basic functionality and edge cases are verified
3. Masking logic is thoroughly tested with variable-length sequences
4. Numerical stability and device compatibility are validated
5. Integration with other components is tested
6. V1/V2+ compatibility is maintained

## V1 vs V2+ Testing Considerations

The tests in this guide focus primarily on the V1 placeholder implementation, but include several tests that verify compatibility with future V2+ implementations:

1. **V1 Placeholder Tests**: Verify the simple linear projection behavior and that pair_repr is accepted but unused
2. **V2+ Compatibility Tests**: Ensure the module stores future-relevant parameters and maintains the correct interface
3. **Performance Tests**: While not as critical for V1, these tests help ensure the module can handle realistically sized inputs, which will be important for V2+

When implementing the full V2+ version in the future, additional tests will be needed for:

1. Rotation and translation invariance 
2. Iterative refinement of coordinates
3. Frame-based representation
4. Integration with the full IPA attention mechanism

## Next Steps

After implementing and verifying the IPA module component through these tests:

1. Fix any identified issues
2. Integrate with the transformer block component (already tested)
3. Connect to the loss computation components
4. Build the complete model architecture

Remember that the IPA module is critical for generating the final 3D coordinates from the learned representations, and proper masking is essential for handling variable-length RNA sequences correctly.
