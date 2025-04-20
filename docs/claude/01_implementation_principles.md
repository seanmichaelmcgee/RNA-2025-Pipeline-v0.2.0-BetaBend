# RNA 3D Folding: Core Implementation Principles

This document establishes the foundational patterns and principles to maintain across all components. These principles ensure consistency, compatibility, and quality throughout the implementation.

## 1. Code Organization & Modularity

### 1.1 Separation of Logic and Orchestration
- **Core Logic**: Implement in `src/` modules (data_loading.py, models/, losses.py)
- **Orchestration Logic**: Implement in `scripts/` (train.py, predict.py)
- **Rationale**: This separation enables reuse of core logic in different environments (local vs. Kaggle)

### 1.2 Dependencies Between Components
- Maintain a clean dependency graph:
  - `data_loading.py` has no dependencies on other project modules
  - `models/embeddings.py` depends only on PyTorch
  - `models/transformer_block.py` may depend on embeddings
  - `models/rna_folding_model.py` may depend on all other model components
  - `losses.py` should not depend on model implementation details
- Avoid circular dependencies

### 1.3 Module Structure Pattern
```python
"""
[Module name]: [Brief description]

This module implements [core functionality] for the RNA 3D structure prediction pipeline.

Responsible Instance: [01_data_pipeline/02_model_components/03_integration]
"""
import torch
import numpy as np
# Standard library imports
# Third-party imports
# Local imports (with explicit paths, not relative)

# Constants (UPPER_CASE)
NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# Helper functions (snake_case)
def helper_function(arg1, arg2):
    """Function docstring."""
    pass

# Classes (PascalCase)
class MainClass:
    """Class docstring."""
    
    def __init__(self, param1, param2):
        """Constructor docstring."""
        pass
        
    def method(self, arg):
        """Method docstring."""
        pass
```

### 1.4 Handoff-Ready Organization
- Include version information in module docstrings
- Mark public interfaces explicitly
- Document handoff status in implementation journals
- Group related components for coordinated handoffs
- Maintain clear boundaries between instance responsibilities

## 2. Path Parameterization (CRITICAL)

### 2.1 No Hardcoded Paths
- **NEVER** include hardcoded file paths in `src/` modules
- All paths must be passed as arguments from orchestration scripts
- Use `os.path.join()` for constructing paths
- This is the most critical requirement for Kaggle compatibility

### 2.2 Path Handling Pattern
```python
# CORRECT:
def load_data(data_path, feature_dir):
    """Load data from the specified paths."""
    file_path = os.path.join(feature_dir, "features.npz")
    
# INCORRECT:
def load_data():
    """Load data from hardcoded paths."""
    file_path = "data/features/features.npz"  # NEVER DO THIS
```

### 2.3 Default Parameters
- Do not use default values for path parameters
- Make path dependencies explicit in function signatures

```python
# CORRECT:
def __init__(self, data_dir, config_path):
    self.data_dir = data_dir
    
# INCORRECT:
def __init__(self, data_dir="data/processed"):  # NEVER DO THIS
    self.data_dir = data_dir
```

### 2.4 Path Verification for Handoffs
- Include path validation in verification tests
- Check that components accept path parameters correctly
- Verify components handle missing files appropriately
- Confirm paths are properly joined using os.path.join()

## 3. PyTorch Implementation Patterns

### 3.1 Module Design
- Inherit from `torch.nn.Module` for all model components
- Implement `forward()` method with clear input/output documentation
- Initialize parameters in `__init__` method
- Use `nn.Parameter` for learnable parameters
- Add version and responsible instance information in docstrings

```python
class TransformerBlock(nn.Module):
    """
    Transformer block implementation.
    
    Version: v1.0
    Responsible Instance: 02_model_components
    """
    
    def __init__(self, config):
        super().__init__()
        self.dim = config['hidden_dim']
        self.attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=config['num_heads'],
            batch_first=True
        )
        # Initialize other layers
        
    def forward(self, x, mask=None):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Implementation
        return output
```

### 3.2 Device Management
- Accept device as parameter or detect automatically
- Move tensors to device explicitly
- Check device consistency in complex operations
- Document device handling behavior in interface contracts

```python
def forward(self, inputs, device=None):
    if device is None:
        device = next(self.parameters()).device
    
    # Move inputs to device if necessary
    if isinstance(inputs, torch.Tensor) and inputs.device != device:
        inputs = inputs.to(device)
    
    # Process on correct device
    output = self.process(inputs)
    return output
```

### 3.3 Tensor Shape Documentation
- Document expected tensor shapes in docstrings
- Use explicit dimension names (batch_size, seq_len, hidden_dim)
- Add shape assertions in debug mode
- Include shape information in interface contracts

```python
def process_features(features):
    """
    Process feature tensors.
    
    Args:
        features: Tensor of shape (batch_size, seq_len, feature_dim)
        
    Returns:
        Processed tensor of shape (batch_size, seq_len, output_dim)
    """
    batch_size, seq_len, feature_dim = features.shape
    assert feature_dim == EXPECTED_FEATURE_DIM, f"Expected feature_dim={EXPECTED_FEATURE_DIM}, got {feature_dim}"
    
    # Processing logic
    return output
```

### 3.4 Configuration Management
- Load all hyperparameters from configuration
- Use dictionary access with explicit defaults
- Validate critical parameters
- Document configuration requirements in interface contracts

```python
def __init__(self, config):
    super().__init__()
    self.hidden_dim = config.get('hidden_dim', 128)
    self.num_layers = config.get('num_layers', 4)
    
    # Validate critical parameters
    if self.hidden_dim % config.get('num_heads', 8) != 0:
        raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({config.get('num_heads')})")
```

### 3.5 Handoff-Ready Component Design
- Include version information
- Document responsible instance
- Provide clear public API boundaries
- Implement verification-friendly interfaces
- Package related components together for handoffs

```python
class EmbeddingComponent(nn.Module):
    """
    Embedding component implementation.
    
    Version: v1.0
    Responsible Instance: 02_model_components
    Handoff Status: Ready
    """
    
    def __init__(self, config):
        # Implementation
        
    def forward(self, x):
        # Implementation
        
    def get_verification_inputs(self):
        """
        Generate sample inputs for verification testing.
        
        Returns:
            Dictionary of sample inputs matching expected interface
        """
        # Implementation
```

## 4. Error Handling & Validation

### 4.1 Input Validation Pattern
- Validate inputs early
- Provide clear error messages
- Include expected vs. actual values
- Document error conditions in interface contracts

```python
def process_batch(batch):
    """Process a batch of data."""
    if 'sequence' not in batch:
        raise ValueError("Batch missing 'sequence' key. Available keys: " + 
                         ", ".join(batch.keys()))
    
    if batch['sequence'].dim() != 2:
        raise ValueError(f"Expected sequence tensor of dim 2, got {batch['sequence'].dim()}")
```

### 4.2 Missing Data Handling
- Be robust to missing feature files
- Return default tensors of appropriate shape
- Log warnings (not errors) for missing optional files
- Document default handling in interface contracts

```python
def load_evolutionary_features(target_id, features_dir):
    """Load evolutionary features if available."""
    filepath = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz")
    
    if not os.path.exists(filepath):
        warnings.warn(f"Evolutionary features not found for {target_id}. Using zeros.")
        # Determine shape from sequence length
        seq_len = get_sequence_length(target_id, features_dir)
        return {
            'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32)
        }
    
    # Normal loading logic
```

### 4.3 Shape Inconsistency Handling
- Check tensor shapes for compatibility
- Raise clear errors for shape mismatches
- Include component name in error messages
- Verify shapes at component boundaries

```python
def forward(self, sequence, pair_features):
    """Forward pass with shape validation."""
    batch_size, seq_len = sequence.shape
    if pair_features.shape != (batch_size, seq_len, seq_len):
        raise ValueError(
            f"TransformerBlock: pair_features shape {pair_features.shape} incompatible "
            f"with sequence shape {sequence.shape}. Expected ({batch_size}, {seq_len}, {seq_len})."
        )
```

### 4.4 Cross-Component Error Propagation
- Use specific exception types for different error categories
- Include context information when re-raising exceptions
- Document error handling in interface contracts
- Provide helpful debugging information

```python
def load_features(target_id, features_dir):
    """Load features with improved error context."""
    try:
        # Load features
        return features
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature file for {target_id} not found in {features_dir}: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid feature format for {target_id}: {e}")
```

## 5. Testing Approach

### 5.1 Unit Test Pattern
- Test each component independently
- Use fixtures for common test data
- Test both success cases and failure modes
- Include test cases specific to handoff verification

```python
def test_load_precomputed_features():
    """Test loading precomputed features."""
    # Setup
    mock_target_id = "test_target"
    mock_features_dir = "/path/to/features"
    
    # Exercise
    with patch("numpy.load") as mock_load:
        mock_load.return_value.__enter__.return_value = {
            "features": np.random.rand(10, 4)
        }
        result = load_precomputed_features(mock_target_id, mock_features_dir)
    
    # Verify
    assert "dihedral" in result
    assert result["dihedral"]["features"].shape == (10, 4)
```

### 5.2 Edge Case Testing
- Test boundary conditions
- Test empty/small/large inputs
- Test invalid inputs
- Include common edge cases in verification tests

```python
def test_collate_fn_empty_batch():
    """Test collate_fn with an empty batch."""
    # Should handle gracefully or raise appropriate error
    
def test_collate_fn_single_item():
    """Test collate_fn with a batch of size 1."""
    # Should work correctly
    
def test_collate_fn_variable_lengths():
    """Test collate_fn with sequences of different lengths."""
    # Should pad correctly and generate valid mask
```

### 5.3 Integration Testing
- Test component combinations
- Verify end-to-end flow
- Check memory consumption
- Document integration patterns in handoff documentation

```python
def test_end_to_end_small():
    """Test end-to-end flow with small inputs."""
    dataset = RNADataset(...)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    model = RNAFoldingModel(config)
    
    batch = next(iter(dataloader))
    outputs = model(batch)
    
    # Verify outputs have expected structure and shapes
```

### 5.4 Handoff Verification Testing
- Create specific tests for verifying component interfaces
- Test with exactly the same input shapes expected in production
- Verify mask handling and padding behavior
- Include memory and performance benchmarks
- Test device handling and transfer

```python
def test_component_verification():
    """Verification test for component handoff."""
    # Create inputs matching exactly what the component will receive
    inputs = create_verification_inputs()
    
    # Run component with these inputs
    outputs = component(**inputs)
    
    # Verify outputs match expected format and behavior
    verify_outputs(outputs)
    
    # Test with different batch sizes and sequence lengths
    for batch_size in [1, 4, 16]:
        for seq_len in [10, 100, 500]:
            # Test and verify
            
    # Test device handling
    if torch.cuda.is_available():
        # Test on GPU and verify
```

## 6. Documentation Standards

### 6.1 Docstring Pattern (Google Style)
```python
def function_name(param1, param2):
    """Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When and why this exception is raised
    """
```

### 6.2 Type Annotations
- Use type hints for all function signatures
- Use `Optional[Type]` for nullable parameters
- Use `Union[Type1, Type2]` for multiple types
- Use `Dict[KeyType, ValueType]` for dictionaries

```python
def load_coordinates(
    labels_df: pd.DataFrame, 
    target_id: str
) -> Tuple[np.ndarray, List[str]]:
    """Load coordinates for target."""
```

### 6.3 Code Comments
- Focus on *why*, not *what*
- Explain complex algorithms
- Document non-obvious design decisions
- Flag potential issues or limitations

```python
# Average the predictions from 5 outputs (NOTE: for v1, we're using only
# the first prediction, but this pattern allows extension to the required
# 5 predictions for final submission)
final_prediction = predictions[0]
```

### 6.4 Interface Contract Documentation
- Create formal interface contracts for all components
- Use the template at `docs/claude/03_code-instances/shared/05_interface_contract_template.md`
- Include all required sections:
  - Component identification and version
  - Input/output interfaces with tensor specifications
  - Error conditions and handling
  - Implementation requirements
  - Usage examples and testing expectations

```markdown
# Component Interface Contract

## Component Identification
- **Component Name**: TransformerBlock
- **Version**: v1.0
- **Responsible Instance**: 02_model_components

## Input Interface
| Parameter | Type | Shape | Device | Description | Required |
|-----------|------|-------|--------|-------------|----------|
| `residue_repr` | `torch.Tensor` | `(batch_size, seq_len, residue_dim)` | Any | Residue representations | Yes |
| `pair_repr` | `torch.Tensor` | `(batch_size, seq_len, seq_len, pair_dim)` | Same as residue_repr | Pair representations | Yes |
| `mask` | `torch.Tensor` | `(batch_size, seq_len)` | Same as residue_repr | Boolean mask (True = valid) | Yes |

## Output Interface
| Return Value | Type | Shape | Device | Description |
|--------------|------|-------|--------|-------------|
| `residue_repr` | `torch.Tensor` | `(batch_size, seq_len, residue_dim)` | Same as input | Updated residue representations |
| `pair_repr` | `torch.Tensor` | `(batch_size, seq_len, seq_len, pair_dim)` | Same as input | Updated pair representations |
```

### 6.5 Implementation Journal
- Maintain an implementation journal following the template
- Update after each implementation session
- Document deviations, issues, and decisions
- Track component status and handoffs
- Include interface documentation as it develops

```markdown
# Implementation Session: 2025-04-15

### Components Completed:
- [x] TransformerBlock implementation
  - Implemented pre-norm architecture
  - Added proper mask handling
  - Used simplified pair update for V1

### Deviations from Plan:
- Used standard nn.MultiheadAttention instead of custom attention
- Simplified pair update mechanism uses outer product without triangle multiplication

### Issues/Questions:
- Unclear how mask should be formatted for nn.MultiheadAttention
  - Current approach: Convert boolean mask (True=valid) to attention mask (0=attend, -inf=ignore)
  - Need confirmation from Integration instance this matches expectations

### Next Steps:
- Finish unit tests for TransformerBlock
- Prepare handoff documentation
- Begin IPAModule placeholder implementation
```

## 7. Component Handoff Preparation

### 7.1 Component Readiness Checklist
- Implementation complete with all features working
- Test coverage ≥90% including edge cases
- Documentation complete and accurate
- Interface contract formalized
- Example usage provided
- Verification tests included

### 7.2 Interface Preparation
- Clarify public vs. private interfaces
- Ensure tensor shapes and types are consistent
- Document device handling behavior
- Include mask propagation details
- Specify error conditions and handling

### 7.3 Handoff Documentation
- Follow template at `docs/claude/03_code-instances/shared/component_handoff_template.md`
- Include component identification and status
- Document public API in detail
- List integration points with other components
- Provide testing requirements and verification steps
- Address common debugging scenarios

### 7.4 Verification Requirements
- Create specific verification tests
- Document expected outputs for sample inputs
- Include performance considerations
- Note any limitations or constraints
- Provide troubleshooting guidance

## 8. Performance Considerations

### 8.1 Memory Efficiency
- Use `torch.no_grad()` for inference
- Release unnecessary tensors
- Consider gradient checkpointing for large models
- Document memory usage patterns in handoff documentation

```python
def predict(self, batch):
    """Make prediction without gradient computation."""
    with torch.no_grad():
        outputs = self.forward(batch)
    return outputs
```

### 8.2 Batch Processing
- Process in batches, not individual samples
- Use vectorized operations when possible
- Avoid unnecessary CPU-GPU transfers
- Document batch size scaling behavior

```python
# CORRECT (vectorized)
result = torch.matmul(features, weights)

# INCORRECT (loop-based)
result = torch.zeros_like(features[:, 0])
for i in range(features.shape[1]):
    result += features[:, i] * weights[i]
```

### 8.3 Device Management
- Check for CUDA availability
- Allow device specification
- Ensure all tensors are on same device
- Document device handling in interface contracts

```python
def __init__(self, config, device=None):
    super().__init__()
    if device is None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        self.device = device
    
    self.to(self.device)  # Move model to device
```

### 8.4 Cross-Component Optimization
- Consider tensor lifecycle across component boundaries
- Document performance characteristics in handoff documentation
- Include scaling behavior with sequence length
- Note memory patterns for different batch sizes
- Provide optimization recommendations

```python
# Memory usage note in handoff documentation
"""
Memory Considerations:
- Peak memory usage scales quadratically with sequence length due to pair representations
- Recommended batch sizes:
  - For L≤100: batch_size=16 (2GB VRAM)
  - For L≤300: batch_size=4 (4GB VRAM)
  - For L≤500: batch_size=1 (8GB VRAM)
- Device transfer overhead is significant for larger batches
"""
```

## 9. Conclusion

These implementation principles provide a foundation for consistent, high-quality code across all instances of the RNA 3D folding project. By adhering to these principles, we ensure:

1. Clear component boundaries with well-defined interfaces
2. Robust error handling and validation
3. Comprehensive testing and verification
4. Efficient knowledge transfer through documentation
5. Smooth component handoffs between instances
6. Optimized performance and resource utilization
7. Compatibility across local development and Kaggle environments

As you implement components, regularly refer to the appropriate instance-specific documentation:
- Data Pipeline: `docs/claude/03_code-instances/01_data_pipeline.md`
- Model Components: `docs/claude/03_code-instances/02_model_components.md`
- Integration: `docs/claude/03_code-instances/03_integration.md`
- Testing: `docs/claude/03_code-instances/04_testing.md`

Follow the handoff protocol documented in `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md` when transitioning components between instances, and maintain your implementation journal to ensure knowledge continuity throughout the project.
