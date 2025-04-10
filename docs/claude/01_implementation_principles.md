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

## 3. PyTorch Implementation Patterns

### 3.1 Module Design
- Inherit from `torch.nn.Module` for all model components
- Implement `forward()` method with clear input/output documentation
- Initialize parameters in `__init__` method
- Use `nn.Parameter` for learnable parameters

```python
class TransformerBlock(nn.Module):
    """Transformer block implementation."""
    
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

```python
def __init__(self, config):
    super().__init__()
    self.hidden_dim = config.get('hidden_dim', 128)
    self.num_layers = config.get('num_layers', 4)
    
    # Validate critical parameters
    if self.hidden_dim % config.get('num_heads', 8) != 0:
        raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({config.get('num_heads')})")
```

## 4. Error Handling & Validation

### 4.1 Input Validation Pattern
- Validate inputs early
- Provide clear error messages
- Include expected vs. actual values

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

```python
def load_evolutionary_features(target_id, features_dir):
    """Load evolutionary features if available."""
    filepath = os.path.join(features_dir, f"{target_id}_features.npz")
    
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

## 5. Testing Approach

### 5.1 Unit Test Pattern
- Test each component independently
- Use fixtures for common test data
- Test both success cases and failure modes

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

## 7. Performance Considerations

### 7.1 Memory Efficiency
- Use `torch.no_grad()` for inference
- Release unnecessary tensors
- Consider gradient checkpointing for large models

```python
def predict(self, batch):
    """Make prediction without gradient computation."""
    with torch.no_grad():
        outputs = self.forward(batch)
    return outputs
```

### 7.2 Batch Processing
- Process in batches, not individual samples
- Use vectorized operations when possible
- Avoid unnecessary CPU-GPU transfers

```python
# CORRECT (vectorized)
result = torch.matmul(features, weights)

# INCORRECT (loop-based)
result = torch.zeros_like(features[:, 0])
for i in range(features.shape[1]):
    result += features[:, i] * weights[i]
```

### 7.3 Device Management
- Check for CUDA availability
- Allow device specification
- Ensure all tensors are on same device

```python
def __init__(self, config, device=None):
    super().__init__()
    if device is None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        self.device = device
    
    self.to(self.device)  # Move model to device
```

## 8. Next Steps

After reviewing these principles, proceed to implementing individual components following the roadmap in the master guide. Start with the data loading implementation.

Refer to this document when you encounter questions about project-wide patterns or practices. Component-specific guides will provide more detailed implementation instructions.
