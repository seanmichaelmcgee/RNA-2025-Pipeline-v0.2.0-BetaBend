# Neural Network Components Handoff Document

## 1. Component Identification

**Component Name:** RNA 3D Folding Neural Network Core Components  
**Instance ID:** instance_02_model  
**Primary Functions:** Sequence embedding, transformer-based representation learning, coordinate prediction  
**Repository Path:** `/src/models/embeddings.py`, `/src/models/transformer_block.py`, `/src/models/ipa_module.py`  
**Handoff Date:** 2025-04-20

## 2. Implementation Status

### 2.1 Completed Components

| Component | Status | Tests | Documentation | Last Updated |
|-----------|--------|-------|---------------|--------------|
| SequenceEmbedding | Complete | Complete | Complete | 2025-04-20 |
| PositionalEncoding | Complete | Complete | Complete | 2025-04-20 |
| RelativePositionalEncoding | Complete | Complete | Complete | 2025-04-20 |
| EmbeddingModule | Complete | Complete | Complete | 2025-04-20 |
| TransformerBlock | Complete | Complete | Complete | 2025-04-20 |
| IPAModule (V1 Placeholder) | Complete | Complete | Complete | 2025-04-20 |

### 2.2 Pending Components

| Component | Current State | Dependencies | Priority | Estimated Complexity |
|-----------|---------------|--------------|----------|----------------------|
| Full IPAModule (V2) | Not Started | TransformerBlock, EmbeddingModule | High | High |
| RNAFoldingModel | Not Started | All core components | High | Medium |
| TopologyLayer | Not Started | IPAModule | Medium | Medium |

### 2.3 Known Issues

| Issue | Severity | Components Affected | Potential Solutions |
|-------|----------|---------------------|---------------------|
| Simplified IPA implementation | Medium | IPAModule | Implement full IPA algorithm with frames in V2 |
| Conservation feature handling | Low | EmbeddingModule | Add more robust handling when feature is missing |
| Large sequence scalability | Low | TransformerBlock | Consider linear attention or similar techniques for very long sequences |

## 3. Interface Contracts

### 3.1 Public API

The key public components are:

```python
class EmbeddingModule(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize the embedding module.
        
        Args:
            config: Dictionary containing model parameters
        """
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input features to create initial residue and pair representations.
        
        Args:
            batch: Dictionary of input tensors from the data loader
            
        Returns:
            Tuple of (residue_repr, pair_repr, mask):
                - residue_repr: Tensor of shape (batch_size, seq_len, residue_dim)
                - pair_repr: Tensor of shape (batch_size, seq_len, seq_len, pair_dim)
                - mask: Boolean mask of shape (batch_size, seq_len)
        """
```

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize transformer block.
        
        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings
                - num_attention_heads: Number of attention heads
                - dropout: Dropout probability
                - ffn_dim: Hidden dimension for feed-forward networks
        """
        
    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
            
        Returns:
            Tuple of:
            - Updated residue representations of shape (batch_size, seq_len, residue_dim)
            - Updated pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        """
```

```python
class IPAModule(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize IPA module.
        
        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings (unused in V1)
                - ipa_dim: Hidden dimension for IPA module projection (optional)
        """
        
    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the IPA module.
        
        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
                       Not used in V1 but included in interface for future versions
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
            
        Returns:
            Predicted coordinates of shape (batch_size, seq_len, 3)
        """
```

### 3.2 Data Structures

The main data structures used by these components:

```python
# Configuration dictionary for initializing components
{
    "num_embeddings": int,  # Number of distinct nucleotides (default: 5)
    "seq_embed_dim": int,  # Dimension of sequence embeddings (default: 32)
    "padding_idx": int,  # Padding index in embedding table (default: 4)
    "residue_embed_dim": int,  # Dimension of residue representations (default: 128)
    "pair_embed_dim": int,  # Dimension of pair representations (default: 64)
    "max_relative_position": int,  # Maximum relative position to encode (default: 32)
    "rel_pos_dim": int,  # Dimension of relative position embeddings (default: 32)
    "use_conservation": bool,  # Whether to use conservation features (default: True)
    "num_attention_heads": int,  # Number of attention heads (default: 4)
    "dropout": float,  # Dropout probability (default: 0.1)
    "ffn_dim": int,  # Hidden dimension for feed-forward networks (default: 4*residue_dim)
    "ipa_dim": int,  # Hidden dimension for IPA module (default: residue_dim/2)
    "num_ipa_iterations": int,  # Number of IPA iterations (default: 1, used in V2)
}
```

```python
# Input batch dictionary expected by EmbeddingModule
{
    "sequence_int": torch.Tensor,  # Integer-encoded sequences (batch_size, seq_len)
    "dihedral_features": torch.Tensor,  # Dihedral angles (batch_size, seq_len, 4)
    "pairing_probs": torch.Tensor,  # Pairing probabilities (batch_size, seq_len, seq_len)
    "positional_entropy": torch.Tensor,  # Positional entropy (batch_size, seq_len)
    "coupling_matrix": torch.Tensor,  # Coupling matrix (batch_size, seq_len, seq_len)
    "accessibility": torch.Tensor,  # Accessibility (batch_size, seq_len)
    "mask": torch.Tensor,  # Boolean mask (batch_size, seq_len)
    "conservation": torch.Tensor,  # Optional conservation (batch_size, seq_len)
}
```

### 3.3 Integration Points

| Consumer Component | Integration Function | Expected Behavior | Error Handling |
|-------------------|---------------------|-------------------|----------------|
| RNAFoldingModel | EmbeddingModule.forward() | Processes batch data into residue and pair representations | Handles missing conservation feature, applies masks |
| RNAFoldingModel | TransformerBlock.forward() | Updates residue and pair representations | Propagates mask to ensure padding consistency |
| RNAFoldingModel | IPAModule.forward() | Predicts 3D coordinates from representations | Applies mask to ensure padding consistency |
| Training Loop | TransformerBlock.forward() | Enables gradient flow for backpropagation | N/A |
| Visualization | IPAModule.forward() | Provides 3D coordinates for structure visualization | N/A |

## 4. Environment and Dependencies

### 4.1 Runtime Requirements

- Python version: 3.10+
- Memory requirements: 8GB+ (16GB+ recommended for transformer with long sequences)
- GPU requirements: CUDA compatible GPU recommended for training
- Environment variables: None required

### 4.2 Package Dependencies

| Package | Version | Purpose | Installation Command |
|---------|---------|---------|---------------------|
| PyTorch | 2.1+ | Neural network operations | `conda install pytorch -c pytorch` |
| einops | 0.6+ | Tensor operations for transformers | `conda install -c conda-forge einops` |

### 4.3 File Dependencies

| File Path | Purpose | Source |
|-----------|---------|--------|
| src/models/embeddings.py | Sequence and pair embedding | This implementation |
| src/models/transformer_block.py | Representation learning | This implementation |
| src/models/ipa_module.py | Coordinate prediction | This implementation |
| src/data_loading.py | Data loading interface | Data pipeline component |
| src/losses.py | Loss functions (for integration) | Losses component |

## 5. Testing Requirements

### 5.1 Test Coverage

| Component | Unit Tests | Integration Tests | Manual Tests Required |
|-----------|------------|-------------------|----------------------|
| EmbeddingModule | Yes | Yes | No |
| TransformerBlock | Yes | Yes | No |
| IPAModule | Yes | Yes | No |

### 5.2 Critical Test Cases

| Test Case | Purpose | Command | Expected Output |
|-----------|---------|---------|----------------|
| test_embedding_module.py | Tests embedding functionality | `python -m pytest tests/test_embedding_module.py -v` | All tests pass |
| test_transformer_block.py | Tests transformer functionality | `python -m pytest tests/test_transformer_block.py -v` | All tests pass |
| test_ipa_module.py | Tests IPA module functionality | `python -m pytest tests/test_ipa_module.py -v` | All tests pass |
| test_model_gpu.py | Tests GPU compatibility | `python -m pytest tests/test_model_gpu.py -v` | All tests pass (if GPU available) |

### 5.3 Integration Verification

To verify successful integration:

1. Run all component tests:
   ```python
   python -m pytest tests/test_embedding_module.py tests/test_transformer_block.py tests/test_ipa_module.py -v
   ```
2. Create a mini-batch and verify correct shape processing:
   ```python
   # Create a data loader
   loader = create_data_loader(...)
   batch = next(iter(loader))
   
   # Initialize components
   embed_module = EmbeddingModule(config)
   transformer = TransformerBlock(config)
   ipa = IPAModule(config)
   
   # Process batch
   residue_repr, pair_repr, mask = embed_module(batch)
   residue_repr, pair_repr = transformer(residue_repr, pair_repr, mask)
   coords = ipa(residue_repr, pair_repr, mask)
   
   # Verify shape: coords should be (batch_size, seq_len, 3)
   assert coords.shape == (batch_size, seq_len, 3)
   ```
3. Verify mask handling across components:
   - Check that padded positions remain zeroed in all outputs
   - Verify that gradients don't flow through padded positions

4. Test with both CPU and GPU (if available):
   ```python
   # Move components to GPU
   embed_module.to('cuda')
   transformer.to('cuda')
   ipa.to('cuda')
   
   # Move batch to GPU
   batch = {k: v.to('cuda') for k, v in batch.items() if isinstance(v, torch.Tensor)}
   
   # Process batch on GPU
   residue_repr, pair_repr, mask = embed_module(batch)
   residue_repr, pair_repr = transformer(residue_repr, pair_repr, mask)
   coords = ipa(residue_repr, pair_repr, mask)
   ```

## 6. Implementation Details

### 6.1 Architecture Overview

The neural network component architecture follows:

```
Data Loader → EmbeddingModule → TransformerBlock(s) → IPAModule → Coordinates
    ↓             ↓                    ↓                  ↓
  Batch → Residue/Pair Repr → Updated Repr → 3D Coordinates
```

Each component processes:
- **EmbeddingModule**: Converts raw features to learned representations
- **TransformerBlock**: Refines representations through self-attention
- **IPAModule**: Projects representations to 3D coordinates

### 6.2 Algorithms and Data Structures

- **Sequence Embedding**: Standard embedding lookup for nucleotide tokens
- **Positional Encoding**: Sinusoidal positional encodings for absolute positions
- **Relative Positional Encoding**: Learned embeddings for relative positions between residues
- **Self-Attention**: Multi-head attention for modeling global relationships
- **Pair Representation**: Outer product-based updates for capturing residue-pair interactions
- **IPA V1 (Placeholder)**: Simple MLP projection for initial coordinate prediction
- **Pre-Normalization**: LayerNorm before each sub-module for better gradient flow

### 6.3 Performance Considerations

- **Time complexity**:
  - EmbeddingModule: O(L + L²) where L is sequence length
  - TransformerBlock: O(L²) for self-attention, O(L²) for pair updates
  - IPAModule V1: O(L) for coordinate prediction
- **Space complexity**:
  - EmbeddingModule: O(L²) for pair representations
  - TransformerBlock: O(L²) for attention weights and pair representations
  - IPAModule V1: O(L) for coordinates
- **Bottlenecks**:
  - Self-attention for long sequences (quadratic complexity)
  - Pair representation memory usage (quadratic in sequence length)
- **Optimization opportunities**:
  - Linear attention variants for TransformerBlock
  - Sparse representation for pair matrices
  - Mixed precision training for GPU acceleration

## 7. Extension and Maintenance

### 7.1 Anticipated Extensions

1. **Full IPA Implementation**: Replace the placeholder with the complete Invariant Point Attention algorithm
2. **Iterative Refinement**: Add support for iterative coordinate refinement
3. **Frame-based Representations**: Extend the model to predict full backbone frames
4. **Linear Attention**: Implement efficient attention variants for long sequences
5. **Embedding Factorization**: Factorize pair representation for memory efficiency

### 7.2 Maintenance Considerations

- **Regular updates**: Review attention mechanisms against latest research
- **Monitoring**: Track memory usage with large batches
- **Technical debt**:
  - IPAModule is a simplified placeholder requiring replacement with full implementation
  - Attention implementation could be optimized for better performance

## 8. Common Debugging Scenarios

| Symptom | Likely Cause | Debugging Steps | Solution |
|---------|--------------|----------------|----------|
| CUDA out of memory | Large sequence batch | Monitor memory usage, reduce batch size | Use gradient accumulation or smaller batches |
| NaN loss values | Numerical instability in attention | Check attention weights for extreme values | Add epsilon values, use layer normalization |
| Vanishing/exploding gradients | Improper initialization | Monitor gradient norms during training | Adjust initialization, use gradient clipping |
| Feature mismatch errors | Incorrect feature dimensions | Check data loader output shapes | Ensure consistent feature shapes |
| Poor mask handling | Incorrect mask propagation | Verify that padded positions stay zero | Apply mask consistently at each layer |

## 9. Decision Log

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| Pre-norm architecture | Better training stability | Post-norm architecture | 2025-04-20 |
| Simplified IPA (V1) | Establish interface first | Implementing full IPA directly | 2025-04-20 |
| Optional conservation features | Flexibility for datasets without conservation | Requiring all features | 2025-04-20 |
| Explicit mask handling | Consistency for variable-length sequences | Implicit masking via attention | 2025-04-20 |
| Pre-computed relative positions | Efficiency vs computing on-the-fly | Dynamic computation | 2025-04-20 |

## 10. Handoff Checklist

- [x] All code pushed to repository
- [x] All tests passing
- [x] Documentation updated
- [x] Interface contracts finalized
- [x] Known issues documented
- [x] Integration points verified
- [x] Handoff template completed
- [ ] Knowledge transfer session completed
- [ ] Receiving agent has run tests successfully
- [ ] Receiving agent has access to all necessary resources

## 11. Contact Information

**Handoff Agent:** instance_02_model  
**Receiving Agent:** instance_03_integration  
**Supervisor:** RNA 2025 Project Lead  
**Knowledge Transfer Session Date:** To be scheduled  

---

## Implementation Notes on Embedding Module

The `EmbeddingModule` establishes the foundation for the RNA 3D folding model by integrating multiple feature types:

1. **Sequence Features**:
   - Integer-encoded nucleotide sequences converted to learned embeddings
   - Positional information through sinusoidal encoding
   - Relative positional information between nucleotides

2. **Structural Features**:
   - Dihedral angles (4 per residue)
   - Pairing probabilities (2D matrix)
   - Positional entropy and accessibility

3. **Evolutionary Features**:
   - Coupling matrix from multiple sequence alignment
   - Conservation profiles (optional)

The implementation:
- Handles optional features gracefully
- Applies consistent masking for padded positions
- Projects concatenated features to fixed dimensionality
- Establishes the dual-representation approach (residue + pair)

## Implementation Notes on Transformer Block

The `TransformerBlock` implements a pre-norm transformer architecture that processes both residue and pair representations:

1. **Residue Update**:
   - LayerNorm → Self-attention → Dropout → Residual connection
   - LayerNorm → Feed-forward network → Dropout → Residual connection

2. **Pair Update**:
   - LayerNorm → Outer product of residue representations
   - Concatenate [h_i, h_j, pair_ij] → MLP → Dropout → Residual connection

The implementation:
- Uses PyTorch's native MultiheadAttention
- Properly handles masks for padded positions
- Ensures efficient tensor operations
- Maintains consistent dimensions throughout

## Implementation Notes on IPA Module (V1)

The `IPAModule` provides a simplified V1 implementation:

1. **Current Implementation**:
   - Simple MLP projection from residue representations to 3D coordinates
   - Proper mask handling for padded positions
   - Interface designed for backwards compatibility with future versions

2. **Future V2 Extensions**:
   - Frame-based representations
   - Iterative coordinate refinement
   - Full invariant point attention implementation

This placeholder implementation establishes the expected interface while allowing for incremental development of the more complex components.