# Neural Network Components Interface Exports

This document provides a detailed specification of all public interfaces exposed by the neural network components for the RNA 3D Folding model. These interfaces are designed to be used by the integration components to build the full model.

## 1. Embedding Components

### 1.1 SequenceEmbedding

```python
class SequenceEmbedding(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize sequence embedding layer.

        Args:
            config: Dictionary containing model parameters:
                - num_embeddings: Number of distinct nucleotides (5 for A,C,G,U,N)
                - seq_embed_dim: Dimension of embedding vectors
                - padding_idx: Index used for padding (default: 4)
        """
        
    def forward(self, sequence_int: torch.Tensor) -> torch.Tensor:
        """
        Convert integer-encoded sequences to embeddings.

        Args:
            sequence_int: Integer tensor of shape (batch_size, seq_len)

        Returns:
            Embedded sequences of shape (batch_size, seq_len, embedding_dim)
        """
```

### 1.2 PositionalEncoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize positional encoding.

        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Embedding dimension
                - max_len: Maximum sequence length to pre-compute (default: 500)
        """
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encodings for sequences of length seq_len.

        Args:
            seq_len: Sequence length to retrieve encodings for

        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
```

### 1.3 RelativePositionalEncoding

```python
class RelativePositionalEncoding(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize relative positional encoding.

        Args:
            config: Dictionary containing model parameters:
                - max_relative_position: Maximum relative distance to consider
                - rel_pos_dim: Dimension of the relative position embedding
        """
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative positional encodings for all position pairs.

        Args:
            seq_len: Sequence length

        Returns:
            Tensor of shape (seq_len, seq_len, rel_pos_dim) with relative
            position embeddings for each position pair
        """
```

### 1.4 EmbeddingModule

```python
class EmbeddingModule(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize the embedding module.

        Args:
            config: Dictionary containing model parameters:
                - seq_embed_dim: Dimension of sequence embeddings
                - residue_embed_dim: Dimension of residue representations
                - pair_embed_dim: Dimension of pair representations
                - use_conservation: Whether to use conservation features
                - Other parameters passed to submodules
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

## 2. Transformer Block

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

## 3. IPA Module

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
                - num_ipa_iterations: Number of iterations (default: 1, for V2)
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

## 4. Configuration Parameters

The complete set of configuration parameters used by the neural network components:

```python
{
    # Sequence embedding parameters
    "num_embeddings": 5,  # Number of nucleotide tokens (A, C, G, U, N/padding)
    "seq_embed_dim": 32,  # Dimension of sequence embeddings
    "padding_idx": 4,  # Index for padding token
    
    # Positional encoding parameters
    "residue_embed_dim": 128,  # Dimension of residue representations
    "max_len": 500,  # Maximum sequence length for positional encodings
    
    # Relative positional encoding parameters
    "max_relative_position": 32,  # Maximum relative distance to consider
    "rel_pos_dim": 32,  # Dimension of relative position embeddings
    
    # Embedding module parameters
    "pair_embed_dim": 64,  # Dimension of pair representations
    "use_conservation": True,  # Whether to use conservation features
    
    # Transformer parameters
    "num_attention_heads": 4,  # Number of attention heads
    "dropout": 0.1,  # Dropout probability
    "ffn_dim": 512,  # Hidden dimension for feed-forward networks (default: 4*residue_dim)
    
    # IPA module parameters
    "ipa_dim": 64,  # Hidden dimension for IPA module
    "num_ipa_iterations": 1,  # Number of IPA iterations (for V2)
}
```

## 5. Usage Examples

### 5.1 Complete Pipeline Example

```python
import torch
from typing import Dict

# Initialize components
config = {
    "num_embeddings": 5,
    "seq_embed_dim": 32,
    "residue_embed_dim": 128,
    "pair_embed_dim": 64,
    "num_attention_heads": 4,
    "dropout": 0.1,
    "use_conservation": True,
    "max_relative_position": 32,
    "rel_pos_dim": 32,
    "ipa_dim": 64,
}

embed_module = EmbeddingModule(config)
transformer_block = TransformerBlock(config)
ipa_module = IPAModule(config)

# Create a batch (normally from data loader)
batch = {
    "sequence_int": torch.randint(0, 5, (2, 10)),  # (batch_size, seq_len)
    "dihedral_features": torch.randn(2, 10, 4),  # (batch_size, seq_len, 4)
    "pairing_probs": torch.rand(2, 10, 10),  # (batch_size, seq_len, seq_len)
    "positional_entropy": torch.rand(2, 10),  # (batch_size, seq_len)
    "coupling_matrix": torch.rand(2, 10, 10),  # (batch_size, seq_len, seq_len)
    "accessibility": torch.rand(2, 10),  # (batch_size, seq_len)
    "conservation": torch.rand(2, 10),  # (batch_size, seq_len)
    "mask": torch.ones(2, 10, dtype=torch.bool),  # (batch_size, seq_len)
}

# Process through the pipeline
residue_repr, pair_repr, mask = embed_module(batch)
residue_repr, pair_repr = transformer_block(residue_repr, pair_repr, mask)
coords = ipa_module(residue_repr, pair_repr, mask)

print(f"Coordinates shape: {coords.shape}")  # Expected: (2, 10, 3)
```

### 5.2 Multi-Layer Transformer Example

```python
def create_transformer_stack(config: Dict, num_layers: int = 3):
    """Create a stack of transformer blocks."""
    return nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

transformer_stack = create_transformer_stack(config, num_layers=3)

# Process through multiple transformer layers
for transformer in transformer_stack:
    residue_repr, pair_repr = transformer(residue_repr, pair_repr, mask)

# Final coordinate prediction
coords = ipa_module(residue_repr, pair_repr, mask)
```

## 6. Integration Notes

### 6.1 Core Integration Requirements

1. **Consistent Configuration**:
   - All components must be initialized with a consistent config dictionary
   - Configuration parameters should be validated for compatibility between components
   - Example: `residue_embed_dim` must be the same across all components

2. **Tensor Shape Contracts**:
   - **Residue representations**: Always (batch_size, seq_len, residue_dim)
   - **Pair representations**: Always (batch_size, seq_len, seq_len, pair_dim)
   - **Masks**: Always (batch_size, seq_len) with dtype=torch.bool
   - **Coordinates**: Always (batch_size, seq_len, 3)

3. **Batch Structure**:
   - EmbeddingModule expects the exact batch structure from the data_loading module
   - Required fields: 'sequence_int', 'dihedral_features', 'pairing_probs', 'positional_entropy', 'coupling_matrix', 'accessibility', 'mask'
   - Optional fields: 'conservation' (controlled by use_conservation config)

### 6.2 Mask Handling Protocol

1. **Mask Format**:
   - Boolean tensors where True indicates valid positions, False indicates padding
   - Shape: (batch_size, seq_len)
   - Must be passed consistently through all components

2. **Mask Propagation**:
   - **EmbeddingModule**: Creates 1D and 2D masks for residue and pair representations
   - **TransformerBlock**: Uses mask for both attention and element-wise operations
   - **IPAModule**: Applies mask to zero out coordinates for padded positions

3. **Common Mask-Related Issues**:
   - Forgetting to use key_padding_mask in nn.MultiheadAttention (True/False semantics reversed)
   - Failing to apply the mask after residual connections
   - Not converting bool mask to float for element-wise operations

### 6.3 Device Handling Protocol

1. **Component Movement**:
   - All components inherit from nn.Module and support .to(device) operation
   - All components handle CPU and CUDA tensors appropriately
   - Example: `model.to('cuda')` moves all parameters and buffers to GPU

2. **Tensor Device Consistency**:
   - Input tensors must be on the same device as the model
   - Batch dictionary should be moved to the correct device before passing to EmbeddingModule
   - Example:
     ```python
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model = model.to(device)
     batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
     ```

3. **Device-Specific Optimizations**:
   - For CUDA, consider enabling mixed precision with torch.cuda.amp
   - For CPU, ensure operations utilize multiple cores effectively

### 6.4 Gradient Flow and Training

1. **Gradient Propagation**:
   - All operations support autograd for end-to-end training
   - No gradient detachment occurs within components unless explicitly noted
   - Mask operations are implemented to preserve gradient flow on valid positions

2. **Numerical Stability**:
   - Pre-normalization architecture provides better gradient stability
   - All components include epsilon terms in sensitive operations
   - Consider gradient clipping for stable training: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 6.5 Integration Example for Full Model

For the full RNAFoldingModel integration, components should be combined as follows:

```python
class RNAFoldingModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        
        # Initialize components
        self.embedding_module = EmbeddingModule(config)
        
        # Multiple transformer layers
        self.num_layers = config.get("num_layers", 4)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.num_layers)
        ])
        
        # Coordinate prediction
        self.ipa_module = IPAModule(config)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Get initial representations from embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        
        # Process through transformer blocks
        for transformer in self.transformer_blocks:
            residue_repr, pair_repr = transformer(residue_repr, pair_repr, mask)
        
        # Predict coordinates
        coords = self.ipa_module(residue_repr, pair_repr, mask)
        
        # Return predictions
        return {
            "predicted_coords": coords,
            "residue_repr": residue_repr,
            "pair_repr": pair_repr,
            "mask": mask
        }
```

This full model implementation maintains all the necessary tensor shape contracts, mask propagation, and device handling requirements established by the individual components.