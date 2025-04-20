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

When integrating these components:

1. **Initialization**: All components should be initialized with a consistent config dictionary
2. **Batch Structure**: The EmbeddingModule expects the batch structure from the data loader
3. **Mask Propagation**: Always propagate the mask through all components to handle variable-length sequences
4. **Device Handling**: All components support moving to GPU with `.to('cuda')`
5. **Gradient Flow**: All operations support autograd for end-to-end training

For the full model integration, these components should be combined as shown in the usage examples, with the number of transformer blocks determined by model complexity requirements.