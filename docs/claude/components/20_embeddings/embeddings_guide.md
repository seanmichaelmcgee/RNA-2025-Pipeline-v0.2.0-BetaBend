# Embeddings Implementation Guide

## Component Overview

The embeddings component is responsible for transforming input RNA sequence data and precomputed features into learned representations suitable for processing by the transformer backbone. This component forms the foundation of the neural network's ability to understand RNA sequences and their properties, creating both per-residue and per-pair embeddings that capture the essential information needed for 3D structure prediction.

## Requirements Reference

From the Product Requirements Document (`4_Product_Requirements_V1.md`):

- **MA-03**: Implement input embedding layers: `SequenceEmbedding`, `PositionalEncoding`, `RelativePositionalEncoding`.
- **MA-04**: Implement input linear projection layers for residue features (seq, dihedral, pair status, etc.) and pair features (pair probs, MI, rel pos).

## Technical Background

### Embedding Concepts

1. **Sequence Embeddings**: Learned representations that map discrete nucleotide tokens (A, C, G, U) to continuous vector spaces, allowing the model to capture semantic relationships between nucleotides.

2. **Positional Encodings**: Fixed or learned patterns that provide information about the position of each nucleotide in the sequence, enabling the attention mechanism to be aware of sequential order.

3. **Relative Positional Encodings**: Representations of the relative distance or relationship between pairs of nucleotides, critical for understanding RNA structure where distant residues in sequence may be close in 3D space.

4. **Feature Projections**: Linear transformations that map heterogeneous input features (with varying dimensions) to a common embedding space with uniform dimensionality.

## Interfaces

### Input Interface

The embedding components take various inputs from the data loading pipeline:

```python
# For sequence embedding
sequence_int: torch.Tensor  # Shape: (batch_size, seq_len), dtype: torch.long
                           # Integer-encoded RNA sequence (A=0, C=1, G=2, U=3, N=4)

# For positional encoding
seq_len: int  # Length of the sequence

# For feature projections
dihedral_features: torch.Tensor  # Shape: (batch_size, seq_len, 4), dtype: torch.float32
pairing_probs: torch.Tensor      # Shape: (batch_size, seq_len, seq_len), dtype: torch.float32
positional_entropy: torch.Tensor # Shape: (batch_size, seq_len), dtype: torch.float32
coupling_matrix: torch.Tensor    # Shape: (batch_size, seq_len, seq_len), dtype: torch.float32
mask: torch.Tensor               # Shape: (batch_size, seq_len), dtype: torch.bool
```

### Output Interface

The embedding components produce:

```python
# Sequence embedding output
sequence_embedding: torch.Tensor  # Shape: (batch_size, seq_len, seq_embed_dim)

# Positional encoding output
positional_encoding: torch.Tensor  # Shape: (1, max_len, residue_embed_dim)
                                  # Can be expanded to batch dimension later

# Residue representation (after projection)
residue_repr: torch.Tensor  # Shape: (batch_size, seq_len, residue_embed_dim)

# Pair representation (after projection)
pair_repr: torch.Tensor  # Shape: (batch_size, seq_len, seq_len, pair_embed_dim)
```

## Implementation Steps

### 1. Implement `SequenceEmbedding` Module

```python
class SequenceEmbedding(nn.Module):
    """
    Embedding layer for RNA nucleotide sequences.
    Maps integer-encoded nucleotides to learned embeddings.
    """
    def __init__(self, num_embeddings=5, embedding_dim=32):
        """
        Initialize sequence embedding layer.
        
        Args:
            num_embeddings: Number of distinct nucleotides (5 for A,C,G,U,N)
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0  # Assuming 0 is used for padding
        )
    
    def forward(self, sequence_int):
        """
        Convert integer-encoded sequences to embeddings.
        
        Args:
            sequence_int: Integer tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded sequences of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(sequence_int)
```

### 2. Implement `PositionalEncoding` Module

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for RNA sequences.
    Provides position information to the model.
    """
    def __init__(self, embed_dim=128, max_len=500):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension, should match residue_embed_dim
            max_len: Maximum sequence length to pre-compute
        """
        super().__init__()
        
        # Create constant positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, seq_len):
        """
        Get positional encodings for sequences of length seq_len.
        
        Args:
            seq_len: Sequence length to retrieve encodings for
            
        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
        return self.pe[:, :seq_len]
```

### 3. Implement `RelativePositionalEncoding` Module

```python
class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for pairs of nucleotides.
    Encodes the distance between positions in the sequence.
    """
    def __init__(self, max_relative_position=32, num_units=32):
        """
        Initialize relative positional encoding.
        
        Args:
            max_relative_position: Maximum relative distance to consider
            num_units: Dimension of the relative position embedding
        """
        super().__init__()
        
        # Create embedding for relative positions
        self.max_relative_position = max_relative_position
        # Total embeddings: 2*max_rel_pos + 1 (to account for -max to +max)
        num_embeddings = 2 * max_relative_position + 1
        self.embeddings = nn.Embedding(num_embeddings, num_units)
        
        # Initialize with sinusoidal pattern
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights with sinusoidal pattern."""
        position = torch.arange(
            -self.max_relative_position,
            self.max_relative_position + 1
        ).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embeddings.embedding_dim, 2).float() * 
            (-math.log(10000.0) / self.embeddings.embedding_dim)
        )
        
        pe = torch.zeros(2 * self.max_relative_position + 1, self.embeddings.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Set embedding weights
        with torch.no_grad():
            self.embeddings.weight.copy_(pe)
    
    def forward(self, seq_len):
        """
        Compute relative positional encodings for all position pairs.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tensor of shape (seq_len, seq_len, embedding_dim) with relative
            position embeddings for each position pair
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.embeddings.weight.device)
        
        # Compute relative positions between all position pairs
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip relative positions to max_relative_position
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position,
            0, 2 * self.max_relative_position
        )
        
        # Get embeddings for all position pairs
        return self.embeddings(relative_positions)
```

### 4. Implement Residue and Pair Feature Projection Functions

These functions will be part of the main model but are crucial for processing embeddings:

```python
def project_residue_features(features_dict, config):
    """
    Project per-residue features to a common embedding space.
    
    Args:
        features_dict: Dictionary of input features
        config: Model configuration
        
    Returns:
        Projected residue representations
    """
    # Extract dimensions from config
    residue_dim = config['residue_embed_dim']
    seq_embed_dim = config['seq_embed_dim']
    
    # Get batch size and sequence length
    batch_size, seq_len = features_dict['sequence_int'].shape
    
    # Get sequence embeddings and positional encodings
    seq_embedding = features_dict['sequence_embedding']  # (B, L, seq_embed_dim)
    pos_encoding = features_dict['positional_encoding']  # (1, L, residue_dim)
    
    # Concatenate all per-residue features
    # Shape: (B, L, in_features)
    residue_features = [
        seq_embedding,                             # (B, L, seq_embed_dim)
        features_dict['dihedral_features'],        # (B, L, 4)
        features_dict['positional_entropy'].unsqueeze(-1),  # (B, L, 1)
        features_dict['accessibility'].unsqueeze(-1),       # (B, L, 1)
    ]
    
    # Optional features if available
    if 'conservation' in features_dict:
        residue_features.append(features_dict['conservation'].unsqueeze(-1))
    
    # Concatenate along feature dimension
    residue_features = torch.cat(residue_features, dim=-1)
    
    # Project to residue embedding dimension
    residue_projection = nn.Linear(residue_features.shape[-1], residue_dim)
    residue_repr = residue_projection(residue_features)
    
    # Add positional encodings
    pos_encoding = pos_encoding.expand(batch_size, -1, -1)
    residue_repr = residue_repr + pos_encoding
    
    return residue_repr

def project_pair_features(features_dict, config):
    """
    Project per-pair features to a common embedding space.
    
    Args:
        features_dict: Dictionary of input features
        config: Model configuration
        
    Returns:
        Projected pair representations
    """
    # Extract dimensions from config
    pair_dim = config['pair_embed_dim']
    
    # Get batch size and sequence length
    batch_size, seq_len = features_dict['sequence_int'].shape
    
    # Get relative positional encodings
    rel_pos = features_dict['relative_pos_encoding']  # (L, L, rel_pos_dim)
    
    # Concatenate all pair features
    # Shape: (B, L, L, in_features)
    pair_features = [
        features_dict['pairing_probs'].unsqueeze(-1),   # (B, L, L, 1)
        features_dict['coupling_matrix'].unsqueeze(-1), # (B, L, L, 1)
        rel_pos.expand(batch_size, -1, -1, -1)          # (B, L, L, rel_pos_dim)
    ]
    
    # Concatenate along feature dimension
    pair_features = torch.cat(pair_features, dim=-1)
    
    # Project to pair embedding dimension
    pair_projection = nn.Linear(pair_features.shape[-1], pair_dim)
    pair_repr = pair_projection(pair_features)
    
    return pair_repr
```

## Critical Aspects

### 1. Feature Dimensionality

- **Input Tracking**: Keep track of the exact dimensionality of all input features to ensure correct sizes in linear projections.
- **Residue Features**: The dimension of combined residue features is the sum of:
  - Sequence embedding dim
  - Dihedral features dim (4 for sin/cos of 2 angles)
  - Positional entropy (1)
  - Accessibility (1)
  - Conservation (1, if available)
- **Pair Features**: The dimension of combined pair features is the sum of:
  - Pairing probability (1)
  - Coupling matrix (1)
  - Relative positional encoding dim

### 2. Device Management

- All embedding components should respect the device specified or inferred from input tensors.
- For pre-computed tensors like positional encodings, ensure they're moved to the correct device before use.

```python
# Example of proper device handling
def forward(self, x):
    device = x.device
    pe = self.pe.to(device)
    # Use pe with x
```

### 3. Initialization

- Proper initialization is crucial for model convergence.
- Sequence embeddings should use default PyTorch initialization.
- Positional encodings use fixed sinusoidal patterns.
- Consider Glorot/Xavier or Kaiming initialization for projection layers.

### 4. Configuration Parameters

The embeddings components rely on several key configuration parameters:

```python
config = {
    'seq_embed_dim': 32,         # Dimension of sequence embeddings
    'residue_embed_dim': 128,    # Dimension of residue representations
    'pair_embed_dim': 64,        # Dimension of pair representations
    'max_relative_position': 32, # Maximum relative position for relative encodings
}
```

## Testing Requirements

### Basic Functionality Tests

1. **SequenceEmbedding**:
   - Verify output shape is correct for different batch sizes and sequence lengths
   - Check if padding index is handled correctly
   - Ensure gradients flow properly

2. **PositionalEncoding**:
   - Verify output shape is correct
   - Check if encodings are consistent for the same positions
   - Test with different maximum lengths

3. **RelativePositionalEncoding**:
   - Verify output shape is correct
   - Test symmetry properties (relative pos i->j should relate to j->i)
   - Check boundary handling

4. **Projection Functions**:
   - Verify output shapes match expected dimensions
   - Test with different input feature combinations
   - Ensure mask values don't contribute to the result

### Integration Tests

1. Test combining all embedding components to create initial residue and pair representations
2. Verify these representations can be properly consumed by the transformer block
3. Test with realistic input data from the data loading component

## Example Usage

Here's a complete example showing how to use the embedding components:

```python
import torch
import torch.nn as nn
import math

# Create embedding components
sequence_embedding = SequenceEmbedding(
    num_embeddings=5,  # A, C, G, U, N/padding
    embedding_dim=32   # From config
)

positional_encoding = PositionalEncoding(
    embed_dim=128,     # From config, matches residue_embed_dim
    max_len=500        # Maximum expected sequence length
)

relative_pos_encoding = RelativePositionalEncoding(
    max_relative_position=32,  # From config
    num_units=32               # From config
)

# Create projection layers (these would be in the main model)
residue_projection = nn.Linear(
    in_features=32 + 4 + 1 + 1,  # seq_emb + dihedral + entropy + accessibility
    out_features=128             # residue_embed_dim
)

pair_projection = nn.Linear(
    in_features=1 + 1 + 32,  # pairing_prob + coupling + rel_pos
    out_features=64          # pair_embed_dim
)

# Mock batch of data
batch_size = 2
seq_len = 10
mock_batch = {
    'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
    'dihedral_features': torch.rand(batch_size, seq_len, 4),
    'positional_entropy': torch.rand(batch_size, seq_len),
    'accessibility': torch.rand(batch_size, seq_len),
    'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
    'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
    'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
}

# Apply embeddings
seq_emb = sequence_embedding(mock_batch['sequence_int'])
pos_enc = positional_encoding(seq_len)
rel_pos = relative_pos_encoding(seq_len)

# Combine features for residue representation
features_to_concat = [
    seq_emb,
    mock_batch['dihedral_features'],
    mock_batch['positional_entropy'].unsqueeze(-1),
    mock_batch['accessibility'].unsqueeze(-1)
]
combined_features = torch.cat(features_to_concat, dim=-1)

# Project to residue embedding dimension
residue_repr = residue_projection(combined_features)

# Add positional encodings
residue_repr = residue_repr + pos_enc.expand(batch_size, -1, -1)

# Create pair representation
pair_features = [
    mock_batch['pairing_probs'].unsqueeze(-1),
    mock_batch['coupling_matrix'].unsqueeze(-1),
    rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
]
combined_pair_features = torch.cat(pair_features, dim=-1)

# Project to pair embedding dimension
pair_repr = pair_projection(combined_pair_features)

print(f"Residue representation shape: {residue_repr.shape}")  # [2, 10, 128]
print(f"Pair representation shape: {pair_repr.shape}")        # [2, 10, 10, 64]
```

## Related Documentation

- **Architecture Specification**: `docs/3_Architecture_Specification.md` - See "Feature Embedding and Representation Initialization" section
- **Product Requirements**: `docs/4_Product_Requirements_V1.md` - Requirements MA-03 and MA-04
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md` - For device management and module design

## Next Steps

1. Implement the embedding modules in `src/models/embeddings.py`
2. Write unit tests in `tests/test_embeddings.py`
3. Proceed to implementing the transformer block component, which will consume these embeddings

After implementing the embeddings component, it will be used by the main RNA folding model to create initial representations of the RNA sequence and its features, which will then be processed by the transformer blocks to capture complex dependencies required for 3D structure prediction.
