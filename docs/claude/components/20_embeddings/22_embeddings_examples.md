# Embeddings Component Examples

This document provides concrete code examples for implementing and using the embedding components in the RNA 3D folding pipeline. These examples illustrate best practices, common patterns, and integration scenarios.

## Basic Implementation Examples

### Sequence Embedding Module

```python
import torch
import torch.nn as nn

class SequenceEmbedding(nn.Module):
    """
    Embedding layer for RNA nucleotide sequences.
    
    Maps integer-encoded nucleotides (A=0, C=1, G=2, U=3, N=4) to learned embeddings.
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
        
        # Optional: Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
    
    def forward(self, sequence_int):
        """
        Convert integer-encoded sequences to embeddings.
        
        Args:
            sequence_int: Integer tensor of shape (batch_size, seq_len)
            
        Returns:
            Embedded sequences of shape (batch_size, seq_len, embedding_dim)
        """
        # Check input shape
        if sequence_int.dim() != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, seq_len), got shape {sequence_int.shape}")
            
        return self.embedding(sequence_int)

# Usage example
seq_embedding = SequenceEmbedding(num_embeddings=5, embedding_dim=32)
sequence_int = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 1, 0, 4]])  # Batch of 2 sequences
embedded_seq = seq_embedding(sequence_int)
print(f"Sequence shape: {sequence_int.shape}")  # torch.Size([2, 5])
print(f"Embedding shape: {embedded_seq.shape}")  # torch.Size([2, 5, 32])
```

### Positional Encoding Module

```python
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for RNA sequences.
    
    Creates fixed patterns that encode position information for the transformer.
    """
    def __init__(self, embed_dim=128, max_len=500, dropout=0.1):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension, should match residue_embed_dim
            max_len: Maximum sequence length to pre-compute
            dropout: Dropout rate applied to encodings
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create constant positional encoding matrix
        # Shape: (1, max_len, embed_dim)
        pe = torch.zeros(1, max_len, embed_dim)
        
        # Create position tensor: [[0], [1], [2], ...] 
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term to compute sin/cos at different frequencies
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Fill pe with sin for even indices and cos for odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
        
    def forward(self, x=None, seq_len=None):
        """
        Get positional encodings.
        
        Args:
            x: Optional input tensor to determine sequence length and device
            seq_len: Optional explicit sequence length
            
        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
        if x is not None:
            seq_len = x.size(1)
            return self.pe[:, :seq_len].to(x.device)
        elif seq_len is not None:
            return self.pe[:, :seq_len]
        else:
            raise ValueError("Either x or seq_len must be provided")

# Usage example
pos_encoding = PositionalEncoding(embed_dim=128, max_len=500)

# Method 1: Provide tensor to determine length and device
x = torch.zeros(2, 10, 64)  # Batch of 2, length 10
pos_enc = pos_encoding(x)
print(f"Positional encoding shape: {pos_enc.shape}")  # torch.Size([1, 10, 128])

# Method 2: Provide explicit sequence length
pos_enc = pos_encoding(seq_len=15)
print(f"Positional encoding shape: {pos_enc.shape}")  # torch.Size([1, 15, 128])

# Apply to embeddings
batch_size = 2
seq_len = 10
embed_dim = 128
embeddings = torch.rand(batch_size, seq_len, embed_dim)
pos_enc = pos_encoding(embeddings)
embeddings_with_pos = embeddings + pos_enc

print(f"Embeddings with positions: {embeddings_with_pos.shape}")  # torch.Size([2, 10, 128])
```

### Relative Positional Encoding Module

```python
import torch
import torch.nn as nn
import math

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
        self.max_relative_position = max_relative_position
        
        # Total embeddings: 2*max_rel_pos + 1 (to account for -max to +max)
        num_embeddings = 2 * max_relative_position + 1
        self.embeddings = nn.Embedding(num_embeddings, num_units)
        
        # Initialize with sinusoidal pattern
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights with sinusoidal pattern."""
        # Create positions from -max_pos to +max_pos
        position_range = torch.arange(
            -self.max_relative_position,
            self.max_relative_position + 1
        ).unsqueeze(1).float()
        
        # Compute frequency terms
        div_term = torch.exp(
            torch.arange(0, self.embeddings.embedding_dim, 2).float() * 
            (-math.log(10000.0) / self.embeddings.embedding_dim)
        )
        
        # Create sinusoidal pattern
        pe = torch.zeros(2 * self.max_relative_position + 1, self.embeddings.embedding_dim)
        pe[:, 0::2] = torch.sin(position_range * div_term)
        pe[:, 1::2] = torch.cos(position_range * div_term)
        
        # Set embedding weights
        with torch.no_grad():
            self.embeddings.weight.copy_(pe)
    
    def forward(self, length=None, device=None):
        """
        Compute relative positional encodings for all position pairs.
        
        Args:
            length: Sequence length
            device: Optional device to place tensors on
            
        Returns:
            Tensor of shape (length, length, embedding_dim) with relative
            position embeddings for each position pair
        """
        if not length:
            raise ValueError("Sequence length must be provided")
            
        # Use provided device or embedding device
        device = device or self.embeddings.weight.device
        
        # Create position indices: [0, 1, 2, ..., length-1]
        positions = torch.arange(length, device=device)
        
        # Compute relative positions between all pairs:
        # [
        #   [0, -1, -2, ...],
        #   [1, 0, -1, ...],
        #   [2, 1, 0, ...],
        #   ...
        # ]
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Shift and clip to embedding range [0, 2*max_pos]
        clipped_positions = torch.clamp(
            relative_positions + self.max_relative_position,
            0, 2 * self.max_relative_position
        )
        
        # Get embeddings for all position pairs
        embeddings = self.embeddings(clipped_positions)
        
        return embeddings

# Usage example
rel_pos_encoding = RelativePositionalEncoding(
    max_relative_position=32,
    num_units=64
)

seq_len = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get relative position encodings for a sequence of length 10
rel_pos = rel_pos_encoding(length=seq_len, device=device)
print(f"Relative position encoding shape: {rel_pos.shape}")  # torch.Size([10, 10, 64])

# Check position (0,5) and (5,0) - these should have related but different embeddings
print(f"Position (0,5) encoding: {rel_pos[0, 5, 0:3]}")
print(f"Position (5,0) encoding: {rel_pos[5, 0, 0:3]}")
```

## Feature Projection Examples

### Projecting Residue Features

```python
import torch
import torch.nn as nn

class ResidueFeatureProjection(nn.Module):
    """
    Projects multiple residue features to a unified embedding dimension.
    """
    def __init__(self, config):
        """
        Initialize feature projection layer.
        
        Args:
            config: Configuration dictionary with keys:
                - residue_embed_dim: Output dimension for residue projections
                - seq_embed_dim: Dimension of sequence embeddings
        """
        super().__init__()
        
        # Extract config parameters
        self.residue_dim = config['residue_embed_dim']
        self.seq_dim = config['seq_embed_dim']
        
        # Calculate input dimension based on feature types
        # Sequence embedding + dihedral (4) + positional entropy (1) + accessibility (1) + conservation (1, optional)
        self.input_dim = self.seq_dim + 4 + 1 + 1
        self.with_conservation = config.get('use_conservation', True)
        if self.with_conservation:
            self.input_dim += 1
            
        # Projection layer
        self.projection = nn.Linear(self.input_dim, self.residue_dim)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(self.residue_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, features_dict):
        """
        Project residue features to unified dimension.
        
        Args:
            features_dict: Dictionary with keys:
                - sequence_embedding: (B, L, seq_dim)
                - dihedral_features: (B, L, 4)
                - positional_entropy: (B, L)
                - accessibility: (B, L)
                - conservation: (B, L), optional
                - mask: (B, L), boolean mask (True for valid positions)
                
        Returns:
            Projected features of shape (B, L, residue_dim)
        """
        # Get sequence embeddings
        seq_emb = features_dict['sequence_embedding']
        batch_size, seq_len, _ = seq_emb.shape
        
        # Prepare list of features to concatenate
        feat_list = [
            seq_emb,  # (B, L, seq_dim)
            features_dict['dihedral_features'],  # (B, L, 4)
            features_dict['positional_entropy'].unsqueeze(-1),  # (B, L, 1)
            features_dict['accessibility'].unsqueeze(-1),  # (B, L, 1)
        ]
        
        # Add conservation if available and configured
        if self.with_conservation and 'conservation' in features_dict:
            feat_list.append(features_dict['conservation'].unsqueeze(-1))
        
        # Concatenate features
        combined = torch.cat(feat_list, dim=-1)
        
        # Project to residue dimension
        projected = self.projection(combined)
        
        # Apply layer normalization
        projected = self.layer_norm(projected)
        
        # Apply mask if provided
        if 'mask' in features_dict:
            mask = features_dict['mask'].unsqueeze(-1)  # (B, L, 1)
            projected = projected * mask
            
        return projected

# Usage example
config = {
    'residue_embed_dim': 128,
    'seq_embed_dim': 32,
    'use_conservation': True
}

projection = ResidueFeatureProjection(config)

# Mock batch of features
batch_size = 2
seq_len = 10
features = {
    'sequence_embedding': torch.rand(batch_size, seq_len, 32),
    'dihedral_features': torch.rand(batch_size, seq_len, 4),
    'positional_entropy': torch.rand(batch_size, seq_len),
    'accessibility': torch.rand(batch_size, seq_len),
    'conservation': torch.rand(batch_size, seq_len),
    'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
}

# Project features
projected = projection(features)
print(f"Projected residue features shape: {projected.shape}")  # torch.Size([2, 10, 128])
```

### Projecting Pair Features

```python
import torch
import torch.nn as nn

class PairFeatureProjection(nn.Module):
    """
    Projects multiple pair features to a unified embedding dimension.
    """
    def __init__(self, config):
        """
        Initialize pair feature projection layer.
        
        Args:
            config: Configuration dictionary with keys:
                - pair_embed_dim: Output dimension for pair projections
                - rel_pos_dim: Dimension of relative position encodings
        """
        super().__init__()
        
        # Extract config parameters
        self.pair_dim = config['pair_embed_dim']
        self.rel_pos_dim = config.get('rel_pos_dim', 32)
        
        # Calculate input dimension:
        # Pairing probability (1) + coupling matrix (1) + relative position (rel_pos_dim)
        self.input_dim = 1 + 1 + self.rel_pos_dim
        
        # Optional: Add extra pair features if configured
        self.use_contact_prior = config.get('use_contact_prior', False)
        if self.use_contact_prior:
            self.input_dim += 1
            
        # Projection layer
        self.projection = nn.Linear(self.input_dim, self.pair_dim)
        
        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(self.pair_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, features_dict):
        """
        Project pair features to unified dimension.
        
        Args:
            features_dict: Dictionary with keys:
                - pairing_probs: (B, L, L)
                - coupling_matrix: (B, L, L)
                - relative_pos_encoding: (L, L, rel_pos_dim)
                - mask: (B, L), boolean mask (True for valid positions)
                
        Returns:
            Projected features of shape (B, L, L, pair_dim)
        """
        # Get batch size and sequence length
        batch_size = features_dict['pairing_probs'].shape[0]
        seq_len = features_dict['pairing_probs'].shape[1]
        
        # Prepare pair features
        pairing_probs = features_dict['pairing_probs'].unsqueeze(-1)  # (B, L, L, 1)
        coupling_matrix = features_dict['coupling_matrix'].unsqueeze(-1)  # (B, L, L, 1)
        
        # Get relative position encodings and expand to batch dimension
        rel_pos = features_dict['relative_pos_encoding']  # (L, L, rel_pos_dim)
        rel_pos = rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, L, L, rel_pos_dim)
        
        # List of features to concatenate
        pair_features = [pairing_probs, coupling_matrix, rel_pos]
        
        # Add contact prior if configured and available
        if self.use_contact_prior and 'contact_prior' in features_dict:
            contact_prior = features_dict['contact_prior'].unsqueeze(-1)  # (B, L, L, 1)
            pair_features.append(contact_prior)
        
        # Concatenate features
        combined = torch.cat(pair_features, dim=-1)  # (B, L, L, input_dim)
        
        # Project to pair dimension
        projected = self.projection(combined)  # (B, L, L, pair_dim)
        
        # Apply layer normalization
        projected = self.layer_norm(projected)
        
        # Apply mask if provided
        if 'mask' in features_dict:
            # Create 2D mask from 1D mask
            mask_2d = torch.einsum(
                'bi,bj->bij', 
                features_dict['mask'], 
                features_dict['mask']
            )  # (B, L, L)
            
            # Apply mask
            projected = projected * mask_2d.unsqueeze(-1)
            
        return projected

# Usage example
config = {
    'pair_embed_dim': 64,
    'rel_pos_dim': 32,
    'use_contact_prior': False
}

projection = PairFeatureProjection(config)

# Mock batch of features
batch_size = 2
seq_len = 10
features = {
    'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
    'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
    'relative_pos_encoding': torch.rand(seq_len, seq_len, 32),
    'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
}

# Project features
projected = projection(features)
print(f"Projected pair features shape: {projected.shape}")  # torch.Size([2, 10, 10, 64])
```

## Integration Examples

### Complete Embedding Pipeline

The following example shows how all embedding components work together:

```python
import torch
import torch.nn as nn
import math

class EmbeddingPipeline(nn.Module):
    """
    Complete embedding pipeline for RNA 3D structure prediction.
    
    Combines sequence embedding, positional encoding, relative positional
    encoding, and feature projections.
    """
    def __init__(self, config):
        """
        Initialize complete embedding pipeline.
        
        Args:
            config: Configuration dictionary with embedding parameters
        """
        super().__init__()
        
        # Extract dimensions from config
        self.seq_embed_dim = config.get('seq_embed_dim', 32)
        self.residue_embed_dim = config.get('residue_embed_dim', 128)
        self.pair_embed_dim = config.get('pair_embed_dim', 64)
        self.max_relative_position = config.get('max_relative_position', 32)
        self.rel_pos_dim = config.get('rel_pos_dim', 32)
        max_seq_len = config.get('max_seq_len', 500)
        
        # Create embedding components
        self.sequence_embedding = SequenceEmbedding(
            num_embeddings=5,  # A, C, G, U, N/padding
            embedding_dim=self.seq_embed_dim
        )
        
        self.positional_encoding = PositionalEncoding(
            embed_dim=self.residue_embed_dim,
            max_len=max_seq_len
        )
        
        self.relative_pos_encoding = RelativePositionalEncoding(
            max_relative_position=self.max_relative_position,
            num_units=self.rel_pos_dim
        )
        
        # Calculate residue feature input dimension
        # Sequence + dihedral + entropy + accessibility (+ conservation optional)
        residue_input_dim = self.seq_embed_dim + 4 + 1 + 1
        if config.get('use_conservation', True):
            residue_input_dim += 1
            
        # Calculate pair feature input dimension
        # Pairing prob + coupling + rel_pos (+ contact_prior optional)
        pair_input_dim = 1 + 1 + self.rel_pos_dim
        if config.get('use_contact_prior', False):
            pair_input_dim += 1
        
        # Create projection layers
        self.residue_projection = nn.Linear(residue_input_dim, self.residue_embed_dim)
        self.pair_projection = nn.Linear(pair_input_dim, self.pair_embed_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.residue_projection.weight)
        nn.init.zeros_(self.residue_projection.bias)
        nn.init.xavier_uniform_(self.pair_projection.weight)
        nn.init.zeros_(self.pair_projection.bias)
    
    def forward(self, batch):
        """
        Process a batch through the complete embedding pipeline.
        
        Args:
            batch: Dictionary from data loader with keys:
                - sequence_int: (B, L) integer-encoded sequence
                - dihedral_features: (B, L, 4)
                - positional_entropy: (B, L)
                - accessibility: (B, L)
                - pairing_probs: (B, L, L)
                - coupling_matrix: (B, L, L)
                - mask: (B, L) boolean mask
                - conservation: (B, L) optional
                
        Returns:
            Dictionary with keys:
                - residue_repr: (B, L, residue_embed_dim)
                - pair_repr: (B, L, L, pair_embed_dim)
        """
        # Extract sequence and dimensions
        sequence_int = batch['sequence_int']
        batch_size, seq_len = sequence_int.shape
        device = sequence_int.device
        
        # 1. Apply sequence embedding
        seq_emb = self.sequence_embedding(sequence_int)  # (B, L, seq_embed_dim)
        
        # 2. Get positional encodings
        pos_enc = self.positional_encoding(seq_len=seq_len).to(device)  # (1, L, residue_dim)
        
        # 3. Get relative positional encodings
        rel_pos = self.relative_pos_encoding(length=seq_len, device=device)  # (L, L, rel_pos_dim)
        
        # 4. Collect features for residue representation
        residue_features = [
            seq_emb,  # (B, L, seq_embed_dim)
            batch['dihedral_features'],  # (B, L, 4)
            batch['positional_entropy'].unsqueeze(-1),  # (B, L, 1)
            batch['accessibility'].unsqueeze(-1)  # (B, L, 1)
        ]
        
        # Add conservation if available
        if 'conservation' in batch:
            residue_features.append(batch['conservation'].unsqueeze(-1))  # (B, L, 1)
        
        # Concatenate residue features
        residue_inputs = torch.cat(residue_features, dim=-1)  # (B, L, residue_input_dim)
        
        # 5. Project residue features
        residue_repr = self.residue_projection(residue_inputs)  # (B, L, residue_embed_dim)
        
        # 6. Add positional encodings to residue representation
        residue_repr = residue_repr + pos_enc.expand(batch_size, -1, -1)
        
        # 7. Collect features for pair representation
        pair_features = [
            batch['pairing_probs'].unsqueeze(-1),  # (B, L, L, 1)
            batch['coupling_matrix'].unsqueeze(-1),  # (B, L, L, 1)
            rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, L, L, rel_pos_dim)
        ]
        
        # Add contact prior if available
        if 'contact_prior' in batch:
            pair_features.append(batch['contact_prior'].unsqueeze(-1))  # (B, L, L, 1)
            
        # Concatenate pair features
        pair_inputs = torch.cat(pair_features, dim=-1)  # (B, L, L, pair_input_dim)
        
        # 8. Project pair features
        pair_repr = self.pair_projection(pair_inputs)  # (B, L, L, pair_embed_dim)
        
        # 9. Apply mask if provided
        if 'mask' in batch:
            mask = batch['mask']
            # Apply 1D mask to residue representation
            residue_repr = residue_repr * mask.unsqueeze(-1)
            
            # Create and apply 2D mask to pair representation
            mask_2d = torch.einsum('bi,bj->bij', mask, mask)
            pair_repr = pair_repr * mask_2d.unsqueeze(-1)
        
        return {
            'residue_repr': residue_repr,
            'pair_repr': pair_repr
        }

# Usage with complete batch from data loader
config = {
    'seq_embed_dim': 32,
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'max_relative_position': 32,
    'rel_pos_dim': 32,
    'max_seq_len': 500,
    'use_conservation': True
}

embedding_pipeline = EmbeddingPipeline(config)

# Create a mock batch similar to what would come from the data loader
batch = {
    'sequence_int': torch.randint(0, 5, (2, 10)),
    'dihedral_features': torch.rand(2, 10, 4),
    'positional_entropy': torch.rand(2, 10),
    'accessibility': torch.rand(2, 10),
    'pairing_probs': torch.rand(2, 10, 10),
    'coupling_matrix': torch.rand(2, 10, 10),
    'conservation': torch.rand(2, 10),
    'mask': torch.ones(2, 10, dtype=torch.bool)
}

# Move batch to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch = {k: v.to(device) for k, v in batch.items()}

# Process batch
embedding_pipeline = embedding_pipeline.to(device)
outputs = embedding_pipeline(batch)

print(f"Residue representation shape: {outputs['residue_repr'].shape}")  # [2, 10, 128]
print(f"Pair representation shape: {outputs['pair_repr'].shape}")        # [2, 10, 10, 64]
```

## Common Patterns and Best Practices

### Device Management

Proper device management is critical for GPU acceleration:

```python
def get_device_for_input(input_tensor=None):
    """Determine device based on input tensor or CUDA availability."""
    if input_tensor is not None:
        return input_tensor.device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyEmbedding(nn.Module):
    def forward(self, x):
        # Get device from input
        device = x.device
        
        # Move any registered buffers to this device
        pe = self.pe.to(device)
        
        # Process on the correct device
        return x + pe
```

### Handling Variable-Length Sequences

Accommodating different sequence lengths with pre-computed positional encodings:

```python
class FlexiblePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        # Pre-compute encodings up to max_len
        self.register_buffer('pe', self._create_encodings(max_len, embed_dim))
        
    def _create_encodings(self, max_len, embed_dim):
        # Implementation details...
        return pe
        
    def forward(self, seq_len, device=None):
        """Support any length up to max_len."""
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pe.size(1)}")
            
        # Return slice of pre-computed encodings
        pe = self.pe[:, :seq_len]
        if device is not None:
            pe = pe.to(device)
        return pe
```

### Symmetry in Pair Representations

Ensuring pair representations have appropriate symmetry properties:

```python
def ensure_pair_symmetry(pair_repr):
    """
    Enforce symmetry in pair representations.
    
    Args:
        pair_repr: Tensor of shape (B, L, L, D)
        
    Returns:
        Symmetrized representation with same shape
    """
    # Average with the transposed version
    # For each feature dimension, matrix should be symmetric
    return 0.5 * (pair_repr + pair_repr.transpose(1, 2))

# Usage in model
pair_repr = self.pair_projection(pair_inputs)

# Enforce symmetry if needed
if self.enforce_symmetry:
    pair_repr = ensure_pair_symmetry(pair_repr)
```

### Efficient Outer Product for Pair Features

Creating pairwise combinations of residue features using outer products:

```python
def create_pair_features_from_residue(residue_repr):
    """
    Create pair features using outer product.
    
    Args:
        residue_repr: Tensor of shape (B, L, D)
        
    Returns:
        Pair features of shape (B, L, L, 2*D)
    """
    batch_size, seq_len, feat_dim = residue_repr.shape
    
    # Create "outgoing" and "incoming" versions using broadcasting
    hi = residue_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D)
    hj = residue_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D)
    
    # Concatenate for each pair
    pair_feat = torch.cat([hi, hj], dim=-1)  # (B, L, L, 2*D)
    
    return pair_feat

# Example: Create pairwise features and combine with existing pair information
residue_repr = torch.rand(2, 10, 64)  # (B, L, D)
pair_repr = torch.rand(2, 10, 10, 32)  # (B, L, L, D_pair)

# Create pairwise residue features
residue_pairs = create_pair_features_from_residue(residue_repr)  # (B, L, L, 2*D)

# Concatenate with existing pair representations
enhanced_pairs = torch.cat([pair_repr, residue_pairs], dim=-1)  # (B, L, L, D_pair + 2*D)
```

### Configurable Feature Selection

Make embedding components configurable to handle different feature sets:

```python
class ConfigurableEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define which features to use
        self.use_features = {
            'dihedral': config.get('use_dihedral', True),
            'conservation': config.get('use_conservation', True),
            'accessibility': config.get('use_accessibility', True)
        }
        
        # Calculate input dimension based on enabled features
        input_dim = config['seq_embed_dim']
        if self.use_features['dihedral']:
            input_dim += 4
        if self.use_features['conservation']:
            input_dim += 1
        if self.use_features['accessibility']:
            input_dim += 1
            
        # Create projection
        self.projection = nn.Linear(input_dim, config['residue_embed_dim'])
        
    def forward(self, batch):
        features = [batch['sequence_embedding']]
        
        # Add features based on configuration
        if self.use_features['dihedral'] and 'dihedral_features' in batch:
            features.append(batch['dihedral_features'])
            
        if self.use_features['conservation'] and 'conservation' in batch:
            features.append(batch['conservation'].unsqueeze(-1))
            
        if self.use_features['accessibility'] and 'accessibility' in batch:
            features.append(batch['accessibility'].unsqueeze(-1))
            
        # Concatenate and project
        combined = torch.cat(features, dim=-1)
        return self.projection(combined)
```

## Common Errors and Solutions

### Shape Mismatch in Projection Layers

**Problem:** Input dimension to projection layers doesn't match the actual concatenated features.

**Solution:** Calculate input dimension explicitly and verify:

```python
def check_feature_dimensions(features_dict, config):
    """Validate feature dimensions before creating projection layers."""
    # Check sequence embedding
    if 'sequence_embedding' in features_dict:
        seq_dim = features_dict['sequence_embedding'].size(-1)
        if seq_dim != config['seq_embed_dim']:
            print(f"Warning: sequence_embedding dimension ({seq_dim}) doesn't match config ({config['seq_embed_dim']})")
    
    # Check dihedral features
    if 'dihedral_features' in features_dict:
        dihedral_dim = features_dict['dihedral_features'].size(-1)
        if dihedral_dim != 4:
            print(f"Warning: dihedral_features should have dimension 4, got {dihedral_dim}")
    
    # Calculate total input dimension for residue projection
    residue_input_dim = config['seq_embed_dim'] + 4 + 1 + 1  # seq + dihedral + entropy + accessibility
    if config.get('use_conservation', True):
        residue_input_dim += 1
        
    print(f"Expected residue input dimension: {residue_input_dim}")
    
    # Similar checks for pair features...
```

### Device Inconsistency

**Problem:** Tensors on different devices causing runtime errors.

**Solution:** Implement consistent device management:

```python
class DeviceAwareModule(nn.Module):
    def forward(self, *args, **kwargs):
        # Determine device from first tensor argument
        device = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break
                
        if device is None:
            for arg in kwargs.values():
                if isinstance(arg, torch.Tensor):
                    device = arg.device
                    break
        
        # Ensure all registered buffers are on the same device
        for buffer_name, buffer in self.named_buffers():
            if buffer.device != device:
                self._buffers[buffer_name] = buffer.to(device)
                
        # Continue with forward computation...
```

### NaN Values in Positional Encodings

**Problem:** NaN values appear in sinusoidal encodings due to numerical issues.

**Solution:** Use numerical stability techniques:

```python
def create_stable_positional_encoding(max_len, embed_dim):
    """Create numerically stable positional encodings."""
    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len).float().unsqueeze(1)
    
    # Use smaller denominator to avoid extreme values
    # log(10000.0) ≈ 9.21 -> can use smaller value if needed
    denom_scale = 9.0  # Slightly smaller than log(10000.0)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * 
        (-denom_scale / embed_dim)
    )
    
    # Compute encodings with numerical checks
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Check for NaN values
    if torch.isnan(pe).any():
        print("Warning: NaN values in positional encoding, using fallback")
        # Fallback to simpler encoding as failsafe
        for pos in range(max_len):
            for i in range(embed_dim):
                if i % 2 == 0:
                    pe[pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                else:
                    pe[pos, i] = math.cos(pos / (10000 ** ((i - 1) / embed_dim)))
    
    return pe
```

## Integration with the Main Model

Example showing how the embedding components integrate with the main `RNAFoldingModel`:

```python
class RNAFoldingModel(nn.Module):
    """Main RNA 3D structure prediction model."""
    
    def __init__(self, config):
        super().__init__()
        
        # Create embedding components
        self.sequence_embedding = SequenceEmbedding(
            num_embeddings=5,
            embedding_dim=config['seq_embed_dim']
        )
        
        self.positional_encoding = PositionalEncoding(
            embed_dim=config['residue_embed_dim'],
            max_len=config.get('max_seq_len', 500)
        )
        
        self.relative_pos_encoding = RelativePositionalEncoding(
            max_relative_position=config.get('max_relative_position', 32),
            num_units=config.get('rel_pos_dim', 32)
        )
        
        # Create feature projections
        self.residue_projection = nn.Linear(
            config['seq_embed_dim'] + 4 + 1 + 1 + (1 if config.get('use_conservation', True) else 0),
            config['residue_embed_dim']
        )
        
        self.pair_projection = nn.Linear(
            1 + 1 + config.get('rel_pos_dim', 32),
            config['pair_embed_dim']
        )
        
        # Additional model components (transformer blocks, etc.)
        # ...
        
    def forward(self, batch):
        # Apply sequence embedding
        seq_emb = self.sequence_embedding(batch['sequence_int'])
        
        # Get sequence length and device
        batch_size, seq_len = batch['sequence_int'].shape
        device = batch['sequence_int'].device
        
        # Get positional encodings
        pos_enc = self.positional_encoding(seq_len=seq_len).to(device)
        rel_pos = self.relative_pos_encoding(length=seq_len, device=device)
        
        # Create intermediate feature dictionary with embeddings
        features = dict(batch)
        features['sequence_embedding'] = seq_emb
        features['relative_pos_encoding'] = rel_pos
        
        # Project residue features
        residue_repr = self._project_residue_features(features)
        
        # Add positional encodings
        residue_repr = residue_repr + pos_enc.expand(batch_size, -1, -1)
        
        # Project pair features
        pair_repr = self._project_pair_features(features)
        
        # Continue with transformer blocks, etc.
        # ...
        
        return {
            'residue_repr': residue_repr,
            'pair_repr': pair_repr,
            # Additional outputs...
        }
    
    def _project_residue_features(self, features):
        """Project residue features to unified dimension."""
        # Implementation...
        
    def _project_pair_features(self, features):
        """Project pair features to unified dimension."""
        # Implementation...
```
