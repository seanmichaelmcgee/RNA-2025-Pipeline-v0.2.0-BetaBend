# Transformer Block Implementation Guide

## Component Overview

The Transformer Block is a core architectural component in our RNA 3D folding pipeline, responsible for processing and updating both residue (per-nucleotide) and pair (nucleotide-pair) representations. Drawing inspiration from attention-based architectures like AlphaFold, this component enables the model to capture complex dependencies between RNA residues needed for accurate 3D structure prediction.

In V1, we implement a simplified transformer block with standard multi-head attention and a basic pair update mechanism, laying the foundation for more advanced implementations in future versions.

## Requirements Reference

From the Product Requirements Document (`4_Product_Requirements_V1.md`):

- **MA-05**: Implement `TransformerBlock` module containing: LayerNorm, **standard Multi-Head Attention** (`batch_first=True`), FFN, and **simplified pair update MLP**.
- **MA-06**: Stack multiple `TransformerBlock` instances using `nn.ModuleList` in the main model backbone.

The architecture is outlined in `3_Architecture_Specification.md` under the "Backbone Model: Transformer-Based Multimodal Fusion" section.

## Technical Background

### Transformer Overview

The transformer block in this implementation has two key components:

1. **Residue Update Path**: Updates per-nucleotide representations through self-attention, allowing each residue to attend to all other residues based on their features.

2. **Pair Update Path**: Updates pairwise representations by incorporating information from the current residue representations, capturing relationships between residue pairs.

### Pre-Norm vs. Post-Norm Architecture

We use the **pre-normalization** pattern (applying layer normalization before attention/FFN) rather than the traditional post-normalization. This approach:
- Improves training stability for deeper networks
- Leads to smoother gradients
- Helps avoid exploding/vanishing gradient issues

### V1 Simplifications

In the V1 implementation, we make these simplifications:
- Use standard PyTorch `nn.MultiheadAttention` without custom modifications
- Implement a simplified pair update mechanism using MLPs
- Defer more complex features (e.g., pair bias, triangle multiplication) to future versions

## Interfaces

### Input Interface

The transformer block takes inputs from the embedding layer or previous transformer block:

```python
# Inputs
residue_repr: torch.Tensor  # Shape: (batch_size, seq_len, residue_dim)
                           # Residue (per-nucleotide) representations
                           
pair_repr: torch.Tensor    # Shape: (batch_size, seq_len, seq_len, pair_dim)
                          # Pair (nucleotide-pair) representations
                          
mask: torch.Tensor         # Shape: (batch_size, seq_len), dtype: torch.bool
                          # Boolean mask indicating valid positions (True for valid)
```

### Output Interface

The transformer block produces updated residue and pair representations:

```python
# Outputs
updated_residue_repr: torch.Tensor  # Shape: (batch_size, seq_len, residue_dim)
                                   # Updated residue representations
                                   
updated_pair_repr: torch.Tensor     # Shape: (batch_size, seq_len, seq_len, pair_dim)
                                   # Updated pair representations
```

## Implementation Steps

### 1. Define the `TransformerBlock` Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TransformerBlock(nn.Module):
    """
    Transformer block for RNA folding model with residue and pair updates.
    
    This block processes both residue-level and pair-level representations
    and updates them through attention and feed-forward networks.
    """
    
    def __init__(self, config: dict):
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
        super().__init__()
        
        # Extract parameters from config
        self.residue_dim = config['residue_embed_dim']
        self.pair_dim = config['pair_embed_dim']
        self.num_heads = config['num_attention_heads']
        self.dropout_rate = config.get('dropout', 0.1)
        self.ffn_dim = config.get('ffn_dim', self.residue_dim * 4)
        
        # Validate dimensions
        if self.residue_dim % self.num_heads != 0:
            raise ValueError(
                f"residue_embed_dim ({self.residue_dim}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )
        
        # Initialize components for residue update
        self._init_residue_update_components()
        
        # Initialize components for pair update
        self._init_pair_update_components()
    
    def _init_residue_update_components(self):
        """Initialize layers for residue representation update."""
        # Pre-normalization for attention
        self.residue_attn_norm = nn.LayerNorm(self.residue_dim)
        
        # Multi-head attention
        self.residue_attention = nn.MultiheadAttention(
            embed_dim=self.residue_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Dropout after attention
        self.residue_attn_dropout = nn.Dropout(self.dropout_rate)
        
        # Pre-normalization for feed-forward
        self.residue_ffn_norm = nn.LayerNorm(self.residue_dim)
        
        # Feed-forward network for residue update
        self.residue_ffn = nn.Sequential(
            nn.Linear(self.residue_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.ffn_dim, self.residue_dim)
        )
        
        # Dropout after feed-forward
        self.residue_ffn_dropout = nn.Dropout(self.dropout_rate)
    
    def _init_pair_update_components(self):
        """Initialize layers for pair representation update."""
        # Pre-normalization for pair update
        self.pair_norm = nn.LayerNorm(self.pair_dim)
        
        # Calculate input dimension for pair MLP
        # Concatenate: h_i, h_j, and pair_repr
        pair_input_dim = 2 * self.residue_dim + self.pair_dim
        
        # MLP for pair update
        self.pair_update_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, self.pair_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.pair_dim, self.pair_dim)
        )
        
        # Dropout for pair update
        self.pair_dropout = nn.Dropout(self.dropout_rate)
    
    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
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
        # Update residue representations using self-attention
        residue_repr = self._update_residue_repr(residue_repr, mask)
        
        # Update pair representations
        pair_repr = self._update_pair_repr(residue_repr, pair_repr, mask)
        
        return residue_repr, pair_repr
    
    def _update_residue_repr(
        self,
        residue_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update residue representations through self-attention and feed-forward network.
        
        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            mask: Boolean mask of shape (batch_size, seq_len)
            
        Returns:
            Updated residue representations of shape (batch_size, seq_len, residue_dim)
        """
        # Prepare attention mask from boolean mask
        attn_mask = None
        key_padding_mask = None
        
        if mask is not None:
            # Invert mask for PyTorch attention (False = keep, True = mask)
            key_padding_mask = ~mask
        
        # Pre-normalization
        res_norm = self.residue_attn_norm(residue_repr)
        
        # Self-attention
        attn_output, _ = self.residue_attention(
            query=res_norm,
            key=res_norm,
            value=res_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Residual connection with dropout
        residue_repr = residue_repr + self.residue_attn_dropout(attn_output)
        
        # Feed-forward network with pre-normalization
        res_norm = self.residue_ffn_norm(residue_repr)
        ffn_output = self.residue_ffn(res_norm)
        
        # Residual connection with dropout
        residue_repr = residue_repr + self.residue_ffn_dropout(ffn_output)
        
        return residue_repr
    
    def _update_pair_repr(
        self,
        residue_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update pair representations using residue representations.
        
        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
            mask: Boolean mask of shape (batch_size, seq_len)
            
        Returns:
            Updated pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        """
        batch_size, seq_len, _ = residue_repr.shape
        
        # Pre-normalization
        pair_norm = self.pair_norm(pair_repr)
        
        # Create outer product of residue representations
        # For each pair (i,j), we'll concatenate h_i, h_j, and pair_repr[i,j]
        
        # Expand residue representations for broadcasting
        h_i = residue_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D_res)
        h_j = residue_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D_res)
        
        # Concatenate along the feature dimension
        pair_inputs = torch.cat([h_i, h_j, pair_norm], dim=-1)  # (B, L, L, 2*D_res + D_pair)
        
        # Apply MLP to update pair representations
        pair_update = self.pair_update_mlp(pair_inputs)
        
        # Apply mask to update if provided
        if mask is not None:
            # Create 2D mask for pairs
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
            pair_mask = pair_mask.unsqueeze(-1)  # (B, L, L, 1)
            pair_update = pair_update * pair_mask
        
        # Residual connection with dropout
        pair_repr = pair_repr + self.pair_dropout(pair_update)
        
        return pair_repr
```

### 2. Implement Helper Method for Attention Mask Creation

For more explicit control over attention masking, you might want to implement a separate helper method:

```python
def _create_attention_masks(
    self,
    mask: Optional[torch.Tensor]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Create attention masks from boolean position mask.
    
    Args:
        mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
        
    Returns:
        Tuple of:
        - attention_mask: Mask for attention scores (None for V1 implementation)
        - key_padding_mask: Mask for padding tokens with shape (batch_size, seq_len)
    """
    attn_mask = None  # Not used in V1
    key_padding_mask = None
    
    if mask is not None:
        # PyTorch attention expects False = keep, True = mask out
        key_padding_mask = ~mask
    
    return attn_mask, key_padding_mask
```

### 3. Implement Symmetrization for Pair Representations (Optional)

For some applications, you might want to ensure the pair representations maintain symmetry (i.e., pair_repr[i,j] = pair_repr[j,i]):

```python
def _ensure_pair_symmetry(self, pair_repr: torch.Tensor) -> torch.Tensor:
    """
    Ensure pair representations are symmetric.
    
    Args:
        pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        
    Returns:
        Symmetrized pair representations with shape (batch_size, seq_len, seq_len, pair_dim)
    """
    # Average with the transposed version
    return 0.5 * (pair_repr + pair_repr.transpose(1, 2))
```

## Critical Aspects

### 1. Masking Implementation

Proper masking is critical for handling variable-length sequences:

```python
def _apply_residue_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply 1D mask to a residue tensor."""
    return tensor * mask.unsqueeze(-1)

def _apply_pair_mask(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply 2D mask to a pair tensor."""
    # Create 2D mask from 1D mask
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    # Apply to tensor
    return tensor * pair_mask.unsqueeze(-1)
```

### 2. Attention Configuration

Understanding the PyTorch MultiheadAttention parameters:

- `embed_dim`: Must match the residue embedding dimension
- `num_heads`: Number of attention heads, must divide embed_dim evenly
- `batch_first=True`: Important to match our tensor formats (batch dimension first)
- `key_padding_mask`: Used to mask out padding tokens (boolean tensor where True = mask)
- `attn_mask`: Used to mask specific attention connections (not used in V1)

### 3. Pre-normalization Architecture

The pre-normalization pattern is crucial for training stability:

1. Apply Layer Normalization to the input
2. Process the normalized input through attention/FFN
3. Add the processed output to the original input (residual connection)

This differs from the traditional transformer architecture which performs normalization after the residual connection.

### 4. Device Management

All operations should work on the device of the input tensors:

```python
def forward(self, residue_repr, pair_repr, mask=None):
    # Get device from input tensor
    device = residue_repr.device
    
    # Ensure mask is on the correct device
    if mask is not None and mask.device != device:
        mask = mask.to(device)
    
    # Proceed with computation...
```

### 5. Configuration Parameters

The transformer block relies on these key configuration parameters:

```python
config = {
    'residue_embed_dim': 128,   # Dimension of residue representations
    'pair_embed_dim': 64,       # Dimension of pair representations
    'num_attention_heads': 4,   # Number of attention heads
    'dropout': 0.1,             # Dropout probability
    'ffn_dim': 512              # Feed-forward network hidden dimension
}
```

## Integration with Other Components

### Integration with Embeddings

The transformer block receives inputs from the embedding layer:

```python
# In main model's forward method
# After processing embeddings...

residue_repr, pair_repr = embeddings_output['residue_repr'], embeddings_output['pair_repr']
mask = batch['mask']

# Apply transformer blocks
for block in self.transformer_blocks:
    residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
```

### Stacking Multiple Blocks

To create a deep transformer model, multiple blocks are stacked:

```python
def __init__(self, config):
    # ...
    
    # Create a stack of transformer blocks
    num_blocks = config.get('num_transformer_blocks', 4)
    self.transformer_blocks = nn.ModuleList([
        TransformerBlock(config) for _ in range(num_blocks)
    ])
```

## Testing Requirements

The transformer block should be tested for:

1. Correct shapes before and after processing
2. Proper handling of masks
3. Integration with embedding components
4. Minimal function with a simple test case
5. Device compatibility (CPU/CUDA)

Refer to `tests/test_transformer_block.py` for detailed testing.

## Performance Considerations

### Memory Usage

The pair representation update is the most memory-intensive part of the transformer block. For a sequence of length L:

- The outer product operation creates tensors of shape (B, L, L, D_res)
- The concatenated tensor has shape (B, L, L, 2*D_res + D_pair)

For large sequences, this can consume significant memory. Consider:

- Using gradient checkpointing for very long sequences
- Starting with smaller model dimensions for initial testing

### Computational Efficiency

For improved computational efficiency:

- The most expensive operation is the outer product for pair updates. This scales as O(L²)
- Consider optimizing this operation for large sequences in future versions
- For V1, the simplified implementation is sufficient for our testing needs

## Example Usage

Here's a complete example showing how to use the TransformerBlock:

```python
import torch
import torch.nn as nn

# Create configuration
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'ffn_dim': 512
}

# Create transformer block
transformer_block = TransformerBlock(config)

# Create dummy inputs
batch_size = 2
seq_len = 10
residue_dim = config['residue_embed_dim']
pair_dim = config['pair_embed_dim']

residue_repr = torch.rand(batch_size, seq_len, residue_dim)
pair_repr = torch.rand(batch_size, seq_len, seq_len, pair_dim)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[0, -2:] = False  # Mask out last two positions in first sequence

# Forward pass
updated_residue_repr, updated_pair_repr = transformer_block(residue_repr, pair_repr, mask)

# Check output shapes
print(f"Residue repr shape: {updated_residue_repr.shape}")  # (2, 10, 128)
print(f"Pair repr shape: {updated_pair_repr.shape}")        # (2, 10, 10, 64)

# Check masking effect
print(f"Masked positions are zero: {torch.all(updated_residue_repr[0, -2:] == 0)}")
```

## Next Steps

After implementing the transformer block:

1. Write unit tests to verify functionality
2. Integrate with the RNA folding model
3. Connect to the IPA module placeholder
4. Consider future enhancements for V2+:
   - Pair bias in attention mechanism
   - Triangle multiplication for pair updates
   - Axial attention or other efficiency improvements

## Related Documentation

- **Architecture Specification**: `docs/3_Architecture_Specification.md` - See "Backbone Model: Transformer-Based Multimodal Fusion" section
- **Product Requirements**: `docs/4_Product_Requirements_V1.md` - Requirements MA-05 and MA-06
- **Embeddings Guide**: `docs/claude/components/20_embeddings/guide.md` - For understanding input format
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md` - For device management and module design

## Conclusion

The transformer block is a critical component that iteratively refines both residue and pair representations, enabling the model to capture complex dependencies required for RNA 3D structure prediction. This V1 implementation provides a solid foundation with standard multi-head attention and simplified pair updates, which can be extended in future versions.
