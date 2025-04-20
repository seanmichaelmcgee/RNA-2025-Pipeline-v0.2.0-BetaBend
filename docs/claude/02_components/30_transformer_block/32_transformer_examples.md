# Transformer Block Examples

This document provides concrete code examples for implementing and using the Transformer Block component in the RNA 3D folding pipeline. These examples illustrate best practices, common patterns, and integration scenarios with a focus on proper masking implementation and shape handling.

## Basic Implementation Examples

### Complete `TransformerBlock` Implementation

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
        Initialize transformer block with configuration parameters.
        
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

### Simplified Attention Mask Helper

```python
def create_attention_masks(mask: torch.Tensor) -> torch.Tensor:
    """
    Create attention masks from boolean position mask.
    
    Args:
        mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
        
    Returns:
        key_padding_mask: Inverted mask for PyTorch attention (False = keep, True = mask)
    """
    # PyTorch attention expects False = keep, True = mask out
    key_padding_mask = ~mask
    
    return key_padding_mask
```

### Ensuring Pair Symmetry

```python
def ensure_pair_symmetry(pair_repr: torch.Tensor) -> torch.Tensor:
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

## Masking and Shape Management Examples

### Creating 2D Mask from 1D Mask

```python
def create_pair_mask(sequence_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a 2D pair mask from a 1D sequence mask.
    
    Args:
        sequence_mask: Boolean mask of shape (batch_size, seq_len) 
                      where True indicates valid positions
        
    Returns:
        pair_mask: Boolean mask of shape (batch_size, seq_len, seq_len)
                  where True indicates valid pairs
    """
    # Outer product of mask with itself using broadcasting
    pair_mask = sequence_mask.unsqueeze(1) & sequence_mask.unsqueeze(2)
    
    return pair_mask

# Example usage
batch_size, seq_len = 2, 5
sequence_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
# Set last two positions in first sequence as padding
sequence_mask[0, -2:] = False

pair_mask = create_pair_mask(sequence_mask)
print(f"Sequence mask shape: {sequence_mask.shape}")  # (2, 5)
print(f"Pair mask shape: {pair_mask.shape}")  # (2, 5, 5)

# Visualize the mask matrix for the first sequence
print("First sequence mask:")
print(sequence_mask[0])  # [True, True, True, False, False]

print("First sequence pair mask:")
print(pair_mask[0])
# Will show a 5x5 matrix where the last two rows and last two columns are False
```

### Applying Masks to Residue and Pair Representations

```python
def apply_masks(
    residue_repr: torch.Tensor,
    pair_repr: torch.Tensor,
    mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply masks to residue and pair representations.
    
    Args:
        residue_repr: Tensor of shape (batch_size, seq_len, residue_dim)
        pair_repr: Tensor of shape (batch_size, seq_len, seq_len, pair_dim)
        mask: Boolean mask of shape (batch_size, seq_len)
        
    Returns:
        Tuple of masked residue and pair representations
    """
    # Apply 1D mask to residue representation
    masked_residue = residue_repr * mask.unsqueeze(-1)
    
    # Create and apply 2D mask to pair representation
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
    masked_pair = pair_repr * pair_mask.unsqueeze(-1)  # (B, L, L, D)
    
    return masked_residue, masked_pair

# Example usage
batch_size, seq_len = 2, 5
residue_dim, pair_dim = 128, 64

# Create dummy representations
residue_repr = torch.rand((batch_size, seq_len, residue_dim))
pair_repr = torch.rand((batch_size, seq_len, seq_len, pair_dim))

# Create a mask with padding in the first sequence
mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
mask[0, -2:] = False  # Last two positions in first sequence are padding

# Apply masks
masked_residue, masked_pair = apply_masks(residue_repr, pair_repr, mask)

# Verify masking
assert torch.all(masked_residue[0, -2:] == 0)  # Masked positions in residue_repr are 0
assert torch.all(masked_pair[0, -2:, :] == 0)  # Last two rows in pair_repr are 0
assert torch.all(masked_pair[0, :, -2:] == 0)  # Last two columns in pair_repr are 0
```

### Complete Forward Pass with Shape Checking

```python
def forward_with_shape_check(
    transformer_block: TransformerBlock,
    residue_repr: torch.Tensor,
    pair_repr: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform forward pass through transformer block with shape validation.
    
    Args:
        transformer_block: TransformerBlock instance
        residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
        pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions
              
    Returns:
        Tuple of:
        - Updated residue representations of shape (batch_size, seq_len, residue_dim)
        - Updated pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
    """
    # Check input shapes
    batch_size, seq_len, residue_dim = residue_repr.shape
    
    if pair_repr.shape[:-1] != (batch_size, seq_len, seq_len):
        raise ValueError(
            f"pair_repr shape {pair_repr.shape} is incompatible with "
            f"residue_repr shape {residue_repr.shape}. Expected first 3 dimensions "
            f"to be ({batch_size}, {seq_len}, {seq_len})"
        )
    
    if mask is not None and mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"mask shape {mask.shape} is incompatible with "
            f"residue_repr shape {residue_repr.shape}. Expected "
            f"({batch_size}, {seq_len})"
        )
    
    # Forward pass
    new_residue_repr, new_pair_repr = transformer_block(residue_repr, pair_repr, mask)
    
    # Check output shapes
    if new_residue_repr.shape != residue_repr.shape:
        raise ValueError(
            f"Output residue_repr shape {new_residue_repr.shape} does not match "
            f"input shape {residue_repr.shape}"
        )
    
    if new_pair_repr.shape != pair_repr.shape:
        raise ValueError(
            f"Output pair_repr shape {new_pair_repr.shape} does not match "
            f"input shape {pair_repr.shape}"
        )
    
    return new_residue_repr, new_pair_repr
```

## Integration Examples

### Integrating with Embedding Outputs

```python
def process_batch_through_transformer(
    embedding_outputs: dict,
    transformer_blocks: nn.ModuleList,
    batch: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a batch through a stack of transformer blocks.
    
    Args:
        embedding_outputs: Dictionary containing:
            - residue_repr: (batch_size, seq_len, residue_dim)
            - pair_repr: (batch_size, seq_len, seq_len, pair_dim)
        transformer_blocks: ModuleList of TransformerBlock instances
        batch: Dictionary containing:
            - mask: (batch_size, seq_len) boolean mask
            
    Returns:
        Tuple of final residue and pair representations
    """
    # Extract representations from embedding outputs
    residue_repr = embedding_outputs['residue_repr']
    pair_repr = embedding_outputs['pair_repr']
    
    # Extract mask from batch
    mask = batch.get('mask')
    
    # Process through each transformer block in sequence
    for i, block in enumerate(transformer_blocks):
        residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        print(f"Block {i+1} output shapes: residue {residue_repr.shape}, pair {pair_repr.shape}")
    
    return residue_repr, pair_repr

# Example usage
import torch.nn as nn

# Create config
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'ffn_dim': 512
}

# Mock embedding outputs
batch_size, seq_len = 2, 10
embedding_outputs = {
    'residue_repr': torch.rand(batch_size, seq_len, config['residue_embed_dim']),
    'pair_repr': torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
}

# Mock batch with mask
batch = {
    'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
}
batch['mask'][0, -2:] = False  # Last two positions in first sequence are padding

# Create stack of transformer blocks
num_blocks = 3
transformer_blocks = nn.ModuleList([
    TransformerBlock(config) for _ in range(num_blocks)
])

# Process through transformer blocks
final_residue, final_pair = process_batch_through_transformer(
    embedding_outputs, transformer_blocks, batch
)

print(f"Final output shapes:")
print(f"- Residue representation: {final_residue.shape}")
print(f"- Pair representation: {final_pair.shape}")
```

### Integrating with IPA Module (Placeholder)

```python
class IPAModulePlaceholder(nn.Module):
    """
    Placeholder for the Invariant Point Attention module.
    Just projects residue representation to 3D coordinates.
    """
    
    def __init__(self, config):
        super().__init__()
        self.residue_dim = config['residue_embed_dim']
        
        # Simple projection to 3D coordinates
        self.coord_projection = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, 3)
        )
    
    def forward(self, residue_repr, pair_repr=None, mask=None):
        """
        Predict 3D coordinates from residue representations.
        
        Args:
            residue_repr: (batch_size, seq_len, residue_dim)
            pair_repr: (batch_size, seq_len, seq_len, pair_dim), unused in placeholder
            mask: (batch_size, seq_len) boolean mask
            
        Returns:
            coords: (batch_size, seq_len, 3) predicted coordinates
        """
        # Project to 3D coordinates
        coords = self.coord_projection(residue_repr)
        
        # Apply mask if provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)
        
        return coords

def full_model_forward(
    embedding_outputs: dict,
    transformer_blocks: nn.ModuleList,
    ipa_module: nn.Module,
    batch: dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Complete forward pass from embeddings through transformer to structure.
    
    Args:
        embedding_outputs: Dictionary with residue_repr and pair_repr
        transformer_blocks: Stack of transformer blocks
        ipa_module: IPA module (placeholder) for coordinate prediction
        batch: Dictionary with mask
        
    Returns:
        Tuple of:
        - pred_coords: (batch_size, seq_len, 3) predicted coordinates
        - pair_repr: Final pair representation
    """
    # Extract representations
    residue_repr = embedding_outputs['residue_repr']
    pair_repr = embedding_outputs['pair_repr']
    mask = batch.get('mask')
    
    # Process through transformer blocks
    for block in transformer_blocks:
        residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
    
    # Generate 3D coordinates using IPA module
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    return coords, pair_repr

# Example usage
# Create IPA module
ipa_module = IPAModulePlaceholder(config)

# Complete forward pass
pred_coords, final_pair = full_model_forward(
    embedding_outputs, transformer_blocks, ipa_module, batch
)

print(f"Predicted coordinates shape: {pred_coords.shape}")  # (2, 10, 3)
```

### Complete Model Integration Example

```python
class RNAFoldingModel(nn.Module):
    """
    Complete RNA folding model that integrates embeddings, transformer blocks, 
    and IPA module for 3D structure prediction.
    """
    
    def __init__(self, config):
        super().__init__()
        # Extract dimensions
        self.residue_dim = config['residue_embed_dim']
        self.pair_dim = config['pair_embed_dim']
        
        # Placeholder for embedding components
        # In a real implementation, this would be proper embedding modules
        self.residue_projection = nn.Linear(32, self.residue_dim)  # Example input dim
        self.pair_projection = nn.Linear(16, self.pair_dim)  # Example input dim
        
        # Create transformer blocks
        num_blocks = config.get('num_transformer_blocks', 4)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_blocks)
        ])
        
        # IPA module (placeholder)
        self.ipa_module = IPAModulePlaceholder(config)
        
        # Output heads
        self.confidence_head = nn.Linear(self.residue_dim, 1)
        self.angle_prediction_head = nn.Linear(self.residue_dim, 4)  # sin/cos of two angles
    
    def forward(self, batch):
        """
        Forward pass through the complete model.
        
        Args:
            batch: Dictionary containing:
                - residue_features: (batch_size, seq_len, input_dim_res)
                - pair_features: (batch_size, seq_len, seq_len, input_dim_pair)
                - mask: (batch_size, seq_len) boolean mask
                
        Returns:
            Dictionary containing:
                - pred_coords: (batch_size, seq_len, 3)
                - pred_confidence: (batch_size, seq_len)
                - pred_angles: (batch_size, seq_len, 4)
        """
        # Extract inputs
        residue_features = batch['residue_features']
        pair_features = batch['pair_features']
        mask = batch.get('mask')
        
        # Initial projections (embeddings in real implementation)
        residue_repr = self.residue_projection(residue_features)
        pair_repr = self.pair_projection(pair_features)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # Generate 3D coordinates
        pred_coords = self.ipa_module(residue_repr, pair_repr, mask)
        
        # Confidence prediction
        pred_confidence = self.confidence_head(residue_repr).squeeze(-1)
        
        # Angle prediction
        pred_angles = self.angle_prediction_head(residue_repr)
        
        # Apply mask to outputs
        if mask is not None:
            pred_confidence = pred_confidence * mask
            pred_angles = pred_angles * mask.unsqueeze(-1)
        
        return {
            'pred_coords': pred_coords,
            'pred_confidence': pred_confidence,
            'pred_angles': pred_angles
        }

# Example usage
# Create dummy batch
batch_size, seq_len = 2, 10
input_dim_res, input_dim_pair = 32, 16

batch = {
    'residue_features': torch.rand(batch_size, seq_len, input_dim_res),
    'pair_features': torch.rand(batch_size, seq_len, seq_len, input_dim_pair),
    'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
}
batch['mask'][0, -2:] = False  # Last two positions in first sequence are padding

# Create model
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'ffn_dim': 512,
    'num_transformer_blocks': 3
}

model = RNAFoldingModel(config)

# Forward pass
outputs = model(batch)

# Check outputs
print(f"Output shapes:")
print(f"- Predicted coordinates: {outputs['pred_coords'].shape}")  # (2, 10, 3)
print(f"- Confidence scores: {outputs['pred_confidence'].shape}")  # (2, 10)
print(f"- Predicted angles: {outputs['pred_angles'].shape}")  # (2, 10, 4)

# Verify masking
assert torch.all(outputs['pred_coords'][0, -2:] == 0)
assert torch.all(outputs['pred_confidence'][0, -2:] == 0)
assert torch.all(outputs['pred_angles'][0, -2:] == 0)
```

## Common Patterns and Best Practices

### Stacking Multiple Transformer Blocks

```python
def create_transformer_stack(config, num_blocks=None):
    """
    Create a stack of transformer blocks.
    
    Args:
        config: Model configuration dictionary
        num_blocks: Optional override for number of blocks
        
    Returns:
        ModuleList of transformer blocks
    """
    if num_blocks is None:
        num_blocks = config.get('num_transformer_blocks', 4)
    
    return nn.ModuleList([
        TransformerBlock(config) for _ in range(num_blocks)
    ])

# Usage
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'ffn_dim': 512,
    'num_transformer_blocks': 6  # Default in config
}

# Create with default number from config
transformer_stack = create_transformer_stack(config)
print(f"Created {len(transformer_stack)} transformer blocks")

# Create with custom number
transformer_stack = create_transformer_stack(config, num_blocks=3)
print(f"Created {len(transformer_stack)} transformer blocks")
```

### Handling Device Transfer

```python
def move_to_device(transformer_blocks, device):
    """
    Move transformer blocks to specified device.
    
    Args:
        transformer_blocks: ModuleList of transformer blocks
        device: Target device (e.g., 'cuda', 'cpu', or torch.device)
        
    Returns:
        The transformer blocks on the target device
    """
    return transformer_blocks.to(device)

# Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_blocks = create_transformer_stack(config, num_blocks=3)
transformer_blocks = move_to_device(transformer_blocks, device)

# Ensure inputs are on the same device
batch_size, seq_len = 2, 10
residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], device=device)
pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'], device=device)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

# Now run forward pass with all tensors on same device
for block in transformer_blocks:
    residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
```

### Gradient Checkpointing for Memory Efficiency

```python
def transformer_forward_with_checkpointing(
    transformer_blocks,
    residue_repr,
    pair_repr,
    mask=None
):
    """
    Process representations through transformer blocks with gradient checkpointing.
    
    Args:
        transformer_blocks: ModuleList of transformer blocks
        residue_repr: Residue representations
        pair_repr: Pair representations
        mask: Optional boolean mask
        
    Returns:
        Tuple of updated residue and pair representations
    """
    # Iterate through blocks, using checkpointing if in training mode
    for block in transformer_blocks:
        if block.training:
            # Define a custom forward function for checkpointing
            def custom_forward(res, pair, msk):
                return block(res, pair, msk)
            
            # Apply checkpointing
            residue_repr, pair_repr = torch.utils.checkpoint.checkpoint(
                custom_forward, 
                residue_repr, 
                pair_repr, 
                mask
            )
        else:
            # Regular forward pass during inference
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
    
    return residue_repr, pair_repr

# Example usage
model = RNAFoldingModel(config).to(device)
model.train()  # Set to training mode

# Forward pass with checkpointing
residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'], device=device)
pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'], device=device)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

residue_repr, pair_repr = transformer_forward_with_checkpointing(
    model.transformer_blocks, 
    residue_repr, 
    pair_repr, 
    mask
)

print(f"Processed with checkpointing. Shapes maintained:")
print(f"- Residue representation: {residue_repr.shape}")
print(f"- Pair representation: {pair_repr.shape}")
```

### Monitoring Attention Patterns

```python
def attention_pattern_analysis(transformer_block, residue_repr, mask=None):
    """
    Analyze attention patterns in a transformer block.
    
    Args:
        transformer_block: A TransformerBlock instance
        residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
        mask: Optional boolean mask of shape (batch_size, seq_len)
        
    Returns:
        Dict of attention statistics
    """
    batch_size, seq_len, _ = residue_repr.shape
    
    # Prepare attention inputs
    res_norm = transformer_block.residue_attn_norm(residue_repr)
    
    # Prepare mask for attention
    key_padding_mask = None
    if mask is not None:
        key_padding_mask = ~mask
    
    # Run attention with need_weights=True to get attention weights
    _, attn_weights = transformer_block.residue_attention(
        query=res_norm,
        key=res_norm,
        value=res_norm,
        attn_mask=None,
        key_padding_mask=key_padding_mask,
        need_weights=True
    )
    
    # Analyze attention patterns - attn_weights shape: (batch_size, seq_len, seq_len)
    avg_attention = attn_weights.mean(dim=0)  # Average over batch
    
    # Statistics
    diagonal_attention = torch.diagonal(avg_attention, dim1=0, dim2=1).mean()
    neighbor_attention = torch.sum(attn_weights[:, torch.arange(seq_len-1), torch.arange(1, seq_len)])
    long_range_mask = torch.ones_like(avg_attention, dtype=torch.bool)
    long_range_mask.fill_diagonal_(False)
    for i in range(seq_len):
        for j in range(max(0, i-3), min(seq_len, i+4)):
            long_range_mask[i, j] = False
    long_range_attention = (avg_attention * long_range_mask).sum() / long_range_mask.sum()
    
    return {
        'attention_weights': attn_weights,
        'diagonal_attention': diagonal_attention.item(),
        'neighbor_attention': neighbor_attention.item(),
        'long_range_attention': long_range_attention.item()
    }

# Example usage
block = TransformerBlock(config)
residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[0, -2:] = False  # Mask last two positions in first sequence

attn_stats = attention_pattern_analysis(block, residue_repr, mask)
print(f"Attention Statistics:")
print(f"- Diagonal attention: {attn_stats['diagonal_attention']:.4f}")
print(f"- Neighbor attention: {attn_stats['neighbor_attention']:.4f}")
print(f"- Long-range attention: {attn_stats['long_range_attention']:.4f}")
```

## Common Errors and Solutions

### Dimension Mismatch in Attention

**Problem**: `RuntimeError: embed_dim must be divisible by num_heads` when creating the transformer block.

**Solution**: Ensure the `residue_embed_dim` is divisible by `num_attention_heads` in your configuration:

```python
def validate_transformer_config(config):
    """Validate transformer configuration parameters."""
    if config['residue_embed_dim'] % config['num_attention_heads'] != 0:
        raise ValueError(
            f"residue_embed_dim ({config['residue_embed_dim']}) must be "
            f"divisible by num_attention_heads ({config['num_attention_heads']})"
        )
    
    return True

# Usage before creating transformer blocks
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 6,  # Will cause error with dim 128
    'dropout': 0.1
}

try:
    validate_transformer_config(config)
except ValueError as e:
    print(f"Configuration error: {e}")
    # Fix the configuration
    config['num_attention_heads'] = 4  # Now 128 % 4 = 0
    print(f"Fixed configuration: residue_dim={config['residue_embed_dim']}, "
          f"heads={config['num_attention_heads']}")
    validate_transformer_config(config)  # Should pass now
```

### Mask Format Issues

**Problem**: Incorrect attention behavior due to improper mask format.

**Solution**: Remember that PyTorch's `MultiheadAttention` expects `key_padding_mask` to be `True` for padding positions (which should be masked out), which is the inverse of our Boolean mask:

```python
def fix_attention_mask_format(mask):
    """
    Convert from our mask format (True = valid) to PyTorch's 
    key_padding_mask format (False = valid, True = mask).
    """
    return ~mask  # Invert the mask

# Example demonstrating correct mask handling
batch_size, seq_len = 2, 5
mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
mask[0, -2:] = False  # Last two positions in first sequence are padding

# Wrong way (using mask directly)
wrong_key_padding_mask = mask  # This would mask valid positions!

# Correct way (inverting mask)
correct_key_padding_mask = fix_attention_mask_format(mask)

print("Our mask format (True = valid):")
print(mask[0])  # [True, True, True, False, False]

print("PyTorch key_padding_mask format (False = valid, True = mask):")
print(correct_key_padding_mask[0])  # [False, False, False, True, True]
```

### Pair Representation Shape Errors

**Problem**: Shape errors in pair update step.

**Solution**: Ensure proper broadcasting of residue representations to create the outer product and check shapes at each step:

```python
def debug_pair_update_shapes(residue_repr, pair_repr):
    """
    Debug shape issues in pair update.
    
    Args:
        residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
        pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
    """
    batch_size, seq_len, residue_dim = residue_repr.shape
    
    print(f"Input shapes:")
    print(f"- residue_repr: {residue_repr.shape}")
    print(f"- pair_repr: {pair_repr.shape}")
    
    # Create outer product components with explicit shapes
    print("\nCreating outer product components:")
    
    # First approach - using unsqueeze and expand
    h_i_1 = residue_repr.unsqueeze(2)
    print(f"- h_i after unsqueeze(2): {h_i_1.shape}")  # (B, L, 1, D_res)
    
    h_i_2 = h_i_1.expand(-1, -1, seq_len, -1)
    print(f"- h_i after expand: {h_i_2.shape}")  # (B, L, L, D_res)
    
    h_j_1 = residue_repr.unsqueeze(1)
    print(f"- h_j after unsqueeze(1): {h_j_1.shape}")  # (B, 1, L, D_res)
    
    h_j_2 = h_j_1.expand(-1, seq_len, -1, -1)
    print(f"- h_j after expand: {h_j_2.shape}")  # (B, L, L, D_res)
    
    # Alternative approach - using repeat/tile
    h_i_alt = residue_repr.unsqueeze(2).repeat(1, 1, seq_len, 1)
    h_j_alt = residue_repr.unsqueeze(1).repeat(1, seq_len, 1, 1)
    
    print("\nAlternative approach (repeat/tile):")
    print(f"- h_i_alt: {h_i_alt.shape}")  # (B, L, L, D_res)
    print(f"- h_j_alt: {h_j_alt.shape}")  # (B, L, L, D_res)
    
    # Check equality
    print(f"\nAre approaches equivalent?")
    print(f"- h_i methods equal: {torch.all(h_i_2 == h_i_alt)}")
    print(f"- h_j methods equal: {torch.all(h_j_2 == h_j_alt)}")
    
    # Concatenate for pair input
    pair_input = torch.cat([h_i_2, h_j_2, pair_repr], dim=-1)
    print(f"\nFinal pair input shape: {pair_input.shape}")  
    # (B, L, L, 2*D_res + D_pair)

# Example usage
batch_size, seq_len = 2, 5
residue_dim, pair_dim = 128, 64

residue_repr = torch.rand((batch_size, seq_len, residue_dim))
pair_repr = torch.rand((batch_size, seq_len, seq_len, pair_dim))

debug_pair_update_shapes(residue_repr, pair_repr)
```

### Device Inconsistency

**Problem**: Operations failed because tensors are on different devices.

**Solution**: Ensure all inputs to the transformer block are on the same device and the modules themselves are moved to that device:

```python
def ensure_device_consistency(module, *tensors):
    """
    Ensure module and all tensors are on the same device.
    
    Args:
        module: PyTorch module
        *tensors: Tensors to check/move
        
    Returns:
        List of tensors, all on the module's device
    """
    # Get module device
    module_device = next(module.parameters()).device
    
    # Check and move tensors
    result_tensors = []
    for i, tensor in enumerate(tensors):
        if tensor.device != module_device:
            print(f"Warning: Tensor {i} is on {tensor.device}, "
                  f"moving to {module_device}")
            result_tensors.append(tensor.to(module_device))
        else:
            result_tensors.append(tensor)
    
    return result_tensors

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_block = TransformerBlock(config).to(device)

# Create tensors on CPU
residue_repr_cpu = torch.rand((batch_size, seq_len, config['residue_embed_dim']))
pair_repr_cpu = torch.rand((batch_size, seq_len, seq_len, config['pair_embed_dim']))
mask_cpu = torch.ones((batch_size, seq_len), dtype=torch.bool)

# Ensure device consistency
residue_repr, pair_repr, mask = ensure_device_consistency(
    transformer_block, residue_repr_cpu, pair_repr_cpu, mask_cpu
)

# Now safe to use in forward pass
residue_repr, pair_repr = transformer_block(residue_repr, pair_repr, mask)
```

### Memory Efficiency for Large Sequences

**Problem**: Out of memory errors when processing large RNA sequences.

**Solution**: Use smaller batch sizes, gradient checkpointing, or optimize the implementation for memory efficiency:

```python
def memory_efficient_transformer_pass(
    transformer_blocks,
    residue_repr,
    pair_repr,
    mask=None,
    use_checkpointing=True
):
    """
    Memory-efficient forward pass through transformer blocks.
    
    Args:
        transformer_blocks: List of transformer blocks
        residue_repr: Residue representations
        pair_repr: Pair representations
        mask: Optional boolean mask
        use_checkpointing: Whether to use gradient checkpointing
        
    Returns:
        Updated residue and pair representations
    """
    # Free memory if on CUDA
    if residue_repr.device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Process one block at a time to minimize peak memory
    for i, block in enumerate(transformer_blocks):
        if use_checkpointing and block.training:
            # Use gradient checkpointing for memory efficiency
            def custom_forward(res, pair, msk):
                return block(res, pair, msk)
            
            residue_repr, pair_repr = torch.utils.checkpoint.checkpoint(
                custom_forward, residue_repr, pair_repr, mask
            )
        else:
            # Standard forward pass
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # Optional: Free memory after each block
        if residue_repr.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Print memory usage stats
        if residue_repr.device.type == 'cuda':
            print(f"Block {i+1}/{len(transformer_blocks)} - "
                  f"Memory: {torch.cuda.memory_allocated() / 1e6:.1f}MB, "
                  f"Peak: {torch.cuda.max_memory_allocated() / 1e6:.1f}MB")
    
    return residue_repr, pair_repr

# Example usage for large sequences
batch_size, seq_len = 1, 500  # Large sequence
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'dropout': 0.1,
    'ffn_dim': 512
}

# Create smaller model for large sequences
config['residue_embed_dim'] = 64
config['pair_embed_dim'] = 32
config['ffn_dim'] = 256

# Create mock data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
residue_repr = torch.rand((batch_size, seq_len, config['residue_embed_dim']), device=device)
pair_repr = torch.rand((batch_size, seq_len, seq_len, config['pair_dim']), device=device)
mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

# Create transformer blocks
num_blocks = 2  # Fewer blocks for large sequences
transformer_blocks = nn.ModuleList([
    TransformerBlock(config) for _ in range(num_blocks)
]).to(device)

# Process efficiently
residue_repr, pair_repr = memory_efficient_transformer_pass(
    transformer_blocks, residue_repr, pair_repr, mask
)
```

## Conclusion

This document has provided comprehensive examples for implementing and using the Transformer Block component in the RNA 3D folding pipeline. The examples cover basic implementation, masking and shape management, integration with other components, common patterns and best practices, and solutions to common errors.

By following these examples, you can implement a functional transformer block that correctly processes residue and pair representations while properly handling masking and shape transformations. The transformer block is a critical component that enables the model to capture complex dependencies between RNA residues required for accurate 3D structure prediction.
