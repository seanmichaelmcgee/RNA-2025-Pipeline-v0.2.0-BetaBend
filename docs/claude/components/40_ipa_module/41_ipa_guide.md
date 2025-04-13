# IPA Module Implementation Guide

## Component Overview

The Invariant Point Attention (IPA) module is a critical component in the RNA 3D folding pipeline, responsible for converting the learned representations from the transformer backbone into actual 3D coordinates. Drawing inspiration from AlphaFold, this module is designed to predict the spatial positions of C1' atoms in RNA nucleotides, using both residue-level and pair-level representations processed by the transformer blocks.

For V1, we will implement a **simplified placeholder** that uses a basic linear projection from residue representations to 3D coordinates, while establishing the interface and structure needed for a more sophisticated implementation in future versions. This approach allows us to focus on getting the end-to-end pipeline working quickly while laying the groundwork for later improvements.

## Requirements Reference

From the Product Requirements Document (`4_Product_Requirements_V1.md`):

- **MA-07**: Implement a **placeholder** `IPAModule` that accepts residue features and outputs 3D coordinates linearly (shape `(B, L, 3)`). Document clearly as placeholder.

From the Architecture Specification (`3_Architecture_Specification.md`):

- For V1, create a "placeholder module (e.g., `nn.Linear` predicting $x_i$ directly from $h_i^{(L)}$)" with the intent to implement full IPA later.
- The target future approach is based on Invariant Point Attention, which "iteratively refine[s] residue coordinates and orientations using a spatially aware attention mechanism that is invariant/equivariant to global rotations and translations."

## Technical Background

### Invariant Point Attention Concept

The full IPA module (planned for future versions) is based on these principles:

1. **Invariance and Equivariance**: The module respects physical principles by being invariant to global rotations and translations, meaning the predictions shouldn't change if the entire structure is rotated or moved.

2. **Iterative Refinement**: Multiple cycles of attention and update gradually improve the 3D structure.

3. **Frame-Based Representation**: Each residue has both a position (xyz coordinates) and an orientation (rotation matrix or frame), which together define its full spatial arrangement.

### V1 Placeholder Approach

For V1, we simplify this complexity to:

1. **Direct Linear Projection**: Use a simple MLP to directly convert residue representations to 3D coordinates.
2. **Single-Pass Prediction**: No iterative refinement, just a single forward pass.
3. **Positions Only**: Predict only positions (xyz coordinates), not orientations.

This allows us to establish the complete pipeline while deferring the complexity of the full IPA implementation.

## Interfaces

### Input Interface

The IPA module takes inputs from the final transformer block:

```python
# Primary inputs
residue_repr: torch.Tensor  # Shape: (batch_size, seq_len, residue_dim)
                           # Residue (per-nucleotide) representations

pair_repr: torch.Tensor    # Shape: (batch_size, seq_len, seq_len, pair_dim)
                          # Pair (nucleotide-pair) representations
                          # Not used in V1 but included in interface for future versions
                          
mask: torch.Tensor         # Shape: (batch_size, seq_len), dtype: torch.bool
                          # Boolean mask indicating valid positions (True for valid)
```

### Output Interface

The IPA module produces predicted 3D coordinates:

```python
# Output
predicted_coords: torch.Tensor  # Shape: (batch_size, seq_len, 3)
                              # Predicted coordinates of C1' atoms
```

## Implementation Steps

### 1. Define the `IPAModule` Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class IPAModule(nn.Module):
    """
    Invariant Point Attention module (V1 Placeholder).
    
    This V1 implementation is a simplified placeholder that projects residue
    representations directly to 3D coordinates using a simple MLP. It establishes
    the interface for future versions which will implement the full IPA algorithm
    with iterative, frame-based, coordinate refinement.
    """
    
    def __init__(self, config: dict):
        """
        Initialize IPA module.
        
        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings (unused in V1)
                - ipa_dim: Hidden dimension for IPA module projection (optional)
        """
        super().__init__()
        
        # Extract parameters from config
        self.residue_dim = config['residue_embed_dim']
        self.pair_dim = config.get('pair_embed_dim', 64)  # Unused in V1 but stored for future
        
        # Get hidden dimension for projection (default to half of residue_dim)
        self.ipa_dim = config.get('ipa_dim', self.residue_dim // 2)
        
        # Initialize coordinate prediction MLP
        self.coord_projection = nn.Sequential(
            nn.Linear(self.residue_dim, self.ipa_dim),
            nn.ReLU(),
            nn.Linear(self.ipa_dim, 3)  # Output: x, y, z coordinates
        )
        
        # Store future-related configuration for documentation
        self.num_iterations = config.get('num_ipa_iterations', 1)  # V1 uses 1, future versions will use more
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        # Xavier/Glorot initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
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
        # V1 Implementation: Simple linear projection from residue representations to coordinates
        coords = self.coord_projection(residue_repr)  # (batch_size, seq_len, 3)
        
        # Apply mask if provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1).float()
        
        return coords
```

### 2. Coordinate Initialization Helper (For Future V2+ Implementation)

This helper will be useful in future versions but is included for documentation:

```python
def _initialize_coordinates(
    self,
    residue_repr: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Initialize 3D coordinates from residue representations.
    
    This is a helper method for future versions that will use iterative refinement.
    In V1, this is equivalent to the forward method.
    
    Args:
        residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
        mask: Boolean mask of shape (batch_size, seq_len)
        
    Returns:
        Initial coordinates of shape (batch_size, seq_len, 3)
    """
    # Project residue features to coordinates
    coords = self.coord_projection(residue_repr)  # (batch_size, seq_len, 3)
    
    # Apply mask if provided
    if mask is not None:
        coords = coords * mask.unsqueeze(-1).float()
    
    return coords
```

## Critical Aspects

### 1. Interface Design for Future Compatibility

Despite the simplified V1 implementation, the interface is designed to support the full IPA in future versions:

- **Input Arguments**: The V1 `forward` method includes `pair_repr` even though it's unused, to maintain a stable interface for V2+.
- **Masking Support**: Properly handles masks to ensure padding positions are set to zero in the output coordinates.
- **Configuration Parameters**: Stores parameters like `num_ipa_iterations` that will be used in future versions.

### 2. Coordinate System and Scale

- The model outputs raw coordinates in an arbitrary scale and reference frame. 
- In a full implementation, we might normalize or standardize the output coordinates, but for V1 we keep it simple.
- If used for scoring (e.g., TM-score), the evaluation typically aligns the predicted structure to the reference structure, making the absolute coordinate system less relevant.

### 3. Placeholder Nature of V1

It's important to clearly document and communicate that the V1 implementation is a placeholder:

- The direct linear projection will not capture proper 3D structure as effectively as the full IPA approach.
- It doesn't guarantee physically realistic structures (e.g., proper bond lengths, angles).
- It lacks the rotation/translation invariance properties of the full IPA.

### 4. Memory and Computation Efficiency

Even this placeholder implementation must handle the core shape transformations efficiently:

- The projections should use batched operations for efficiency.
- The mask application ensures we don't waste computation on padding positions.

## Testing Requirements

### Basic Functionality Tests

1. **Output Shape Test**: Verify the output tensor has the expected shape `(batch_size, seq_len, 3)`.
2. **Masking Test**: Confirm that masked positions (padding) have zero coordinates in the output.
3. **Device Compatibility Test**: Ensure the module works correctly on both CPU and CUDA devices.
4. **Gradient Flow Test**: Check that gradients flow correctly through the module during backpropagation.
5. **Configuration Handling Test**: Test that the module correctly uses configuration parameters.

### Integration Tests

1. **Integration with Transformer Block**: Verify that the IPA module correctly processes the outputs from the transformer blocks.
2. **End-to-End Forward Pass**: Test the entire forward pass from embedding to transformer to IPA module.

### Example Test Case

```python
import torch
import torch.nn as nn
from src.models.ipa_module import IPAModule

def test_ipa_module_basic():
    """Test basic functionality of the IPAModule."""
    # Create configuration
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        'ipa_dim': 64
    }
    
    # Create module
    ipa_module = IPAModule(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False  # Mask last two positions in first sequence
    
    # Forward pass
    coords = ipa_module(residue_repr, pair_repr, mask)
    
    # Check output shape
    assert coords.shape == (batch_size, seq_len, 3)
    
    # Check masking
    assert torch.all(coords[0, -2:] == 0)
    assert not torch.all(coords[0, :-2] == 0)
    assert not torch.all(coords[1] == 0)
```

## Example Usage

Here's an example showing how to use the IPA module in the context of the full model:

```python
import torch
import torch.nn as nn
from src.models.transformer_block import TransformerBlock
from src.models.ipa_module import IPAModule

# Create configuration
config = {
    'residue_embed_dim': 128,
    'pair_embed_dim': 64,
    'num_attention_heads': 4,
    'ipa_dim': 64,
    'dropout': 0.1
}

# Create transformer blocks and IPA module
transformer_blocks = nn.ModuleList([
    TransformerBlock(config) for _ in range(4)
])
ipa_module = IPAModule(config)

# Create dummy inputs (as if from embedding layer)
batch_size = 2
seq_len = 10
residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
mask[0, -2:] = False  # Mask last two positions in first sequence

# Process through transformer blocks
for block in transformer_blocks:
    residue_repr, pair_repr = block(residue_repr, pair_repr, mask)

# Generate 3D coordinates using IPA module
coords = ipa_module(residue_repr, pair_repr, mask)

print(f"Generated coordinates shape: {coords.shape}")  # (2, 10, 3)
```

## Future Implementation (V2+)

For future versions, the IPA module will be expanded to include:

### 1. Frame-Based Representation

Instead of just coordinates, each residue will have:
- A position vector (3D coordinates)
- A rotation matrix or frame defining its orientation

### 2. Iterative Refinement

Multiple cycles of:
1. **IPA Attention**: Attention mechanism that considers relative positions and orientations
2. **Equivariant Update**: Updates positions and orientations in a way that respects rotation/translation equivariance

### 3. Full Interface for V2+

```python
def forward_v2(
    self,
    residue_repr: torch.Tensor,
    pair_repr: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Full IPA forward pass (V2+ implementation).
    
    Args:
        residue_repr: Residue representations (batch_size, seq_len, residue_dim)
        pair_repr: Pair representations (batch_size, seq_len, seq_len, pair_dim)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        Predicted coordinates (batch_size, seq_len, 3)
    """
    # Initialize coordinates and frames
    coords = self._initialize_coordinates(residue_repr, mask)
    frames = self._initialize_frames(residue_repr, mask)
    
    # Iterative refinement
    for _ in range(self.num_iterations):
        # IPA Attention: Update residue representations based on current coords/frames
        residue_repr = self._ipa_attention(residue_repr, pair_repr, coords, frames, mask)
        
        # Update coordinates and frames using updated representations
        coords, frames = self._update_coordinates_and_frames(residue_repr, coords, frames, mask)
    
    return coords
```

## Related Documentation

- **Architecture Specification**: `docs/3_Architecture_Specification.md` - See "Structure Prediction Module (3D Coordinate Generation)" section
- **Product Requirements**: `docs/4_Product_Requirements_V1.md` - Requirement MA-07
- **Transformer Block Guide**: `docs/claude/components/30_transformer_block/transformer_guide.md` - For understanding of upstream component interfaces

## Next Steps

1. Implement the V1 placeholder `IPAModule` in `src/models/ipa_module.py`
2. Write unit tests in `tests/test_ipa_module.py`
3. Integrate with the main RNA folding model
4. Begin planning for the full IPA implementation in future versions:
   - Research frame-based representation and equivariant network design
   - Design the iteration mechanism for coordinate refinement
   - Develop the invariant attention mechanism

After implementing the IPA module placeholder, the main model will be able to predict 3D coordinates from RNA sequences, completing the core prediction pipeline of the RNA 3D folding project. While these initial predictions may not capture all structural details, they provide a foundation for the more sophisticated IPA implementation in future versions.
