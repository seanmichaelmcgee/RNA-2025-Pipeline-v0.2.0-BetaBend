# RNAFoldingModel Component Handoff

## Component Information

- **Component Name**: RNAFoldingModel
- **Version**: 1.0
- **Last Updated**: 2025-04-20
- **Implementation Status**: Complete
- **Test Coverage**: 100%
- **Location**: `src/models/rna_folding_model.py`

## Component Description

The RNAFoldingModel is the main model architecture for RNA 3D structure prediction. It integrates embeddings, transformer blocks, and the IPA module into a cohesive model that predicts 3D coordinates, confidence scores, and torsion angles for RNA molecules. This V1 implementation focuses on establishing the core architecture and proper data flow between components.

## Public API

### Class Definition

```python
class RNAFoldingModel(nn.Module):
    def __init__(self, config: Dict):
        """
        Initialize the RNA folding model.
        
        Args:
            config: Dictionary containing model parameters
                - num_blocks: Number of transformer blocks
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings
                - num_attention_heads: Number of attention heads in transformer
                - dropout: Dropout probability
                - confidence_output_dim: Dimension of confidence output (usually 1)
                - angles_output_dim: Dimension of angle output (usually 4)
                - ffn_dim: Hidden dimension for feed-forward networks
                - seq_embed_dim: Dimension of sequence embeddings
                - max_relative_position: Maximum relative distance to consider for positional encoding
        """
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNA folding model.
        
        Args:
            batch: Dictionary of input tensors from the data loader
                - sequence_int: Integer-encoded RNA sequence (batch_size, seq_len)
                - dihedral_features: Dihedral angle features (batch_size, seq_len, 4)
                - pairing_probs: Pairing probabilities (batch_size, seq_len, seq_len)
                - positional_entropy: Positional entropy (batch_size, seq_len)
                - accessibility: Solvent accessibility (batch_size, seq_len)
                - coupling_matrix: Evolutionary coupling (batch_size, seq_len, seq_len)
                - mask: Boolean mask (batch_size, seq_len)
        
        Returns:
            Dictionary of output tensors:
                - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
                - pred_confidence: Predicted confidence scores (batch_size, seq_len)
                - pred_angles: Predicted torsion angles (batch_size, seq_len, 4)
        """
```

## Data Structures and Tensor Specifications

### Input Format

The model expects a dictionary with the following keys and tensor shapes:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| sequence_int | (batch_size, seq_len) | LongTensor | Integer-encoded RNA sequence (0=A, 1=C, 2=G, 3=U, 4=N/padding) |
| dihedral_features | (batch_size, seq_len, 4) | FloatTensor | Dihedral angles in sin/cos encoding [sin(η), cos(η), sin(θ), cos(θ)] |
| pairing_probs | (batch_size, seq_len, seq_len) | FloatTensor | Base pairing probabilities |
| positional_entropy | (batch_size, seq_len) | FloatTensor | Positional entropy from sequence alignment |
| accessibility | (batch_size, seq_len) | FloatTensor | Solvent accessibility predictions |
| coupling_matrix | (batch_size, seq_len, seq_len) | FloatTensor | Evolutionary coupling matrix |
| mask | (batch_size, seq_len) | BoolTensor | Boolean mask (True for valid positions, False for padding) |
| coordinates | (batch_size, seq_len, 3) | FloatTensor | True 3D coordinates for training (optional for inference) |

### Output Format

The model returns a dictionary with the following keys and tensor shapes:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| pred_coords | (batch_size, seq_len, 3) | FloatTensor | Predicted 3D coordinates (x, y, z) |
| pred_confidence | (batch_size, seq_len) | FloatTensor | Predicted confidence scores (logits) |
| pred_angles | (batch_size, seq_len, 4) | FloatTensor | Predicted dihedral angles [sin(η), cos(η), sin(θ), cos(θ)] |

## Integration Points

### Component Dependencies

- **EmbeddingModule** (`src/models/embeddings.py`): Processes input features to create initial residue and pair representations
- **TransformerBlock** (`src/models/transformer_block.py`): Processes residue and pair representations through attention mechanisms
- **IPAModule** (`src/models/ipa_module.py`): Predicts 3D coordinates from residue representations

### Expected Behavior

1. The `__init__` method initializes model components and prediction heads
2. The `forward` method processes input batch through:
   - Embedding module to create initial representations
   - Stack of transformer blocks for iterative refinement
   - IPA module for 3D coordinate prediction
   - Prediction heads for confidence and angles
3. Masks are propagated throughout the model to handle variable-length sequences
4. Predicted outputs are returned for loss calculation

## Environment and Dependencies

- PyTorch >= 2.1.0
- Numpy >= 1.24.0
- Python >= 3.9

## Testing Requirements

### Unit Tests

Run all model-specific unit tests:
```bash
python -m pytest tests/test_model.py -v
```

Run a specific test:
```bash
python -m pytest tests/test_model.py::TestRNAFoldingModel::test_forward_shapes -v
```

### Test Coverage

The model has 100% test coverage including tests for:
- Initialization
- Forward pass shapes
- Mask propagation
- Input validation
- End-to-end integration with loss functions
- Gradient flow

## Implementation Details

### Model Architecture

1. **Input Processing**: 
   - Embedding module converts raw inputs to residue and pair representations
   - Positional encoding is added to capture sequence position information

2. **Representation Processing**:
   - Series of transformer blocks refine the representations
   - Each block processes both residue-level and pair-level information

3. **Structure Prediction**:
   - IPA module converts residue representations to 3D coordinates
   - Confidence head predicts per-residue confidence scores
   - Angle head predicts auxiliary dihedral angles

### Performance Considerations

- The model supports dynamic batch sizes and sequence lengths
- Mask handling ensures efficient processing of variable-length sequences
- Tensor dimensions scale linearly with sequence length for most operations
- Pair representations scale quadratically with sequence length (O(L²))
- Memory usage is dominated by pair representations for long sequences

## Extension and Maintenance

### Adding Features

To add features to the model:
1. Update the embedding module to process new input features
2. Modify the forward method to handle new inputs in the batch
3. Add new prediction heads if needed
4. Update the tests to verify the new functionality

### Future Improvements

Areas for future enhancement in V2:
- Replace IPA placeholder with full invariant point attention
- Add iterative coordinate refinement
- Incorporate attention-based structure module
- Support alternative input features for improved generalization

## Common Debugging

### Shape Mismatch Issues

**Issue**: If tensor shape mismatch errors occur:
- Check that `mask` is properly propagated
- Verify input feature dimensions match embedding expectations
- Ensure batch sizes and sequence lengths are consistent

**Resolution**: Add explicit shape validation or reshape operations

### Memory Issues

**Issue**: For long sequences, memory consumption is high:
- Pair representations scale quadratically (O(L²))
- Attention matrices in transformer blocks also scale quadratically

**Resolution**: 
- Use gradient checkpointing for very long sequences
- Consider chunk-wise processing for extreme cases

## Decision Log

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| Implement V1 as simple feed-forward | Establish interfaces and data flow | More complex IPA | 2025-04-20 |
| Separate angle and confidence heads | Allow independent optimization | Joint prediction | 2025-04-20 |
| Direct projection from transformer | Simplify V1 implementation | Multi-stage refinement | 2025-04-20 |
| Dictionary input/output format | Flexibility for future extensions | Fixed tensor inputs | 2025-04-20 |

## Usage Examples

### Basic Model Initialization

```python
import torch
from src.models.rna_folding_model import RNAFoldingModel

# Define model configuration
config = {
    "num_blocks": 4,
    "residue_embed_dim": 128,
    "pair_embed_dim": 64,
    "num_attention_heads": 4,
    "dropout": 0.1,
    "ffn_dim": 512,
    "seq_embed_dim": 32,
    "max_relative_position": 32,
    "max_len": 500,
    "num_embeddings": 5,
    "padding_idx": 4
}

# Initialize model
model = RNAFoldingModel(config)

# Optionally move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Forward Pass and Loss Calculation

```python
import torch
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_combined_loss

# Initialize model (as above)
model = RNAFoldingModel(config).to(device)

# Create or load a batch of data
batch = {
    "sequence_int": sequence_int.to(device),
    "dihedral_features": dihedral_features.to(device),
    "pairing_probs": pairing_probs.to(device),
    "positional_entropy": positional_entropy.to(device),
    "accessibility": accessibility.to(device),
    "coupling_matrix": coupling_matrix.to(device),
    "coordinates": coordinates.to(device),
    "mask": mask.to(device)
}

# Forward pass
outputs = model(batch)

# Calculate loss
loss_weights = {"fape": 1.0, "confidence": 0.1, "angle": 0.5}
total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)

# Backward pass for training
total_loss.backward()
```