# CLAUDE.md - Model Components Instance

This file provides guidance to Claude Code (claude.ai/code) when working with neural network components in this repository.

## Core Implementation Patterns

### PyTorch Module Structure
- All components inherit from `nn.Module`
- Configuration-driven initialization with `config` dictionary
- Forward methods with consistent tensor shapes and types
- Explicit mask handling throughout

### Neural Network Component Testing
```python
# Test component with dummy data
def test_component_forward():
    config = {
        "residue_embed_dim": 128,
        "pair_embed_dim": 64,
        "num_attention_heads": 4,
    }
    
    # Initialize component
    component = YourComponent(config)
    
    # Create dummy inputs with appropriate shapes
    batch_size, seq_len = 2, 10
    residue_repr = torch.randn(batch_size, seq_len, config["residue_embed_dim"])
    pair_repr = torch.randn(batch_size, seq_len, seq_len, config["pair_embed_dim"])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Run forward pass
    outputs = component(residue_repr, pair_repr, mask)
    
    # Assert expected shapes
    assert outputs[0].shape == residue_repr.shape
    assert outputs[1].shape == pair_repr.shape
```

## Commands

### Development Commands
- Run core model tests: `python -m pytest tests/test_*module*.py -v`
- Run type checking: `mypy src/models/`
- Run formatting: `black src/models/ tests/` and `isort src/models/ tests/`
- View model architecture: `python -c "import torch; from src.models.transformer_block import TransformerBlock; config = {'residue_embed_dim': 128, 'pair_embed_dim': 64, 'num_attention_heads': 4}; print(TransformerBlock(config))"`

### GPU Compatibility Check
```python
# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourModel(config).to(device)
batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
outputs = model(batch)
```

## Debugging Tips

### Common Shape Issues
- Ensure mask dimensions match the corresponding tensor dimensions
- Check for accidental squeezing or unsqueezing of dimensions
- Verify that batch dimension is preserved throughout

### Memory Optimization
- Use smaller embedding dimensions during development
- Consider batch size reduction for large sequence lengths
- Profile memory usage with `torch.cuda.max_memory_allocated()`

### Gradient Flow
```python
# Check gradient flow through the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
output = model(inputs)
loss = criterion(output, targets)
loss.backward()

# Verify gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean().item()}")
    else:
        print(f"{name}: No gradient")
```

## Model Component Configuration Reference

```python
# Complete configuration reference for neural network components
config = {
    # Sequence embedding parameters
    "num_embeddings": 5,         # Number of nucleotide tokens (A, C, G, U, N/padding)
    "seq_embed_dim": 32,         # Dimension of sequence embeddings
    "padding_idx": 4,            # Index for padding token
    
    # Positional encoding parameters
    "residue_embed_dim": 128,    # Dimension of residue representations
    "max_len": 500,              # Maximum sequence length for positional encodings
    
    # Relative positional encoding parameters
    "max_relative_position": 32, # Maximum relative distance to consider
    "rel_pos_dim": 32,           # Dimension of relative position embeddings
    
    # Embedding module parameters
    "pair_embed_dim": 64,        # Dimension of pair representations
    "use_conservation": True,    # Whether to use conservation features
    
    # Transformer parameters
    "num_attention_heads": 4,    # Number of attention heads
    "dropout": 0.1,              # Dropout probability
    "ffn_dim": 512,              # Hidden dimension for feed-forward networks
    
    # IPA module parameters
    "ipa_dim": 64,               # Hidden dimension for IPA module
    "num_ipa_iterations": 1,     # Number of IPA iterations (for V2)
    
    # Main model parameters
    "num_layers": 4,             # Number of transformer layers
}
```