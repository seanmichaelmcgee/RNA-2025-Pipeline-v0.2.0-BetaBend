# Integration Instance CLAUDE.md

This document provides guidance for Claude Code when working with the Integration Instance (03) of the RNA 3D folding model pipeline.

## Purpose

This instance is responsible for assembling the complete RNA 3D folding model architecture by integrating components from the Data Pipeline (01) and Model Components (02) instances into a cohesive model for training.

## Build/Test Commands

- Run unit tests: `python -m pytest tests/test_losses.py -v`
- Run model tests: `python -m pytest tests/test_model.py -v`  
- Run integration tests: `python -m pytest tests/test_integration.py -v`
- Run specific test: `python -m pytest tests/test_losses.py::TestCombinedLoss::test_combined_loss_calculation -v`
- Check tensor shapes: `python scripts/debug_model_shapes.py`
- Memory profiling: `python scripts/profile_memory.py`
- Lint: `black src/ tests/`
- Type checking: `mypy src/`

## Integration Patterns

### Component Assembly

```python
# Assembly pattern for model components
model = RNAFoldingModel(config)

# Input batch format from data loader
batch = {
    'sequence_int': torch.LongTensor,        # shape: (B, L)
    'dihedral_features': torch.FloatTensor,  # shape: (B, L, 4) 
    'pairing_probs': torch.FloatTensor,      # shape: (B, L, L)
    'positional_entropy': torch.FloatTensor, # shape: (B, L)
    'accessibility': torch.FloatTensor,      # shape: (B, L)
    'coupling_matrix': torch.FloatTensor,    # shape: (B, L, L)
    'coordinates': torch.FloatTensor,        # shape: (B, L, 3)
    'mask': torch.BoolTensor,                # shape: (B, L)
}

# Forward pass and loss calculation
outputs = model(batch)
# outputs = {
#    'pred_coords': torch.FloatTensor,     # shape: (B, L, 3)
#    'pred_confidence': torch.FloatTensor, # shape: (B, L)
#    'pred_angles': torch.FloatTensor      # shape: (B, L, 4)
# }

loss_weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}
total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)
```

### Numerical Stability Approaches

- Use small epsilon values (e.g., 1e-8) for all divisions
- Apply clamping for distancesc (e.g., torch.clamp(distance, 0.0, 10.0))
- Handle potential NaN/Inf values with torch.nan_to_num
- Validate tensor shapes before operations
- Apply stable implementations for coordinate alignment (Kabsch)
- Ensure proper mask handling throughout to avoid invalid computations

### Memory Optimization Techniques

- Use in-place operations where possible (x.add_() instead of x = x + y)
- Clear intermediate variables when no longer needed
- Apply masking early to avoid calculations on padding tokens
- Use torch.utils.checkpoint for gradient checkpointing if needed
- Regularly monitor peak memory usage with torch.cuda.max_memory_allocated()

## Loss Function Implementation

```python
# FAPE Loss: Frame-aligned point error (simplified for V1)
fape_loss = compute_stable_fape_loss(
    pred_coords=outputs['pred_coords'],  # (B, L, 3)
    true_coords=batch['coordinates'],    # (B, L, 3)
    mask=batch['mask']                   # (B, L)
)

# Confidence loss: Train model to predict per-residue confidence
conf_loss = compute_confidence_loss(
    pred_confidence=outputs['pred_confidence'],  # (B, L)
    pred_coords=outputs['pred_coords'],          # (B, L, 3)
    true_coords=batch['coordinates'],            # (B, L, 3)
    mask=batch['mask']                           # (B, L)
)

# Angle loss: Auxiliary loss for angle prediction
angle_loss = compute_angle_loss(
    pred_angles=outputs['pred_angles'],      # (B, L, 4)
    true_angles=batch['dihedral_features'],  # (B, L, 4)
    mask=batch['mask']                       # (B, L)
)
```

## Component Handoff Procedures

### Receiving Components

1. Verify interface documentation against code implementation
2. Run unit tests to confirm component functionality
3. Test component behavior with masked inputs
4. Verify gradient flow through component
5. Check tensor shapes and device handling
6. Validate error handling for edge cases
7. Document any integration challenges

### Providing Components

1. Complete implementation with comprehensive tests
2. Verify component functionality in isolation
3. Create detailed interface documentation
4. Prepare examples demonstrating usage
5. Create formal handoff documentation
6. Support testing instance with clarifications

## Interface Verification Methods

- Tensor shape consistency checks
- Type annotation verification
- Input validation in forward methods
- Device consistency checks
- Masking propagation tests
- Gradient flow verification
- Memory usage profiling
- Numerical stability tests
