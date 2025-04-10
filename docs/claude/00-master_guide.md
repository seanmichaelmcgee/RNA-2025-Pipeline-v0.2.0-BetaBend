# RNA 3D Structure Prediction: Master Implementation Guide

## Project Overview

You are assisting with implementing a PyTorch-based machine learning pipeline for predicting the 3D structure of RNA molecules from their sequences. This project targets the Stanford RNA 3D Folding Kaggle competition and includes a custom Transformer-based architecture with specialized components for RNA feature processing.

### Key Objectives

1. Create a modular, well-tested implementation following the architecture in `3_Architecture_Specification.md`
2. Ensure compatibility with both local development and Kaggle submission environments
3. Maintain strict adherence to core principles like path parameterization and modularity
4. Generate a working V1 system with scaled-down parameters that can be iterated upon

## Implementation Roadmap

Follow this sequence to implement the full system:

### Phase 1: Data Pipeline (Start Here)
1. **Data Loading** (`src/data_loading.py`)
   - Implement `RNADataset` class and helper functions
   - Create `collate_fn` for variable-length sequences
   - Focus on proper feature loading from .npz files

### Phase 2: Model Components
2. **Embedding Layers** (`src/models/embeddings.py`)
   - Implement sequence, positional, and relative positional embeddings
   - Create input projection layers for features

3. **Transformer Block** (`src/models/transformer_block.py`)
   - Implement standard multi-head attention
   - Create simplified pair update mechanism
   - Ensure proper residual connections and layer normalization

4. **IPA Module Placeholder** (`src/models/ipa_module.py`)
   - Create simple linear layer placeholder for structure generation
   - Maintain correct tensor shapes for integration

5. **Loss Functions** (`src/losses.py`)
   - Implement simplified FAPE loss proxy
   - Create confidence and angle prediction losses
   - Handle proper masking for variable-length sequences

### Phase 3: Integration
6. **Model Integration** (`src/models/rna_folding_model.py`)
   - Combine all components into coherent model
   - Implement forward pass with correct data flow
   - Manage tensor shapes throughout pipeline

7. **Pipeline Testing**
   - Create end-to-end test script
   - Validate with small batches on GPU
   - Verify memory usage and performance

## Core Implementation Principles

### Critical: Path Parameterization
- **NEVER use hardcoded paths** in `src/` modules
- All file/directory paths must be passed as arguments from the orchestration layer
- Always use `os.path.join()` for constructing paths
- This enables both local development and Kaggle compatibility

```python
# CORRECT - paths as arguments
def load_features(target_id: str, features_dir: str):
    feature_path = os.path.join(features_dir, f"{target_id}_features.npz")
    # ...

# INCORRECT - hardcoded paths
def load_features(target_id: str):
    feature_path = f"data/features/{target_id}_features.npz"  # NEVER DO THIS
    # ...
```

### Modularity and Code Organization
- Keep core logic in `src/` modules, orchestration in `scripts/`
- Design components with clear interfaces
- Follow PyTorch conventions (nn.Module, device handling)
- Enable independent testing of components

### Test-Driven Development
- Write tests alongside implementation
- Verify functionality before integration
- Use appropriate fixtures and mocks

### Memory Efficiency
- Start with scaled-down parameters
- Use PyTorch best practices (e.g., `torch.no_grad()` for evaluation)
- Monitor GPU memory usage

## Common Implementation Pitfalls

### Data Loading Issues
- **Missing Feature Handling**: Some sequences may lack certain feature files (especially evolutionary data). Provide default zero tensors of correct shape.
- **Shape Inconsistencies**: Verify sequence lengths match across features, return clear errors on mismatch.
- **Padding & Masking**: Ensure proper padding of variable-length sequences and correct attention masks.

### Model Implementation Challenges
- **Device Management**: Remember to move all tensors to the target device (`tensor.to(device)`).
- **Dimension Tracking**: Keep track of tensor shapes through transformations. Document expected shapes in comments.
- **Gradient Accumulation**: Use separate graphs for different loss components, detach when needed.

### Integration Pain Points
- **End-to-End Flow**: The most common issue is interfaces between components. Document expected input/output formats clearly.
- **Memory Leaks**: Monitor memory utilization on GPU, use `torch.cuda.empty_cache()` when debugging.
- **Configuration Management**: Ensure config parameters are used consistently across components.

## User Interaction Patterns

### Implementation Requests
When asked to implement a component:
1. Determine which component in the roadmap is requested
2. Review the specific component guide document
3. Implement according to specifications with proper error handling
4. Include comprehensive docstrings and type hints
5. Create unit tests for the implementation

### Debugging Assistance
When debugging is requested:
1. Identify which component is causing issues
2. Review error messages carefully
3. Check common pitfalls for that component
4. Ask for specific error messages if not provided
5. Suggest specific fixes with explanations

### Integration Questions
For questions about connecting components:
1. Reference both component interfaces
2. Explain how data should flow between them
3. Highlight potential shape transformations
4. Suggest clean interface patterns

## Documentation Navigation

For detailed implementation guidance, refer to these component-specific guides:

- **Data Loading**: `docs/claude/components/10_data_loading/guide.md`
- **Embeddings**: `docs/claude/components/20_embeddings/guide.md`
- **Transformer Block**: `docs/claude/components/30_transformer_block/guide.md`
- **IPA Module**: `docs/claude/components/40_ipa_module/guide.md`
- **Loss Functions**: `docs/claude/components/50_losses/guide.md`

For integration and workflows:

- **Model Integration**: `docs/claude/workflows/60_model_integration.md`
- **Pipeline Testing**: `docs/claude/workflows/70_pipeline_testing.md`
- **Debugging**: `docs/claude/workflows/80_debugging.md`

For technical references:

- **Feature Formats**: `docs/claude/reference/feature_formats.md`
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md`

For detailed specifications, these original documents are available:

- **Project Setup**: `docs/1_Context_and_Setup.md`
- **Feature Specification**: `docs/2_Feature_Specification.md`
- **Architecture Specification**: `docs/3_Architecture_Specification.md`
- **Product Requirements**: `docs/4_Product_Requirements_V1.md`
- **Implementation Plan**: `docs/6_Tactical_Plan_V1.md`
- **Agent Rules**: `docs/7_AI_Agent_Rules.md`

## Next Steps

Begin implementation with the data loading component by reviewing:
1. `docs/claude/components/10_data_loading/guide.md`
2. The feature specifications in `docs/2_Feature_Specification.md`

Then implement the `src/data_loading.py` module following test-driven development principles.
