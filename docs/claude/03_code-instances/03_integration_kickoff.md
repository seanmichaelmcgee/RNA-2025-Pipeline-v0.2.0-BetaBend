# Integration Claude Code Instructions

## Instance Purpose

You are the Integration instance responsible for assembling the complete RNA 3D folding model architecture by combining individual components into a cohesive pipeline. Your primary focus is implementing the main `RNAFoldingModel` class that integrates embeddings, transformer blocks, and prediction heads, along with the loss functions needed for optimization. You serve as the critical bridge between component development and end-to-end functionality, ensuring proper data flow, correct assembly, and appropriate loss computation for the training process.

## Core Responsibilities

- Implement `src/models/rna_folding_model.py`:
  - Main `RNAFoldingModel` class (MA-01, MA-02) as a PyTorch `nn.Module`
  - Input projection layers for residue and pair features (MA-04)
  - Assembly of transformer blocks into backbone (MA-06)
  - Confidence prediction head (MA-08)
  - Angle prediction head for multi-task learning (MA-09)
  - Complete forward method ensuring proper data flow (MA-10, MA-11)
  - Config-driven parameter initialization

- Implement `src/losses.py`:
  - Coordinate loss function (`compute_fape_loss` - simplified proxy for V1) (LF-01)
  - Confidence prediction loss function (`compute_confidence_loss`) (LF-02)
  - Auxiliary angle prediction loss function (`compute_angle_loss`) (LF-03)
  - Proper mask handling in all loss functions (LF-04)
  - Combined loss function for training

- Create comprehensive unit tests:
  - `tests/test_model.py`: Test model initialization, forward pass, output shapes
  - `tests/test_losses.py`: Test loss function correctness, numerical stability, mask handling

- Support basic integration testing:
  - Facilitate end-to-end data flow verification (data→model→loss)
  - Assist with developing validation scripts for the complete pipeline

## Implementation Order

1. **Loss Functions Implementation** (`src/losses.py`):
   - Implement `compute_fape_loss` function (simplified Kabsch-based proxy)
   - Implement `compute_confidence_loss` function (lDDT proxy)
   - Implement `compute_angle_loss` function (sin/cos comparison)
   - Implement `compute_combined_loss` helper function
   - Write tests for all loss functions (`tests/test_losses.py`)

2. **RNAFoldingModel Implementation** (`src/models/rna_folding_model.py`):
   - Define model structure and initialize components from config
   - Calculate input dimensions and implement projection layers
   - Stack transformer blocks from instance 02
   - Implement confidence and angle prediction heads
   - Implement complete forward pass with proper tensor flow
   - Ensure proper mask propagation through components
   - Write tests for the model (`tests/test_model.py`)

3. **Integration Tests Support**:
   - Assist with creating basic test script(s) showing end-to-end flow
   - Verify compatibility between data loading, model, and loss functions
   - Support development of memory profiling and gradient flow checks

## Reference Documents

### Architecture and Requirements
- `docs/3_Architecture_Specification.md` - See "Backbone Model" and "Loss Functions" sections
- `docs/4_Product_Requirements_V1.md` - Requirements MA-01, MA-02, MA-04, MA-06, MA-08-11 and LF-01 to LF-04
- `docs/6_Tactical_Plan_V1.md` - Sections III.4 and IV for implementation guidance

### Implementation Guides
- `docs/claude/workflows/60_model_integration.md` - Complete integration workflow
- `docs/claude/components/50_losses/guide.md` - Detailed loss implementation guide
- `docs/claude/components/50_losses/examples.md` - Example implementations for losses
- `docs/claude/components/50_losses/testing.md` - Testing strategies for loss functions

### Component Interface Contracts
- `docs/claude/code-instances/shared/interface_specifications/RNADataset_v1.0.md` - Data batch format
- `docs/claude/code-instances/shared/interface_specifications/SequenceEmbedding_v1.0.md` - Embedding interface
- `docs/claude/code-instances/shared/interface_specifications/TransformerBlock_v1.0.md` - Transformer interface
- `docs/claude/code-instances/shared/interface_specifications/IPAModule_v1.0.md` - IPA placeholder interface

### Code Guidelines
- `docs/claude/01_implementation_principles.md` - Core implementation principles
- `docs/claude/7_AI_Agent_Rules.md` - Focus on rules 2.4 (Modularity) and 7.2 (Path Parameterization)

## Communication Guidelines

### When to Communicate with Other Instances

- **Instance 01 (Data Pipeline)**:
  - Request clarification on batch format, tensor shapes, or mask conventions
  - Verify handling of variable-length sequences and padding
  - Confirm expected data types and device handling

- **Instance 02 (Model Components)**:
  - Verify transformer block interface and expected input/output formats
  - Confirm embedding dimensions and processing requirements
  - Clarify IPA module placeholder functionality and parameter needs

- **Instance 04 (Testing)**:
  - Provide interface documentation for the complete model
  - Collaborate on integration testing strategies
  - Support debugging of end-to-end pipeline issues

### Handoff Protocol

- **Receiving Component Handoffs**:
  1. Review interface contracts from instances 01 and 02
  2. Verify tensor shapes, masks, and device handling match expectations
  3. Acknowledge receipt and provide feedback within 48 hours
  4. Document any interface mismatches or concerns
  5. Provide test cases to verify correct integration

- **Providing Component Handoffs**:
  1. Complete implementation of model and loss functions
  2. Write comprehensive documentation of model interface
  3. Provide working examples of model usage with expected inputs/outputs
  4. Demonstrate loss function usage with typical values
  5. Include common error conditions and handling patterns

### Documentation Standards

- Document all tensor shapes explicitly in docstrings and comments
- Specify mask handling for all functions
- Note memory and performance considerations
- Include complete interface contracts for all implemented components
- Update implementation journal with progress, challenges, and decisions

## Code Standards

### General Standards

- Adhere to PEP 8 style guidelines
- Use Google-style docstrings with type hints
- Include examples in docstrings for complex functions
- Add detailed comments for critical sections and calculations
- Apply consistent naming conventions across components

### Model-Specific Standards

1. **Configuration Management**:
   - Access all hyperparameters from config dictionary
   - Provide sensible defaults for optional parameters
   - Validate critical parameters (e.g., dimensions must be compatible)
   - Document all expected config parameters

2. **Shape Management**:
   - Add explicit tensor shape validation for critical operations
   - Document tensor shapes in comments and docstrings
   - Handle variable sequence lengths properly
   - Verify shape consistency at component boundaries

3. **Mask Propagation**:
   - Properly propagate masks through all components
   - Apply masks consistently (True = valid, False = padding)
   - Check mask shape compatibility with tensors
   - Ensure masked positions don't contribute to loss computation

4. **Error Handling**:
   - Validate input tensors early in forward pass
   - Provide detailed error messages with shape information
   - Handle edge cases such as empty/short sequences
   - Avoid silent failures or unexpected behavior

### Loss Function Standards

1. **Numerical Stability**:
   - Add small epsilon when dividing (e.g., mask.sum() + 1e-8)
   - Use clamping to prevent extreme values
   - Handle NaN and Inf values properly
   - Implement proper error propagation

2. **Implementation Details**:
   - Kabsch alignment must preserve gradients
   - lDDT proxy should be meaningful and bounded [0,1]
   - Angle loss must handle NaN values in true angles
   - Loss reduction should correctly account for masks

3. **Testing Requirements**:
   - Test with various batch sizes and sequence lengths
   - Verify mask handling (all valid, all masked, partially masked)
   - Check numerical stability with extreme inputs
   - Ensure loss is always non-negative and finite

## Dependencies and Interfaces

### Dependencies on Other Instances

- **From Instance 01 (Data Pipeline)**:
  - Batch dictionary format with keys: 'sequence_int', 'dihedral_features', 'pairing_probs', 'positional_entropy', 'accessibility', 'coupling_matrix', 'mask', 'coordinates'
  - Tensor shapes and dtypes for all features
  - Mask convention (True = valid, False = padding)
  - Collation strategy for variable-length sequences

- **From Instance 02 (Model Components)**:
  - `SequenceEmbedding` implementation
  - `PositionalEncoding` implementation
  - `RelativePositionalEncoding` implementation
  - `TransformerBlock` implementation (with forward method signature)
  - `IPAModule` implementation (placeholder for V1)

### Interfaces Provided to Others

- **To Instance 04 (Testing)**:
  - `RNAFoldingModel` full interface documentation
  - Loss function interface and expected usage
  - Example forward pass with realistic dimensions
  - Memory usage patterns and optimization strategies

- **To Training Scripts** (future):
  - Complete model with accessible loss computation
  - Device transfer pattern for model and inputs
  - Gradient flow verification approach
  - Expected output format and interpretation

### Critical Interface Points

1. **Model Input Interface**:
   ```python
   # Batch dictionary from DataLoader
   batch = {
       'sequence_int': torch.LongTensor,         # shape: (B, L)
       'dihedral_features': torch.FloatTensor,   # shape: (B, L, 4)
       'pairing_probs': torch.FloatTensor,       # shape: (B, L, L)
       'positional_entropy': torch.FloatTensor,  # shape: (B, L)
       'accessibility': torch.FloatTensor,       # shape: (B, L)
       'coupling_matrix': torch.FloatTensor,     # shape: (B, L, L)
       'coordinates': torch.FloatTensor,         # shape: (B, L, 3)
       'mask': torch.BoolTensor,                 # shape: (B, L)
       'target_ids': List[str],                  # not tensor
       'lengths': torch.LongTensor               # shape: (B,)
   }
   ```

2. **Model Output Interface**:
   ```python
   # Model output dictionary
   outputs = {
       'pred_coords': torch.FloatTensor,      # shape: (B, L, 3)
       'pred_confidence': torch.FloatTensor,  # shape: (B, L)
       'pred_angles': torch.FloatTensor       # shape: (B, L, 4)
   }
   ```

3. **Loss Function Interfaces**:
   ```python
   # Coordinate loss
   fape_loss = compute_fape_loss(
       pred_coords: torch.FloatTensor,  # shape: (B, L, 3)
       true_coords: torch.FloatTensor,  # shape: (B, L, 3)
       mask: torch.BoolTensor           # shape: (B, L)
   ) -> torch.FloatTensor               # shape: scalar

   # Confidence loss
   conf_loss = compute_confidence_loss(
       pred_confidence: torch.FloatTensor,  # shape: (B, L)
       pred_coords: torch.FloatTensor,      # shape: (B, L, 3)
       true_coords: torch.FloatTensor,      # shape: (B, L, 3)
       mask: torch.BoolTensor               # shape: (B, L)
   ) -> torch.FloatTensor                   # shape: scalar

   # Angle loss
   angle_loss = compute_angle_loss(
       pred_angles: torch.FloatTensor,  # shape: (B, L, 4)
       true_angles: torch.FloatTensor,  # shape: (B, L, 4)
       mask: torch.BoolTensor           # shape: (B, L)
   ) -> torch.FloatTensor               # shape: scalar
   ```

## Success Criteria

Your implementation is successful when:

1. **Loss Functions**:
   - All loss functions are implemented and pass unit tests
   - Numerical stability is maintained across a range of inputs
   - Masking works correctly to ignore padding positions
   - Loss values are reasonable and match expected behavior

2. **RNA Folding Model**:
   - Model initializes correctly from configuration
   - All components are properly assembled and connected
   - Forward pass completes successfully with realistic inputs
   - Output shapes match expected format
   - Gradient flow works through the entire model

3. **Integration**:
   - End-to-end data flow (data→model→loss) works without errors
   - Memory usage is reasonable on target hardware
   - Model passes basic verification tests
   - Forward and backward passes complete with consistent results

4. **Documentation**:
   - Complete interface documentation for model and losses
   - Implementation journal updated with progress and decisions
   - All tests are well-documented and cover edge cases
   - Clear usage examples for model and loss functions

### Final Verification Checklist

- [ ] Loss function implementation passes all unit tests
- [ ] Model initialization from config works correctly
- [ ] Forward pass with batch data produces expected output shapes
- [ ] Backward pass with computed loss flows gradients through model
- [ ] Mask handling verified through all components
- [ ] Variable sequence length handling tested and working
- [ ] Memory usage profiled and within target limits
- [ ] Interface contracts formalized and provided to Instance 04

Upon meeting these criteria, your implementation will provide the critical integration layer that brings together the data pipeline and model components into a complete, trainable system ready for the testing phase.
