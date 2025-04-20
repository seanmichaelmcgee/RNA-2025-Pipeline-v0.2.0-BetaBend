# Integration Claude Code Instructions

## Instance Purpose

You are the Integration instance responsible for assembling the complete RNA 3D folding model architecture by combining individual components into a cohesive pipeline. Your primary focus is implementing the main `RNAFoldingModel` class that integrates embeddings, transformer blocks, and prediction heads, along with the loss functions needed for optimization. You serve as the critical bridge between component development and end-to-end functionality, ensuring proper data flow, correct assembly, and appropriate loss computation for the training process.

As the central integration point in the multi-instance architecture, you have unique responsibilities for component handoffs - both receiving components from the Data Pipeline (01) and Model Components (02) instances, and providing integrated components to the Testing (04) instance. Your role in maintaining thorough handoff documentation and verification is essential to the project's success.

## Kickoff Reference
This document is located at: `docs/claude/03_code-instances/03_integration_kickoff.md`

## Claude.md Configuration
This instance should maintain its own `CLAUDE.md` file located at `docs/claude/03_code-instances/instance_03_integration/CLAUDE.md`. This file should contain:
- Integration patterns for connecting model components
- Loss function implementation techniques and numerical stability approaches
- Configuration management for the full model assembly
- Testing strategies for end-to-end verification
- Memory optimization techniques for training loops
- Gradient flow and backpropagation patterns
- Component handoff procedures and verification checklists
- Interface verification methods and templates
- Component compatibility assessment techniques

Update this file throughout development to document integration-specific implementation patterns and commands that should be readily available to Claude Code when working with the assembled model and loss functions.

## Required Documentation Structure

Before beginning implementation, establish these organizational documents:

### 1. Implementation Journal
- **Location**: `docs/claude/03_code-instances/instance_03_integration/implementation_journal.md`
- **Purpose**: Chronological record of all implementation sessions, decisions, and issues
- **Format**: Follow template at `docs/claude/03_code-instances/shared/04_implementation_jorunal_template.md`
- **Content**:
  - Session records with timestamps
  - Component completion tracking
  - Deviations from specifications
  - Challenges and resolutions
  - Questions for other instances
  - Next steps for upcoming sessions
  - **Handoff records**: Document all component handoffs received and provided

### 2. Completed Components List
- **Location**: `docs/claude/03_code-instances/instance_03_integration/completed_components.md`
- **Purpose**: Track progress of individual components with current status
- **Format**:
  ```markdown
  # Completed Components Tracker
  
  | Component | Status | Test Coverage | Interface Doc | Handoff Status | Last Updated |
  |-----------|--------|---------------|--------------|----------------|--------------|
  | [component_name] | [Not Started/In Progress/Completed] | [0-100%] | [Yes/No] | [Received/Verified/Provided] | YYYY-MM-DD |
  ```

### 3. Component Handoff Documentation
- **Location**: `docs/claude/03_code-instances/instance_03_integration/handoffs/`
- **Purpose**: Maintain formal records of component handoffs (both received and provided)
- **Structure**:
  ```
  handoffs/
  ├── received/
  │   ├── data_pipeline/
  │   │   ├── RNADataset_handoff.md
  │   │   └── collate_fn_handoff.md
  │   └── model_components/
  │       ├── embedding_handoff.md
  │       ├── transformer_block_handoff.md
  │       └── ipa_module_handoff.md
  └── provided/
      └── testing/
          ├── rna_folding_model_handoff.md
          └── losses_handoff.md
  ```
- **Format**: Follow template at `docs/claude/03_code-instances/shared/component_handoff_template.md`

## Handoff Protocol and Documentation

As the central integration instance, your handoff responsibilities are bidirectional and critical to project success. Adhere strictly to the formal handoff protocol documented in `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md`.

### Receiving Component Handoffs

1. **Component Reception**
   - Monitor for handoff notifications from Instances 01 and 02
   - Formally acknowledge receipt within 24 hours using the acknowledgment template
   - Document receipt in implementation journal
   - Create component-specific handoff record in `handoffs/received/[instance]/[component]_handoff.md`

2. **Verification Process**
   - Review interface documentation thoroughly
   - Run provided verification tests to confirm functionality
   - Verify tensor shapes, types, and device handling
   - Test mask propagation behavior
   - Check error handling for edge cases
   - Document verification results in handoff record

3. **Integration Assessment**
   - Evaluate compatibility between components from different instances
   - Identify potential interface mismatches or integration challenges
   - Document findings in handoff record and implementation journal
   - Provide feedback to provider instance within 48 hours

4. **Issue Resolution**
   - For minor issues: Document using the issue report template and propose solutions
   - For major issues: Follow formal dispute documentation process
   - Work collaboratively with provider instance to resolve issues
   - Document resolution in handoff record and implementation journal

5. **Final Acceptance**
   - After successful verification, formally accept the component
   - Update component status in completed components tracker
   - Notify provider instance of successful integration
   - Proceed with implementation that depends on the component

### Providing Component Handoffs

1. **Handoff Preparation**
   - Complete implementation with comprehensive tests (≥90% coverage)
   - Verify component functionality in isolation and integration
   - Create detailed interface documentation
   - Prepare working examples demonstrating usage

2. **Formal Handoff**
   - Create handoff notification using the template
   - Include links to all relevant documentation
   - Create component-specific handoff record in `handoffs/provided/testing/[component]_handoff.md`
   - Document handoff in implementation journal

3. **Support and Clarification**
   - Respond to questions from instance 04 within 24 hours
   - Provide additional examples or clarification as needed
   - Address any identified issues promptly
   - Assist with integration testing as required

4. **Handoff Completion**
   - Document final acceptance from instance 04
   - Update component status in completed components tracker
   - Keep record of any integration issues for future reference
   - Maintain handoff documentation for project records

### Handoff Documentation Requirements

Each handoff document must include:
- Component identification and version
- Implementation status and test coverage
- Public API details with complete function signatures and docstrings
- Data structures and tensor specifications
- Integration points and expected behavior
- Environment and dependencies
- Testing requirements and verification methods
- Implementation details including algorithms and performance considerations
- Extension and maintenance guidance
- Common debugging scenarios
- Decision log for key implementation choices

## Core Responsibilities

- **FIRST TASK: Update and Maintain Documentation**:
  - Create organizational documents including handoff documentation structure
  - Ensure all documentation reflects current understanding of interfaces
  - Keep implementation journals current with progress, decisions, and handoff records
  - Maintain interface contracts as implementations evolve

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

- **Component Handoff Management**:
  - Receive and verify components from Instances 01 and 02
  - Document all handoffs thoroughly following the formal protocol
  - Provide well-documented components to Instance 04
  - Assist with integration issues and interface mismatches

## Implementation Order

1. **Documentation and Organization Setup**:
   - Create implementation journal with handoff tracking sections
   - Set up component status tracker with handoff columns
   - Establish handoff documentation structure
   - Create initial interface contracts for your components
   - Document handoff verification criteria for incoming components

2. **Receive and Verify Components from Instances 01 and 02**:
   - Review interface documentation from Instance 01 (Data Pipeline)
   - Verify and document data loading interfaces (RNADataset, collate_fn)
   - Review interface documentation from Instance 02 (Model Components)
   - Verify and document model component interfaces (embeddings, transformer, IPA)
   - Document all verifications following handoff protocol
   - Report any issues or integration challenges

3. **Loss Functions Implementation** (`src/losses.py`):
   - Implement `compute_fape_loss` function (simplified Kabsch-based proxy)
   - Implement `compute_confidence_loss` function (lDDT proxy)
   - Implement `compute_angle_loss` function (sin/cos comparison)
   - Implement `compute_combined_loss` helper function
   - Write tests for all loss functions (`tests/test_losses.py`)
   - Document interfaces for future handoff to Instance 04

4. **RNAFoldingModel Implementation** (`src/models/rna_folding_model.py`):
   - Define model structure and initialize components from config
   - Integrate verified components from Instances 01 and 02
   - Calculate input dimensions and implement projection layers
   - Stack transformer blocks from Instance 02
   - Implement confidence and angle prediction heads
   - Implement complete forward pass with proper tensor flow
   - Ensure proper mask propagation through components
   - Write tests for the model (`tests/test_model.py`)
   - Document interface for future handoff to Instance 04

5. **Integration Verification**:
   - Implement basic integration tests showing end-to-end flow
   - Verify compatibility between all components
   - Document integration challenges and solutions
   - Create memory profiling and gradient flow checks
   - Prepare for handoff to Instance 04

6. **Handoff to Instance 04 (Testing)**:
   - Prepare comprehensive handoff documentation for all components
   - Include examples of end-to-end usage
   - Document known limitations and edge cases
   - Provide verification tests and expected results
   - Follow formal handoff protocol for all provided components

## Reference Documents

### Architecture and Requirements
- `docs/3_Architecture_Specification.md` - See "Backbone Model" and "Loss Functions" sections
- `docs/4_Product_Requirements_V1.md` - Requirements MA-01, MA-02, MA-04, MA-06, MA-08-11 and LF-01 to LF-04
- `docs/6_Tactical_Plan_V1.md` - Sections III.4 and IV for implementation guidance

### Implementation Guides
- `docs/claude/05_workflows/60_model_integration.md` - Complete integration workflow
- `docs/claude/02_components/50_losses/51_losses_guide.md` - Detailed loss implementation guide
- `docs/claude/02_components/50_losses/52_losses_examples.md` - Example implementations for losses
- `docs/claude/02_components/50_losses/53_losses_tests.md` - Testing strategies for loss functions

### Component Interface Contracts
- `docs/claude/03_code-instances/shared/interface_specifications/RNADataset_v1.0.md` - Data batch format
- `docs/claude/03_code-instances/shared/interface_specifications/SequenceEmbedding_v1.0.md` - Embedding interface
- `docs/claude/03_code-instances/shared/interface_specifications/TransformerBlock_v1.0.md` - Transformer interface
- `docs/claude/03_code-instances/shared/interface_specifications/IPAModule_v1.0.md` - IPA placeholder interface

### Handoff Documentation and Protocols
- `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md` - Comprehensive handoff protocol
- `docs/claude/03_code-instances/shared/component_handoff_template.md` - Handoff documentation template
- `docs/claude/03_code-instances/shared/05_interface_contract_template.md` - Interface documentation template

### Code Guidelines
- `docs/claude/01_implementation_principles.md` - Core implementation principles
- `docs/claude/7_AI_Agent_Rules.md` - Focus on rules 2.4 (Modularity) and 7.2 (Path Parameterization)

## Communication Guidelines

### When to Communicate with Other Instances

- **Instance 01 (Data Pipeline)**:
  - Formally acknowledge handoff of components using the acknowledgment template
  - Request clarification on batch format, tensor shapes, or mask conventions
  - Report verification results for received components
  - Document any interface issues using the issue report template
  - Verify handling of variable-length sequences and padding
  - Confirm expected data types and device handling

- **Instance 02 (Model Components)**:
  - Formally acknowledge handoff of components using the acknowledgment template
  - Request clarification on component interfaces and parameter requirements
  - Report verification results for received components
  - Document any integration challenges between model components
  - Verify transformer block interface and expected input/output formats
  - Confirm embedding dimensions and processing requirements
  - Clarify IPA module placeholder functionality

- **Instance 04 (Testing)**:
  - Provide formal handoff notification for integrated components
  - Include comprehensive interface documentation and examples
  - Respond promptly to questions and clarification requests
  - Support testing efforts with additional information as needed
  - Follow up on reported issues with fixes and verifications
  - Coordinate on integration testing strategies

### Handoff Communication Protocol

- **Receiving Components**:
  - Use the handoff acknowledgment template within 24 hours of notification
  - Document verification timeline and initial assessment
  - Ask specific questions about implementation details if needed
  - Provide verification results using the specified template
  - Report any issues using the issue report template
  - Confirm final acceptance once integration is successful

- **Providing Components**:
  - Use the handoff notification template when components are ready
  - Include links to all interface documentation
  - Highlight special considerations or limitations
  - Document expected timelines for verification
  - Be available for questions and clarifications
  - Support integration with additional guidance as needed

### Regular Communication Cadence

- Document all significant communication in the implementation journal
- Provide weekly status updates on component integration progress
- Alert other instances immediately about blocking issues
- Share integration insights that might benefit other instances
- Follow up on pending questions or issues within 24 hours
- Maintain a record of all handoff communications in the handoff documentation

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

### Handoff Documentation Standards

1. **Interface Contracts**:
   - Document all public interfaces with complete type information
   - Explicitly specify tensor shapes, types, and device expectations
   - Detail all error conditions and handling
   - Include examples of typical usage

2. **Component Documentation**:
   - Document component architecture and design decisions
   - Specify critical algorithms and implementation details
   - Include performance considerations and limitations
   - Provide usage examples and integration patterns

3. **Verification Documentation**:
   - Document all verification steps performed
   - Record test results and any issues found
   - Detail resolution of any integration challenges
   - Include test cases for future verification

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
  - Complete handoff documentation for integrated components

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

1. **Documentation and Organization**:
   - Complete documentation structure established and maintained
   - Implementation journal kept up to date with all activities
   - Component status tracker accurately reflects progress
   - Handoff documentation is comprehensive and follows templates

2. **Component Handoff Management**:
   - All received components properly verified and documented
   - Integration issues identified and resolved promptly
   - Provided components documented and handed off successfully
   - Handoff communication follows formal protocol

3. **Loss Functions**:
   - All loss functions implemented and pass unit tests
   - Numerical stability maintained across a range of inputs
   - Masking works correctly to ignore padding positions
   - Loss values are reasonable and match expected behavior

4. **RNA Folding Model**:
   - Model initializes correctly from configuration
   - All components are properly assembled and connected
   - Forward pass completes successfully with realistic inputs
   - Output shapes match expected format
   - Gradient flow works through the entire model

5. **Integration**:
   - End-to-end data flow (data→model→loss) works without errors
   - Memory usage is reasonable on target hardware
   - Model passes basic verification tests
   - Forward and backward passes complete with consistent results

6. **Handoff Completion**:
   - All components received, verified, and accepted
   - All integrated components successfully handed off to Instance 04
   - Handoff documentation comprehensive and follows standards
   - Support provided for integration challenges and questions

### Final Verification Checklist

- [ ] Documentation structure complete and organized
- [ ] Component handoff tracking established and maintained
- [ ] All received components verified and accepted
- [ ] Loss function implementation passes all unit tests
- [ ] Model initialization from config works correctly
- [ ] Forward pass with batch data produces expected output shapes
- [ ] Backward pass with computed loss flows gradients through model
- [ ] Mask handling verified through all components
- [ ] Variable sequence length handling tested and working
- [ ] Memory usage profiled and within target limits
- [ ] Interface contracts formalized for all components
- [ ] All components successfully handed off to Instance 04
- [ ] Handoff documentation complete and follows standards
- [ ] Integration issues resolved and documented
- [ ] End-to-end pipeline functioning correctly

Upon meeting these criteria, your implementation will provide the critical integration layer that brings together the data pipeline and model components into a complete, trainable system ready for comprehensive testing by Instance 04.
