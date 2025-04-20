# RNA 3D Structure Prediction: Master Implementation Guide

## Project Overview

You are assisting with implementing a PyTorch-based machine learning pipeline for predicting the 3D structure of RNA molecules from their sequences. This project targets the Stanford RNA 3D Folding Kaggle competition and uses a multi-instance development architecture with specialized Claude Code instances for different components.

### Key Objectives

1. Create a modular, well-tested implementation following the architecture in `3_Architecture_Specification.md`
2. Ensure compatibility with both local development and Kaggle submission environments
3. Maintain strict adherence to core principles like path parameterization and modularity
4. Generate a working V1 system with scaled-down parameters that can be iterated upon
5. Establish clear interfaces between components with formal contracts
6. Maintain knowledge continuity through implementation journals and handoffs

## Multi-Instance Architecture

This project uses a specialized multi-instance development approach with four distinct Claude Code instances, each responsible for specific components:

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│  01_data_pipeline ├─────►│  03_integration   │
│  (Data Processing)│      │  (Model Assembly) │
│                   │      │                   │
└───────────────────┘      └─────────┬─────────┘
                                     │
┌───────────────────┐                │
│                   │                │
│ 02_model_components ◄───────┐      │
│  (Neural Network)  │        │      │
│                   │        │      │
└─────────┬─────────┘        │      │
          │                  │      │
          ▼                  │      ▼
┌───────────────────┐        │      │
│                   │        │      │
│    04_testing     ◄────────┴──────┘
│  (Verification)   │
│                   │
└───────────────────┘
```

### Instance Roles and Responsibilities

1. **Data Pipeline (01_data_pipeline)**
   - RNA dataset class and preprocessing
   - Feature loading and validation
   - Batch collation and mask generation
   - Variable sequence length handling

2. **Model Components (02_model_components)**
   - Embedding layers (sequence, positional, relative)
   - Transformer blocks with attention
   - IPA module (placeholder for V1)
   - Individual component testing

3. **Integration (03_integration)**
   - Main model architecture assembly
   - Loss function implementation
   - End-to-end data flow coordination
   - Configuration management

4. **Testing (04_testing)**
   - Comprehensive test suite development
   - Component verification and validation
   - Performance benchmarking
   - Integration testing across components

### Documentation Structure

The documentation is organized to support this multi-instance approach:

```
docs/claude/
├── 00-master_guide.md                  # This document - overall guidance
├── 01_implementation_principles.md     # Core patterns for all instances
├── 02_components/                      # Component-specific guides
│   ├── 10_data_loading/
│   ├── 20_embeddings/
│   └── ...
├── 03_code-instances/                  # Multi-instance architecture
│   ├── README.md                       # Overview of multi-instance approach
│   ├── 01_data_pipeline.md             # Data instance instructions
│   ├── 02_model_components.md          # Model instance instructions 
│   ├── 03_integration.md               # Integration instance instructions
│   ├── 04_testing.md                   # Testing instance instructions
│   ├── shared/                         # Shared protocols and templates
│   │   ├── 06_component_handoff_protocol.md
│   │   ├── 05_interface_contract_template.md
│   │   ├── 04_implementation_jorunal_template.md
│   │   └── 07_component_status_tracker.md
│   ├── instance_01_data/               # Data instance workspace
│   ├── instance_02_model/              # Model instance workspace
│   ├── instance_03_integration/        # Integration instance workspace
│   └── instance_04_testing/            # Testing instance workspace
├── 04_reference/                       # Reference documentation
└── 05_workflows/                       # Workflow documentation
```

## Implementation Roadmap

Follow this sequence to implement the full system across instances:

### Phase 1: Data Pipeline (Instance 01)
1. **Data Loading** (`src/data_loading.py`)
   - Implement `RNADataset` class and helper functions
   - Create `collate_fn` for variable-length sequences
   - Focus on proper feature loading from .npz files
   - Prepare handoff documentation for Integration instance

### Phase 2: Model Components (Instance 02)
2. **Embedding Layers** (`src/models/embeddings.py`)
   - Implement sequence, positional, and relative positional embeddings
   - Create input projection layers for features
   - Develop unit tests for shape validation
   - Prepare handoff documentation for Integration instance

3. **Transformer Block** (`src/models/transformer_block.py`)
   - Implement standard multi-head attention
   - Create simplified pair update mechanism
   - Ensure proper residual connections and layer normalization
   - Prepare handoff documentation for Integration instance

4. **IPA Module Placeholder** (`src/models/ipa_module.py`)
   - Create simple linear layer placeholder for structure generation
   - Maintain correct tensor shapes for integration
   - Prepare handoff documentation for Integration instance

### Phase 3: Integration (Instance 03)
5. **Loss Functions** (`src/losses.py`)
   - Implement simplified FAPE loss proxy
   - Create confidence and angle prediction losses
   - Handle proper masking for variable-length sequences
   - Prepare handoff documentation for Testing instance

6. **Model Integration** (`src/models/rna_folding_model.py`)
   - Combine all components into coherent model
   - Implement forward pass with correct data flow
   - Manage tensor shapes throughout pipeline
   - Prepare handoff documentation for Testing instance

### Phase 4: Testing (Instance 04)
7. **Test Suite Development**
   - Create comprehensive unit tests for all components
   - Implement integration tests for end-to-end verification
   - Develop performance benchmarks
   - Verify Kaggle compatibility

8. **Pipeline Validation**
   - Validate end-to-end functionality
   - Benchmark memory and performance
   - Verify gradient flow and backpropagation
   - Document validation results

## Knowledge Management and Handoffs

### Implementation Journals

Each instance maintains an implementation journal following `docs/claude/03_code-instances/shared/04_implementation_jorunal_template.md` to ensure knowledge continuity:

```markdown
# [INSTANCE_NAME] Implementation Journal

## Component Status Tracker
| Component | Status | Tests | Interface Doc | Dependent Instances | Last Updated |
|-----------|--------|-------|---------------|---------------------|--------------|
| [component_name] | ✅ Complete | ✅ | ✅ | [instances] | YYYY-MM-DD |

## Implementation Sessions
### Implementation Session: YYYY-MM-DD
#### Components Completed:
- [x] [component_name] - [implementation details]
#### Deviations from Plan:
- [description of deviation and rationale]
#### Issues/Questions:
- [issue description] - [potential solutions]
#### Next Steps:
- [upcoming implementation tasks]
```

### Component Status Tracking

Overall implementation progress is tracked in `docs/claude/03_code-instances/shared/07_component_status_tracker.md`, providing:

- Implementation status of all components across instances
- Critical path components and dependencies
- Blocked components and resolution plans
- Test coverage metrics
- Interface documentation status
- Handoff status

### Handoff Protocol

Component handoffs between instances follow the formal protocol in `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md`:

1. **Provider Responsibilities**:
   - Complete implementation with ≥90% test coverage
   - Create interface documentation following the contract template
   - Document limitations, edge cases, and design decisions
   - Prepare handoff documentation with verification steps

2. **Consumer Responsibilities**:
   - Verify interface matches documentation
   - Run verification tests on the component
   - Provide structured feedback within 48 hours
   - Formally acknowledge successful handoff

3. **Issue Resolution**:
   - Document specific issues using the issue report template
   - Reference authoritative specifications
   - Assign responsibility for resolution
   - Track resolution progress

4. **Handoff Documentation**:
   - Component identification and version
   - Implementation status with test coverage
   - Public API with complete function signatures
   - Data structures and tensor specifications
   - Integration points and expected behavior
   - Testing requirements and verification steps

### Interface Contracts

All component interfaces are documented using the template in `docs/claude/03_code-instances/shared/05_interface_contract_template.md`:

```markdown
# Component Interface Contract

## Component Identification
- **Component Name**: [e.g., TransformerBlock]
- **Version**: [e.g., v1.0.0]
- **Responsible Instance**: [e.g., 02_model_components]

## Input Interface
| Parameter | Type | Shape | Device | Description | Required |
|-----------|------|-------|--------|-------------|----------|
| `param_name` | `torch.Tensor` | `(batch_size, seq_len, hidden_dim)` | Same as input | Description of the parameter | Yes/No |

## Output Interface
| Return Value | Type | Shape | Device | Description |
|--------------|------|-------|--------|-------------|
| `output_name` | `torch.Tensor` | `(batch_size, seq_len, hidden_dim)` | Same as input | Description of the output |

## Error Conditions
| Error Type | Trigger Condition | Error Message | Recovery Option |
|------------|-------------------|---------------|-----------------|
```

## Core Implementation Principles

### Critical: Path Parameterization
- **NEVER use hardcoded paths** in `src/` modules
- All file/directory paths must be passed as arguments from the orchestration layer
- Always use `os.path.join()` for constructing paths
- This enables both local development and Kaggle compatibility

```python
# CORRECT - paths as arguments
def load_features(target_id: str, features_dir: str):
    feature_path = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz")
    # ...

# INCORRECT - hardcoded paths
def load_features(target_id: str):
    feature_path = f"data/features/mi_features/{target_id}_mi_features.npz"  # NEVER DO THIS
    # ...
```

### Modularity and Code Organization
- Keep core logic in `src/` modules, orchestration in `scripts/`
- Design components with clear interfaces
- Follow PyTorch conventions (nn.Module, device handling)
- Enable independent testing of components
- Prepare components for handoff with proper documentation

### Test-Driven Development
- Write tests alongside implementation
- Verify functionality before integration
- Use appropriate fixtures and mocks
- Develop verification tests for handoffs
- Maintain ≥90% test coverage on all components

### Memory Efficiency
- Start with scaled-down parameters
- Use PyTorch best practices (e.g., `torch.no_grad()` for evaluation)
- Monitor GPU memory usage
- Document memory patterns for component integration

## Common Implementation Pitfalls

### Data Loading Issues
- **Missing Feature Handling**: Some sequences may lack certain feature files (especially evolutionary data). Provide default zero tensors of correct shape.
- **Shape Inconsistencies**: Verify sequence lengths match across features, return clear errors on mismatch.
- **Padding & Masking**: Ensure proper padding of variable-length sequences and correct attention masks.
- **Feature Filtering**: Handle partial datasets where only some sequences have features.

### Model Implementation Challenges
- **Device Management**: Remember to move all tensors to the target device (`tensor.to(device)`).
- **Dimension Tracking**: Keep track of tensor shapes through transformations. Document expected shapes in comments.
- **Gradient Accumulation**: Use separate graphs for different loss components, detach when needed.
- **Mask Propagation**: Ensure mask is properly handled throughout the network.

### Integration Pain Points
- **End-to-End Flow**: The most common issue is interfaces between components. Document expected input/output formats clearly.
- **Memory Leaks**: Monitor memory utilization on GPU, use `torch.cuda.empty_cache()` when debugging.
- **Configuration Management**: Ensure config parameters are used consistently across components.
- **Handoff Mismatches**: Verify interfaces match documentation when integrating components from different instances.

### Handoff Challenges
- **Interface Consistency**: Ensure tensor shapes, types, and device handling match documentation.
- **Verification Gaps**: Create comprehensive verification tests covering normal operation and edge cases.
- **Documentation Incompleteness**: Provide thorough interface contracts with all required sections.
- **Integration Assumptions**: Be explicit about expected behavior and constraints in handoff documentation.

## Instance-Specific Guidance

### Working with Instance 01 (Data Pipeline)
- Focus on correct feature loading and preprocessing
- Be meticulous about path parameterization
- Ensure proper handling of variable-length sequences
- Document tensor shapes and types thoroughly
- Handle missing or incomplete features gracefully

### Working with Instance 02 (Model Components)
- Maintain strict component boundaries
- Provide comprehensive interface documentation
- Focus on tensor shape transformations and device handling
- Implement proper mask propagation
- Design components for integration readiness

### Working with Instance 03 (Integration)
- Verify interface compatibility between components
- Ensure end-to-end data flow works correctly
- Focus on configuration management
- Implement numerically stable loss functions
- Document integration patterns and considerations

### Working with Instance 04 (Testing)
- Create comprehensive verification tests
- Document expected outputs and behaviors
- Benchmark memory and performance
- Validate edge case handling
- Verify Kaggle compatibility

## Documentation Navigation

For detailed implementation guidance, refer to these component-specific guides:

- **Data Loading**: `docs/claude/components/10_data_loading/guide.md`
- **Embeddings**: `docs/claude/components/20_embeddings/guide.md`
- **Transformer Block**: `docs/claude/components/30_transformer_block/guide.md`
- **IPA Module**: `docs/claude/components/40_ipa_module/guide.md`
- **Loss Functions**: `docs/claude/components/50_losses/guide.md`

For instance-specific instructions:

- **Data Pipeline**: `docs/claude/03_code-instances/01_data_pipeline.md`
- **Model Components**: `docs/claude/03_code-instances/02_model_components.md`
- **Integration**: `docs/claude/03_code-instances/03_integration.md`
- **Testing**: `docs/claude/03_code-instances/04_testing.md`

For implementation protocols:

- **Handoff Protocol**: `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md`
- **Interface Contracts**: `docs/claude/03_code-instances/shared/05_interface_contract_template.md`
- **Implementation Journals**: `docs/claude/03_code-instances/shared/04_implementation_jorunal_template.md`
- **Component Tracking**: `docs/claude/03_code-instances/shared/07_component_status_tracker.md`

For workflow guidance:

- **Model Integration**: `docs/claude/workflows/60_model_integration.md`
- **Pipeline Testing**: `docs/claude/workflows/70_pipeline_testing.md`
- **Debugging**: `docs/claude/workflows/80_debugging.md`

For reference information:

- **Feature Formats**: `docs/claude/reference/feature_formats.md`
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md`
- **Configuration**: `docs/claude/reference/configuration.md`

## Next Steps

Begin implementation with the appropriate instance for your current task:

### For Data Pipeline (Instance 01)
1. Review the data pipeline kickoff document: `docs/claude/03_code-instances/01_data_pipeline.md`
2. Study the feature specifications in `docs/2_Feature_Specification.md`
3. Create your implementation journal following the template
4. Start with implementing feature loading functions

### For Model Components (Instance 02)
1. Review the model components kickoff document: `docs/claude/03_code-instances/02_model_components.md`
2. Study the architecture specification in `docs/3_Architecture_Specification.md`
3. Create your implementation journal following the template
4. Start with implementing embedding components

### For Integration (Instance 03)
1. Review the integration kickoff document: `docs/claude/03_code-instances/03_integration.md`
2. Review handoff documentation from Instances 01 and 02
3. Create your implementation journal following the template
4. Begin with implementing loss functions

### For Testing (Instance 04)
1. Review the testing kickoff document: `docs/claude/03_code-instances/04_testing.md`
2. Study the verification process in the handoff protocol
3. Create your implementation journal following the template
4. Start by developing test fixtures and verification utilities

When working within any instance, follow the specifications in the relevant kickoff document, maintain your implementation journal, and prepare comprehensive handoff documentation for components you complete.
