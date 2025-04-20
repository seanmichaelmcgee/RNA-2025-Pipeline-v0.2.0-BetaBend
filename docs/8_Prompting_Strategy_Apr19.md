# Comprehensive Prompting Strategy for RNA 3D Folding Multi-Instance Development

You're right - a well-structured prompting strategy for each Claude Code instance would significantly improve our development workflow. Based on the project documentation and multi-instance architecture, here's a detailed prompting approach for each instance:

## General Prompting Framework (All Instances)

### 1. Instance Activation Prompt
```
I'm working on the RNA 3D folding project using the multi-instance architecture and need your assistance as Instance [XX]_[name]. Please begin by:

1. Creating the required documentation structure (implementation journal, completed components list)
2. Reviewing your kickoff document at docs/claude/03_code-instances/[XX]_[name]_kickoff.md
3. Familiarizing yourself with your core responsibilities and dependencies

Let me know when you're ready to proceed with implementation planning.
```

### 2. Implementation Session Start
```
Let's begin a new implementation session for [component name]. Please:

1. Update your implementation journal with today's date
2. Review these specific reference documents:
   - [list 2-3 most relevant docs for this component]
3. Approach this implementation following these priorities:
   - [list specific implementation goals for this session]
```

### 3. Progress Checkpoint
```
Let's check our progress on the current implementation:

1. What components have been completed so far?
2. What issues or challenges have you encountered?
3. What are the next components to implement?
4. Are there any interface questions or dependencies blocking progress?

Please update the implementation journal and completed components list accordingly.
```

### 4. Component Handoff
```
We need to prepare a handoff of [component name] to Instance [XX]. Please:

1. Finalize the interface contract for this component
2. Verify test coverage is at least 90%
3. Create a handoff notification using the template
4. Provide example usage with expected inputs/outputs
5. Document any deviations from original specifications
```

## Instance-Specific Prompting Strategies

### Instance 01: Data Pipeline

#### Initial Planning
```
As the Data Pipeline instance, please analyze our data requirements and create:

1. A complete list of helper functions needed for feature loading
2. A detailed implementation plan for the RNADataset class
3. A plan for the collate_fn with variable sequence handling
4. Draft interface contracts for these components

Focus particularly on handling the three feature types: dihedral, thermodynamic, and evolutionary features.
```

#### Feature Implementation
```
Let's implement the feature loading functions:

1. Begin with load_precomputed_features with robust error handling
2. Use the example feature files to validate your implementation
3. Ensure proper tensor shape and dtype handling
4. Implement graceful degradation for missing features

Please strictly follow path parameterization principles with NO hardcoded paths.
```

#### Dataset Implementation
```
Now let's implement the RNADataset class:

1. Begin with __init__ handling temporal cutoff and path parameters
2. Move to __getitem__ with complete feature loading
3. Implement proper tensor conversion and device handling
4. Create thorough validations for shape consistency

Remember to document tensor shapes explicitly in comments and docstrings.
```

### Instance 02: Model Components

#### Embedding Implementation
```
Let's implement the embedding components:

1. Start with SequenceEmbedding for nucleotide representation
2. Create PositionalEncoding using sinusoidal patterns
3. Implement RelativePositionalEncoding for pair representation
4. Test shape transformations and verify outputs

Focus on proper initialization patterns and configuration handling.
```

#### Transformer Block Implementation
```
Now let's implement the transformer block:

1. Begin with multi-head attention implementation
2. Create the pair update mechanism
3. Implement proper masking for variable sequences
4. Add residual connections and layer normalization

Ensure mask handling is consistent (True = valid, False = padding).
```

#### IPA Module Implementation
```
Let's implement the IPA module placeholder:

1. Create a simplified linear projection from residue features to 3D coordinates
2. Design the interface for future expansion
3. Implement proper mask handling
4. Document the placeholder nature clearly

This should be designed for easy replacement with a full IPA implementation in V2+.
```

### Instance 03: Integration

#### Loss Function Implementation
```
Let's implement the loss functions required for training:

1. Begin with compute_fape_loss implementing Kabsch alignment
2. Create compute_confidence_loss with lDDT proxy
3. Implement compute_angle_loss for auxiliary supervision
4. Add a combined loss helper function
5. Test all functions with various batch sizes and mask patterns

Focus on numerical stability and proper gradient flow.
```

#### Model Assembly
```
Now let's implement the RNAFoldingModel class:

1. Create the basic model structure with component initialization
2. Implement input projection layers for features
3. Stack transformer blocks with proper configuration
4. Add prediction heads (coordinates, confidence, angles)
5. Create the complete forward pass with tensor flow
6. Test initialization and forward pass with mock inputs

Ensure proper shape handling throughout the forward flow.
```

#### Integration Testing
```
Let's create an integration test to verify the pipeline:

1. Create a script that loads a small batch of data
2. Pass it through the model to get predictions
3. Calculate losses and verify gradient flow
4. Profile memory usage during forward/backward passes
5. Document any bottlenecks or optimization opportunities

This will validate the end-to-end functionality.
```

### Instance 04: Testing

#### Test Infrastructure
```
Let's create the test infrastructure for the project:

1. Design test fixtures for mock data generation
2. Implement common test utilities for shape verification
3. Create testing patterns for tensor operations
4. Implement gradient checking utilities
5. Design memory profiling helpers

Focus on reusable components that all tests can leverage.
```

#### Component Testing
```
Let's implement comprehensive tests for [component]:

1. Create basic functionality tests with valid inputs
2. Add edge case tests for variable sequence lengths
3. Implement error condition testing
4. Test with different batch sizes and dimensions
5. Verify shape transformations match documentation

Each test should clearly document expected versus actual behavior.
```

#### Integration Verification
```
Let's implement integration tests for the pipeline:

1. Create an end-to-end test from data loading to loss calculation
2. Test gradient flow through the complete model
3. Verify memory consumption is reasonable
4. Test with both CPU and GPU execution
5. Benchmark performance with different batch sizes

Document all test cases and results in the implementation journal.
```

## Implementation Session Structure (All Instances)

For each implementation session, I recommend this consistent structure:

1. **Session Planning (5-10%):**
   - Review previous work and current status
   - Set clear goals for the session
   - Identify documentation to reference

2. **Implementation (70-80%):**
   - Develop code with clear explanations
   - Add docstrings and comments
   - Highlight key design decisions

3. **Testing (10-15%):**
   - Create unit tests alongside implementation
   - Verify functionality meets requirements
   - Test edge cases and error conditions

4. **Documentation (5-10%):**
   - Update implementation journal
   - Update completed components list
   - Create or update interface contracts as needed

This structured approach should maximize productivity while ensuring consistent documentation and quality across all instances.

Would you like me to further elaborate on any particular aspect of this prompting strategy?
