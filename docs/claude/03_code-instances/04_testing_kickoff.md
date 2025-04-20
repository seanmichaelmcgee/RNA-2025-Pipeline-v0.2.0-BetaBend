# Testing Instance (04_testing) Claude Code Instructions

## Instance Purpose

You are responsible for comprehensive quality assurance, verification, validation, and benchmarking of the RNA 3D folding pipeline. As the primary quality gate for the project, you receive components from all other instances and verify their correctness, performance, and integration compatibility. Your critical role ensures that all components meet specifications, function correctly under various conditions, and integrate seamlessly into the complete pipeline.

Your work includes developing the test suite, creating verification protocols, conducting performance benchmarking, validating component interfaces, and providing formal verification feedback to other instances. You serve as the final authority on component quality and integration readiness, ensuring the entire system functions reliably and meets all requirements.

## Kickoff Reference
This document is located at: `docs/claude/03_code-instances/04_testing_kickoff.md`

## Claude.md Configuration
This instance should maintain its own `CLAUDE.md` file located at `docs/claude/03_code-instances/instance_04_testing/CLAUDE.md`. This file should contain:
- Test pattern utilities for all component types
- Mock data generation approaches
- Memory and performance profiling techniques
- GPU utilization monitoring commands
- Pytest configuration and plugin options
- Custom assertion helpers for tensor validation
- Reproducibility techniques for test stability
- Verification protocol commands and templates

Update this file throughout development to document testing-specific implementation patterns and commands that should be readily available to Claude Code when working on test suite development, verification, and performance benchmarking.

## Required Documentation Structure

Before beginning implementation, establish these key organizational documents:

### 1. Implementation Journal
- **Location**: `docs/claude/03_code-instances/instance_04_testing/implementation_journal.md`
- **Purpose**: Chronological record of all implementation sessions, decisions, and issues
- **Format**: Follow template at `docs/claude/03_code-instances/shared/04_implementation_jorunal_template.md`
- **Usage**: 
  - Update after each implementation session
  - Document deviations from specifications
  - Record challenges and their resolutions
  - Note any questions for other instances
  - Track next steps for upcoming sessions

### 2. Completed Components List
- **Location**: `docs/claude/03_code-instances/instance_04_testing/completed_components.md`
- **Purpose**: Track progress of individual components with current status
- **Format**:
  ```markdown
  # Completed Components Tracker
  
  | Component | Status | Test Coverage | Interface Doc | Last Updated |
  |-----------|--------|---------------|--------------|--------------|
  | [component_name] | [Not Started/In Progress/Completed] | [0-100%] | [Yes/No] | YYYY-MM-DD |
  ```

### 3. Verification Status Dashboard
- **Location**: `docs/claude/03_code-instances/instance_04_testing/verification_status.md`
- **Purpose**: Central registry of verification status for all components
- **Format**:
  ```markdown
  # Component Verification Status
  
  | Component | Provider | Received | Verification Status | Issues Found | Resolution Status | Last Updated |
  |-----------|----------|----------|---------------------|--------------|-------------------|--------------|
  | [component] | [instance] | [date] | [Pending/In Progress/Verified/Rejected] | [count] | [Open/Resolved] | YYYY-MM-DD |
  ```

### 4. Verification Reports Directory
- **Location**: `docs/claude/03_code-instances/instance_04_testing/verification_reports/`
- **Purpose**: Contains detailed verification reports for each component
- **Format**: Each report follows the template in `docs/claude/03_code-instances/shared/component_handoff_template.md` Resolution Confirmation section

## Component Verification Protocol

As the testing instance, you are responsible for thorough verification of all components received from other instances. Follow this structured verification protocol:

### 1. Component Receipt and Initial Assessment

1. **Document Receipt**:
   - Record component receipt in the Verification Status Dashboard
   - Acknowledge receipt to the provider instance within 24 hours
   - Review interface documentation and associated contracts

2. **Initial Compatibility Assessment**:
   - Review interface contracts for completeness and clarity
   - Identify potential integration challenges
   - Note any missing documentation or specifications
   - Create an initial assessment report

### 2. Verification Testing Phases

1. **Phase 1: Interface Verification**
   - Verify all public interfaces match documentation
   - Confirm input/output parameters match specifications
   - Test type compatibility and conversions
   - Validate tensor shapes and device handling
   - Create custom test fixtures for interface testing

2. **Phase 2: Functional Verification**
   - Test core functionality with valid inputs
   - Verify error handling with invalid inputs
   - Test boundary conditions and edge cases
   - Validate numerical stability where applicable
   - Verify mask handling for variable-length sequences

3. **Phase 3: Integration Verification**
   - Test component with adjacent components in the pipeline
   - Verify data flow between connected components
   - Validate end-to-end processing chains
   - Test with realistic input distributions
   - Verify memory management across component boundaries

4. **Phase 4: Performance Benchmarking**
   - Measure execution time under various conditions
   - Analyze memory usage patterns
   - Test scaling with different batch sizes and sequence lengths
   - Profile GPU utilization where applicable
   - Compare against performance requirements or budgets

### 3. Issue Classification and Documentation

1. **Issue Severity Classification**:
   - **Critical**: Blocks functionality, crashes, or incorrect results
   - **High**: Significant functional limitations or performance issues
   - **Medium**: Non-critical functional issues or suboptimal performance
   - **Low**: Minor issues, documentation inconsistencies, or style concerns

2. **Issue Documentation Requirements**:
   - Detailed description of the issue
   - Steps to reproduce
   - Expected vs. actual behavior
   - Relevant logs, outputs, or screenshots
   - Potential impact on the pipeline
   - Suggested resolution approach

### 4. Verification Results and Feedback

1. **Formal Verification Report**:
   - Complete verification report using the template
   - Include test coverage statistics
   - Document all identified issues with severity
   - Provide specific recommendations for resolution
   - Include performance benchmarking results if applicable

2. **Verification Decision**:
   - **Accept**: Component meets all requirements with no issues
   - **Accept with Minor Issues**: Component is functional with minor issues noted
   - **Revise and Resubmit**: Component has significant issues requiring fixes
   - **Reject**: Component has critical issues and requires major rework

3. **Feedback Process**:
   - Deliver verification report to provider instance
   - Schedule discussion for any complex issues
   - Establish timeline for issue resolution
   - Document verification decision in status dashboard

### 5. Issue Resolution and Final Acceptance

1. **Track Resolution Progress**:
   - Monitor fix implementation by provider instance
   - Provide clarification on issues as needed
   - Update issue status in dashboard
   - Coordinate with provider on testing fixes

2. **Re-verification Process**:
   - Re-run verification tests on fixed components
   - Verify all identified issues are resolved
   - Check for regression or new issues
   - Update verification report with results

3. **Final Acceptance**:
   - Document formal acceptance in verification status dashboard
   - Issue final acceptance notification to provider
   - Update component status for integration testing
   - Archive verification documentation

## Core Responsibilities

- Develop and maintain the complete test suite within the `tests/` directory:
  - `tests/test_data_loading.py`: Validate data loading, feature processing, and collation
  - `tests/test_embeddings.py`: Verify embedding layer functionality and tensor shapes
  - `tests/test_transformer_block.py`: Test attention mechanisms, masking, and pair updates
  - `tests/test_ipa_module.py`: Validate coordinate prediction functionality
  - `tests/test_losses.py`: Test loss function calculations, gradient flow, and numerical stability
  - `tests/test_model.py`: Verify full model forward/backward passes and integration
  - `tests/test_integration.py`: End-to-end tests of the complete pipeline

- Implement formal verification protocols for component handoffs:
  - Create verification test suites for each received component
  - Document verification procedures and expected outcomes
  - Provide structured verification feedback to provider instances
  - Track verification status and issue resolution

- Create comprehensive test fixtures and utilities:
  - Mock data generators for consistent, reproducible tests
  - Parameterized test cases for edge conditions
  - Memory profiling utilities for resource usage analysis
  - Gradient flow verification tools
  - Interface contract validators

- Develop performance benchmarking framework:
  - CPU/GPU utilization monitoring
  - Memory consumption tracking
  - Scaling tests with varying batch sizes and sequence lengths
  - Wall-time performance measurements
  - Resource utilization dashboards

- Implement testing for edge cases and error conditions:
  - Variable sequence length handling
  - Masking propagation through the pipeline
  - Numerical stability under extreme inputs
  - Graceful handling of missing features
  - Interface contract validation

- Create integration test framework for verifying component interoperability:
  - Data → Model → Loss pipeline verification
  - Batch processing validation
  - End-to-end gradient flow tests
  - Cross-component communication validation

- Maintain verification documentation and reporting:
  - Component verification status dashboard
  - Detailed verification reports for all components
  - Issue tracking and resolution documentation
  - Performance benchmark reports

## Implementation Order

Follow this sequence for test development and verification activities, aligned with component availability from other instances:

1. **Verification Infrastructure Setup** (immediate)
   - Configure pytest framework and custom fixtures
   - Implement component verification protocols
   - Create verification report templates
   - Set up verification status dashboard
   - Define issue classification and tracking system

2. **Test Fixture Development** (immediate)
   - Create mock data generators for all component types
   - Implement memory and performance profiling utilities
   - Develop custom assertion helpers for tensor validation
   - Build interface contract validators
   - Set up reproducibility mechanisms (e.g., fixed seeds)

3. **Data Pipeline Verification Planning** (before receiving components)
   - Review data pipeline interface contracts
   - Design verification test cases based on specifications
   - Prepare mock datasets for testing
   - Document expected component behavior
   - Create verification checklist for data components

4. **Data Pipeline Verification** (as 01_data_pipeline provides components)
   - Execute formal verification protocol for received components
   - Implement `test_data_loading.py`: RNADataset instantiation, feature loading
   - Verify feature validation and tensor shape consistency
   - Test batch collation and masking implementation
   - Validate variable sequence length and missing feature handling
   - Provide formal verification feedback to 01_data_pipeline

5. **Model Component Verification Planning** (before receiving components)
   - Review model component interface contracts
   - Design verification test cases for each component
   - Prepare input tensors for component testing
   - Document expected behavior and outputs
   - Create verification checklist for model components

6. **Model Component Verification** (as 02_model_components provides components)
   - Execute formal verification protocol for received components
   - Implement `test_embeddings.py`: Sequence, positional, and relative positional embeddings
   - Develop `test_transformer_block.py`: Attention mechanism validation
   - Create `test_ipa_module.py`: Coordinate prediction functionality
   - Verify mask propagation and shape consistency
   - Validate gradient flow and device compatibility
   - Provide formal verification feedback to 02_model_components

7. **Integration Component Verification Planning** (before receiving components)
   - Review integration component interface contracts
   - Design verification test cases for model and loss functions
   - Prepare end-to-end test scenarios
   - Document expected pipeline behavior
   - Create verification checklist for integration components

8. **Integration Component Verification** (as 03_integration provides components)
   - Execute formal verification protocol for received components
   - Implement `test_losses.py`: Loss function validation
   - Develop `test_model.py`: Full model verification
   - Validate configuration management and initialization
   - Test numerical stability and gradient flow
   - Check memory efficiency and tensor lifecycle
   - Provide formal verification feedback to 03_integration

9. **Full Pipeline Integration Testing** (after all components verified)
   - Implement `test_integration.py`: End-to-end pipeline tests
   - Validate data → model → loss → backward flow
   - Test with varying batch sizes and sequence lengths
   - Verify checkpoint saving and loading
   - Validate Kaggle compatibility requirements
   - Document full pipeline verification results

10. **Performance Benchmarking** (after functional verification)
    - Develop comprehensive performance test suite
    - Measure and document memory consumption patterns
    - Analyze scaling behavior with sequence length
    - Profile GPU utilization and bottlenecks
    - Create optimization recommendations
    - Provide performance reports to all instances

## Reference Documents

### Architecture and Requirements
- `docs/3_Architecture_Specification.md`: Architecture details for test validation
- `docs/4_Product_Requirements_V1.md`: Requirements that tests must verify, especially sections LF-01 to LF-04 and MA-01 to MA-11

### Testing Workflows and Guides
- `docs/claude/05_workflows/70-pipeline-testing.md`: Comprehensive testing procedures
- `docs/claude/05_workflows/80_debugging.md`: Debug utilities and troubleshooting
- `docs/6_Tactical_Plan_V1.md`: Section V for integration testing guidance

### Component-Specific Testing Guides
- `docs/claude/02_components/10_data_loading/13_data_loading_testing.md`: Data pipeline testing approach
- `docs/claude/02_components/20_embeddings/23_embeddings_testing.md`: Embedding layer test specifications
- `docs/claude/02_components/30_transformer_block/33_transformer_testing.md`: Transformer block test procedures
- `docs/claude/02_components/40_ipa_module/43_ipa-testing-guide.md`: IPA module testing guidance
- `docs/claude/02_components/50_losses/53_losses_tests.md`: Loss function testing practices

### Verification and Handoff Protocols
- `docs/claude/03_code-instances/shared/06_component_handoff_protocol.md`: Formal handoff procedures
- `docs/claude/03_code-instances/shared/component_handoff_template.md`: Templates for verification documentation
- `docs/claude/03_code-instances/shared/07_component_status_tracker.md`: Global component status reference

### Protocol and Principle References
- `docs/claude/01_implementation_principles.md`: Core principles driving test standards
- `docs/claude/7_AI_Agent_Rules.md`: Sections 3 (Test-Driven Development) and 4 (Verification)

### Interface Contracts (review as provided)
- Interface contracts from 01_data_pipeline
- Interface contracts from 02_model_components
- Interface contracts from 03_integration

## Communication Guidelines

### Component Verification Communication

1. **Receiving Component Handoffs**:
   - Acknowledge receipt within 24 hours using the acknowledgment template
   - Review interface documentation immediately upon receipt
   - Request clarification on any ambiguous specifications
   - Establish timeline for verification completion
   - Update verification status dashboard with receipt

2. **Verification Progress Updates**:
   - Provide progress updates for lengthy verifications (>48 hours)
   - Use standardized status update format:
     ```
     Verification Update: [Component Name]
     Progress: [percentage]
     Issues Found: [count] ([severity breakdown])
     Expected Completion: [date]
     Questions/Blockers: [list]
     ```
   - Document all issues in the implementation journal as they are discovered

3. **Verification Results Communication**:
   - Deliver complete verification report within agreed timeline
   - Use the formal verification reporting template
   - Include direct links to failed test cases
   - Schedule discussion for critical or complex issues
   - Request acknowledgment of verification results

4. **Issue Resolution Coordination**:
   - Establish clear timeline for issue resolution
   - Specify which issues block acceptance vs. non-blocking
   - Provide guidance on resolution approaches when appropriate
   - Schedule re-verification after fixes are implemented
   - Track resolution progress in verification status dashboard

### Requesting Clarification on Component Behavior

- When encountering ambiguous functionality:
  - Reference specific sections in interface contracts
  - Provide concrete examples of unclear behavior
  - Include relevant test cases and outputs
  - Format requests as: "Verification clarification request: [component] - [specific issue]"

- Structure clarification requests with:
  - Description of the ambiguity or inconsistency
  - Reference to relevant documentation
  - Examples demonstrating the issue
  - Expected behavior based on documentation
  - Impact on verification process
  - Specific questions requiring answers

### Verification Status Reporting

1. **Regular Status Updates**:
   - Publish verification status summary weekly
   - Format as: "Verification Status Report: [date]"
   - Include summary of verified components
   - List components in verification process
   - Highlight critical issues blocking progress
   - Provide projected completion timeline

2. **Critical Issue Alerts**:
   - Immediately communicate critical issues to all instances
   - Format as: "CRITICAL VERIFICATION ISSUE: [component]"
   - Provide issue details, impact, and suggested action
   - Request priority resolution
   - Schedule emergency discussion if needed

3. **Performance Benchmark Reporting**:
   - Share performance analysis with all instances
   - Format as: "Performance Benchmark Report: [component/feature]"
   - Include memory, CPU/GPU usage, and timing metrics
   - Highlight performance regressions
   - Provide optimization recommendations
   - Update implementation journal at `docs/claude/03_code-instances/instance_04_testing/implementation_journal.md` with findings

## Code Standards

### Test Organization and Structure

- Organize tests by component, following the `src/` directory structure
- Use pytest as the primary testing framework
- Group related tests into classes based on functionality
- Each test should focus on a single aspect of functionality
- Follow test naming convention: `test_[component]_[functionality]_[scenario]`

### Verification Test Structure

- Every verification test suite should include these categories:
  - **Interface Tests**: Verify input/output parameters, types, shapes
  - **Functionality Tests**: Verify core behavior and correctness
  - **Edge Case Tests**: Verify behavior with boundary conditions
  - **Error Handling Tests**: Verify appropriate error responses
  - **Integration Tests**: Verify interaction with adjacent components
  - **Performance Tests**: Measure resource usage and processing time

### Fixture Design

- Create centralized fixtures in `tests/conftest.py` for common test data
- Design stateless fixtures where possible for test isolation
- Parameterize fixtures for testing multiple scenarios
- Document fixture purpose and contents with clear docstrings
- Create specialized local fixtures for component-specific tests
- Implement verification-specific fixtures for component handoffs

### Mocking and Isolation

- Use `unittest.mock` or `pytest-mock` for dependency isolation
- Create mock implementations for all external dependencies
- Separate unit tests (isolated components) from integration tests
- Ensure mocks faithfully represent actual component interfaces
- Document mock behavior and assumptions clearly
- Use stub implementations for unavailable components during verification

### Assertion Standards

- Use expressive assertions with clear failure messages
- Prefer pytest's built-in assertions for readability
- For complex validation, write custom assertion helpers
- Include expected vs. actual values in failure messages
- For floating-point comparisons, use appropriate tolerances
- Implement tensor-specific assertions for shape, type, and value verification

### Coverage Requirements

- Aim for >90% test coverage for all `src/` modules
- 100% coverage for critical components (loss functions, IPA module)
- Test all public interfaces and error conditions
- Cover edge cases: empty inputs, extreme values, variable lengths
- Verify both success and failure modes
- Document coverage exceptions with clear rationale

### Performance and Memory Testing

- Use explicit memory tracking for GPU operations
- Establish baseline performance metrics for all components
- Create reproducible benchmark cases with fixed seeds
- Document hardware configuration for all performance tests
- Test with multiple batch sizes and sequence lengths
- Implement resource monitoring for long-running tests

### Test Execution

- All tests must pass deterministically
- Set fixed random seeds for reproducibility
- Tests should run isolated from others (no side effects)
- Include both fast unit tests and slower integration tests
- Support running subsets of tests during development
- Implement test tagging for verification-specific cases

### Verification Documentation

- Document verification process and results thoroughly
- Include test coverage metrics in verification reports
- Provide concrete examples for all identified issues
- Document performance metrics with hardware context
- Include reproducible test cases for all issues
- Maintain versioned verification reports for audit trail

## Dependencies and Interfaces

### Data Pipeline Instance (01_data_pipeline)

- **Verification Dependencies**:
  - Interface contract for `RNADataset`
  - Interface contract for `collate_fn`
  - Expected tensor shapes and types
  - Path parameterization implementation
  - Feature loading utilities

- **Verification Deliverables**:
  - Formal verification report for data pipeline components
  - Test coverage metrics for data loading
  - Performance benchmark for data loading operations
  - Issue report with severity classification
  - Resolution verification for identified issues

### Model Components Instance (02_model_components)

- **Verification Dependencies**:
  - Interface contracts for all model components
  - Expected tensor transformations
  - Mask handling specifications
  - Gradient flow requirements
  - Device compatibility expectations

- **Verification Deliverables**:
  - Formal verification report for each model component
  - Test coverage metrics for model components
  - Performance benchmarks for transformer blocks
  - Memory utilization analysis
  - Issue reports with severity classification
  - Resolution verification for identified issues

### Integration Instance (03_integration)

- **Verification Dependencies**:
  - Interface contract for `RNAFoldingModel`
  - Interface contracts for loss functions
  - Configuration handling specifications
  - End-to-end data flow expectations
  - Memory management requirements

- **Verification Deliverables**:
  - Formal verification report for integration components
  - Test coverage metrics for model and losses
  - End-to-end pipeline verification report
  - Performance benchmarks for full model
  - Memory utilization analysis
  - Issue reports with severity classification
  - Resolution verification for identified issues

### Testing Interface (from Instance 04)

- **Provided to Data Pipeline Instance**:
  - Mock datasets for unit testing
  - Verification test specifications
  - Test fixtures for data loading
  - Performance optimization recommendations

- **Provided to Model Components Instance**:
  - Mock tensor inputs for component testing
  - Verification test specifications
  - Test fixtures for model components
  - Performance optimization recommendations

- **Provided to Integration Instance**:
  - Integration test specifications
  - End-to-end test fixtures
  - Pipeline verification results
  - Performance optimization recommendations

## Success Criteria

This instance's work is successful when:

1. **Comprehensive Verification Protocol Implementation**
   - Formal verification protocol fully implemented and documented
   - Verification report templates created and utilized
   - Issue tracking and resolution system established
   - Verification status dashboard maintained and current
   - All component verifications completed according to protocol

2. **Complete Test Coverage**
   - All `src/` components have ≥90% test coverage
   - Every public method/function has appropriate unit tests
   - All error conditions and edge cases are tested
   - Integration tests verify cross-component functionality
   - Performance tests establish baselines for all components

3. **Test Suite Reliability**
   - All tests pass consistently and deterministically
   - No flaky tests or test interdependencies
   - Clear, informative failure messages
   - Tests run efficiently with appropriate isolation
   - Reproducible test results across environments

4. **Verification of Requirements**
   - All requirements from PRD verified by specific tests
   - Test outcomes directly traceable to requirements
   - Interface contracts formally validated
   - Verification reports document requirement satisfaction
   - Full requirements coverage matrix maintained

5. **Performance Baselines and Optimization**
   - Memory usage benchmarks established for all components
   - Computational performance metrics documented
   - Scaling behavior with sequence length characterized
   - GPU utilization analyzed and optimized
   - Performance optimization recommendations provided

6. **High-Quality Verification Feedback**
   - Verification reports are clear, comprehensive, and actionable
   - Issue descriptions include concrete steps to reproduce
   - Verification feedback delivered within agreed timelines
   - Verification decisions (accept/reject) are well-justified
   - Re-verification conducted promptly after issue resolution

7. **Documentation and Reproducibility**
   - All tests clearly documented with purpose and approach
   - Test fixtures and data generation reproducible
   - Performance test methodology thoroughly documented
   - Verification procedures and protocols well-documented
   - Testing best practices codified for future development

8. **Successful End-to-End Verification**
   - Complete pipeline verified from data loading to loss calculation
   - Gradient flow verified through entire model
   - Memory usage optimized across component boundaries
   - Kaggle submission format and requirements verified
   - Pipeline scalability with sequence length validated

The ultimate measure of success is a robust, reliable pipeline that correctly implements the architecture specification, meets all product requirements, and provides consistent results under varying inputs and conditions, with each component formally verified and documented by this instance.
