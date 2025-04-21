# RNA 3D Folding Project: Prompt Alias Cheat Sheet
Remember, you will have to add these to AutoKey or a similar hotkey or alias management software.

## Core Prompt Aliases (High-Priority)

- **\*reinit** - Instance Reinitialization Protocol
  - Reloads component context, performs deep analysis of documentation & status
  - Provides comprehensive status report on instance responsibilities

- **\*handoff** - Component Handoff Documentation Preparation
  - Creates formal handoff documentation for component transfers between instances
  - Includes interface contracts, verification instructions, usage examples

- **\*ministage** - Quick Implementation Documentation Update
  - Performs rapid documentation checkpoint for implementation journal
  - Updates component status and interface documentation
  - Maintains knowledge continuity with minimal disruption

- **\*fullstage** - Comprehensive Implementation Documentation Update
  - Exhaustively updates all knowledge artifacts & documentation
  - Performs cross-reference verification and gap analysis
  - Ensures complete knowledge preservation across instances

- **\*debug** - Basic Code Debugging Protocol
  - Systematic approach for identifying and fixing code issues
  - Static analysis, execution flow tracing, root cause isolation
  - Provides specific fix with clear reasoning

- **\*fulldebug** - Multi-Instance Debugging Protocol
  - Diagnoses issues across instance boundaries
  - Verifies interface contracts and performs cross-instance analysis
  - Proposes solutions that maintain architectural integrity

## Implementation Session Prompts

- **\*session** - Implementation Session Execution Protocol
  - Structured approach for component implementation
  - Covers planning, implementation, testing, and documentation phases
  - Maintains quality and documentation standards

- **\*verify** - Cross-Instance Verification Protocol
  - Systematically verifies components handed off from other instances
  - Tests functionality, integration compatibility, edge cases
  - Generates verification report with next steps

## Best Practices Reference

- **Instance Boundary Management**: Maintain interface contracts, document dependencies
- **Knowledge Continuity**: Update journals, document decisions with rationales
- **Path Parameterization**: Never use hardcoded paths, use `os.path.join()`
- **Interface Definition**: Document tensor shapes, types, devices, and mask handling
- **Tensor Flow Management**: Verify shapes at boundaries, add assertions

## Instance-Specific Focus Areas

- **01 Data Pipeline**: Path parameterization, feature detection, memory efficiency
- **02 Model Components**: Mask propagation, composition, numerical stability
- **03 Integration**: Clean tensor flows, configuration management, error handling
- **04 Testing**: Comprehensive test suites, benchmarking, integration tests
