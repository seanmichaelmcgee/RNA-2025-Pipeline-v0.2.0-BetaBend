#Prompt guide for working with multi-instance cloud code.
##These aliases are saved with AutoKey to input without hard return or anything else when the alias is typed
##Some do have specifiers that need to be selected before they are put in to the coding agent.

# Enhanced RNA 3D Folding Project: Prompt Aliases

I've consolidated and enhanced the prompting strategies to focus on high-value prompts that incorporate self-documentation and trigger appropriate computational depth. Here are the improved prompt aliases:

## 1. Core Prompt Aliases (High-Priority)

```markdown
PROMPT *reinit : # Instance Reinitialization and Context Loading Prompt (ULTRATHINK)

## RNA 3D Folding Project: Instance Reinitialization Protocol

I need you to fully reinitialize as Instance [01_data_pipeline/02_model_components/03_integration/04_testing] for our RNA 3D folding project. Using structured introspection, please:

### 1. Self-Context Discovery
First, locate and analyze these documents in order:
- Your instance kickoff document at `docs/claude/03_code-instances/[01/02/03/04]_*_kickoff.md`
- The multi-instance architecture overview at `docs/claude/03_code-instances/README.md`
- Project implementation principles at `docs/claude/01_implementation_principles.md`
- Your implementation journal at `docs/claude/03_code-instances/instance_[01/02/03/04]_*/implementation_journal.md`

### 2. Component Responsibility Assessment
- Extract your specific component responsibilities from the kickoff document
- Identify your implementation boundaries and interface points with other instances
- Note any specific implementation standards emphasized for your instance

### 3. Implementation Status Analysis
- Review your implementation journal to identify completed, in-progress, and pending components
- Check your instance's completed_components.md document
- Examine the global component status tracker at `docs/claude/03_code-instances/shared/07_component_status_tracker.md`

### 4. Dependency and Artifact Inventory
- Scan for all available source files related to your components
- Identify which components from other instances your work depends on
- Check for documented interfaces or handoffs from other instances

### 5. Architecture Integration Analysis
- Identify where your components fit in the overall RNA 3D folding pipeline
- Analyze tensor flows between your components and others
- Note any critical design patterns you must maintain

### 6. Status Report
After this thorough analysis, please provide a comprehensive status report including:
1. Your instance's core purpose and responsibilities
2. Current implementation status with component-level details
3. Dependencies on other instances and their current status
4. Interface contracts that need to be maintained
5. Immediate implementation priorities based on the critical path

This deep reinitialization ensures proper alignment and continuity within our multi-instance architecture.
```

```markdown
PROMPT *handoff : # Component Handoff Documentation Preparation Prompt THINK HARDER

## RNA 3D Folding Project: Component Handoff Protocol

I need to prepare a formal handoff of [COMPONENT_NAME] from Instance [SOURCE] to Instance [TARGET]. Please help me create comprehensive handoff documentation by executing this structured protocol:

### 1. Component Verification
First, analyze the component's implementation status:
- Verify test coverage meets or exceeds 90%
- Confirm all public methods have complete docstrings
- Check that tensor shapes and types are explicitly documented
- Verify mask handling is properly implemented
- Confirm the component adheres to path parameterization principles

### 2. Interface Contract Documentation
Use the template found at `docs/claude/03_code-instances/shared/05_interface_contract_template.md`
Create a formal interface contract at `docs/claude/03_code-instances/instance_[XX]_[source]/handoffs/YYYY-MM-DD_[component_name]_handoff.md` with:

- **Component Identification**:
  - Full component name and version
  - Location in source tree
  - Primary responsibility

- **Input Interface**:
  - Parameter names, types, shapes, and descriptions
  - Default values and behavior
  - Required vs. optional parameters
  - Device expectations

- **Output Interface**:
  - Return types, shapes, and descriptions
  - Error conditions and handling
  - Device guarantees

- **Implementation Requirements**:
  - Critical algorithms or patterns
  - Performance considerations
  - Memory usage patterns
  - Error handling approaches

- **Verification Instructions**:
  - Step-by-step test commands
  - Expected outputs or behaviors
  - Edge cases to verify

- **Current Limitations**:
  - Known issues or constraints
  - Planned future improvements
  - Optimization opportunities

### 3. Example Usage Documentation
Provide clear, executable examples showing:
- Basic usage pattern
- Handling of edge cases
- Error condition management
- Integration points with other components

### 4. Implementation Decision Record
Document key implementation decisions:
- Deviations from original specifications with rationale
- Algorithm choices with justifications
- Performance tradeoffs made
- Future-proofing considerations

### 5. Handoff Status Update
Update the status in tracking documents:
- Mark component as "Ready for Handoff" in your instance's completed_components.md
- Update the global component tracker at `docs/claude/03_code-instances/shared/07_component_status_tracker.md`
- Note the handoff in your implementation journal

### 6. Integration Guidance
Provide specific guidance for the receiving instance on:
- How to properly integrate this component
- Known pitfalls to avoid
- Testing recommendations
- Performance optimization opportunities

This comprehensive handoff protocol ensures knowledge continuity across instances and smooth integration of components.
```

```markdown
PROMPT *ministage : # Quick Implementation Documentation Update THINK

## RNA 3D Folding Project: Rapid Documentation Checkpoint

Please perform a quick but thorough documentation update for your recent implementation work:

### 1. Implementation Journal Update
Locate your implementation journal at `docs/claude/03_code-instances/instance_[XX]_*/implementation_journal.md` and:
- Add an entry for today's date if needed
- Document components completed or advanced
- Note any implementation challenges or decisions
- Record deviations from specifications with rationale
- List next steps based on current progress

### 2. Component Status Update
Update status tracking documents:
- Update your instance's completed_components.md
- For any newly completed components, update the global tracker at `docs/claude/03_code-instances/shared/07_component_status_tracker.md`

### 3. Interface Documentation
For components nearing completion:
- Document public interfaces with parameter specifications
- Note any changes from originally planned interfaces
- Highlight integration points with other components

### 4. Special Considerations
Note any:
- Cross-instance dependencies identified
- Performance considerations discovered
- Memory optimization opportunities
- Path parameterization verifications

### 5. Documentation Summary
Summarize your documentation updates to maintain a clear record of progression. Once you are satisfied with the update COMMIT and PUSH to git.

This efficient checkpoint ensures we maintain knowledge continuity while focusing development time on implementation.
```

```markdown
PROMPT *fullstage : # Comprehensive Implementation Documentation Update ULTRATHINK
## RNA 3D Folding Project: Deep Documentation Protocol

Please perform an exhaustive documentation update across all relevant artifacts to ensure complete knowledge preservation:

### 1. Implementation Analysis
First, perform deep introspection on recent implementation work:
- Which components have been implemented or modified?
- What level of completion has been achieved?
- What algorithmic decisions or optimizations were made?
- What deviations from specifications occurred and why?
- What integration points with other instances were established?
- What performance or memory considerations emerged?

### 2. Knowledge Artifact Updates
Systematically update all knowledge artifacts:

#### 2.1 Implementation Journal
Locate and update your implementation journal at `docs/claude/03_code-instances/instance_[XX]_*/implementation_journal.md`:
- Create a detailed session entry with timestamp
- Document all components touched with specific progress details
- Record all technical decisions with supporting rationale
- Document challenges encountered and solutions implemented
- Note cross-instance integration points established
- List explicit next steps with dependencies and priorities

#### 2.2 Component Status Tracker
Update status tracking with precise metrics:
- Update your instance's completed_components.md with current status, test coverage percentage, and interface documentation status
- For any completed components, update the global tracker at `docs/claude/03_code-instances/shared/07_component_status_tracker.md` with current status, dependencies, and handoff readiness

#### 2.3 Interface Documentation
For all components that have reached at least 50% completion:
- Create or update interface documentation with explicit tensor shapes, types, and device specifications
- Document mask handling behavior in detail
- Specify error conditions and handling procedures
- Note performance characteristics and memory patterns

#### 2.4 Handoff Documentation
For components approaching readiness for handoff:
- Create preliminary handoff documentation using the template at `docs/claude/03_code-instances/shared/08_component_handoff_template.md`
- Include verification instructions with expected outputs
- Document integration patterns and requirements

#### 2.5 Technical Documentation
Update component-specific technical documentation:
- Add newly discovered implementation patterns to your instance's CLAUDE.md
- Document performance optimization techniques
- Record any critical algorithms or numerical stability considerations

### 3. Cross-Reference Verification
Perform consistency verification across documents:
- Ensure status is consistently reported across all trackers
- Verify interface specifications match implementation
- Confirm next steps align with global component dependencies

### 4. Global Documentation Assessment
Assess if updates are needed for project-wide documentation:
- README.md implementation status section
- index.md file structure if changes occurred
- docs/claude/03_code-instances/README.md if architectural patterns changed

### 5. Documentation Gap Analysis
Identify any remaining documentation gaps:
- Components without sufficient interface documentation
- Missing rationales for key decisions
- Undocumented cross-instance dependencies
- Incomplete verification instructions

### 6. Documentation Update Summary
Provide a comprehensive summary of all documentation updates performed, noting:
- Which documents were updated and how
- Which documents were analyzed but required no changes
- Remaining documentation tasks and their priority

### 7. Comprehensive stage and commit
Perform a thorough review of global documents and component or instance specific documents to stage and commit
- Be certain to include all documents altered as the part of this review
- Once satisfied with thorough staging, COMMIT and PUSH to git

This exhaustive documentation protocol ensures complete knowledge continuity across the multi-instance architecture, preserving critical implementation details and decisions.
```
```markdown
PROMPT *debug ;; Please debug this code using a systematic approach: (1) Analyze the code statically to identify suspicious patterns or potential issues. (2) Walk through the execution flow step-by-step, highlighting the current state at each key point. (3) Identify any inconsistencies between expected and actual behavior. (4) Isolate the root cause with evidence from the code. (5) Propose a specific fix with clear reasoning. (6) Explain how your solution aligns with best practices and prevents similar issues. If multiple approaches exist, compare their tradeoffs briefly.

```

```markdown
PROMPT *fulldebug : # Multi-Instance Debugging Protocol THINK HARDER

## RNA 3D Folding Project: Cross-Instance Debug Framework

Please deploy a systematic debugging methodology to diagnose issues across instance boundaries:

### 1. Issue Characterization
First, precisely characterize the issue:
- What is the observed behavior vs. expected behavior?
- Which components are involved?
- Which instances own these components?
- At what interface boundary does the issue appear?

### 2. Static Analysis
Perform comprehensive static analysis:
- Analyze tensor shape transformations across component boundaries
- Verify mask propagation through the component chain
- Check device management across instance boundaries
- Examine data type handling and potential precision issues
- Verify configuration parameter consistency

### 3. Sequential Execution Tracing
Perform step-by-step execution analysis:
- Trace input tensors through each transformation
- Document the state at each component boundary
- Identify where expectations diverge from actuality
- Note any performance bottlenecks or memory issues

### 4. Interface Contract Verification
Verify adherence to defined interfaces:
- Check if interfaces are fully specified and documented
- Validate implementation against interface contracts
- Identify any interface mismatches or miscommunications
- Verify tensor shape, device, and type assumptions

### 5. Root Cause Isolation
Using evidence from previous steps, isolate the root cause:
- Identify the specific component and code location
- Determine whether it's an implementation or interface issue
- Pinpoint any misunderstood requirements or assumptions
- Determine if the issue crosses instance boundaries

### 6. Solution Development
Propose a specific, actionable solution:
- Provide exact code changes needed
- Explain the rationale behind the solution
- Address both immediate fix and prevention of similar issues
- Consider impact on other components and instances

### 7. Cross-Instance Consideration
If the issue spans multiple instances:
- Document required changes for each affected instance
- Specify interface adjustments needed
- Outline coordination requirements for the fix
- Propose verification steps for each instance

### 8. Documentation Updates
Identify documentation that needs updating:
- Interface contracts that require clarification
- Implementation journal entries to record the issue and solution
- Component status changes if applicable

This structured debugging protocol ensures systematic resolution of issues across instance boundaries while maintaining architectural integrity.
```

## 2. Implementation Session Prompts

```markdown
PROMPT *session : # Implementation Session Execution Protocol THINK

## RNA 3D Folding Project: Structured Implementation Session
## NEEDS WORK

Let's conduct a focused implementation session for [COMPONENT_NAME]. This session will follow our structured implementation methodology:

### 1. Planning Phase (15%)
First, let's establish our implementation plan:

#### 1.1 Component Context
- **Purpose**: [Brief description of component's role]
- **Primary Interfaces**:
  - Inputs: [Expected input tensors with shapes]
  - Outputs: [Expected output tensors with shapes]
- **Key Requirements**:
  - [List critical requirements]

#### 1.2 Dependency Analysis
- **Prerequisites**: [List components this depends on]
- **Consumers**: [List components that will use this]
- **Interface Constraints**: [Note any interface requirements]

#### 1.3 Implementation Strategy
- **Implementation Order**:
  1. [Step-by-step implementation plan]
  2. [...]
- **Validation Approach**:
  - [Verification strategy]
- **Potential Challenges**:
  - [Anticipated issues]

### 1.4 Develop the implentation plan
Given the component context, dependency analysis, and implementation strategy above for the reference component, reference the appropriate documentation to get us up to speed and populate the plan.

### 2. Implementation Phase (70%)
Now let's implement systematically:

#### 2.1 Component Interface Definition
- Define class/function signatures with complete type annotations
- Add comprehensive docstrings following our documentation standard
- Establish parameter validation

#### 2.2 Core Logic Implementation
- Implement functionality incrementally in logical units
- Add detailed comments for complex operations
- Verify tensor shapes at critical points

#### 2.3 Edge Case Handling
- Implement robust error handling
- Add edge case detection and processing
- Ensure proper mask handling throughout

### 3. Testing Phase (10%)
Let's verify our implementation:

#### 3.1 Basic Functionality Testing
- Test with typical inputs
- Verify output correctness
- Check mask handling behavior

#### 3.2 Edge Case Testing
- Test with edge cases (empty, small, large inputs)
- Verify error handling behavior
- Check numerical stability

#### 3.3 Integration Validation
- Verify compatibility with dependent components
- Test end-to-end flows if applicable

### 4. Documentation Phase (5%)
Finally, let's document our work:

#### 4.1 Implementation Journal Update
- Record today's progress
- Document key decisions and rationales
- Note challenges and solutions

#### 4.2 Component Status Update
- Update status in tracking documents
- Document next steps

### 5. Session Summary
Let's conclude with a concise summary of accomplishments, challenges, and next steps.

This structured approach ensures efficient implementation while maintaining quality and documentation standards.
```

```markdown
PROMPT *verify : # Cross-Instance Verification Protocol THINK HARD

## RNA 3D Folding Project: Component Verification Framework

I need to verify the [COMPONENT_NAME] that was handed off from Instance [SOURCE]. Please help me execute a systematic verification process:

### 1. Interface Contract Review
First, locate and analyze the handoff documentation:
- Find the component's handoff document at `docs/claude/03_code-instances/instance_[XX]_[source]/handoffs/`
- Review the interface contract specifications
- Note any implementation requirements or constraints
- Identify verification instructions provided

### 2. Functionality Verification
Execute a step-by-step verification of core functionality:
- Verify each public method behaves as documented
- Test with the provided example inputs
- Confirm outputs match expected patterns
- Verify proper error handling for invalid inputs

### 3. Integration Compatibility
Check compatibility with components in this instance:
- Verify tensor shape compatibility
- Confirm mask handling consistency
- Check device management alignment
- Validate configuration parameter handling

### 4. Edge Case Validation
Test specified edge cases:
- Verify behavior with empty inputs
- Test with minimum and maximum expected sizes
- Check performance with large inputs
- Confirm numerical stability in extreme cases

### 5. Performance Assessment
If performance requirements are specified:
- Measure execution time for typical operations
- Check memory usage patterns
- Verify scalability with input size
- Identify any potential bottlenecks

### 6. Verification Report
Generate a detailed verification report:
- List tests performed and results
- Document any discrepancies between expected and actual behavior
- Note any implementation concerns or optimization opportunities
- Provide an overall verification status (Verified/Issues Found)

### 7. Next Steps
Based on verification results:
- If verified successfully, update the global status tracker
- If issues found, document specific problems for the source instance
- Identify any needed interface clarifications
- Plan integration steps for this component

This structured verification process ensures components work correctly across instance boundaries and meet all specified requirements.
```

## 3. Best Practices for Multi-Instance Development

### 3.1 Critical Implementation Guidelines

1. **Instance Boundary Management**
   - Maintain strict adherence to interface contracts
   - Never modify code owned by another instance
   - Use the handoff protocol for component transitions
   - Document cross-instance dependencies explicitly

2. **Knowledge Continuity**
   - Update implementation journals after each session
   - Document all key decisions with rationales
   - Create interface documentation early
   - Use the fullstage protocol before major milestones

3. **Path Parameterization (Critical)**
   - NEVER use hardcoded paths in `src/` modules
   - Pass all file/directory paths as parameters
   - Use `os.path.join()` for all path construction
   - Verify path parameterization in all components

4. **Interface Definition**
   - Document tensor shapes, types, and devices explicitly
   - Specify mask handling behavior
   - Include error conditions and responses
   - Note any performance characteristics

5. **Tensor Flow Management**
   - Verify shape transformations at component boundaries
   - Document tensor shapes in comments
   - Add shape assertions in key functions
   - Ensure consistent device management

### 3.2 Instance-Specific Focus Areas

1. **Instance 01: Data Pipeline**
   - Focus on proper file handling with path parameterization
   - Implement robust feature availability detection
   - Ensure memory-efficient operations for large sequences
   - Document feature metadata comprehensively

2. **Instance 02: Model Components**
   - Prioritize mask propagation through all operations
   - Design for composition via explicit interfaces
   - Implement numerically stable operations
   - Document tensor transformations thoroughly

3. **Instance 03: Integration**
   - Focus on clean tensor flows between components
   - Implement robust configuration management
   - Ensure proper error handling at component boundaries
   - Document end-to-end data paths

4. **Instance 04: Testing**
   - Develop comprehensive test suites
   - Implement performance benchmarking
   - Create integration tests for cross-component verification
   - Document testing approach for each component

By using these enhanced prompts and following these guidelines, we can maintain effective multi-instance development with proper knowledge continuity and clear component boundaries.
