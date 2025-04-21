# Issue Classification System

This document defines the standardized issue classification system used by the testing instance for component verification. It ensures consistent evaluation and prioritization of issues across all components.

## 1. Issue Severity Levels

### Critical
- **Definition**: Issues that completely block functionality, cause crashes, or produce incorrect results that would significantly impact model performance.
- **Examples**:
  - Component crashes during normal operation
  - Incorrect coordinate calculations affecting structural prediction
  - Memory leaks causing out-of-memory errors
  - Data corruption or loss
  - Gradient flow completely broken
- **Resolution Timeline**: Immediate (blocking)
- **Re-verification**: Required before acceptance

### High
- **Definition**: Significant functional limitations, performance issues, or incorrect behaviors that impact component usability but have workarounds.
- **Examples**:
  - Significantly degraded performance (>50% slower than requirements)
  - Numerical instability in specific cases
  - Incorrect handling of edge cases
  - GPU/CPU incompatibilities
  - Memory usage exceeds budget by >25%
- **Resolution Timeline**: Within 48 hours
- **Re-verification**: Required before acceptance

### Medium
- **Definition**: Non-critical functional issues, minor performance problems, or documentation inconsistencies that don't significantly impact core functionality.
- **Examples**:
  - Minor performance issues (<50% degradation)
  - Unclear error messages
  - Inconsistent parameter naming
  - Incomplete documentation
  - Memory usage exceeds budget by <25%
- **Resolution Timeline**: Within 1 week
- **Re-verification**: Required but not blocking

### Low
- **Definition**: Minor issues, style concerns, or potential future problems that don't impact current functionality.
- **Examples**:
  - Code style inconsistencies
  - Minor documentation gaps
  - Non-optimal implementations
  - Unused parameters or code paths
  - Potential future compatibility issues
- **Resolution Timeline**: Before final release
- **Re-verification**: Optional

## 2. Issue Type Categories

### Interface Issues
- **Definition**: Problems with component interfaces, including parameter types, shapes, names, or return values.
- **Subcategories**:
  - Parameter type mismatch
  - Return value inconsistency
  - Shape inconsistency
  - Missing parameters
  - Device handling problems
  - Mask handling problems

### Functional Issues
- **Definition**: Problems with core functionality not behaving as specified.
- **Subcategories**:
  - Incorrect calculations
  - Logic errors
  - Missing functionality
  - Incorrect transformations
  - Boundary condition failures

### Performance Issues
- **Definition**: Problems related to execution time, memory usage, or resource utilization.
- **Subcategories**:
  - Excessive execution time
  - Memory inefficiency
  - GPU underutilization
  - Scaling problems
  - Unnecessary computations

### Integration Issues
- **Definition**: Problems that only appear when the component is integrated with others.
- **Subcategories**:
  - Data flow incompatibility
  - Module interaction failure
  - Conflicting dependencies
  - State management problems
  - Context inconsistencies

### Documentation Issues
- **Definition**: Problems with documentation completeness, accuracy, or clarity.
- **Subcategories**:
  - Missing documentation
  - Incorrect documentation
  - Ambiguous descriptions
  - Outdated examples
  - Missing edge case documentation

## 3. Issue Tracking

### Issue Identification
Each issue is assigned a unique identifier with the format:
```
[COMPONENT_PREFIX]-[ISSUE_NUMBER]
```

Component prefixes:
- DATA: Data loading components
- EMB: Embedding components
- TRF: Transformer block components
- IPA: IPA module components
- LOSS: Loss function components
- MODEL: Full model components
- INT: Integration components

Example: `IPA-001` for the first issue found in the IPA module.

### Issue Status
Issues progress through the following statuses:
1. **Open**: Issue identified but not yet addressed
2. **In Progress**: Provider instance actively working on resolution
3. **Review**: Resolution submitted for verification
4. **Resolved**: Issue fixed and verified
5. **Won't Fix**: Issue acknowledged but will not be fixed (requires rationale)
6. **Deferred**: Issue will be addressed in a future version

### Issue Documentation
Each issue requires the following documentation:
1. Full description with context
2. Steps to reproduce
3. Expected vs. actual behavior
4. Impact assessment
5. Recommended resolution approach

## 4. Resolution Verification

### Resolution Requirements
For an issue to be considered resolved:
1. The root cause must be addressed (not just symptoms)
2. Test coverage must be added to prevent regression
3. Documentation must be updated to reflect changes
4. Original reproduction steps must no longer trigger the issue

### Verification Process
1. Re-run the original reproduction case
2. Verify that the issue no longer occurs
3. Run regression tests to ensure no new issues
4. Review code changes for completeness and correctness
5. Update issue status in tracking system

## 5. Issue Prioritization Matrix

| Severity | Functional | Interface | Performance | Integration | Documentation |
|----------|------------|-----------|-------------|-------------|---------------|
| Critical | Priority 1 | Priority 1 | Priority 1  | Priority 1  | Priority 2    |
| High     | Priority 1 | Priority 2 | Priority 2  | Priority 2  | Priority 3    |
| Medium   | Priority 2 | Priority 3 | Priority 3  | Priority 3  | Priority 4    |
| Low      | Priority 3 | Priority 4 | Priority 4  | Priority 4  | Priority 5    |

**Priority Definitions**:
- **Priority 1**: Immediate resolution required (blocking)
- **Priority 2**: Resolve before next verification cycle
- **Priority 3**: Resolve before final component acceptance
- **Priority 4**: Resolve before project completion
- **Priority 5**: Address if time permits

## 6. Issue Reporting Templates

See the [issue_report_template.md](verification_reports/issue_report_template.md) file for the standardized template to use when reporting issues.

## 7. Common Issue Patterns

The following are common issue patterns to watch for in specific components:

### Data Loading Components
- Inconsistent padding handling
- Missing feature validation
- Inefficient data loading
- Incorrect mask generation
- Memory leaks with large datasets

### Embedding Components
- Incorrect positional encoding
- Shape inconsistencies
- Device transfer inefficiencies
- Numerical instability in embeddings
- Normalization issues

### Transformer Components
- Attention mechanism errors
- Mask propagation failures
- Gradient vanishing/exploding
- Memory inefficiency in self-attention
- Incorrect residual connections

### IPA Module
- Coordinate frame transformation errors
- Numerical instability in IPA operations
- Incorrect update gate behavior
- Poor performance scaling with sequence length
- Device inconsistencies

### Loss Functions
- Numerical instability
- Incorrect gradient computation
- Mask handling issues
- Weight scaling problems
- Edge case failures

### Full Model Components
- Integration errors between components
- Configuration handling issues
- Checkpoint saving/loading problems
- Device placement inconsistencies
- Memory management across components