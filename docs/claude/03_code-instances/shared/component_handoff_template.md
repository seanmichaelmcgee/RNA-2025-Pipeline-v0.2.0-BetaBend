# Component Handoff Protocol Template

This document serves as a standardized handoff protocol template for coding agent transitions. It ensures critical information is properly communicated when transitioning work between different coding agent instances.

## 1. Component Identification

**Component Name:** [e.g., Data Pipeline]  
**Instance ID:** [e.g., instance_01_data]  
**Primary Functions:** [Brief list of core responsibilities]  
**Repository Path:** [Path to main implementation files]  
**Handoff Date:** [YYYY-MM-DD]

## 2. Implementation Status

### 2.1 Completed Components

| Component | Status | Tests | Documentation | Last Updated |
|-----------|--------|-------|---------------|--------------|
| [Component 1] | [Complete/Partial] | [Complete/Partial/Missing] | [Complete/Partial/Missing] | [YYYY-MM-DD] |
| [Component 2] | [Complete/Partial] | [Complete/Partial/Missing] | [Complete/Partial/Missing] | [YYYY-MM-DD] |
| ... | ... | ... | ... | ... |

### 2.2 Pending Components

| Component | Current State | Dependencies | Priority | Estimated Complexity |
|-----------|---------------|--------------|----------|----------------------|
| [Component 1] | [Not Started/In Progress] | [List dependencies] | [High/Medium/Low] | [High/Medium/Low] |
| [Component 2] | [Not Started/In Progress] | [List dependencies] | [High/Medium/Low] | [High/Medium/Low] |
| ... | ... | ... | ... | ... |

### 2.3 Known Issues

| Issue | Severity | Components Affected | Potential Solutions |
|-------|----------|---------------------|---------------------|
| [Issue 1] | [Critical/High/Medium/Low] | [List components] | [Brief description of solutions] |
| [Issue 2] | [Critical/High/Medium/Low] | [List components] | [Brief description of solutions] |
| ... | ... | ... | ... |

## 3. Interface Contracts

### 3.1 Public API

List all public functions, classes, and interfaces that other components rely on:

```python
# Function/class signature with docstring
def example_function(param1: type, param2: type) -> return_type:
    """
    Brief description.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return value
    """
```

### 3.2 Data Structures

Define all data structures shared with other components:

```python
# Example data structure
{
    "field1": type,  # Description
    "field2": type,  # Description
    ...
}
```

### 3.3 Integration Points

| Consumer Component | Integration Function | Expected Behavior | Error Handling |
|-------------------|---------------------|-------------------|----------------|
| [Component 1] | [Function name] | [Expected behavior] | [Error handling strategy] |
| [Component 2] | [Function name] | [Expected behavior] | [Error handling strategy] |
| ... | ... | ... | ... |

## 4. Environment and Dependencies

### 4.1 Runtime Requirements

- Python version: [e.g., 3.10+]
- Memory requirements: [e.g., 8GB+]
- GPU requirements: [if applicable]
- Environment variables: [if applicable]

### 4.2 Package Dependencies

| Package | Version | Purpose | Installation Command |
|---------|---------|---------|---------------------|
| [Package 1] | [Version] | [Purpose] | [Command] |
| [Package 2] | [Version] | [Purpose] | [Command] |
| ... | ... | ... | ... |

### 4.3 File Dependencies

| File Path | Purpose | Source |
|-----------|---------|--------|
| [Path 1] | [Purpose] | [Source/Origin] |
| [Path 2] | [Purpose] | [Source/Origin] |
| ... | ... | ... |

## 5. Testing Requirements

### 5.1 Test Coverage

| Component | Unit Tests | Integration Tests | Manual Tests Required |
|-----------|------------|-------------------|----------------------|
| [Component 1] | [Yes/No] | [Yes/No] | [Yes/No - Description] |
| [Component 2] | [Yes/No] | [Yes/No] | [Yes/No - Description] |
| ... | ... | ... | ... |

### 5.2 Critical Test Cases

| Test Case | Purpose | Command | Expected Output |
|-----------|---------|---------|----------------|
| [Test 1] | [Purpose] | [Command] | [Expected output] |
| [Test 2] | [Purpose] | [Command] | [Expected output] |
| ... | ... | ... | ... |

### 5.3 Integration Verification

List steps required to verify successful integration:

1. [Step 1]
2. [Step 2]
3. ...

## 6. Implementation Details

### 6.1 Architecture Overview

Brief description of the component architecture, with diagram if possible:

```
[Component 1] → [Component 2] → [Component 3]
    ↓              ↓               ↓
[Component 4] → [Component 5] → [Component 6]
```

### 6.2 Algorithms and Data Structures

Explain key algorithms and data structures used in the implementation:

- [Algorithm/DS 1]: [Explanation]
- [Algorithm/DS 2]: [Explanation]
- ...

### 6.3 Performance Considerations

Document critical performance characteristics:

- Time complexity: [e.g., O(n) for operation X]
- Space complexity: [e.g., O(n²) for operation Y]
- Bottlenecks: [Description of identified bottlenecks]
- Optimization opportunities: [Description of potential optimizations]

## 7. Extension and Maintenance

### 7.1 Anticipated Extensions

List likely future extensions:

1. [Extension 1]: [Description]
2. [Extension 2]: [Description]
3. ...

### 7.2 Maintenance Considerations

Document maintenance requirements:

- Regular updates: [Description]
- Monitoring: [Description]
- Technical debt: [Description]

## 8. Common Debugging Scenarios

| Symptom | Likely Cause | Debugging Steps | Solution |
|---------|--------------|----------------|----------|
| [Symptom 1] | [Cause] | [Steps] | [Solution] |
| [Symptom 2] | [Cause] | [Steps] | [Solution] |
| ... | ... | ... | ... |

## 9. Decision Log

Document key implementation decisions and their rationales:

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| [Decision 1] | [Rationale] | [Alternatives] | [YYYY-MM-DD] |
| [Decision 2] | [Rationale] | [Alternatives] | [YYYY-MM-DD] |
| ... | ... | ... | ... |

## 10. Handoff Checklist

- [ ] All code pushed to repository
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Interface contracts finalized
- [ ] Known issues documented
- [ ] Integration points verified
- [ ] Handoff template completed
- [ ] Knowledge transfer session completed
- [ ] Receiving agent has run tests successfully
- [ ] Receiving agent has access to all necessary resources

## 11. Contact Information

**Handoff Agent:** [Agent ID]  
**Receiving Agent:** [Agent ID]  
**Supervisor:** [Name]  
**Knowledge Transfer Session Date:** [YYYY-MM-DD]  

---

This template should be completed by the agent handing off the component and reviewed by the receiving agent. Any questions or clarifications should be addressed before considering the handoff complete.