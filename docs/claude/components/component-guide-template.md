# [Component Name] Implementation Guide

## Component Overview
Brief description of the component's purpose and role in the RNA 3D folding pipeline.

## Requirements Reference
List of requirements from the PRD that this component must satisfy:
- **[Requirement ID]**: Description of the requirement

## Technical Background
Concise explanation of the technical concepts needed to understand this component:
- Concept 1: Explanation
- Concept 2: Explanation

## Interfaces

### Input Interface
Description of the input data that this component receives:
```python
# Example input format
input_data = {
    "field1": torch.Tensor(...),  # Shape, type, description
    "field2": torch.Tensor(...),  # Shape, type, description
}
```

### Output Interface
Description of the output data that this component produces:
```python
# Example output format
output_data = {
    "field1": torch.Tensor(...),  # Shape, type, description
    "field2": torch.Tensor(...),  # Shape, type, description
}
```

## Implementation Steps

1. **Step 1**: Description
   ```python
   # Example code snippet
   def function_name(...):
       # Implementation
       pass
   ```

2. **Step 2**: Description
   ```python
   # Example code snippet
   ```

## Critical Aspects
Key points to pay special attention to:
- **Path Parameterization**: Reminder about no hardcoded paths
- **Error Handling**: Specific error cases to handle
- **Memory Efficiency**: Considerations for large inputs

## Testing Requirements
Description of tests that should be implemented:
1. Test case 1: Description
2. Test case 2: Description

## Example Usage
Complete example showing how this component will be used:
```python
# Usage example
```

## Related Documentation
- Architecture: [Link to relevant architecture doc]
- Reference: [Link to relevant reference doc]
- Integration: [Link to relevant integration doc]

## Next Steps
- Implement [Next Component] after completing this one
- See [Integration Guide] for connecting this with other components
