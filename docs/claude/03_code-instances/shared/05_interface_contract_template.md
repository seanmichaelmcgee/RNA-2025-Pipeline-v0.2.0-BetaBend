# RNA 3D Folding Project - Interface Contract Template

```markdown
# Component Interface Contract

## Component Identification
- **Component Name**: [Name of the component, e.g., TransformerBlock]
- **Version**: [e.g., v1.0.0]
- **Responsible Instance**: [e.g., 02_model_components]
- **Last Updated**: [YYYY-MM-DD]
- **Status**: [Draft/Proposed/Approved/Implemented]

## Component Description
[Brief description of what this component does, its role in the pipeline, and key functionality]

## Dependencies
- **Imports**: [List required imports]
- **Related Components**: [List components this directly interacts with]

## Input Interface

| Parameter | Type | Shape | Device | Description | Required |
|-----------|------|-------|--------|-------------|----------|
| `param_name` | `torch.Tensor` | `(batch_size, seq_len, hidden_dim)` | Same as input | Description of the parameter | Yes/No |
| `mask` | `torch.Tensor` | `(batch_size, seq_len)` | Same as input | Attention mask (True = valid, False = padding) | Yes |
| `config` | `Dict[str, Any]` | N/A | N/A | Configuration dictionary with hyperparameters | No |

### Input Constraints
- [List any constraints on the inputs, e.g., value ranges, relationships between parameters]
- [Specify if parameters need to be on the same device]
- [Note any required dtype precision (float32, float16, etc.)]

## Output Interface

| Return Value | Type | Shape | Device | Description |
|--------------|------|-------|--------|-------------|
| `output_name` | `torch.Tensor` | `(batch_size, seq_len, hidden_dim)` | Same as input | Description of the output |
| `auxiliary_output` | `torch.Tensor` | `(batch_size, seq_len, aux_dim)` | Same as input | Description of any auxiliary outputs |

### Output Guarantees
- [List guarantees about the outputs, e.g., normalization properties, value ranges]
- [Specify tensor properties that will be maintained]
- [Note any backward compatibility guarantees]

## Error Conditions

| Error Type | Trigger Condition | Error Message | Recovery Option |
|------------|-------------------|---------------|-----------------|
| `ValueError` | Input tensor shapes don't match expected dimensions | "Expected input shape (B, L, D), got {shape}" | None, caller must ensure correct shapes |
| `RuntimeError` | CUDA out of memory | "CUDA OOM during {operation}" | Try reducing batch size or sequence length |
| `TypeError` | Incorrect parameter type | "Expected {param} to be torch.Tensor, got {type}" | None, caller must ensure correct types |

## Implementation Requirements
- [List architectural constraints that implementations must respect]
- [Note performance requirements or expectations]
- [Specify required behavior for edge cases]
- [Document any side effects]

### Memory Considerations
- [Document expected memory footprint]
- [Note tensor lifecycle management requirements]
- [Specify if gradient accumulation is supported/required]

### Numerical Stability
- [Document approaches required for numerical stability]
- [Note any normalization or initialization requirements]
- [Specify handling of outlier values, NaNs, etc.]

## Example Usage

```python
# Example code showing typical usage of the component
import torch
from models.component_name import ComponentName

# Setup inputs
batch_size, seq_len, hidden_dim = 32, 128, 256
input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # All positions valid

# Initialize component
component = ComponentName(hidden_dim=hidden_dim)

# Forward pass
output = component(input_tensor, mask=mask)

# Expected output shape check
assert output.shape == (batch_size, seq_len, hidden_dim)

# Example error handling
try:
    invalid_input = torch.randn(batch_size, seq_len + 1, hidden_dim)
    component(invalid_input, mask=mask)
except ValueError as e:
    print(f"Caught expected error: {e}")
```

## Testing Expectations
- [Document required unit tests]
- [Specify edge cases that must be tested]
- [Note performance benchmarks expected]

## Compatibility Notes
- [Document backward compatibility commitments]
- [Note any deprecation plans]
- [Specify minimum PyTorch version]

## Handoff Checklist
- [ ] Interface documentation completed and reviewed
- [ ] Unit tests written covering normal operation
- [ ] Unit tests written covering all error conditions
- [ ] Example code verified to work
- [ ] Performance benchmarks completed
- [ ] Consumer instance has acknowledged interface

## Version History

| Version | Date | Changes | Author | Reviewed By |
|---------|------|---------|--------|-------------|
| v1.0.0 | YYYY-MM-DD | Initial version | [Name] | [Name] |
| v1.0.1 | YYYY-MM-DD | Updated error handling | [Name] | [Name] |
```

## Usage Instructions

This Interface Contract Template serves as a formal specification document for component interfaces in our RNA 3D folding pipeline. Follow these guidelines when completing the template:

1. **Be precise about tensor shapes**: Always use variables like `batch_size`, `seq_len`, etc. rather than concrete numbers.

2. **Document device handling**: Explicitly state expectations about tensor devices (CPU/CUDA) and if operations should preserve input devices.

3. **Include complete error information**: For each potential error, document the exact condition that triggers it, the specific error message, and any recovery options.

4. **Provide working examples**: Example code should be complete, runnable, and demonstrate typical usage patterns.

5. **Use consistent terminology**: Maintain consistency with other interface documents in parameter naming and descriptions.

6. **Document both normal and edge cases**: Ensure all potential use cases are covered in the documentation.

7. **Be explicit about numerical considerations**: Document any requirements related to initialization, normalization, or precision that affect numerical stability.

8. **Version everything**: Maintain clear version numbers and history to track changes over time.

This template should be placed in `docs/claude/code-instances/shared/interface_specifications/template.md` and referenced by all Claude Code instances when creating component interfaces.
