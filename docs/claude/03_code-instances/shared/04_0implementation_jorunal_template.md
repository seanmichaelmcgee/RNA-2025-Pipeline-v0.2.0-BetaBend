# Implementation Journal Template for Claude Code Instances

This template provides a standardized format for tracking implementation progress across all Claude Code instances in the RNA 3D folding project. Maintaining detailed implementation journals ensures knowledge continuity, facilitates handoffs, and provides transparency about component status.

```markdown
# [INSTANCE_NAME] Implementation Journal

## Component Status Tracker

| Component | Status | Tests | Interface Doc | Dependent Instances | Last Updated |
|-----------|--------|-------|---------------|---------------------|--------------|
| [component_name] | ⬜ Pending | ⬜ | ⬜ | [instances] | YYYY-MM-DD |
| [component_name] | 🟡 Partial | 🟡 | ⬜ | [instances] | YYYY-MM-DD |
| [component_name] | ✅ Complete | ✅ | ✅ | [instances] | YYYY-MM-DD |
| [component_name] | ❌ Blocked | ⬜ | ⬜ | [instances] | YYYY-MM-DD |

**Status Legend:**
- ⬜ Pending: Not yet started
- 🟡 Partial: Implementation in progress
- ✅ Complete: Fully implemented
- ❌ Blocked: Implementation blocked by dependency or issue

## Implementation Sessions

### Implementation Session: YYYY-MM-DD

#### Components Completed:
- [x] [component_name] - [brief description of what was implemented]
  - [key implementation details]
  - [notable techniques or approaches used]
- [x] [another_component] - [implementation details]

#### Deviations from Plan:
- [description of deviation from original implementation plan]
- [rationale for the deviation]
- [impact on timeline or other components]

#### Issues/Questions:
- [issue description] - [current understanding/approach]
  - Possible solutions: [list of potential solutions]
  - Instances affected: [list instances affected by this issue]
- [question about specification/requirement] - [impact on implementation]

#### Next Steps:
- Complete [component_name] implementation
- Begin work on [next_component]
- Request clarification on [unclear_specification]
- Address [issue_to_resolve]

## Interface Documentation

### [Component Name] Interface

**Version:** v1.0
**Last Updated:** YYYY-MM-DD

#### Input Interface:
- [parameter_name]: [type] - [description]
- [parameter_name]: [type] - [description]

#### Output Interface:
- [return_value]: [type] - [description]
- [tensor_shape]: [dimensions] - [description]

#### Error Conditions:
- [condition]: [resulting error and handling]
- [condition]: [resulting error and handling]

#### Implementation Requirements:
- [requirement_1]
- [requirement_2]

## Cross-Instance Communication Log

### Communication with [OTHER_INSTANCE]: YYYY-MM-DD

#### Topic:
[brief description of communication topic]

#### Key Points:
- [decision or information point]
- [decision or information point]

#### Action Items:
- [this_instance]: [action to take]
- [other_instance]: [action to take]
```

## Example Implementation Journal Entry

Below is an example of how to use this template for an actual implementation session:

```markdown
# 01_Data_Pipeline Implementation Journal

## Component Status Tracker

| Component | Status | Tests | Interface Doc | Dependent Instances | Last Updated |
|-----------|--------|-------|---------------|---------------------|--------------|
| load_coordinates() | ✅ Complete | ✅ | ✅ | 03_Integration | 2025-04-15 |
| load_precomputed_features() | 🟡 Partial | 🟡 | ⬜ | 03_Integration | 2025-04-15 |
| RNADataset.__init__() | ⬜ Pending | ⬜ | ⬜ | 03_Integration | 2025-04-14 |
| RNADataset.__getitem__() | ⬜ Pending | ⬜ | ⬜ | 03_Integration | 2025-04-14 |
| collate_fn() | ❌ Blocked | ⬜ | ⬜ | 03_Integration | 2025-04-14 |

## Implementation Sessions

### Implementation Session: 2025-04-15

#### Components Completed:
- [x] load_coordinates() function
  - Implemented efficient loading using numpy.load for .npz files
  - Added normalization option to standardize coordinates to unit sphere
- [x] load_precomputed_features() function (partial)
  - Basic loading functionality implemented
  - Still need to implement error handling for missing features

#### Deviations from Plan:
- Added expanded error handling for missing MI files that wasn't in original specification
- Changed tensor dtype to float32 throughout for consistency with model components
- Implemented custom caching mechanism to improve loading performance

#### Issues/Questions:
- Unclear how to handle NaN values in angle data
  - Current approach: replace with zeros and create separate mask tensor
  - Need confirmation from 02_Model_Components on preferred approach
- Need specification of expected matrix symmetry
  - Instances affected: 02_Model_Components, 03_Integration

#### Next Steps:
- Complete feature loading with proper error handling
- Begin RNADataset class implementation
- Request clarification on angle data handling from 02_Model_Components

## Interface Documentation

### load_coordinates() Interface

**Version:** v1.0  
**Last Updated:** 2025-04-15

#### Input Interface:
- filepath: str - Path to the coordinates .npz file
- normalize: bool - Whether to normalize coordinates (default: True)

#### Output Interface:
- coordinates: torch.Tensor - Shape: (seq_len, 3, 3), dtype: torch.float32

#### Error Conditions:
- File not found: Raises FileNotFoundError with path information
- Invalid file format: Raises ValueError with details on expected format
- Empty file: Raises ValueError if no valid coordinates found

#### Implementation Requirements:
- Must handle variable sequence lengths
- Must return tensor on CPU (device placement handled by caller)
- All NaN values must be replaced and tracked in separate mask

## Cross-Instance Communication Log

### Communication with 02_Model_Components: 2025-04-15

#### Topic:
Tensor shape convention for coordinates

#### Key Points:
- Agreed on (seq_len, 3, 3) shape for coordinate tensors
- Confirmed float32 as standard dtype
- Discussed handling of padding for variable length sequences

#### Action Items:
- This instance to update documentation with agreed shape convention
- 02_Model_Components to ensure embedding layer handles this format
```

## Usage Guidelines

1. **Create one journal per instance** - Each Claude Code instance should maintain its own implementation journal.

2. **Update regularly** - Make entries after each implementation session to ensure knowledge continuity.

3. **Keep the component tracker current** - Update component status whenever it changes.

4. **Document all interfaces** - Completed components should have documented interfaces for other instances.

5. **Record cross-instance communications** - Note all significant discussions with other instances.

6. **Link to external resources** - Reference relevant specifications, discussions, or code examples.

7. **Highlight blockers immediately** - Mark components as blocked and document dependencies clearly.

8. **Be specific about deviations** - Explain why implementation differs from the original plan.

This template balances comprehensive tracking with practical usability. By maintaining detailed implementation journals, each Claude Code instance contributes to a cohesive knowledge base that preserves insights and facilitates smooth collaboration throughout the RNA 3D folding project.
