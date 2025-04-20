Claude Code Instance Structure Plan
Based on my understanding of the Claude Code research preview and your project needs, here's my recommended plan for structuring the instruction files:
Directory Structure
Create a dedicated directory within your documentation for Claude Code instance instructions:
rna_3d_project/
└── docs/
    └── claude/
        ├── components/  (existing)
        ├── workflows/   (existing)
        ├── reference/   (existing)
        └── code-instances/  (new)
            ├── README.md  (explains the multi-instance approach)
            ├── 01_data_pipeline.md
            ├── 02_model_components.md
            ├── 03_integration.md
            └── 04_testing.md
Instruction File Format
Each instruction file should follow a consistent structure:
# [Component Group] Claude Code Instructions

## Instance Purpose
[Clear statement of this instance's responsibilities and boundaries]

## Core Responsibilities
- [List of specific components to implement]
- [Key functionality for this instance]

## Implementation Order
1. [First component to implement]
2. [Next component]
3. [...]

## Reference Documents
- [Links to relevant architecture docs]
- [Links to relevant requirement docs]
- [Links to relevant example docs]

## Communication Guidelines
- [How to request clarification]
- [When to defer to other instances]
- [How to report completion of a component]

## Code Standards
- [Reminder of Python style guidelines]
- [Testing requirements]
- [Documentation requirements]
Cross-Reference System
To ensure instances know about each other:
    1. In the README.md, explain the division of responsibilities and when to hand off work 
    2. Add a section in each instruction file listing the other instances and their domains 
    3. Use consistent component naming across all instances 
In-Code References
Add comments in the code to reference the relevant Claude Code instance:
# Implemented by Claude Code Instance: 01_data_pipeline
class RNADataset(Dataset):
    """
    Dataset class for RNA 3D structure prediction.
    """
Activation Pattern
When activating each instance, use:
    1. A standardized prompt that references the instruction file 
    2. Initial loading of relevant documents from the content window 
    3. A clear first task to begin implementation 
Example:
I'm working on the RNA 3D folding project and would like your help implementing components according to docs/claude/code-instances/01_data_pipeline.md. Please begin by reviewing that file along with the data loading specification documents, then start implementing the RNADataset class.
Sample First Instruction File
Here's a sample for the Data Pipeline instance:
# Data Pipeline Claude Code Instructions

## Instance Purpose
You will implement the data loading pipeline for the RNA 3D folding project, focusing on efficient loading, preprocessing, and batching of RNA sequence data and features.

## Core Responsibilities
- `src/data_loading.py` with RNADataset class
- Helper functions for loading coordinates and features
- Collate function for handling variable-length sequences
- Unit tests in `tests/test_data_loading.py`

## Implementation Order
1. Helper functions (`load_coordinates`, `load_precomputed_features`)
2. `RNADataset.__init__` and supporting methods
3. `collate_fn` for batching
4. Unit tests for all components

## Reference Documents
- `docs/2_Feature_Specification.md` (primary reference)
- `docs/claude/components/10_data_loading/guide.md`
- `docs/claude/components/10_data_loading/examples.md`
- `docs/claude/components/10_data_loading/testing.md`

## Communication Guidelines
- Request clarification on feature formats if specifications are ambiguous
- Notify when key components are complete
- Flag any potential interface issues affecting Model Components

## Code Standards
- Strictly follow path parameterization principle - NO hardcoded paths
- Include comprehensive docstrings in Google style
- Implement proper error handling for missing files
- Write unit tests covering normal operation and edge cases
This structure provides clear guidance for each Claude Code instance while maintaining the organization and accessibility needed for effective development across multiple AI assistants.

