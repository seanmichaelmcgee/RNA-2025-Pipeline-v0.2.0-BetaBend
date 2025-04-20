# Data Pipeline Claude Code Instructions

## Instance Purpose
You are the Claude Code instance responsible for implementing the data loading pipeline for the RNA 3D folding project. Your primary purpose is to develop robust, efficient code for loading, preprocessing, and batching precomputed RNA features and corresponding labels. You will focus exclusively on the data ingestion and preprocessing components, creating a foundation that other instances will build upon. Your implementation must adhere strictly to path parameterization principles and handle variable-length RNA sequences appropriately. Additionally, you must design the pipeline with V2 readiness in mind, incorporating safeguards that will facilitate the transition to the full IPA model in future versions. Importantly, the initial implementation needs to handle a partial dataset where only some sequences have features available (though those that do have all feature types), with the ability to easily incorporate more sequences as their features become available.

## Kickoff Reference
This document is located at: `docs/claude/03_code-instances/01_data_pipeline_kickoff.md`

## Claude.md Configuration
This instance should maintain its own `CLAUDE.md` file located at `docs/claude/03_code-instances/instance_01_data/CLAUDE.md`. This file should contain:

- Common bash commands for testing and running the data loader
- Specific import patterns and code style guidance for data handling
- Key file locations and naming conventions for feature files
- Testing commands specific to data pipeline components
- Links to reference feature files for testing and verification

Update this file throughout development to document project-specific details and commands that should be readily available to Claude Code when working with data pipeline components.

## Required Documentation Structure

Before beginning implementation, establish these three key organizational documents:

### 1. Implementation Journal
- **Location**: `docs/claude/03_code-instances/instance_[XX]_[name]/implementation_journal.md`
- **Purpose**: Chronological record of all implementation sessions, decisions, and issues
- **Format**: Follow template at `docs/claude/03_code-instances/shared/04_implementation_jorunal_template.md`
- **Usage**: 
  - Update after each implementation session
  - Document deviations from specifications
  - Record challenges and their resolutions
  - Note any questions for other instances
  - Track next steps for upcoming sessions

### 2. Completed Components List
- **Location**: `docs/claude/03_code-instances/instance_[XX]_[name]/completed_components.md`
- **Purpose**: Track progress of individual components with current status
- **Format**:
  ```markdown
  # Completed Components Tracker
  
  | Component | Status | Test Coverage | Interface Doc | Last Updated |
  |-----------|--------|---------------|--------------|--------------|
  | [component_name] | [Not Started/In Progress/Completed] | [0-100%] | [Yes/No] | YYYY-MM-DD |

## Core Responsibilities
You are responsible for implementing the following components:

- **Helper Functions for Feature Loading**:
  - `load_coordinates(labels_df, target_id)`: Function to load C1' coordinates from labels DataFrame
  - `load_precomputed_features(target_id, features_dir)`: Function to load all precomputed feature arrays from .npz files
  - `get_dihedral_tensors(target_id, features_dir)`: Function that returns both input and target dihedral tensors (even if zeros for inference)
  - `check_features_availability(target_id, features_dir)`: Function to verify which features are available for a given target
  - Additional helper functions as needed for feature processing

- **Sequence Filtering and Management**:
  - Logic to filter sequences based on available features
  - Mechanism to easily update/extend the dataset when new features become available
  - Tracking of which sequences have features and which are pending

- **RNA Dataset Class**:
  - `RNADataset` class inheriting from `torch.utils.data.Dataset`
  - Constructor with parameterized paths and pluggable split logic for flexible temporal cutoffs
  - Feature availability filtering for partial dataset loading
  - Temporal cutoff filtering for training data
  - Validation set handling
  - Methods for sequence conversion and preprocessing
  - Metadata flag generation for feature presence/absence

- **Batch Processing Utilities**:
  - Composable padding utility functions in `src/utils/padding.py`:
    ```python
    def pad_1d(x: torch.Tensor, max_len: int, pad_value=0) -> torch.Tensor: ...
    def pad_2d(x: torch.Tensor, max_len: int, pad_value=0) -> torch.Tensor: ...
    def pad_tensor(x: torch.Tensor, target_shape: Tuple[int,...], pad_value=0) -> torch.Tensor: ...
    ```
  - `collate_fn` for batching variable-length sequences using these utilities
  - Mask generation for attention mechanisms
  - Batch metadata generation (has_dihedrals, has_msa, etc.)

- **DataLoader Integration**:
  - `create_data_loader` function with appropriate parameters
  - Support for future DistributedSampler integration
  - Configuration-driven creation of data loaders
  - Option to filter based on feature availability

- **Comprehensive Unit Tests**:
  - `tests/test_data_loading.py` with test coverage for all implemented components
  - Tests for edge cases (missing files, variable lengths, etc.)
  - Tests for mixed-presence features (e.g., dihedral present but thermo missing)
  - Tests for sequences with no features at all
  - Tests for updating the dataset with newly available features
  - Tests for long sequences (e.g., L=500) to verify padding and memory warnings
  - Tests for partial MI or NaN-only channels
  - Tests for interface consistency with other components

## Implementation Order
Implement components in this specific sequence to ensure logical progression:

1. **Initial Helper Functions**:
   - `load_coordinates` for loading C1' coordinates from labels CSV
   - `sequence_to_int` for converting nucleotide sequences to integer indices
   - `check_features_availability` to detect which features exist for each target

2. **Feature Loading Functions with V2 Readiness**:
   - `load_precomputed_features` with specific handling for:
     - Dihedral features (`{target_id}_dihedral_features.npz`)
     - Thermodynamic features (`{target_id}_thermo_features.npz`)
     - Evolutionary/MI features (`{target_id}_mi_features.npz`)
   - `get_dihedral_tensors` that always returns both input and target tensors
   - Implement robust error handling and default values for missing features
   - Include feature presence detection for metadata flags

3. **Sequence Filtering Logic**:
   - Implement function to scan features directory and identify which sequences have features
   - Create mechanism to filter sequences based on feature availability
   - Design update method to incorporate newly available features

4. **Padding Utilities**:
   - Create `src/utils/padding.py` with composable utility functions
   - Implement modular padding for 1D, 2D, and future tensor shapes
   - Design for extensibility to handle new tensor types in V2

5. **RNADataset Core Implementation**:
   - `__init__` with proper path handling, pluggable split function, and feature availability filtering:
     ```python
     def __init__(self, 
                 sequences_csv_path: str, 
                 labels_csv_path: Optional[str],
                 features_dir: str,
                 split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                 temporal_cutoff: Optional[str] = None,
                 use_validation_set: bool = False,
                 require_features: bool = True):
         """Initialize RNA dataset with pluggable split logic and feature filtering."""
     ```
   - `__len__` method
   - Basic `__getitem__` implementation
   - Method to update dataset with newly available features:
     ```python
     def update_available_features(self):
         """Scan features directory and update available sequences."""
     ```

6. **Feature Integration in Dataset**:
   - Expand `__getitem__` to load and process all feature types
   - Generate metadata flags (has_dihedrals, has_msa, etc.)
   - Convert all data to appropriate PyTorch tensors
   - Implement shape consistency checks
   - Handle cases where a sequence has no features

7. **Batch Processing Implementation**:
   - `collate_fn` using the composable padding utilities
   - Attention mask generation
   - Metadata aggregation in batches
   - Tensor stacking with proper handling of non-tensor items

8. **DataLoader Integration**:
   - `create_data_loader` function with appropriate parameters and feature filtering
   - Configuration for future DistributedSampler support

9. **Unit Test Development**:
   - Tests for each helper function
   - Tests for RNADataset functionality including feature filtering
   - Tests for updating the dataset with new features
   - Tests for collate_fn and batching
   - Tests for edge cases (mixed feature presence, long sequences, no features)
   - End-to-end tests for the full data loading pipeline

10. **Integration Smoke Test**:
    - Create `tests/test_integration_smoke.py` with basic pipeline test:
      ```python
      def test_pipeline_smoke():
          loader = create_data_loader(...)
          batch = next(iter(loader))
          # Verify batch structure and tensor shapes
          assert "meta" in batch
          assert "has_msa" in batch["meta"]
          # More assertions...
      ```

## Reference Documents
Refer to these documents for implementation details:

- **Primary References**:
  - `docs/2_Feature_Specification.md` - Detailed specification of feature formats
  - `docs/claude/04_reference/feature_formats.md` - Reference guide for feature formats
  - `docs/claude/components/10_data_loading/11_data_loading_guide.md` - Implementation guide
  - `docs/claude/components/10_data_loading/12_data_loading_examples.md` - Usage examples
  - `docs/claude/components/10_data_loading/13_data_loading_testing.md` - Testing strategy
  - `docs/data_examples/train_features_example.md` - Example feature file structure
  - `docs/claude/03_code-instances/15_data_loading_v2_readiness_risks.md` - V2 readiness considerations

- **Supporting Documents**:
  - `docs/claude/01_implementation_principles.md` - Core implementation principles
  - `docs/4_Product_Requirements_V1.md` - Requirements DL-01 to DL-08
  - `docs/3_Architecture_Specification.md` - Overall architecture

## Communication Guidelines
Follow these guidelines for communication with other instances:

- **Progress Updates**: Document completed components in your implementation journal in `docs/claude/03_code-instances/instance_01_data/implementation_journal.md`
- **Interface Documentation**: Create formal interface documentation for completed components, especially for:
  - Tensor shapes and types output by RNADataset.__getitem__
  - Batch structure from collate_fn including metadata flags
  - Expected behavior with variable-length sequences
  - Handling of missing features and feature presence flags
  - Feature availability filtering and dataset updates

- **Questions and Clarifications**:
  - Direct questions about feature specifications to the user
  - Flag ambiguities in requirements or specifications immediately
  - Document assumptions made during implementation

- **Handoff to Instance 03 (Integration)**:
  - Prepare detailed documentation about batch format and tensor shapes
  - Document mask format for attention mechanisms (True for valid positions)
  - Explain handling of variable-length sequences
  - Specify which features may be missing and their default values
  - Document metadata flags and their meanings (has_dihedrals, has_msa, etc.)
  - Explain feature availability filtering and how to handle dataset updates

- **Coordination with Instance 04 (Testing)**:
  - Share test strategies and edge cases being tested
  - Clarify expected behavior for error conditions
  - Document handling of mixed-presence features
  - Explain testing approach for partial feature availability

## Code Standards
Adhere to these standards throughout implementation:

- **Path Parameterization (CRITICAL)**:
  - **NO hardcoded paths** in any functions or classes in `src/`
  - ALL paths must be passed as arguments from orchestration scripts
  - Use `os.path.join()` for path construction
  - No default values for path parameters
  - This is non-negotiable for Kaggle compatibility

- **V2 Readiness Standards**:
  - Design interfaces with future extensions in mind
  - Implement composable utilities rather than monolithic functions
  - Include metadata flags for feature presence/absence
  - Support for pluggable components via function parameters
  - Facilitate easy updating with new feature data

- **Documentation**:
  - Google-style docstrings for all functions and classes
  - Type hints for function signatures (`typing` module)
  - Inline comments for complex logic
  - Example usage in docstrings where appropriate
  - Document V2 extension points and design decisions
  - Document feature availability filtering and update mechanisms

- **Error Handling**:
  - Robust handling of missing or invalid feature files
  - Clear error messages with context (file paths, expected formats)
  - Graceful handling of optional features (e.g., evolutionary features)
  - Shape consistency checks with informative errors
  - Proper handling of sequences with no features

- **Testing**:
  - Minimum 90% code coverage
  - Tests for normal operation, edge cases, and error conditions
  - Explicit tests for variable-length sequences
  - Tests for mixed-presence features (some present, some missing)
  - Tests for sequences with no features at all
  - Tests for updating the dataset with newly available features
  - Tests for extreme cases (very long sequences, uniform MI matrices)
  - Performance tests for memory efficiency with large sequences

- **Performance Considerations**:
  - Efficient batch processing for variable-length sequences
  - Minimize memory usage for large tensors
  - Avoid unnecessary data duplication
  - Efficient feature availability checking to avoid filesystem bottlenecks

- **Style and Formatting**:
  - Follow PEP 8
  - Consistent naming conventions
  - Maximum line length of 100 characters
  - Use spaces around operators

## Dependencies and Interfaces

### Dependencies
The data loading component should depend only on:
- Standard Python libraries (os, warnings, etc.)
- PyTorch (torch, torch.utils.data)
- NumPy (for array handling)
- Pandas (for CSV loading)

### Interfaces for Other Instances

#### For Integration Instance (03):
- **Batch Structure**: Provide a dictionary with:
  - `target_ids`: List of target IDs in the batch
  - `sequence_int`: Tensor of integer-encoded sequences (shape: [batch_size, max_seq_len])
  - `dihedral_features`: Tensor of dihedral features (shape: [batch_size, max_seq_len, 4])
  - `pairing_probs`: Tensor of base-pairing probabilities (shape: [batch_size, max_seq_len, max_seq_len])
  - `positional_entropy`: Tensor of positional entropy (shape: [batch_size, max_seq_len])
  - `coupling_matrix`: Tensor of evolutionary coupling scores (shape: [batch_size, max_seq_len, max_seq_len])
  - `coordinates`: Tensor of C1' coordinates (shape: [batch_size, max_seq_len, 3])
  - `mask`: Boolean mask indicating valid positions (shape: [batch_size, max_seq_len], True for valid)
  - `lengths`: Tensor of sequence lengths (shape: [batch_size])
  - `meta`: Dictionary containing metadata flags:
    ```python
    "meta": {
        "has_dihedrals": torch.Tensor,  # shape: [batch_size], dtype: bool
        "has_msa": torch.Tensor,        # shape: [batch_size], dtype: bool
        # Add more metadata flags as needed
    }
    ```

- **DataLoader Function**: Provide a function with this signature:
  ```python
  def create_data_loader(
      sequences_csv_path: str,
      labels_csv_path: Optional[str],
      features_dir: str,
      batch_size: int,
      split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
      temporal_cutoff: Optional[str] = None,
      use_validation_set: bool = False,
      require_features: bool = True,
      shuffle: bool = True,
      num_workers: int = 4,
      distributed: bool = False
  ) -> torch.utils.data.DataLoader:
      """Create data loader for RNA structure prediction with flexible splitting and feature filtering."""
  ```

- **Dataset Update Mechanism**: Document how to update the dataset when new features become available:
  ```python
  # Example usage
  dataset = RNADataset(...)
  
  # After new features are added to the features directory:
  dataset.update_available_features()
  # This rescans the features directory and updates the list of available sequences
  ```

#### For Testing Instance (04):
- **Test Hooks**: Ensure all core functions are testable in isolation
- **Error Conditions**: Document expected error types and messages
- **Edge Cases**: Document handling of missing features, variable lengths, no features, etc.
- **Padding Utilities**: Share utility functions for testing various padding scenarios
- **Feature Availability Testing**: Provide methods to test feature availability detection and dataset updates

## Success Criteria
Your implementation will be considered successful when:

1. **Complete Implementation**:
   - All required components are implemented:
     - Helper functions for feature loading with V2 readiness
     - Composable padding utilities
     - RNADataset class with pluggable split logic and feature availability filtering
     - Feature update mechanism for incorporating newly available features
     - collate_fn using modular padding utilities
     - create_data_loader function with feature availability filtering option
   - Comprehensive test suite in tests/test_data_loading.py

2. **V2 Readiness**:
   - Always emits both dihedral_input and dihedral_target tensors
   - Includes metadata flags for feature presence/absence
   - Uses composable padding utilities that can be extended for new tensor types
   - Supports pluggable split functions in RNADataset.__init__
   - Handles mixed-presence features gracefully
   - Supports updating the dataset when new features become available

3. **Passing Tests**:
   - All unit tests pass
   - Edge cases are correctly handled
   - Tests for mixed-presence features pass
   - Tests for sequences with no features pass
   - Tests for updating the dataset with new features pass
   - Tests for long sequences pass
   - No memory leaks or performance issues

4. **Interface Compliance**:
   - Batch output matches specified tensor shapes and types
   - Metadata flags are correctly generated
   - All paths are properly parameterized
   - Feature loading correctly handles all specified formats
   - Feature availability filtering works correctly

5. **Documentation Quality**:
   - Code is well-documented with docstrings and comments
   - Interface documentation is clear and complete
   - V2 extension points are clearly documented
   - Feature availability filtering and update mechanisms are documented
   - Implementation journal records key decisions and deviations

6. **Robustness**:
   - Gracefully handles missing optional features
   - Properly handles sequences with no features
   - Properly validates input data
   - Produces informative error messages
   - Handles uniform MI matrices appropriately
   - Efficiently updates available features when new data is added

The most critical criterion is strict adherence to path parameterization - there must be NO hardcoded paths in any `src/` modules. All paths must be passed as parameters and constructed using `os.path.join()`. Additionally, the implementation must follow the V2 readiness guidelines to facilitate future extensions and properly handle the scenario where only some sequences have features available.
