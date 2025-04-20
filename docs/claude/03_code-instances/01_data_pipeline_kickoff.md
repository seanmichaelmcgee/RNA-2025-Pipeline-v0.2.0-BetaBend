# Data Pipeline Claude Code Instructions

## Instance Purpose
You are the Claude Code instance responsible for implementing the data loading pipeline for the RNA 3D folding project. Your primary purpose is to develop robust, efficient code for loading, preprocessing, and batching precomputed RNA features and corresponding labels. You will focus exclusively on the data ingestion and preprocessing components, creating a foundation that other instances will build upon. Your implementation must adhere strictly to path parameterization principles and handle variable-length RNA sequences appropriately.

## Core Responsibilities
You are responsible for implementing the following components:

- **Helper Functions for Feature Loading**:
  - `load_coordinates(labels_df, target_id)`: Function to load C1' coordinates from labels DataFrame
  - `load_precomputed_features(target_id, features_dir)`: Function to load all precomputed feature arrays from .npz files
  - Additional helper functions as needed for feature processing

- **RNA Dataset Class**:
  - `RNADataset` class inheriting from `torch.utils.data.Dataset`
  - Constructor with parameterized paths (sequences_csv_path, labels_csv_path, features_dir)
  - Temporal cutoff filtering for training data
  - Validation set handling
  - Methods for sequence conversion and preprocessing

- **Batch Processing**:
  - `collate_fn` for batching variable-length sequences
  - Padding logic for sequence-length-dependent tensors
  - Mask generation for attention mechanisms
  - Tensor stacking with appropriate handling of different shapes

- **DataLoader Integration**:
  - `create_data_loader` function with appropriate parameters
  - Support for future DistributedSampler integration
  - Configuration-driven creation of data loaders

- **Comprehensive Unit Tests**:
  - `tests/test_data_loading.py` with test coverage for all implemented components
  - Tests for edge cases (missing files, variable lengths, etc.)
  - Tests for interface consistency with other components

## Implementation Order
Implement components in this specific sequence to ensure logical progression:

1. **Initial Helper Functions**:
   - `load_coordinates` for loading C1' coordinates from labels CSV
   - `sequence_to_int` for converting nucleotide sequences to integer indices

2. **Feature Loading Functions**:
   - `load_precomputed_features` with specific handling for:
     - Dihedral features (`{target_id}_dihedral_features.npz`)
     - Thermodynamic features (`{target_id}_thermo_features.npz`)
     - Evolutionary/MI features (`{target_id}_mi_features.npz`)
   - Implement robust error handling and default values for missing features

3. **RNADataset Core Implementation**:
   - `__init__` with proper path handling and sequence filtering
   - `__len__` method
   - Basic `__getitem__` implementation

4. **Feature Integration in Dataset**:
   - Expand `__getitem__` to load and process all feature types
   - Convert all data to appropriate PyTorch tensors
   - Implement shape consistency checks

5. **Batch Processing Implementation**:
   - `collate_fn` function for variable-length sequences
   - Padding logic for 1D, 2D-N, and 2D-NxN tensors
   - Attention mask generation
   - Tensor stacking with proper handling of non-tensor items

6. **DataLoader Integration**:
   - `create_data_loader` function with appropriate parameters
   - Configuration for future DistributedSampler support

7. **Unit Test Development**:
   - Tests for each helper function
   - Tests for RNADataset functionality
   - Tests for collate_fn and batching
   - End-to-end tests for the full data loading pipeline

## Reference Documents
Refer to these documents for implementation details:

- **Primary References**:
  - `docs/2_Feature_Specification.md` - Detailed specification of feature formats
  - `docs/claude/reference/feature_formats.md` - Reference guide for feature formats
  - `docs/claude/components/10_data_loading/guide.md` - Implementation guide
  - `docs/claude/components/10_data_loading/examples.md` - Usage examples
  - `docs/claude/components/10_data_loading/testing.md` - Testing strategy

- **Supporting Documents**:
  - `docs/claude/01_implementation_principles.md` - Core implementation principles
  - `docs/4_Product_Requirements_V1.md` - Requirements DL-01 to DL-08
  - `docs/3_Architecture_Specification.md` - Overall architecture

## Communication Guidelines
Follow these guidelines for communication with other instances:

- **Progress Updates**: Document completed components in your implementation journal
- **Interface Documentation**: Create formal interface documentation for completed components, especially for:
  - Tensor shapes and types output by RNADataset.__getitem__
  - Batch structure from collate_fn
  - Expected behavior with variable-length sequences
  - Handling of missing features

- **Questions and Clarifications**:
  - Direct questions about feature specifications to the user
  - Flag ambiguities in requirements or specifications immediately
  - Document assumptions made during implementation

- **Handoff to Instance 03 (Integration)**:
  - Prepare detailed documentation about batch format and tensor shapes
  - Document mask format for attention mechanisms (True for valid positions)
  - Explain handling of variable-length sequences
  - Specify which features may be missing and their default values

- **Coordination with Instance 04 (Testing)**:
  - Share test strategies and edge cases being tested
  - Clarify expected behavior for error conditions

## Code Standards
Adhere to these standards throughout implementation:

- **Path Parameterization (CRITICAL)**:
  - **NO hardcoded paths** in any functions or classes in `src/`
  - ALL paths must be passed as arguments from orchestration scripts
  - Use `os.path.join()` for path construction
  - No default values for path parameters
  - This is non-negotiable for Kaggle compatibility

- **Documentation**:
  - Google-style docstrings for all functions and classes
  - Type hints for function signatures (`typing` module)
  - Inline comments for complex logic
  - Example usage in docstrings where appropriate

- **Error Handling**:
  - Robust handling of missing or invalid feature files
  - Clear error messages with context (file paths, expected formats)
  - Graceful handling of optional features (e.g., evolutionary features)
  - Shape consistency checks with informative errors

- **Testing**:
  - Minimum 90% code coverage
  - Tests for normal operation, edge cases, and error conditions
  - Explicit tests for variable-length sequences
  - Performance tests for memory efficiency with large sequences

- **Performance Considerations**:
  - Efficient batch processing for variable-length sequences
  - Minimize memory usage for large tensors
  - Avoid unnecessary data duplication

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

- **DataLoader Function**: Provide a function with this signature:
  ```python
  def create_data_loader(
      sequences_csv_path: str,
      labels_csv_path: str,
      features_dir: str,
      batch_size: int,
      temporal_cutoff: Optional[str] = None,
      use_validation_set: bool = False,
      shuffle: bool = True,
      num_workers: int = 4,
      distributed: bool = False
  ) -> torch.utils.data.DataLoader:
      """Create data loader for RNA structure prediction."""
  ```

#### For Testing Instance (04):
- **Test Hooks**: Ensure all core functions are testable in isolation
- **Error Conditions**: Document expected error types and messages
- **Edge Cases**: Document handling of missing features, variable lengths, etc.

## Success Criteria
Your implementation will be considered successful when:

1. **Complete Implementation**:
   - All required components are implemented:
     - Helper functions for feature loading
     - RNADataset class
     - collate_fn for batching
     - create_data_loader function
   - Comprehensive test suite in tests/test_data_loading.py

2. **Passing Tests**:
   - All unit tests pass
   - Edge cases are correctly handled
   - No memory leaks or performance issues

3. **Interface Compliance**:
   - Batch output matches specified tensor shapes and types
   - All paths are properly parameterized
   - Feature loading correctly handles all specified formats

4. **Documentation Quality**:
   - Code is well-documented with docstrings and comments
   - Interface documentation is clear and complete
   - Implementation journal records key decisions and deviations

5. **Robustness**:
   - Gracefully handles missing optional features
   - Properly validates input data
   - Produces informative error messages

The most critical criterion is strict adherence to path parameterization - there must be NO hardcoded paths in any `src/` modules. All paths must be passed as parameters and constructed using `os.path.join()`.
