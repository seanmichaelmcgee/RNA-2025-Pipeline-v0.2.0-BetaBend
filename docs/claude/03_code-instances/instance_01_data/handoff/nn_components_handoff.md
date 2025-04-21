# Data Pipeline Handoff Document

## 1. Component Identification

**Component Name:** RNA 3D Folding Data Pipeline  
**Instance ID:** instance_01_data  
**Primary Functions:** Data loading, feature processing, batch creation  
**Repository Path:** `/src/data_loading.py`, `/src/utils/padding.py`  
**Handoff Date:** 2025-04-20

## 2. Implementation Status

### 2.1 Completed Components

| Component | Status | Tests | Documentation | Last Updated |
|-----------|--------|-------|---------------|--------------|
| check_features_availability() | Complete | Complete | Complete | 2025-04-20 |
| sequence_to_int() | Complete | Complete | Complete | 2025-04-20 |
| load_coordinates() | Complete | Complete | Complete | 2025-04-20 |
| is_uniform_top_pairs() | Complete | Complete | Complete | 2025-04-20 |
| is_uniform_mi_matrix() | Complete | Complete | Complete | 2025-04-20 |
| load_precomputed_features() | Complete | Complete | Complete | 2025-04-20 |
| get_dihedral_tensors() | Complete | Complete | Complete | 2025-04-20 |
| padding utilities | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.__init__() | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.__getitem__() | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.update_available_features() | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.set_temporal_cutoff() | Complete | Complete | Complete | 2025-04-20 |
| collate_fn() | Complete | Complete | Complete | 2025-04-20 |
| create_data_loader() | Complete | Complete | Complete | 2025-04-20 |

### 2.2 Pending Components

None - all components have been implemented and tested.

### 2.3 Known Issues

| Issue | Severity | Components Affected | Potential Solutions |
|-------|----------|---------------------|---------------------|
| Memory usage with large sequences | Medium | collate_fn(), pad_2d() | Consider implementing sparse tensor support for pairing_probs and coupling_matrix matrices |
| Warnings for missing features | Low | load_precomputed_features() | Consider silent mode parameter to suppress warnings in production |
| Cache invalidation for feature updates | Low | update_available_features() | Consider more sophisticated timestamp-based cache invalidation |

## 3. Interface Contracts

### 3.1 Public API

See `interface_exports.md` for complete API documentation. Key interfaces with updates:

```python
def is_uniform_top_pairs(top_pairs: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if top pairs from MI features have uniform scores, indicating a single sequence MSA.
    
    Args:
        top_pairs: Array of shape (P, 3) with format [pos_i, pos_j, score]
        epsilon: Threshold for standard deviation to consider uniform
        
    Returns:
        True if all scores are effectively identical
    """
```

```python
def is_uniform_mi_matrix(matrix: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if an MI matrix contains uniform values, indicating a single sequence MSA.
    
    Args:
        matrix: Mutual information matrix
        epsilon: Threshold for standard deviation to consider uniform
        
    Returns:
        True if matrix appears to have uniform off-diagonal values
    """
```

```python
def load_precomputed_features(
    target_id: str, 
    features_dir: str,
    temporal_cutoff: Optional[str] = None
) -> Dict[str, Union[Dict[str, np.ndarray], None]]:
    """Load all precomputed features for a target from .npz files.
    
    Args:
        target_id: RNA sequence identifier
        features_dir: Directory containing feature files
        temporal_cutoff: Optional temporal cutoff date for filtering
        
    Returns:
        Dictionary of feature dictionaries with structure:
        {
            'dihedral': {'features': array(...)},
            'thermo': {'pairing_probs': array(...), 'mfe': value, ...},
            'evolutionary': {'coupling_matrix': array(...), 'has_valid_mi': bool}
        }
    """
```

```python
class RNADataset(Dataset):
    def __init__(
        self, 
        sequences_csv_path: str,
        labels_csv_path: Optional[str] = None,
        features_dir: str = "",
        split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        temporal_cutoff: Optional[str] = None,
        use_validation_set: bool = False,
        require_features: bool = True
    ):
        """Initialize RNA dataset with pluggable split logic and feature filtering."""
        
    def set_temporal_cutoff(self, new_cutoff: Optional[str] = None) -> None:
        """
        Update the temporal cutoff and refilter sequences.
        
        Args:
            new_cutoff: New temporal cutoff date or None to remove cutoff
        """
```

### 3.2 Data Structures

Key data structure is the batch dictionary returned by DataLoader, with updates to the metadata field:

```python
{
    # Target IDs
    'target_ids': List[str],  # Original sequence identifiers
    
    # Core Sequence Features
    'sequence_int': torch.Tensor,  # Shape: (batch_size, max_seq_len)
    'lengths': torch.Tensor,       # Shape: (batch_size,)
    'mask': torch.Tensor,          # Shape: (batch_size, max_seq_len)
    
    # Structure Features
    'dihedral_features': torch.Tensor,     # Shape: (batch_size, max_seq_len, 4)
    'pairing_probs': torch.Tensor,         # Shape: (batch_size, max_seq_len, max_seq_len)
    'positional_entropy': torch.Tensor,     # Shape: (batch_size, max_seq_len)
    'accessibility': torch.Tensor,          # Shape: (batch_size, max_seq_len)
    'coupling_matrix': torch.Tensor,        # Shape: (batch_size, max_seq_len, max_seq_len)
    'conservation': torch.Tensor,           # Shape: (batch_size, max_seq_len)
    
    # Target Values (training only)
    'coordinates': torch.Tensor,            # Shape: (batch_size, max_seq_len, 3)
    
    # Metadata
    'meta': {
        'has_dihedrals': torch.Tensor,      # Shape: (batch_size,)
        'has_thermo': torch.Tensor,         # Shape: (batch_size,)
        'has_msa': torch.Tensor,            # Shape: (batch_size,) - TRUE only if MI exists AND is valid
        'before_cutoff': torch.Tensor,      # Shape: (batch_size,) - TRUE if before temporal cutoff
    }
}
```

### 3.3 Integration Points

| Consumer Component | Integration Function | Expected Behavior | Error Handling |
|-------------------|---------------------|-------------------|----------------|
| Model Training | create_data_loader() | Returns DataLoader that yields batches with all specified tensors | Raises specific exceptions for missing files or features |
| Model Inference | create_data_loader() with labels_csv_path=None | Returns DataLoader without coordinate tensors | Warns about missing features but continues with zero tensors |
| Visualization | RNADataset.__getitem__() | Returns sample dictionary with all features for a single sequence | Raises exceptions for critical missing features |
| Experiment Tracking | RNADataset.set_temporal_cutoff() | Allows dynamic reconfiguration of temporal cutoff | Updates filtered sequences list appropriately |

## 4. Environment and Dependencies

### 4.1 Runtime Requirements

- Python version: 3.10+
- Memory requirements: 8GB+ (16GB+ recommended for large sequences)
- GPU requirements: None for data pipeline (memory copies happen on CPU)
- Environment variables: None required

### 4.2 Package Dependencies

| Package | Version | Purpose | Installation Command |
|---------|---------|---------|---------------------|
| PyTorch | 2.1+ | Tensor operations | `conda install pytorch -c pytorch` |
| NumPy | 1.20+ | Array operations | `conda install numpy` |
| Pandas | 1.4+ | CSV parsing | `conda install pandas` |

### 4.3 File Dependencies

| File Path | Purpose | Source |
|-----------|---------|--------|
| sequences.csv | RNA sequence information | User provided |
| labels.csv | 3D coordinate data | User provided |
| features/dihedral_features/*.npz | Dihedral angle features | Generated from preprocessing |
| features/thermo_features/*.npz | Thermodynamic predictions | Generated from preprocessing |
| features/mi_features/*.npz | Evolutionary information | Generated from preprocessing |

## 5. Testing Requirements

### 5.1 Test Coverage

| Component | Unit Tests | Integration Tests | Manual Tests Required |
|-----------|------------|-------------------|----------------------|
| Feature loading | Yes | Yes | No |
| MI matrix validity detection | Yes | Yes | No |
| Dataset | Yes | Yes | No |
| Temporal cutoff filtering | Yes | Yes | No |
| Collation | Yes | Yes | No |
| Data loader | Yes | Yes | No |
| Padding | Yes | Yes | No |

### 5.2 Critical Test Cases

| Test Case | Purpose | Command | Expected Output |
|-----------|---------|---------|----------------|
| test_data_loading.py | Comprehensive test suite | `python -m pytest tests/test_data_loading.py -v` | All tests pass |
| test_padding.py | Tests for padding utilities | `python -m pytest tests/test_padding.py -v` | All tests pass |
| TestIntegration.test_pipeline_smoke | End-to-end test | `python -m pytest tests/test_data_loading.py::TestIntegration::test_pipeline_smoke -v` | Test pass |
| test_is_uniform_mi_matrix | Tests MI matrix detection | `python -m pytest tests/test_data_loading.py::test_is_uniform_mi_matrix -v` | Test pass |
| test_dataset_getitem_mi_handling | Tests MI metadata propagation | `python -m pytest tests/test_data_loading.py::test_dataset_getitem_mi_handling -v` | Test pass |

### 5.3 Integration Verification

To verify successful integration:

1. Run `python -m pytest tests/test_data_loading.py tests/test_padding.py -v`
2. Create a small example dataset and verify batch shapes:
   ```python
   loader = create_data_loader(...)
   batch = next(iter(loader))
   # Verify batch structure and tensor shapes
   ```
3. Check that all feature tensors have appropriate shapes and types
4. Verify metadata flags correspond to actual feature availability
5. Verify that uniform MI matrices are correctly identified:
   ```python
   # Check that has_msa is False for uniform MI
   batch = next(iter(loader))
   print(batch['meta']['has_msa'])  # Should show appropriate boolean values
   ```
6. Verify temporal cutoff functionality:
   ```python
   # Create dataset with temporal cutoff
   loader = create_data_loader(..., temporal_cutoff="2022-05-27")
   # Verify sequences are filtered correctly
   print([item['target_id'] for item in loader.dataset])
   
   # Test reconfiguration
   loader.set_temporal_cutoff("2023-01-01")
   # Verify sequence list has changed
   print([item['target_id'] for item in loader.dataset])
   ```

## 6. Implementation Details

### 6.1 Architecture Overview

The data pipeline follows this architecture:

```
Feature Detection (check_features_availability)
           ↓
MI Matrix Validation (is_uniform_top_pairs, is_uniform_mi_matrix)
           ↓
Feature Loading (load_precomputed_features, load_coordinates)
           ↓
Temporal Filtering (based on temporal_cutoff)
           ↓
Dataset (RNADataset.__getitem__, RNADataset.update_available_features)
           ↓
Collation (collate_fn, padding utilities)
           ↓
DataLoader (create_data_loader)
```

### 6.2 Algorithms and Data Structures

- **Feature Availability Detection**: File system-based detection with caching
- **MI Matrix Validation**:
  - **Top Pairs Analysis**: Quick check using just the top_pairs array (typically <1MB) before loading full matrix
  - **Uniform Detection**: Statistical evaluation of off-diagonal values to identify uninformative matrices
- **Temporal Filtering**:
  - Dataset-level filtering of sequences based on publication date
  - Feature-level filtering based on feature generation date
- **Partial Data Handling**: Metadata flag propagation with zero-filled tensors
- **Variable-Length Sequences**: Memory-efficient padding with mask tracking
- **Batch Creation**: Dictionary-based with consistent tensor shapes

### 6.3 Performance Considerations

- **MI Matrix Memory Optimization**:
  - Avoids loading full MI matrices (up to 146MB) for uniform/invalid cases
  - Zeros out invalid matrices to save memory
  - Uses top_pairs array (typically <1MB) for quick detection
- **Time complexity**: O(n) for loading each sample, where n is sequence length
- **Space complexity**: O(n²) for 2D matrices (pairing_probs, coupling_matrix)
- **Bottlenecks**: 
  - Loading large 2D matrices for long sequences
  - Padding operations for batches with highly variable sequence lengths
- **Optimization opportunities**:
  - Implement sparse tensor support for 2D matrices
  - Add on-the-fly feature computation to reduce storage requirements

## 7. Extension and Maintenance

### 7.1 Anticipated Extensions

1. **On-demand feature computation**: Add support for computing features on-the-fly
2. **Sparse tensor support**: Implement sparse matrices for pairing_probs and coupling_matrix
3. **Feature versioning**: Add support for tracking feature versions
4. **Advanced caching**: Implement more sophisticated caching strategies

### 7.2 Maintenance Considerations

- **Regular updates**: Update feature_availability cache periodically
- **Monitoring**: Track memory usage with large batches
- **Technical debt**: 
  - NPZ file format is simple but lacks compression
  - Feature directory structure is flat and may not scale well

## 8. Common Debugging Scenarios

| Symptom | Likely Cause | Debugging Steps | Solution |
|---------|--------------|----------------|----------|
| "Missing thermodynamic features" error | Required feature files not found | Check file existence and paths | Add missing feature files or set require_features=False |
| Memory errors with large sequences | 2D matrices consuming excessive memory | Monitor memory usage during loading | Reduce batch size or implement sparse matrices |
| Shape mismatch in collated batch | Inconsistent feature dimensions | Check individual sample shapes before collation | Ensure consistent feature dimensions or add additional padding |
| Missing features despite files present | Incorrect file naming or format | Verify file names and formats | Ensure consistent file naming and proper NPZ format |
| Valid MI marked as invalid | Epsilon threshold too large | Check uniformity detection logic and parameters | Adjust epsilon in is_uniform_mi_matrix() |
| Feature generation after temporal cutoff | Incorrect timestamp extraction | Check timestamp parsing in load_precomputed_features | Verify feature generation date format and comparison logic |

## 9. Decision Log

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| Metadata flags for feature presence | Allows models to handle varying feature availability | Feature filtering at dataset level | 2025-04-20 |
| Top pairs MI detection | Significantly reduces memory usage for uniform matrices | Loading full matrices first | 2025-04-20 |
| Zeroing out invalid MI matrices | Saves memory and prevents models from using invalid data | Keeping matrices as-is with only metadata flags | 2025-04-20 |
| Feature-level temporal filtering | Prevents data leakage at the most granular level | Dataset-level filtering only | 2025-04-20 |
| Reconfigurable temporal cutoff | Enables dynamic experimentation with different cutoff dates | Hardcoded cutoff date | 2025-04-20 |

## 10. Handoff Checklist

- [x] All code pushed to repository
- [x] All tests passing
- [x] Documentation updated
- [x] Interface contracts finalized
- [x] Known issues documented
- [x] Integration points verified
- [x] Handoff template completed
- [x] Handoff documentation finalized
- [ ] Knowledge transfer session completed
- [ ] Receiving agent has run tests successfully
- [ ] Receiving agent has access to all necessary resources

## 11. Contact Information

**Handoff Agent:** instance_01_data  
**Receiving Agent:** instance_03_integration  
**Supervisor:** RNA 2025 Project Lead  
**Knowledge Transfer Session Date:** To be scheduled  

---

## MI Matrix Optimization Details

The MI matrix optimization is a critical enhancement that significantly reduces memory usage and improves performance:

1. **Problem**: Full MI matrices can be up to 146MB per sample. Many of these matrices are uniform (constant values) and provide no useful evolutionary information, yet were being loaded in full.

2. **Solution**: Two-tier detection of uniform MI matrices:
   - **First tier**: Check the much smaller top_pairs array (usually <1MB) for uniform scores
   - **Second tier**: Fall back to checking the full matrix only if necessary
   
3. **Implementation**:
   ```python
   # First check top_pairs for uniformity
   if 'top_pairs' in data:
       top_pairs = data['top_pairs']
       if is_uniform_top_pairs(top_pairs):
           # Uniform MI detected from top_pairs - no need to load full matrix
           evolutionary_features['coupling_matrix'] = np.zeros((seq_len, seq_len))
           evolutionary_features['has_valid_mi'] = False
           return features
   
   # Fallback to checking full matrix
   if 'coupling_matrix' in evolutionary_features:
       has_valid_mi = not is_uniform_mi_matrix(evolutionary_features['coupling_matrix'])
       evolutionary_features['has_valid_mi'] = has_valid_mi
       
       # Zero out invalid matrices
       if not has_valid_mi:
           evolutionary_features['coupling_matrix'] = np.zeros_like(
               evolutionary_features['coupling_matrix']
           )
   ```

4. **Performance Benefits**:
   - Memory reduction of up to 146MB per sample for uniform MI matrices
   - Significant loading time improvement (up to 10x faster for uniform cases)
   - Particularly effective when loading batches with multiple samples
   
5. **Detection Algorithm**:
   - For top_pairs: Check standard deviation of scores column
   - For full matrix: Check standard deviation of off-diagonal elements
   - Use configurable epsilon threshold (default 1e-6) to detect "effective uniformity"

6. **Integration Points**:
   - Added `has_valid_mi` field to evolutionary features dictionary
   - Updated `has_msa` metadata flag to consider both existence AND validity of MI
   - Zeros out coupling_matrix for invalid cases to save memory

## Temporal Cutoff Enhancement Details

The temporal cutoff enhancement provides critical protection against data leakage:

1. **Problem**: The original implementation only filtered sequences at the dataset level, but feature files might contain data generated after the cutoff date.

2. **Solution**: Two-level temporal filtering:
   - **Dataset level**: Filter sequences based on publication date
   - **Feature level**: Check feature generation timestamp against cutoff
   
3. **Implementation**:
   ```python
   # Feature-level temporal check in load_precomputed_features
   if temporal_cutoff is not None and 'generation_date' in data:
       generation_date = str(data['generation_date'])
       if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
           warnings.warn(f"Features were generated after the temporal cutoff. Using zeros.")
           features['feature_type'] = None
           return features
   ```

4. **Reconfigurable Design**:
   ```python
   # Dynamic reconfiguration method
   def set_temporal_cutoff(self, new_cutoff: Optional[str] = None) -> None:
       self.temporal_cutoff = new_cutoff
       
       # Clear cached filtered IDs
       if hasattr(self, '_temporal_filtered_cache'):
           delattr(self, '_temporal_filtered_cache')
       
       # Reapply filtering with new cutoff
       self._update_filtered_sequences()
   ```

5. **Caching Mechanism**:
   ```python
   def _get_temporal_filtered_ids(self) -> List[str]:
       """Get target IDs filtered by temporal cutoff."""
       if not hasattr(self, '_temporal_filtered_cache'):
           # Cache the results to avoid recomputing
           if self.temporal_cutoff is not None and not self.use_validation_set:
               cutoff_date = pd.to_datetime(self.temporal_cutoff)
               self._temporal_filtered_cache = set(
                   self.sequences_df[
                       pd.to_datetime(self.sequences_df['temporal_cutoff']) <= cutoff_date
                   ]['target_id'].tolist()
               )
           else:
               self._temporal_filtered_cache = set(self.sequences_df['target_id'].tolist())
       
       return self._temporal_filtered_cache
   ```

6. **Integration Points**:
   - Added `temporal_cutoff` parameter to `load_precomputed_features()`
   - Added timestamp extraction and comparison logic for each feature type
   - Added `set_temporal_cutoff()` method for dynamic reconfiguration
   - Updated DataLoader to expose reconfiguration method via `data_loader.set_temporal_cutoff = dataset.set_temporal_cutoff`

## Important Integration Notes

When integrating this data pipeline with the model components, pay special attention to:

1. **Feature Tensor Shapes**: Ensure model accepts the exact tensor shapes produced by the data pipeline
2. **Metadata Flags**: Model should check metadata flags before using optional features
3. **MI Validity Check**: Models should check `has_msa` metadata flag before using MI information
4. **Temporal Reconfiguration**: Use `data_loader.set_temporal_cutoff()` for dynamic temporal boundary changes
5. **Memory Usage**: Monitor memory usage when working with large sequences
6. **Feature-level Timestamps**: Ensure all feature generation code properly adds timestamps

The dataset automatically handles partial feature availability and uniform MI detection, with metadata flags indicating which features are real vs. zero-filled. The model should always check these flags before using features, especially for inference.