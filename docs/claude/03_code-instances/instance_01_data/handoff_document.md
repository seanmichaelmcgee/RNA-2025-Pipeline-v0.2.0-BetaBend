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
| load_precomputed_features() | Complete | Complete | Complete | 2025-04-20 |
| get_dihedral_tensors() | Complete | Complete | Complete | 2025-04-20 |
| padding utilities | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.\_\_init\_\_() | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.\_\_getitem\_\_() | Complete | Complete | Complete | 2025-04-20 |
| RNADataset.update_available_features() | Complete | Complete | Complete | 2025-04-20 |
| collate_fn() | Complete | Complete | Complete | 2025-04-20 |
| create_data_loader() | Complete | Complete | Complete | 2025-04-20 |

### 2.2 Pending Components

None - all components have been implemented and tested.

### 2.3 Known Issues

| Issue | Severity | Components Affected | Potential Solutions |
|-------|----------|---------------------|---------------------|
| Memory usage with large sequences | Medium | collate_fn(), pad_2d() | Consider implementing sparse tensor support for pairing_probs and coupling_matrix matrices |
| Warnings for missing features | Low | load_precomputed_features() | Consider silent mode parameter to suppress warnings in production |
| Cache invalidation for feature updates | Low | update_available_features() | Implement timestamp-based cache invalidation |

## 3. Interface Contracts

### 3.1 Public API

See `interface_exports.md` for complete API documentation. Key interfaces:

```python
def create_data_loader(
    sequences_csv_path: str,
    labels_csv_path: Optional[str] = None,
    features_dir: str = "",
    batch_size: int = 32,
    split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    temporal_cutoff: Optional[str] = None,
    use_validation_set: bool = False,
    require_features: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False
) -> torch.utils.data.DataLoader:
    """Create data loader for RNA structure prediction."""
```

### 3.2 Data Structures

Key data structure is the batch dictionary returned by DataLoader:

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
        'has_msa': torch.Tensor,            # Shape: (batch_size,)
    }
}
```

### 3.3 Integration Points

| Consumer Component | Integration Function | Expected Behavior | Error Handling |
|-------------------|---------------------|-------------------|----------------|
| Model Training | create_data_loader() | Returns DataLoader that yields batches with all specified tensors | Raises specific exceptions for missing files or features |
| Model Inference | create_data_loader() with labels_csv_path=None | Returns DataLoader without coordinate tensors | Warns about missing features but continues with zero tensors |
| Visualization | RNADataset.\_\_getitem\_\_() | Returns sample dictionary with all features for a single sequence | Raises exceptions for critical missing features |

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
| Dataset | Yes | Yes | No |
| Collation | Yes | Yes | No |
| Data loader | Yes | Yes | No |
| Padding | Yes | Yes | No |

### 5.2 Critical Test Cases

| Test Case | Purpose | Command | Expected Output |
|-----------|---------|---------|----------------|
| test_data_loading.py | Comprehensive test suite | `python -m pytest tests/test_data_loading.py -v` | All tests pass |
| test_padding.py | Tests for padding utilities | `python -m pytest tests/test_padding.py -v` | All tests pass |
| TestIntegration.test_pipeline_smoke | End-to-end test | `python -m pytest tests/test_data_loading.py::TestIntegration::test_pipeline_smoke -v` | Test pass |

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

## 6. Implementation Details

### 6.1 Architecture Overview

The data pipeline follows this architecture:

```
Feature Detection (check_features_availability)
           ↓
Feature Loading (load_precomputed_features, load_coordinates)
           ↓
Dataset (RNADataset.__getitem__, RNADataset.update_available_features)
           ↓
Collation (collate_fn, padding utilities)
           ↓
DataLoader (create_data_loader)
```

### 6.2 Algorithms and Data Structures

- **Feature Availability Detection**: File system-based detection with caching
- **Partial Data Handling**: Metadata flag propagation with zero-filled tensors
- **Variable-Length Sequences**: Memory-efficient padding with mask tracking
- **Batch Creation**: Dictionary-based with consistent tensor shapes

### 6.3 Performance Considerations

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

## 9. Decision Log

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|-------------------------|------|
| Metadata flags for feature presence | Allows models to handle varying feature availability | Feature filtering at dataset level | 2025-04-20 |
| Memory-efficient padding | Reduces memory usage for variable-length sequences | Fixed-size tensors | 2025-04-20 |
| Directory-based feature organization | Simple structure with path parameterization | Database storage, single file per feature type | 2025-04-20 |
| NPZ file format | Simple, familiar format with good compatibility | HDF5, custom binary format | 2025-04-20 |

## 10. Handoff Checklist

- [x] All code pushed to repository
- [x] All tests passing
- [x] Documentation updated
- [x] Interface contracts finalized
- [x] Known issues documented
- [x] Integration points verified
- [x] Handoff template completed
- [ ] Knowledge transfer session completed
- [ ] Receiving agent has run tests successfully
- [ ] Receiving agent has access to all necessary resources

## 11. Contact Information

**Handoff Agent:** instance_01_data  
**Receiving Agent:** instance_03_integration  
**Supervisor:** RNA 2025 Project Lead  
**Knowledge Transfer Session Date:** To be scheduled  

---

## Important Integration Notes

When integrating this data pipeline with the model components, pay special attention to:

1. **Feature Tensor Shapes**: Ensure model accepts the exact tensor shapes produced by the data pipeline
2. **Metadata Flags**: Model should check metadata flags before using optional features
3. **Missing Feature Handling**: Implement fallback strategies for missing features
4. **Padding Mask**: Use the mask tensor to ignore padding in loss calculations
5. **Memory Usage**: Monitor memory usage when working with large sequences

The dataset automatically handles partial feature availability, with metadata flags indicating which features are real vs. zero-filled. The model should check these flags before using features, especially for inference.