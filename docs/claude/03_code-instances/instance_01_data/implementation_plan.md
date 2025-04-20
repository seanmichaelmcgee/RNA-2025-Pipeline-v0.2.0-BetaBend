# Data Pipeline Implementation Plan

## Overview

This implementation plan outlines the development strategy for the RNA 3D folding data pipeline. The plan addresses core requirements including partial data handling, feature availability detection, and V2 readiness while maintaining strict path parameterization.

## Timeline and Milestones

### Phase 1: Foundation & Core Feature Detection (Week 1)

#### Milestone 1.1: Core Feature Detection (Days 1-2)
- [  ] Create directory and file structure for data loading components
- [  ] Implement path validation and resolution utilities
- [  ] Develop `check_features_availability()` function
- [  ] Create sequence conversion utilities (`sequence_to_int`)
- [  ] Add unit tests for feature detection
- [  ] Document feature detection interface

#### Milestone 1.2: Feature Loading System (Days 3-5)
- [  ] Implement `load_coordinates()` function
- [  ] Create `load_precomputed_features()` for all feature types
- [  ] Implement `get_dihedral_tensors()` with future-proof design
- [  ] Add comprehensive error handling for missing features
- [  ] Create tensor conversion utilities with shape verification
- [  ] Add unit tests for feature loading
- [  ] Document feature loading interfaces

### Phase 2: Dataset Implementation & Partial Data Handling (Week 2)

#### Milestone 2.1: RNA Dataset Core (Days 1-3)
- [  ] Create `RNADataset` class with path parameterization
- [  ] Implement `__init__` with pluggable split function
- [  ] Develop feature availability filtering logic
- [  ] Add caching system for efficient repeated access
- [  ] Implement `__len__` and basic `__getitem__`
- [  ] Create dataset update mechanism
- [  ] Add unit tests for dataset initialization
- [  ] Document dataset interface

#### Milestone 2.2: Feature Integration (Days 4-5)
- [  ] Expand `__getitem__` to load all feature types
- [  ] Generate metadata flags for feature presence
- [  ] Implement shape consistency validation
- [  ] Add tensor conversion with proper device handling
- [  ] Create comprehensive error handling for edge cases
- [  ] Add unit tests for feature integration
- [  ] Document feature integration interfaces

### Phase 3: Batch Processing & Integration (Week 3)

#### Milestone 3.1: Padding Utilities (Days 1-2)
- [  ] Create `src/utils/padding.py` module
- [  ] Implement `pad_1d` for 1D tensor padding
- [  ] Implement `pad_2d` for 2D tensor padding
- [  ] Implement generalized `pad_tensor` for N-dimensional tensors
- [  ] Develop memory-efficient padding strategies
- [  ] Add unit tests for padding utilities
- [  ] Document padding utility interfaces

#### Milestone 3.2: Batch Collation (Days 3-4)
- [  ] Implement `collate_fn` using padding utilities
- [  ] Generate attention masks for transformer compatibility
- [  ] Aggregate metadata in batches
- [  ] Handle tensor stacking with non-tensor items
- [  ] Add unit tests for batch collation
- [  ] Document batch collation interface

#### Milestone 3.3: DataLoader Integration (Day 5)
- [  ] Create `create_data_loader` function
- [  ] Add feature filtering options
- [  ] Incorporate distributed training support
- [  ] Create end-to-end tests for data loading pipeline
- [  ] Document DataLoader interface

### Phase 4: Comprehensive Testing & Documentation (Week 4)

#### Milestone 4.1: Edge Case Testing (Days 1-2)
- [  ] Test variable-length sequence handling
- [  ] Test mixed-presence feature scenarios
- [  ] Test sequences with no features
- [  ] Test feature update mechanism
- [  ] Test extremely long sequences
- [  ] Test memory efficiency
- [  ] Document test results

#### Milestone 4.2: Integration Testing (Day 3)
- [  ] Create integration smoke test
- [  ] Verify tensor shapes match specifications
- [  ] Validate partial data handling
- [  ] Test end-to-end pipeline with real data
- [  ] Document integration test results

#### Milestone 4.3: Documentation & Handoff (Days 4-5)
- [  ] Complete implementation journal
- [  ] Update interface documentation
- [  ] Create examples for component usage
- [  ] Prepare handoff package
- [  ] Draft final documentation

## Risk Assessment

### High Priority Risks

1. **Partial Data Handling Complexity**
   - **Risk**: Feature availability detection and filtering may become complex
   - **Mitigation**: Implement modular system with caching, clear interfaces
   - **Fallback**: Simplify to all-or-nothing approach if necessary

2. **Memory Usage for Large Sequences**
   - **Risk**: Large RNA sequences (500+ nucleotides) may cause memory issues
   - **Mitigation**: Implement memory-efficient padding, early warnings
   - **Fallback**: Add configurable sequence length limits

3. **Path Parameterization Violations**
   - **Risk**: Accidental hardcoded paths may be introduced
   - **Mitigation**: Strict code review, static analysis, comprehensive tests
   - **Fallback**: Automated scanning for path string patterns

### Medium Priority Risks

1. **Feature Format Changes**
   - **Risk**: Feature file formats may evolve during development
   - **Mitigation**: Version-aware loaders, extensible parsing
   - **Fallback**: Feature format adapters

2. **Cross-Instance Dependencies**
   - **Risk**: Interface changes may impact other instances
   - **Mitigation**: Clear interface contracts, early communication
   - **Fallback**: Backward compatibility layers

3. **Kaggle Compatibility Issues**
   - **Risk**: Kaggle environment may differ from development
   - **Mitigation**: Test with Kaggle-like constraints, no local dependencies
   - **Fallback**: Simplified Kaggle-specific version

## Critical Implementation Patterns

### 1. Feature Availability Tracking

```python
def check_features_availability(target_id: str, features_dir: str) -> Dict[str, bool]:
    """Check which features are available for a given target.
    
    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories
        
    Returns:
        Dictionary mapping feature types to availability (True/False)
    """
    availability = {
        "dihedral": False,
        "thermo": False,
        "mi": False
    }
    
    # Check each feature type
    for feature_type in availability.keys():
        feature_path = os.path.join(
            features_dir, 
            f"{feature_type}_features", 
            f"{target_id}_{feature_type}_features.npz"
        )
        availability[feature_type] = os.path.exists(feature_path)
        
    return availability
```

### 2. Dataset Update Mechanism

```python
def update_available_features(self):
    """Update the list of available features.
    
    Rescans the features directory to identify newly available feature files
    and updates the filtered sequence list accordingly.
    
    Returns:
        Number of sequences with complete feature sets
    """
    # Reset availability cache
    self._availability_cache = {}
    
    # Check availability for all sequences
    for target_id in self.target_ids:
        self._availability_cache[target_id] = check_features_availability(
            target_id, self.features_dir)
            
    # Update filtered sequences based on requirements
    self._update_filtered_sequences()
    
    # Return number of valid sequences
    return len(self.filtered_sequences)
```

### 3. Memory-Efficient Padding

```python
def pad_2d(tensor: torch.Tensor, max_len: int, pad_value: float = 0) -> torch.Tensor:
    """Memory-efficient 2D tensor padding.
    
    Args:
        tensor: Input tensor of shape (L, L)
        max_len: Target length for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len, max_len)
    """
    # Handle empty case
    if tensor.shape[0] == 0:
        return torch.full((max_len, max_len), pad_value, 
                        dtype=tensor.dtype, device=tensor.device)
    
    # Pre-allocate full tensor instead of concatenating (memory efficient)
    result = torch.full((max_len, max_len), pad_value, 
                       dtype=tensor.dtype, device=tensor.device)
                       
    # Copy only what's needed
    src_len = min(tensor.shape[0], max_len)
    result[:src_len, :src_len] = tensor[:src_len, :src_len]
    
    return result
```

This implementation plan provides a structured approach to developing the data pipeline with clear milestones, risk assessment, and critical implementation patterns.